import os
import logging
import traceback
from pathlib import Path

import torch
import soundfile as sf
from tqdm import tqdm
from audio_separator.separator import Separator

from amphion.models.vc.vevo.vevo_utils import (
    VevoInferencePipeline,
    save_audio as v1_save_audio,
)
from amphion.models.svc.vevo2.vevo2_utils import (
    Vevo2InferencePipeline,
    save_audio as v2_save_audio,
)

from project_root import PROJECT_ROOT
from utils.manifest import VCManifest
from vevo.download_models import download_vevo_ckpts, download_vevo2_ckpts


def load_vocal_separator(model: Path, cache_dir: Path) -> Separator:
    vocal_separator = Separator(
        log_level=logging.WARNING,
        output_single_stem="vocals",
        output_dir=cache_dir,
        model_file_dir=str(model.parent),
    )
    vocal_separator.load_model(model.name)
    assert vocal_separator.model_instance is not None, (
        "Failed to load audio separation model."
    )

    return vocal_separator


def separate_audio(separator: Separator, audio: Path) -> Path:
    outputs = separator.separate(str(audio))
    assert len(outputs) > 0, "Audio separation failed."
    return Path(separator.output_dir) / outputs[0]


def vevo_voice(
    manifests: list[VCManifest],
    device: str | torch.device,
    overwrite_output: bool = False,
    cache_dir: Path = PROJECT_ROOT() / "vevo-cache",
) -> list[tuple[VCManifest, str]]:
    ckpts = download_vevo_ckpts()

    inference_pipeline = VevoInferencePipeline(
        content_tokenizer_ckpt_path=ckpts.content_tokenizer_ckpt_path,
        content_style_tokenizer_ckpt_path=ckpts.content_style_tokenizer_ckpt_path,
        ar_cfg_path=ckpts.ar_cfg_path,
        ar_ckpt_path=ckpts.ar_ckpt_path,
        fmt_cfg_path=ckpts.fmt_cfg_path,
        fmt_ckpt_path=ckpts.fmt_ckpt_path,
        vocoder_cfg_path=ckpts.vocoder_cfg_path,
        vocoder_ckpt_path=ckpts.vocoder_ckpt_path,
        device=device,
    )

    vocal_separator = load_vocal_separator(ckpts.vocal_separator_path, cache_dir)
    separation_cache: dict[Path, Path] = dict()

    errored_manifests: list[tuple[VCManifest, str]] = []
    for manifest in tqdm(manifests, desc="VC"):
        try:
            if manifest.output_path.exists() and not overwrite_output:
                tqdm.write(f"Skipping {manifest.output_path}: file already exists")
                continue

            if manifest.voice_reference_path not in separation_cache:
                with tqdm.external_write_mode():
                    separation_cache[manifest.voice_reference_path] = separate_audio(
                        vocal_separator, manifest.voice_reference_path
                    )
            voice_reference = separation_cache[manifest.voice_reference_path]

            gen_sample_rate = 24000
            for i in range(5):
                if i > 0:
                    tqdm.write(f"Attempting voice conversion. Attempt ({i + 1}/5)")
    
                gen_audio = inference_pipeline.inference_fm(
                    src_wav_path=manifest.source_audio_path,
                    timbre_ref_wav_path=voice_reference,
                    flow_matching_steps=256,
                )

                # Hallucinates sometimes
                # gen_audio = inference_pipeline.inference_ar_and_fm(
                #     src_wav_path=manifest.source_audio_path,
                #     src_text=None,
                #     style_ref_wav_path=manifest.source_audio_path,
                #     style_ref_wav_text=manifest.source_audio_text,
                #     timbre_ref_wav_path=voice_reference,
                #     use_global_guided_inference=False,
                #     flow_matching_steps=32,
                # )

                source_duration = sf.info(manifest.source_audio_path).duration
                if (
                    abs(len(gen_audio.squeeze(0)) / gen_sample_rate - source_duration)
                    < source_duration * 0.1
                ):
                    break
            else:
                raise RuntimeError(
                    "Could not convert input within 5 attempts. The resulting audio length is off by too much"
                )

            silence = torch.zeros(int(1 * gen_sample_rate), dtype=gen_audio.dtype)
            padded_audio = torch.concatenate(
                [silence, gen_audio.squeeze(0), silence]
            ).unsqueeze(0)

            os.makedirs(manifest.output_path.parent, exist_ok=True)
            v1_save_audio(
                padded_audio, sr=gen_sample_rate, output_path=manifest.output_path
            )
        except Exception:
            exception = traceback.format_exc()
            errored_manifests.append((manifest, exception))
            print(exception)

    return errored_manifests


def vevo_voice2(
    manifests: list[VCManifest],
    device: str | torch.device,
    overwrite_output: bool = False,
    cache_dir: Path = PROJECT_ROOT() / "vevo-cache",
) -> list[tuple[VCManifest, str]]:
    ckpts = download_vevo2_ckpts()

    inference_pipeline = Vevo2InferencePipeline(
        content_style_tokenizer_ckpt_path=ckpts.content_style_tokenizer_ckpt_path,
        prosody_tokenizer_ckpt_path=ckpts.prosody_tokenizer_ckpt_path,
        fmt_cfg_path=ckpts.fmt_cfg_path,
        fmt_ckpt_path=ckpts.fmt_ckpt_path,
        ar_cfg_path=ckpts.ar_cfg_path,
        ar_ckpt_path=ckpts.ar_ckpt_path,
        vocoder_cfg_path=ckpts.vocoder_cfg_path,
        vocoder_ckpt_path=ckpts.vocoder_ckpt_path,
        device=device,
    )

    vocal_separator = load_vocal_separator(ckpts.vocal_separator_path, cache_dir)
    separation_cache: dict[Path, Path] = dict()

    errored_manifests: list[tuple[VCManifest, str]] = []
    for manifest in tqdm(manifests, desc="VC"):
        try:
            if manifest.output_path.exists() and not overwrite_output:
                tqdm.write(f"Skipping {manifest.output_path}: file already exists")
                continue

            if manifest.voice_reference_path not in separation_cache:
                with tqdm.external_write_mode():
                    separation_cache[manifest.voice_reference_path] = separate_audio(
                        vocal_separator, manifest.voice_reference_path
                    )

            voice_reference = separation_cache[manifest.voice_reference_path]
            gen_sample_rate = 24000
            for i in range(5):
                if i > 0:
                    tqdm.write(f"Attempting voice conversion. Attempt ({i + 1}/5)")
    
                gen_audio = inference_pipeline.inference_fm(
                    src_wav_path=manifest.source_audio_path,
                    src_wav_text=manifest.source_audio_text or "",
                    timbre_ref_wav_path=voice_reference,
                    flow_matching_steps=64,
                    use_pitch_shift=True,
                    whisper_spec_perturb=True,
                )

                # Hallucinates too much
                # gen_audio = inference_pipeline.inference_ar_and_fm(
                #     target_text=manifest.source_audio_text,
                #     prosody_wav_path=manifest.source_audio_path,
                #     style_ref_wav_path=manifest.source_audio_path,
                #     style_ref_wav_text=manifest.source_audio_text or "",
                #     timbre_ref_wav_path=manifest.voice_reference_path,
                #     use_prosody_code=True,
                #     use_pitch_shift=True,
                #     temperature=1,
                #     top_k=25,
                #     top_p=0.8,
                #     flow_matching_steps=32,
                # )

                source_duration = sf.info(manifest.source_audio_path).duration
                if (
                    abs(len(gen_audio.squeeze(0)) / gen_sample_rate - source_duration)
                    < source_duration * 0.1
                ):
                    break
            else:
                raise RuntimeError(
                    "Could not convert input within 5 attempts. The resulting audio length is off by too much"
                )

            silence = torch.zeros(int(1 * gen_sample_rate), dtype=gen_audio.dtype)
            padded_audio = torch.concatenate(
                [silence, gen_audio.squeeze(0), silence]
            ).unsqueeze(0)

            os.makedirs(manifest.output_path.parent, exist_ok=True)
            v2_save_audio(
                padded_audio, sr=gen_sample_rate, output_path=manifest.output_path
            )
        except Exception:
            exception = traceback.format_exc()
            errored_manifests.append((manifest, exception))
            print(exception)

    return errored_manifests
