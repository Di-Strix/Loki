import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from pathlib import Path
import os
import json
from tqdm import tqdm
import traceback

from utils.manifest import TTSManifest


def qwen_tts(
    manifests: list[TTSManifest],
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
    emotion_prompts_path: Path = Path(__file__).parent / "emotion_prompts.json",
    overwrite_output: bool = False,
) -> list[tuple[TTSManifest, str]]:
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    # Languages ['auto', 'chinese', 'english', 'french', 'german', 'italian', 'japanese', 'korean', 'portuguese', 'russian', 'spanish']

    with open(emotion_prompts_path) as f:
        prompts = json.load(f)

    errored_manifests: list[tuple[TTSManifest, str]] = []
    for manifest in tqdm(manifests, desc="TTS"):
        try:
            if manifest.target_emotion not in prompts:
                raise ValueError(
                    f"No prompt found for target emotion '{manifest.target_emotion}'. Known emotions: {list(dict(prompts).keys())}"
                )

            if manifest.output_path.exists() and not overwrite_output:
                print(f"Skipping {manifest.output_path}: file already exists")
                continue

            wavs, sr = model.generate_custom_voice(
                text=manifest.text,
                language=manifest.language,
                speaker="Ryan",
                instruct=prompts[manifest.target_emotion],
            )

            os.makedirs(manifest.output_path.parent, exist_ok=True)
            sf.write(manifest.output_path, wavs[0], sr)
        except Exception:
            exception = traceback.format_exc()
            errored_manifests.append((manifest, exception))
            print(exception)

    return errored_manifests
