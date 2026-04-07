import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)

import torch
import argparse
from memo.inference import MemoInferenceModels, inference, EmotionCode
from pathlib import Path
import json
import traceback
import sys
import onnxruntime as ort


class ManifestEntry:
    def __init__(
        self,
        input_audio_path: Path,
        reference_image_path: Path,
        output_path: Path,
        original_entry: str,
        target_emotion: None | EmotionCode = None
    ):
        self.input_audio_path = input_audio_path
        self.reference_image_path = reference_image_path
        self.output_path = output_path
        self.original_entry = original_entry
        self.target_emotion = target_emotion


if __name__ == "__main__":
    # region Parser setup
    parser = argparse.ArgumentParser(description="Video-Generation pipeline")
    parser.add_argument("manifest", type=Path, help="Path to the .jsonl manifest file")
    parser.add_argument("--device", type=str, help="Torch accelerator device")
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default="bfloat16",
    )
    parser.add_argument(
        "--output_resolution",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--num_generated_frames_per_clip",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num_init_past_frames",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num_past_frames",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=3.5,
    )
    parser.add_argument(
        "--use_xformers",
        action="store_true",
    )
    args = parser.parse_args()
    # endregion

    # region Read parsed args
    manifest_file: Path = args.manifest.resolve()
    assert manifest_file.is_file()
    device = torch.device(args.device)
    overwrite: bool = args.overwrite
    seed: int = args.seed
    weight_dtype: torch.dtype = getattr(torch, args.weight_dtype)
    assert isinstance(weight_dtype, torch.dtype)
    output_resolution: int = args.output_resolution
    fps: int = args.fps
    num_generated_frames_per_clip: int = args.num_generated_frames_per_clip
    num_init_past_frames: int = args.num_init_past_frames
    num_past_frames: int = args.num_past_frames
    inference_steps: int = args.inference_steps
    cfg_scale: float = args.cfg_scale
    use_xformers = args.use_xformers
    # endregion

    # region Parse manifest
    manifests: list[ManifestEntry] = []
    with open(manifest_file) as f:
        for i, line in enumerate(f.readlines()):
            manifest_entry = json.loads(line)
            input_audio_path = manifest_entry["input_audio_path"]
            reference_image_path = manifest_entry["reference_image_path"]
            output_path = manifest_entry["output_path"]
            target_emotion = manifest_entry["target_emotion"]

            assert isinstance(input_audio_path, str), (
                f"line {i + 1}: input_audio_path must be a string"
            )
            assert isinstance(reference_image_path, str), (
                f"line {i + 1}: reference_image_path must be a string"
            )
            assert isinstance(output_path, str), (
                f"line {i + 1}: output_path must be a string"
            )

            input_audio_path, reference_image_path, output_path = (
                Path(input_audio_path),
                Path(reference_image_path),
                Path(output_path),
            )

            assert input_audio_path.is_file()
            assert reference_image_path.is_file()
            if target_emotion is not None:
              target_emotion = str(target_emotion).upper()
              assert target_emotion in EmotionCode.__members__, f"target_emotion must be one of the following: {EmotionCode.__members__.keys()}"
              target_emotion = EmotionCode[target_emotion]

            manifests.append(
                ManifestEntry(
                    input_audio_path=input_audio_path,
                    reference_image_path=reference_image_path,
                    output_path=output_path,
                    original_entry=line,
                    target_emotion=target_emotion
                )
            )
    # endregion

    # region Run inference
    errored_manifests: list[tuple[ManifestEntry, str]] = []
    models = MemoInferenceModels(onnxExecutionProviders=["CUDAExecutionProvider", "CPUExecutionProvider"])
    models.preload()
    for manifest in manifests:
        try:
            inference(
                input_image_path=manifest.reference_image_path,
                input_audio_path=manifest.input_audio_path,
                output_video_path=manifest.output_path,
                overwrite_output=overwrite,
                seed=seed,
                weight_dtype=weight_dtype,
                output_resolution=output_resolution,
                fps=fps,
                num_generated_frames_per_clip=num_generated_frames_per_clip,
                num_init_past_frames=num_init_past_frames,
                num_past_frames=num_past_frames,
                inference_steps=inference_steps,
                cfg_scale=cfg_scale,
                device=device,
                enable_xformers_memory_efficient_attention=use_xformers,
                models=models,
                force_emotion=manifest.target_emotion,
            )
        except Exception:
            exception = traceback.format_exc()
            errored_manifests.append((manifest, exception))
            print(exception)
    # endregion

    if len(errored_manifests):
        print(f"\n\nFAILED TO PROCESS {len(errored_manifests)} MANIFESTS")
        try:
            failed_manifests_path = manifest_file.parent / "memo-failed-manifests.jsonl"
            with open(failed_manifests_path, "w+") as f:
                    f.write(
                        '\n'.join(
                            [
                                json.dumps(
                                    dict(
                                        manifest=errored_manifest.original_entry,
                                        stacktrace=trace,
                                    )
                                )
                                for errored_manifest, trace in errored_manifests
                            ]
                        )
                    )
            print("Logs written to " + str(failed_manifests_path))
        except Exception:
            for errored_manifest, trace in errored_manifests:
                manifest = json.loads(errored_manifest.original_entry)
                manifest["stacktrace"] = trace
                print(json.dumps(manifest))

        sys.exit(-1)
