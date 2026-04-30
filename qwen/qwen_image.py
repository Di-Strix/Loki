import os
import torch
from PIL import Image
from diffusers import (
    QwenImageEditPlusPipeline,
    GGUFQuantizationConfig,
    QwenImageTransformer2DModel,
)
from pathlib import Path
import json
from tqdm import tqdm
import traceback

from utils.manifest import IEManifest


def qwen_image(
    manifests: list[IEManifest],
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    emotion_prompts_path: Path = Path(__file__).parent / "image_emotion_prompts.json",
    overwrite_output: bool = False,
) -> list[tuple[IEManifest, str]]:

    transformer = QwenImageTransformer2DModel.from_single_file(
        "https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF/blob/main/qwen-image-edit-2511-Q6_K.gguf",
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        config="Qwen/Qwen-Image-Edit-2511",
        subfolder="transformer",
        torch_dtype=dtype,
    )

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        transformer=transformer,
        torch_dtype=dtype,
    )
    pipeline.enable_model_cpu_offload(device=device)
    pipeline.set_progress_bar_config(position=1, leave=False)

    with open(emotion_prompts_path) as f:
        prompts = json.load(f)

    errored_manifests: list[tuple[IEManifest, str]] = []
    for manifest in tqdm(manifests, desc="Image Edit", position=0):
        try:
            if manifest.target_emotion not in prompts:
                raise ValueError(
                    f"No prompt found for target emotion '{manifest.target_emotion}'. Known emotions: {list(dict(prompts).keys())}"
                )

            if manifest.output_path.exists() and not overwrite_output:
                tqdm.write(f"Skipping {manifest.output_path}: file already exists")
                continue

            image = Image.open(manifest.face_reference_path)
            with torch.inference_mode():
                output = pipeline(
                    image=image,
                    width=image.size[0],
                    height=image.size[1],
                    prompt=prompts[manifest.target_emotion],
                    generator=torch.manual_seed(0),
                    true_cfg_scale=4.0,
                    negative_prompt=" ",
                    num_inference_steps=40,
                    # guidance_scale=1.0,
                    num_images_per_prompt=1,
                )

            os.makedirs(manifest.output_path.parent, exist_ok=True)
            output_image = output.images[0]
            output_image.save(manifest.output_path)

        except Exception:
            exception = traceback.format_exc()
            errored_manifests.append((manifest, exception))
            tqdm.write(exception)

    return errored_manifests
