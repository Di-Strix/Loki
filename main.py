from argparse import ArgumentParser
from pathlib import Path
from collections.abc import Sequence
from typing import Any
import torch
import shutil
from typing import Literal

from utils.manifest import (
    process_manifest,
    TTSManifest,
    VCManifest,
    IEManifest,
    VGManifest,
    AnyManifest,
)
from vevo.vevo_voice import vevo_voice, vevo_voice2
from qwen.qwen_tts import qwen_tts
from qwen.qwen_image import qwen_image
from memo.memo_vg import memo_vg
from project_root import PROJECT_ROOT
import gc


def filter_manifests[T: AnyManifest](
    target_type: type[T],
    candidates: Sequence[AnyManifest],
) -> list[T]:
    return list(filter(lambda x: isinstance(x, target_type), candidates))  # pyright: ignore[reportReturnType]


if __name__ == "__main__":
    parser = ArgumentParser(prog="Loki")
    parser.add_argument("manifest", type=Path, help="Path to the .jsonl manifest file")
    parser.add_argument(
        "--tts-dir",
        default=PROJECT_ROOT() / "outputs" / "tts",
        type=Path,
        help="Directory for text-to-speech synthesis results",
    )
    parser.add_argument(
        "--vc-dir",
        default=PROJECT_ROOT() / "outputs" / "vc",
        type=Path,
        help="Directory for voice conversion results",
    )
    parser.add_argument(
        "--ie-dir",
        default=PROJECT_ROOT() / "outputs" / "ie",
        type=Path,
        help="Directory for image editing results",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        type=str,
        help="Torch accelerator device",
    )
    parser.add_argument(
        "--overwrite_existing",
        default=False,
        action="store_true",
        help="Whether to skip rendering existing output files",
    )
    parser.add_argument(
        "--clean",
        nargs="?",
        const="all",
        default=False,
        choices=["all", "tts", "vc", "ie"],
        help="Clean directories: 'all', 'tts', 'vc' or 'ie'. Default is False. If flag is present, the default is 'all'",
    )

    args = parser.parse_args()

    manifest_path, tts_dir, vc_dir, ie_dir = (
        args.manifest.resolve(),
        args.tts_dir.resolve(),
        args.vc_dir.resolve(),
        args.ie_dir.resolve(),
    )
    device: str = args.device
    overwrite_existing: bool = args.overwrite_existing
    clean_start: Literal["all", "tts", "vc", "ie"] = args.clean

    if clean_start:
        if clean_start == "all" or clean_start == "tts":
            shutil.rmtree(tts_dir, ignore_errors=True)
        if clean_start == "all" or clean_start == "vc":
            shutil.rmtree(vc_dir, ignore_errors=True)
        if clean_start == "all" or clean_start == "ie":
            shutil.rmtree(ie_dir, ignore_errors=True)

    manifests = process_manifest(
        file=manifest_path, tts_dir=tts_dir, vc_dir=vc_dir, ie_dir=ie_dir
    )

    tts_tasks = filter_manifests(TTSManifest, manifests)
    failed_tts_tasks = qwen_tts(
        tts_tasks,
        device=device,
        attn_implementation="flash_attention_2",
        overwrite_output=overwrite_existing,
    )

    vc_tasks = filter_manifests(VCManifest, manifests)
    failed_vc_tasks = vevo_voice(
        vc_tasks, device=device, overwrite_output=overwrite_existing
    )

    ie_tasks = filter_manifests(IEManifest, manifests)
    failed_ie_tasks = qwen_image(
        ie_tasks, device=device, overwrite_output=overwrite_existing
    )

    gc.collect()
    torch.cuda.empty_cache()

    vg_tasks = filter_manifests(VGManifest, manifests)
    failed_vg_tasks = memo_vg(
        vg_tasks, device=device, overwrite_output=overwrite_existing
    )
