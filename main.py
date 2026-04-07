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
    VGManifest,
    AnyManifest,
    manifest_depends,
)
from vevo.vevo_voice import vevo_voice, vevo_voice2
from qwen.qwen_tts import qwen_tts
from memo.memo_vg import memo_vg
from project_root import PROJECT_ROOT
import gc


def depends_on_failed(
    manifest: AnyManifest,
    failed: Sequence[tuple[AnyManifest, Any]],
) -> bool:
    return any(
        manifest_depends(manifest, failed_manifest) for failed_manifest, _ in failed
    )


def filter_manifests[T: AnyManifest](
    target_type: type[T],
    candidates: Sequence[AnyManifest],
    # failed: Sequence[tuple[AnyManifest, Any]] = [],
) -> list[T]:
    # return list(
    #     filter(
    #         lambda x: isinstance(x, target_type) and not depends_on_failed(x, failed),
    #         candidates,
    #     )
    # )  # pyright: ignore[reportReturnType]

    return list(filter(lambda x: isinstance(x, target_type), candidates))  # pyright: ignore[reportReturnType]


if __name__ == "__main__":
    parser = ArgumentParser(prog="Loki")
    parser.add_argument("manifest", type=Path, help="Path to the .jsonl manifest file")
    parser.add_argument(
        "--tts-dir",
        default=PROJECT_ROOT() / "tts-outputs",
        type=Path,
        help="Directory for text-to-speech synthesis results",
    )
    parser.add_argument(
        "--vc-dir",
        default=PROJECT_ROOT() / "vc-outputs",
        type=Path,
        help="Directory for voice conversion results",
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
        choices=["all", "tts", "vc"],
        help="Clean directories: 'all', 'tts', or 'vc'. Default is False. If flag is present, the default is 'all'",
    )

    args = parser.parse_args()

    manifest_path, tts_dir, vc_dir = (
        args.manifest.resolve(),
        args.tts_dir.resolve(),
        args.vc_dir.resolve(),
    )
    device: str = args.device
    overwrite_existing: bool = args.overwrite_existing
    clean_start: Literal["all", "tts", "vc"] | bool = args.clean

    if clean_start:
        if clean_start == "all" or clean_start == "tts":
            shutil.rmtree(tts_dir, ignore_errors=True)
        if clean_start == "all" or clean_start == "vc":
            shutil.rmtree(vc_dir, ignore_errors=True)

    manifests = process_manifest(file=manifest_path, tts_dir=tts_dir, vc_dir=vc_dir)

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

    gc.collect()
    torch.cuda.empty_cache()

    vg_tasks = filter_manifests(VGManifest, manifests)
    failed_vg_tasks = memo_vg(
        vg_tasks, device=device, overwrite_output=overwrite_existing
    )
