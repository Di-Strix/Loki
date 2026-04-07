# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from huggingface_hub import snapshot_download, hf_hub_download
from pathlib import Path
import amphion.models.vc.vevo.config as vevo_config
from project_root import PROJECT_ROOT


class ModelV1Ckpts:
    content_tokenizer_ckpt_path: Path
    content_style_tokenizer_ckpt_path: Path
    ar_cfg_path: Path
    ar_ckpt_path: Path
    fmt_cfg_path: Path
    fmt_ckpt_path: Path
    vocoder_cfg_path: Path
    vocoder_ckpt_path: Path
    vocal_separator_path: Path


class ModelV2Ckpts:
    content_style_tokenizer_ckpt_path: Path
    fmt_cfg_path: Path
    fmt_ckpt_path: Path
    prosody_tokenizer_ckpt_path: Path
    ar_cfg_path: Path
    ar_ckpt_path: Path
    vocoder_cfg_path: Path
    vocoder_ckpt_path: Path
    vocal_separator_path: Path

def download_vocal_separator(cache_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            repo_id="memoavatar/memo",
            filename="misc/vocal_separator/Kim_Vocal_2.onnx",
            cache_dir=cache_dir,
        )
    )

def download_vevo_ckpts() -> ModelV1Ckpts:
    cache_dir = PROJECT_ROOT() / "checkpoints"
    vevo_config_path = Path(vevo_config.__path__[0]).resolve()
    ckpts = ModelV1Ckpts()

    # ===== Content Tokenizer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir=cache_dir,
        allow_patterns=["tokenizer/vq32/*"],
    )
    ckpts.content_tokenizer_ckpt_path = (
        Path(local_dir) / "tokenizer/vq32/hubert_large_l18_c32.pkl"
    )

    # ===== Content-Style Tokenizer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir=cache_dir,
        allow_patterns=["tokenizer/vq8192/*"],
    )
    ckpts.content_style_tokenizer_ckpt_path = Path(local_dir) / "tokenizer/vq8192"

    # ===== Autoregressive Transformer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir=cache_dir,
        allow_patterns=["contentstyle_modeling/Vq32ToVq8192/*"],
    )

    ckpts.ar_cfg_path = vevo_config_path / "Vq32ToVq8192.json"
    ckpts.ar_ckpt_path = Path(local_dir) / "contentstyle_modeling/Vq32ToVq8192"

    # ===== Flow Matching Transformer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir=cache_dir,
        allow_patterns=["acoustic_modeling/Vq8192ToMels/*"],
    )

    ckpts.fmt_cfg_path = vevo_config_path / "Vq8192ToMels.json"
    ckpts.fmt_ckpt_path = Path(local_dir) / "acoustic_modeling/Vq8192ToMels"

    # ===== Vocoder =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir=cache_dir,
        allow_patterns=["acoustic_modeling/Vocoder/*"],
    )

    ckpts.vocoder_cfg_path = vevo_config_path / "Vocoder.json"
    ckpts.vocoder_ckpt_path = Path(local_dir) / "acoustic_modeling/Vocoder"

    ckpts.vocal_separator_path = download_vocal_separator(cache_dir)

    return ckpts


def download_vevo2_ckpts() -> ModelV2Ckpts:
    cache_dir = PROJECT_ROOT() / "ckpts"
    ckpts = ModelV2Ckpts()

    local_dir = snapshot_download(
        repo_id="RMSnow/Vevo2",
        repo_type="model",
        local_dir=cache_dir,
        resume_download=True,
    )

    ckpts.content_style_tokenizer_ckpt_path = (
        Path(local_dir) / "tokenizer" / "contentstyle_fvq16384_12.5hz"
    )
    ckpts.prosody_tokenizer_ckpt_path = (
        Path(local_dir) / "tokenizer" / "prosody_fvq512_6.25hz"
    )

    ckpts.fmt_cfg_path = (
        Path(local_dir)
        / "acoustic_modeling"
        / "fm_emilia101k_singnet7k_repa"
        / "config.json"
    )
    ckpts.fmt_ckpt_path = (
        Path(local_dir) / "acoustic_modeling" / "fm_emilia101k_singnet7k_repa"
    )

    ckpts.ar_cfg_path = (
        Path(local_dir)
        / "contentstyle_modeling"
        / "posttrained"
        / "amphion_config.json"
    )
    ckpts.ar_ckpt_path = Path(local_dir) / "contentstyle_modeling" / "posttrained"

    ckpts.vocoder_cfg_path = Path(local_dir) / "vocoder" / "config.json"
    ckpts.vocoder_ckpt_path = Path(local_dir) / "vocoder"

    ckpts.vocal_separator_path = download_vocal_separator(cache_dir)

    return ckpts


if __name__ == "__main__":
    download_vevo_ckpts()
    download_vevo2_ckpts()
