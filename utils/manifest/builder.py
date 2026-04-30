from utils.manifest.types import TTSManifest, VCManifest, VGManifest
from hashlib import sha256
from pathlib import Path


def build_tts_manifest(
    text: str, language: str, target_emotion: str, tts_dir: Path
) -> TTSManifest:
    text_hash = sha256(text.encode()).hexdigest()[:5]
    filename = f"{target_emotion}.{text_hash}.wav"
    output_path = tts_dir / filename

    return TTSManifest(
        text=text,
        language=language,
        target_emotion=target_emotion,
        output_path=output_path.resolve(),
    )


def build_vc_manifest(
    source_audio_path: Path,
    voice_reference_path: Path,
    vc_dir: Path,
    source_audio_text: str | None = None,
    voice_reference_text: str | None = None,
) -> VCManifest:
    path_hash = sha256(str(source_audio_path).encode())
    path_hash.update(str(voice_reference_path).encode())
    path_hash = path_hash.hexdigest()[:5]

    filename = (
        f"{source_audio_path.stem}_to_{voice_reference_path.stem}.{path_hash}.wav"
    )
    output_path = vc_dir / filename
    return VCManifest(
        source_audio_path=source_audio_path.resolve(),
        source_audio_text=source_audio_text,
        voice_reference_path=voice_reference_path.resolve(),
        voice_reference_text=voice_reference_text,
        output_path=output_path.resolve(),
    )


def build_vg_manifest(
    audio_path: Path,
    face_reference_path: Path,
    output_path: Path,
    target_emotion: str | None = None,
) -> VGManifest:
    return VGManifest(
        audio_path=audio_path.resolve(),
        face_reference_path=face_reference_path.resolve(),
        output_path=output_path.resolve(),
        target_emotion=target_emotion,
    )
