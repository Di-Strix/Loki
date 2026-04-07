import sys
from pathlib import Path
from termcolor import colored
import json

from utils.manifest.types import TTSManifest, VCManifest, VGManifest, _Manifest
from utils.manifest.builder import (
    build_tts_manifest,
    build_vc_manifest,
    build_vg_manifest,
)
from utils.manifest.parser import parse_manifest_entry


def _expand_manifest(
    manifest: _Manifest, tts_dir: Path, vc_dir: Path, file_prefix: str = ""
) -> list[TTSManifest | VCManifest | VGManifest]:
    if manifest.type == "TTS":
        tts_manifest = build_tts_manifest(
            text=manifest.tts_text,
            language=manifest.tts_language,
            target_emotion=manifest.tts_target_emotion,
            tts_dir=tts_dir,
            file_prefix=file_prefix,
        )
        vc_manifest = build_vc_manifest(
            source_audio_path=tts_manifest.output_path,
            source_audio_text=manifest.tts_text,
            voice_reference_path=manifest.vc_voice_reference_path,
            voice_reference_text=manifest.vc_voice_reference_text,
            vc_dir=vc_dir,
            file_prefix=file_prefix,
        )
        vg_manifest = build_vg_manifest(
            audio_path=vc_manifest.output_path,
            face_reference_path=manifest.vg_face_reference_path,
            output_path=manifest.output_path,
            target_emotion=manifest.vg_target_emotion,
        )
        return [tts_manifest, vc_manifest, vg_manifest]
    elif manifest.type == "VC":
        vc_manifest = build_vc_manifest(
            source_audio_path=manifest.vc_source_audio_path,
            source_audio_text=manifest.vc_source_audio_text,
            voice_reference_path=manifest.vc_voice_reference_path,
            voice_reference_text=manifest.vc_voice_reference_text,
            vc_dir=vc_dir,
            file_prefix=file_prefix,
        )
        vg_manifest = build_vg_manifest(
            audio_path=vc_manifest.output_path,
            face_reference_path=manifest.vg_face_reference_path,
            output_path=manifest.output_path,
            target_emotion=manifest.vg_target_emotion,
        )
        return [vc_manifest, vg_manifest]

    print("WARN: encountered manifest of type 'VideoGen', which is not expected")
    print(
        f"audio_path={manifest.vg_audio_path}, face_reference_path={manifest.vg_face_reference_path}, output_path={manifest.output_path}"
    )
    return [
        build_vg_manifest(
            audio_path=manifest.vg_audio_path,
            face_reference_path=manifest.vg_face_reference_path,
            output_path=manifest.output_path,
            target_emotion=manifest.vg_target_emotion,
        )
    ]


def process_manifest(
    file: Path, tts_dir: Path, vc_dir: Path
) -> list[TTSManifest | VCManifest | VGManifest]:
    manifests = []

    with open(file) as manifest_file:
        basepath = file.resolve().parent
        for index, line in enumerate(manifest_file.readlines()):
            try:
                manifest = parse_manifest_entry(json.loads(line), basepath=basepath)
                manifests.extend(
                    _expand_manifest(
                        manifest,
                        tts_dir=tts_dir,
                        vc_dir=vc_dir,
                        file_prefix=f"{index + 1:04d}",
                    )
                )
            except Exception:
                print(
                    colored(
                        f"\nError occurred while processing line {index + 1} of the manifest:",
                        color="red",
                        attrs=["bold"],
                    ),
                    file=sys.stderr,
                )
                raise

    return manifests
