import sys
from pathlib import Path
from termcolor import colored
import json

from utils.manifest.types import _Manifest, AnyManifest
from utils.manifest.builder import (
    build_tts_manifest,
    build_vc_manifest,
    build_vg_manifest,
    build_ie_manifest,
)
from utils.manifest.parser import parse_manifest_entry


def _expand_manifest(
    manifest: _Manifest, tts_dir: Path, vc_dir: Path, ie_dir: Path
) -> list[AnyManifest]:
    manifests: list[AnyManifest] = []

    if manifest.tts_text and manifest.tts_language and manifest.tts_target_emotion:
        tts_manifest = build_tts_manifest(
            text=manifest.tts_text,
            language=manifest.tts_language,
            target_emotion=manifest.tts_target_emotion,
            tts_dir=tts_dir,
        )
        manifest.vc_source_audio_path = tts_manifest.output_path
        manifest.vc_source_audio_text = manifest.tts_text
        manifests.append(tts_manifest)

    vc_manifest = build_vc_manifest(
        source_audio_path=manifest.vc_source_audio_path,
        source_audio_text=manifest.vc_source_audio_text,
        voice_reference_path=manifest.vc_voice_reference_path,
        voice_reference_text=manifest.vc_voice_reference_text,
        vc_dir=vc_dir,
    )
    manifest.vg_audio_path = vc_manifest.output_path
    manifests.append(vc_manifest)

    if manifest.ie_face_reference_path and manifest.ie_target_emotion:
        ie_manifest = build_ie_manifest(
            face_reference_path=manifest.ie_face_reference_path,
            target_emotion=manifest.ie_target_emotion,
            ie_dir=ie_dir,
        )
        manifest.vg_face_reference_path = ie_manifest.output_path
        manifests.append(ie_manifest)

    vg_manifest = build_vg_manifest(
        audio_path=manifest.vg_audio_path,
        face_reference_path=manifest.vg_face_reference_path,
        target_emotion=manifest.vg_target_emotion,
        output_path=manifest.output_path,
    )
    manifests.append(vg_manifest)

    return manifests


def process_manifest(
    file: Path, tts_dir: Path, vc_dir: Path, ie_dir: Path
) -> list[AnyManifest]:
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
                        ie_dir=ie_dir,
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
