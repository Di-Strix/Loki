from utils.manifest.types import _Manifest
from pathlib import Path
from collections.abc import Mapping
from utils.path import resolve_path


def parse_manifest_entry(entry: object, basepath: Path) -> _Manifest:
    if not isinstance(entry, Mapping):
        raise TypeError("Manifest entry is not subscriptable")

    entry = dict(entry)

    output_path = entry.pop("output_path", None)
    voice_reference_path = entry.pop("voice_reference_path", None)
    voice_reference_text = entry.pop("voice_reference_text", None)
    face_reference_path = entry.pop("face_reference_path", None)
    source_audio_path = entry.pop("source_audio_path", None)
    source_audio_text = entry.pop("source_audio_text", None)
    text = entry.pop("tts_text", None)
    language = entry.pop("tts_language", None)
    tts_target_emotion = entry.pop("tts_target_emotion", None)
    adjust_face_reference_emotion = entry.pop("adjust_face_reference_emotion", None)
    vg_target_emotion = entry.pop("vg_target_emotion", None)

    if len(unexpected_keys := entry.keys()):
        print(f"Warning: found unexpected keys {list(unexpected_keys)}")

    manifest = _Manifest()

    assert output_path is not None, "'output_path' must be specified"
    output_path = resolve_path(output_path)
    manifest.output_path = basepath / output_path
    assert not manifest.output_path.is_dir() and manifest.output_path.suffix, (
        "'output_path' must have file extension and point to a file (not necessarily existent)"
    )

    assert voice_reference_path is not None, "'voice_reference_path' must be specified"
    voice_reference_path = resolve_path(voice_reference_path)
    manifest.vc_voice_reference_path = basepath / voice_reference_path
    manifest.vc_voice_reference_text = voice_reference_text
    assert manifest.vc_voice_reference_path.is_file(), (
        "'voice_reference_path' must point to a file"
    )

    assert face_reference_path is not None, "'face_reference_path' must be specified"
    face_reference_path = resolve_path(face_reference_path)
    manifest.vg_face_reference_path = basepath / face_reference_path
    assert manifest.vg_face_reference_path.is_file(), (
        "'face_reference_path' must point to a file"
    )

    assert vg_target_emotion is None or isinstance(vg_target_emotion , str)
    manifest.vg_target_emotion = vg_target_emotion

    assert adjust_face_reference_emotion is None or isinstance(adjust_face_reference_emotion , bool) or isinstance(adjust_face_reference_emotion , str), "'adjust_face_reference_emotion' must be either None, bool or string"
    if adjust_face_reference_emotion is None and vg_target_emotion is not None:
        # Enable by default, given prerequisites are satisfied
        adjust_face_reference_emotion = True 

    if adjust_face_reference_emotion is not None:
        manifest.ie_face_reference_path = face_reference_path

        if isinstance(adjust_face_reference_emotion, str):
            manifest.ie_target_emotion = adjust_face_reference_emotion
        elif adjust_face_reference_emotion is True:
            assert vg_target_emotion is not None, "if 'adjust_face_reference_emotion' is true, 'vg_target_emotion' must be specified"
            manifest.ie_target_emotion = vg_target_emotion
        else:
            manifest.ie_target_emotion = None

    if source_audio_path is not None:
        assert text is None and tts_target_emotion is None, (
            " Cannot specify 'source_path' AND 'tts_text'/'tts_target_emotion' simultaneously"
        )
        source_audio_path = resolve_path(source_audio_path)
        assert source_audio_path.is_file(), (
            "'source_audio_path' must point to a file"
        )

        manifest.vc_source_audio_path = basepath / source_audio_path
        manifest.vc_source_audio_text = source_audio_text
        return manifest
    else:
        assert (
            text is not None and tts_target_emotion is not None and language is not None
        ), "Provide either 'source_path' OR all 'tts_text', 'tts_target_emotion' and 'tts_language'"

        manifest.tts_text = str(text)
        manifest.tts_language = str(language)
        manifest.tts_target_emotion = str(tts_target_emotion)
        return manifest
