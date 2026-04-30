from pathlib import Path


class _Manifest:
    tts_text: str | None = None
    tts_language: str | None = None
    tts_target_emotion: str | None = None
    vc_source_audio_path: Path
    vc_source_audio_text: str | None = None
    vc_voice_reference_path: Path
    vc_voice_reference_text: str | None = None
    ie_face_reference_path: Path | None = None
    ie_target_emotion: str | None = None
    vg_audio_path: Path
    vg_face_reference_path: Path
    vg_target_emotion: str | None
    output_path: Path


class TTSManifest:
    text: str
    language: str
    target_emotion: str
    output_path: Path

    def __init__(
        self, text: str, language: str, target_emotion: str, output_path: Path
    ):
        self.text = text
        self.language = language
        self.target_emotion = target_emotion
        self.output_path = output_path


class VCManifest:
    source_audio_path: Path
    source_audio_text: str | None
    voice_reference_path: Path
    voice_reference_text: str | None
    output_path: Path

    def __init__(
        self,
        source_audio_path: Path,
        voice_reference_path: Path,
        output_path: Path,
        source_audio_text: str | None = None,
        voice_reference_text: str | None = None,
    ):
        self.source_audio_path = source_audio_path
        self.source_audio_text = source_audio_text
        self.voice_reference_path = voice_reference_path
        self.voice_reference_text = voice_reference_text
        self.output_path = output_path


class VGManifest:
    def __init__(self, audio_path: Path, face_reference_path: Path, output_path: Path, target_emotion: str | None = None):
        self.audio_path = audio_path
        self.face_reference_path = face_reference_path
        self.output_path = output_path
        self.target_emotion = target_emotion

class IEManifest:
    def __init__(self, face_reference_path: Path, target_emotion: str, output_path: Path,):
        self.face_reference_path = face_reference_path
        self.output_path = output_path
        self.target_emotion = target_emotion


type AnyManifest = TTSManifest | VCManifest | IEManifest | VGManifest
