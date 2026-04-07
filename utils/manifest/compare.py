from pathlib import Path
from utils.manifest.types import TTSManifest, VCManifest, VGManifest, AnyManifest


def get_manifest_input_output(
    m: AnyManifest,
) -> tuple[Path | None, Path]:
    match m:
        case TTSManifest():
            return None, m.output_path
        case VCManifest():
            return m.source_audio_path, m.output_path
        case VGManifest():
            return m.audio_path, m.output_path


def manifest_depends(a: AnyManifest, b: AnyManifest) -> bool:
    """
    Tests if `a` depends on `b`. I.e that `a` cannot run if `b` has failed

    Args:
        a (AnyManifest)
        b (AnyManifest)

    Returns:
        bool: True if `a` depends on `b`
    """
    processing_order: list[type[AnyManifest]] = [TTSManifest, VCManifest, VGManifest]

    if processing_order.index(type(a)) < processing_order.index(type(b)):
        # Manifest that comes earlier in the pipeline cannot depend on the one that comes later
        print(a, "does not depend on", b, "since latter goes later in the pipeline")
        return False

    # True if a's output == b's input
    return get_manifest_input_output(a)[1] == get_manifest_input_output(b)[0]
