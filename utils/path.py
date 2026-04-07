from pathlib import Path

def resolve_path(basepath: Path|str, other: Path|str = "") -> Path:
    _basepath = Path(basepath).expanduser()
    _other = Path(other).expanduser()
    return (_basepath / _other.expanduser()).resolve()

