import pathlib


def get_root_dir() -> pathlib.Path:
    root_dir = pathlib.Path(__file__).parent.parent
    return root_dir


def get_scripts_dir() -> pathlib.Path:
    return get_root_dir() / "scripts"
