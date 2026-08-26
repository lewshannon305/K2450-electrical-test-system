"""Portable paths shared by source and packaged builds."""

import sys
from pathlib import Path


def application_root():
    """Return the folder containing the public app files."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def config_directory():
    return application_root() / "configs"


def readme_path():
    return application_root() / "Readme.md"


def default_data_directory(folder_name):
    return str(Path.home() / "Documents" / "K2450_Data" / folder_name)
