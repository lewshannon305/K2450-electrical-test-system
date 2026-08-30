"""Portable paths shared by source and packaged builds."""

import sys
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal


def application_root():
    """Return the folder containing the public app files."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_path(*parts):
    """Return a bundled resource in both source and PyInstaller builds."""
    if getattr(sys, "frozen", False):
        root = Path(getattr(sys, "_MEIPASS", application_root()))
    else:
        root = application_root()
    return root.joinpath(*parts)


def config_directory():
    return application_root() / "configs"


def readme_path():
    return application_root() / "Readme.md"


def default_data_directory(folder_name):
    return str(Path(default_data_root()) / folder_name)


def default_data_root():
    return str(Path("C:/Users/Public/Documents/K2450_Data"))


class DataRootSettings(QObject):
    """Shared, editable data root used by every measurement page."""

    changed = pyqtSignal(str)

    def __init__(self, root=None, parent=None):
        super().__init__(parent)
        self._root = str(Path(root or default_data_root()).expanduser())

    @property
    def root(self):
        return self._root

    def set_root(self, root):
        value = str(Path(str(root).strip() or default_data_root()).expanduser())
        if value != self._root:
            self._root = value
            self.changed.emit(value)

    def resolve(self, subfolder):
        value = str(subfolder).strip()
        path = Path(value).expanduser() if value else Path()
        if path.is_absolute():
            return str(path)
        return str(Path(self._root) / path)
