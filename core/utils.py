import pyqtgraph as pg
from PyQt6.QtWidgets import QComboBox

G0 = 7.748091729e-5

_0, _1, _2, _3, _4, _5, _6, _7, _8 = 0, 1, 2, 3, 4, 5, 6, 7, 8


def configure_pyqtgraph(use_opengl=False):
    if use_opengl:
        try:
            import OpenGL  # noqa: F401
            pg.setConfigOptions(useOpenGL=True)
        except ImportError:
            pass

    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')


class NoScrollComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()
