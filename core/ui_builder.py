from PyQt6.QtWidgets import QGroupBox, QVBoxLayout, QPushButton, QTextEdit, QWidget, QHBoxLayout
from PyQt6.QtCore import Qt


def create_log_group(title, ui_font, bold_font, on_clear=None):
    log_group = QGroupBox(title)
    log_group.setFont(bold_font)

    log_layout = QVBoxLayout(log_group)
    log_layout.setContentsMargins(5, 5, 5, 5)

    log_text = QTextEdit()
    log_text.setReadOnly(True)
    log_text.setFont(ui_font)
    log_text.setStyleSheet("background-color: #FFF0F0; color: #333333;")
    log_text.setFixedHeight(60)
    log_layout.addWidget(log_text)

    btn_clear_log = QPushButton("清除信息")
    btn_clear_log.setFont(bold_font)
    btn_clear_log.setFixedWidth(100)
    btn_clear_log.setFixedHeight(30)
    if on_clear is not None:
        btn_clear_log.clicked.connect(on_clear)
    log_layout.addWidget(btn_clear_log, alignment=Qt.AlignmentFlag.AlignCenter)

    return log_group, log_text, btn_clear_log


def create_bottom_buttons(bold_font, on_start=None, on_stop=None, on_force=None):
    btn_widget = QWidget()
    btn_layout = QHBoxLayout(btn_widget)
    btn_layout.setContentsMargins(0, 10, 0, 10)

    start_btn = QPushButton("开始")
    start_btn.setFixedSize(100, 30)
    start_btn.setFont(bold_font)
    if on_start is not None:
        start_btn.clicked.connect(on_start)

    stop_btn = QPushButton("停止")
    stop_btn.setFixedSize(100, 30)
    stop_btn.setFont(bold_font)
    stop_btn.setEnabled(False)
    if on_stop is not None:
        stop_btn.clicked.connect(on_stop)

    force_stop_btn = QPushButton("强制终止")
    force_stop_btn.setFixedSize(100, 30)
    force_stop_btn.setFont(bold_font)
    force_stop_btn.setStyleSheet("color: #AA0000;")
    if on_force is not None:
        force_stop_btn.clicked.connect(on_force)

    btn_layout.addWidget(start_btn)
    btn_layout.addStretch()
    btn_layout.addWidget(stop_btn)
    btn_layout.addStretch()
    btn_layout.addWidget(force_stop_btn)

    return btn_widget, start_btn, stop_btn, force_stop_btn
