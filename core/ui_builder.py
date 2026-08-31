from pathlib import Path

from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QWidget,
)
from PyQt6.QtCore import QCoreApplication, QEvent, QTimer, Qt
from PyQt6.QtGui import QFontMetrics


CURRENT_RANGES = (
    ('10 nA', 10e-9), ('100 nA', 100e-9), ('1 µA', 1e-6),
    ('10 µA', 10e-6), ('100 µA', 100e-6), ('1 mA', 1e-3),
    ('10 mA', 10e-3), ('100 mA', 100e-3), ('1 A', 1.0),
)

VOLTAGE_RANGES = (
    ('0.2 V', 0.2), ('2 V', 2.0), ('20 V', 20.0), ('200 V', 200.0),
)

PARAMETER_LABEL_WIDTH = 140


def configure_parameter_grid(grid, *, margins=None):
    """Use the same two equal parameter columns throughout the application."""
    if margins is not None:
        grid.setContentsMargins(*margins)
    grid.setHorizontalSpacing(10)
    grid.setVerticalSpacing(6)
    grid.setColumnMinimumWidth(0, PARAMETER_LABEL_WIDTH)
    grid.setColumnMinimumWidth(2, PARAMETER_LABEL_WIDTH)
    grid.setColumnStretch(1, 1)
    grid.setColumnStretch(3, 1)


def style_parameter_label(label, ui_font):
    label.setFont(ui_font)
    label.setStyleSheet('font-weight: normal;')
    label.setFixedWidth(PARAMETER_LABEL_WIDTH)
    label.setAlignment(
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    )


def style_parameter_control(control, ui_font):
    control.setFont(ui_font)
    control.setStyleSheet('font-weight: normal;')
    control.setMinimumWidth(0)
    control.setSizePolicy(
        QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
    )


def create_current_range_combo(default='AUTO', include_auto=True, ui_font=None):
    combo = QComboBox()
    combo.setProperty('config_uses_data', True)
    if include_auto:
        combo.addItem('AUTO', 'AUTO')
    for label, value in CURRENT_RANGES:
        combo.addItem(label, value)
    if ui_font is not None:
        combo.setFont(ui_font)
    index = combo.findData(
        'AUTO' if str(default).upper() == 'AUTO' else float(default)
    )
    combo.setCurrentIndex(max(0, index))
    return combo


def create_voltage_range_combo(default=20.0, ui_font=None):
    """Create a fixed Keithley 2450 source-voltage range selector."""
    combo = QComboBox()
    combo.setProperty('config_uses_data', True)
    for label, value in VOLTAGE_RANGES:
        combo.addItem(label, value)
    if ui_font is not None:
        combo.setFont(ui_font)
    index = combo.findData(float(default))
    combo.setCurrentIndex(max(0, index))
    return combo


def bind_range_to_limit(combo, limit_edit):
    """Only a user selection updates the still-editable current limit."""
    def update_limit(_index):
        value = combo.currentData()
        if isinstance(value, (int, float)):
            limit_edit.setText(f'{float(value) * 1.05:.8g}')
    combo.activated.connect(update_limit)


def update_scroll_area_layout(scroll_area, update_widgets, repaint_widget=None):
    """Apply one atomic dynamic-panel change after Qt settles its layouts."""
    repaint_target = repaint_widget or scroll_area
    scrollbar = scroll_area.verticalScrollBar()
    old_position = scrollbar.value()
    generation = int(
        repaint_target.property('_layout_update_generation') or 0
    ) + 1
    repaint_target.setProperty('_layout_update_generation', generation)
    if not repaint_target.property('_layout_update_pending'):
        repaint_target.setProperty(
            '_layout_updates_were_enabled', repaint_target.updatesEnabled()
        )
        repaint_target.setProperty('_layout_update_pending', True)
        repaint_target.setUpdatesEnabled(False)

    def settle_layouts():
        content = scroll_area.widget()
        if content is not None:
            content.updateGeometry()
            if content.layout() is not None:
                content.layout().activate()
        scroll_area.updateGeometry()
        if repaint_target.layout() is not None:
            repaint_target.layout().activate()

    try:
        update_widgets()
        settle_layouts()
    finally:
        def finish_update():
            if int(
                repaint_target.property('_layout_update_generation') or 0
            ) != generation:
                return
            QCoreApplication.sendPostedEvents(
                None, QEvent.Type.LayoutRequest
            )
            settle_layouts()
            scrollbar.setValue(min(old_position, scrollbar.maximum()))
            updates_were_enabled = bool(
                repaint_target.property('_layout_updates_were_enabled')
            )
            repaint_target.setProperty('_layout_update_pending', False)
            if updates_were_enabled:
                repaint_target.setUpdatesEnabled(True)
                repaint_target.update()

        QTimer.singleShot(0, finish_update)


def combo_config_value(combo):
    return combo.currentData() if combo.property('config_uses_data') else combo.currentText()


def set_combo_config_value(combo, value):
    if combo.property('config_uses_data'):
        target = 'AUTO' if str(value).upper() == 'AUTO' else float(value)
        index = combo.findData(target)
    else:
        index = combo.findText(str(value))
    if index >= 0:
        combo.setCurrentIndex(index)
    elif combo.isEditable():
        combo.setEditText(str(value))


def create_status_group(status_items, ui_font, bold_font):
    """Build the same 50/50 aligned status panel on all seven pages."""
    group = QGroupBox('实时状态显示')
    group.setFont(bold_font)
    group.setFixedHeight(135)
    outer = QHBoxLayout(group)
    outer.setContentsMargins(10, 8, 10, 8)
    outer.setSpacing(20)
    names_by_half = [[], []]
    for text, _key, _row, col in status_items:
        raw_text = str(text).rstrip()
        if raw_text.endswith(':'):
            raw_text = raw_text[:-1].rstrip()
        names_by_half[0 if col < 2 else 1].append(raw_text)
    metrics = QFontMetrics(ui_font)
    name_widths = [
        max((metrics.horizontalAdvance(text) for text in names), default=0) + 4
        for names in names_by_half
    ]

    halves = []
    for _ in range(2):
        pane = QWidget()
        pane.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        grid = QGridLayout(pane)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 0)
        grid.setColumnStretch(2, 1)
        grid.setHorizontalSpacing(4)
        halves.append(grid)
        outer.addWidget(pane, 1)
    labels = {}
    for text, key, row, col in status_items:
        grid = halves[0 if col < 2 else 1]
        raw_text = str(text).rstrip()
        has_colon = raw_text.endswith(':')
        name = QLabel(raw_text[:-1].rstrip() if has_colon else raw_text)
        name.setFont(ui_font)
        name.setStyleSheet('font-weight: normal;')
        name.setFixedWidth(name_widths[0 if col < 2 else 1])
        name.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        colon = QLabel(':' if has_colon else '')
        colon.setFont(ui_font)
        colon.setStyleSheet('font-weight: normal;')
        colon.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        value = QLabel('-')
        value.setFont(bold_font)
        value.setStyleSheet('color: #0055A4;')
        value.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(name, row, 0)
        grid.addWidget(colon, row, 1)
        grid.addWidget(value, row, 2)
        labels[key] = value
    return group, labels


def configure_output_path(
    owner, path_grid, folder_edit, filename_edit, data_settings,
    default_subfolder, *, filename_is_prefix=False, hint='',
):
    """Collapse folder and filename controls into one root-relative path row."""
    owner.data_settings = data_settings
    folder_edit.setText(default_subfolder)
    owner.output_subfolder_input = folder_edit
    folder_edit.hide()
    if filename_edit is not None:
        filename_edit.hide()

    normal_font = folder_edit.font()
    hint_label = QLabel(hint)
    hint_label.setFont(normal_font)
    hint_label.setStyleSheet('font-weight: normal; color: #666666;')
    hint_label.setWordWrap(True)
    owner.output_hint_label = hint_label
    label = QLabel('保存路径 (根目录下)：')
    label.setFont(normal_font)
    label.setStyleSheet('font-weight: normal;')
    combined = QLineEdit()
    combined.setFont(normal_font)
    combined.setStyleSheet('font-weight: normal;')
    combined.setPlaceholderText('例如 Break/Break.txt')
    browse = QPushButton('浏览')
    browse.setFont(normal_font)
    browse.setStyleSheet('font-weight: normal;')
    browse.setFixedWidth(70)
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(combined, 1)
    row.addWidget(browse)
    path_grid.addWidget(label, 0, 0)
    path_grid.addLayout(row, 0, 1)
    if hint:
        hint_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        path_grid.addWidget(hint_label, 1, 0, 1, 2)
    else:
        hint_label.hide()
    path_grid.setColumnStretch(1, 1)
    owner.combined_output_input = combined

    def resolved_folder():
        return data_settings.resolve(folder_edit.text())

    def apply_combined(text):
        value = str(text).strip()
        if not value:
            return
        path = Path(value)
        parent = '' if str(path.parent) == '.' else str(path.parent)
        folder_edit.setText(parent)
        if filename_edit is not None and path.name:
            name = path.name
            if filename_is_prefix and name.lower().endswith('.txt'):
                name = name[:-4]
            filename_edit.setText(name)

    def refresh_from_parts(*_args):
        name = filename_edit.text().strip() if filename_edit is not None else ''
        if filename_is_prefix and name and not name.lower().endswith('.txt'):
            name += '.txt'
        folder = folder_edit.text().strip().replace('\\', '/')
        value = '/'.join(part for part in (folder, name) if part)
        combined.blockSignals(True)
        combined.setText(value)
        combined.blockSignals(False)

    def browse_folder():
        directory = QFileDialog.getExistingDirectory(
            owner, '选择保存文件夹', resolved_folder()
        )
        if directory:
            folder_edit.setText(directory)
            refresh_from_parts()

    def set_combined_path(value):
        combined.setText(str(value))
        refresh_from_parts()

    owner.resolved_output_folder = resolved_folder
    owner.refresh_combined_output_from_parts = refresh_from_parts
    owner.set_combined_output_path = set_combined_path
    combined.textChanged.connect(apply_combined)
    combined.editingFinished.connect(refresh_from_parts)
    browse.clicked.connect(browse_folder)
    refresh_from_parts()
