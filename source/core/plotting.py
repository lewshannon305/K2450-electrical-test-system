import copy
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib as mpl
from PyQt6.QtCore import QObject, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFontComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.paths import config_directory
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter
import pyqtgraph as pg


MODULE_NAMES = {
    'break_junction': '断裂结',
    'iv_curve': '循环IV特性扫描',
    'isd_vg_setvsd': '栅压特性扫描',
    'mapping_scan': '二维Mapping扫描',
    'it_step_setgate': 'It特性扫描',
    'bias_switch': '偏压开关测试',
    'gate_switch': '栅压开关测试',
    'arbitrary_bias': '任意偏压波形测试',
    'arbitrary_gate': '任意栅压波形测试',
}


def select_preview_paths(image_paths, module_id=None):
    """Return the PNGs that should be shown after a completed measurement."""
    paths = [
        str(path) for path in image_paths
        if str(path).lower().endswith('.png') and os.path.exists(path)
    ]
    if module_id == 'mapping_scan':
        differential = [
            path for path in paths
            if os.path.basename(path).startswith(
                'differential_conductance_mapping'
            )
        ]
        if differential:
            return differential
    return paths


def default_plot_settings():
    module_defaults = {
        'break_junction': ('Break junction', 'Voltage (V)', 'Current / Conductance'),
        'iv_curve': ('Current–voltage characteristics', 'Bias voltage (V)', 'Current (A)'),
        'isd_vg_setvsd': ('Single-molecule transistor transfer', 'Gate voltage (V)', 'Current (A)'),
        'mapping_scan': ('Single-Molecule Transistor Stability Diagram', 'Gate voltage (V)', 'Bias voltage (mV)'),
        'it_step_setgate': ('Current-Time Characteristics', 'Time (s)', 'Current (A)'),
        'bias_switch': ('Bias Switching Response', 'Time (s)', 'Voltage / Current'),
        'gate_switch': ('Gate Switching Response', 'Time (s)', 'Voltage / Current'),
        'arbitrary_bias': ('Arbitrary Bias Response', 'Time (s)', 'Voltage / Current'),
        'arbitrary_gate': ('Arbitrary Gate Response', 'Time (s)', 'Voltage / Current'),
    }
    dimensions = {
        'break_junction': (89.0, 110.0),
        'iv_curve': (89.0, 80.0),
        'isd_vg_setvsd': (89.0, 90.0),
        'mapping_scan': (120.0, 95.0),
        'it_step_setgate': (183.0, 80.0),
        'bias_switch': (120.0, 100.0),
        'gate_switch': (120.0, 100.0),
        'arbitrary_bias': (120.0, 100.0),
        'arbitrary_gate': (120.0, 100.0),
    }
    return {
        'plot_schema_version': 2,
        'auto_plot': True,
        'plot_partial': True,
        'show_preview': True,
        'format': 'png',
        'dpi': 300,
        'width': 9.0,
        'height': 6.0,
        'grid': True,
        'font_family': 'Arial',
        'title_size': 14,
        'label_size': 11,
        'tick_size': 10,
        'primary_color': '#1565C0',
        'secondary_color': '#D32F2F',
        'background_color': '#FFFFFF',
        'grid_color': '#B0B0B0',
        'line_width': 1.8,
        'output_mode': 'plots_subfolder',
        'output_folder': '',
        'modules': {
            module_id: {
                'enabled': True,
                'current_scale': 'linear',
                'x_scale': 'linear',
                'title': module_defaults[module_id][0],
                'x_label': module_defaults[module_id][1],
                'y_label': module_defaults[module_id][2],
                'x_min': '',
                'x_max': '',
                'y_min': '',
                'y_max': '',
                'primary_color': '#1565C0',
                'secondary_color': '#D32F2F',
                'line_width': 1.0,
                'marker': 'none',
                'colormap': (
                    'RdBu_r' if module_id == 'mapping_scan' else 'viridis'
                ),
                'width_mm': dimensions[module_id][0],
                'height_mm': dimensions[module_id][1],
                'size_preset': (
                    'single' if dimensions[module_id][0] == 89.0
                    else 'double' if dimensions[module_id][0] == 183.0
                    else 'one_half'
                ),
                'formats': ['svg', 'pdf', 'png'],
                'dpi': 450 if module_id == 'mapping_scan' else 300,
                'font_family': 'Arial',
                'title_size': 7,
                'label_size': 7,
                'tick_size': 7,
                'grid': False,
                'legend': True,
                'legend_location': 'best',
                'top_spine': False,
                'right_spine': False,
                'minor_ticks': False,
                'tick_direction': 'out',
                'tick_length': 4.0,
                'tick_width': 0.8,
                'iv_file_mode': 'group',
                'iv_gate_mode': 'per_gate',
                'iv_cycle_mode': 'per_cycle',
                'isd_mode': 'overlay',
                'isd_offset_mode': 'auto',
                'isd_offset': 0.0,
                'mapping_full_iv': False,
                'mapping_full_iv_summary': False,
                'savgol_window': 11,
                'savgol_order': 2,
                'it_bin_method': 'fd',
                'it_bins': 50,
                'it_bin_width': 0.0,
                'it_hist_norm': 'density',
                'it_show_mean': True,
                'it_show_median': True,
                'it_show_std': True,
            }
            for module_id in MODULE_NAMES
        },
    }


def merge_plot_settings(value):
    settings = default_plot_settings()
    if not isinstance(value, dict):
        return settings
    try:
        incoming_version = int(value.get('plot_schema_version', 1))
    except (TypeError, ValueError):
        incoming_version = 1
    for key in (
        'auto_plot', 'plot_partial', 'show_preview', 'format', 'dpi',
        'width', 'height', 'grid', 'output_mode', 'output_folder',
        'font_family', 'title_size', 'label_size', 'tick_size',
        'primary_color', 'secondary_color', 'background_color',
        'grid_color', 'line_width',
    ):
        if key in value:
            settings[key] = value[key]
    modules = value.get('modules', {})
    if isinstance(modules, dict):
        for module_id, module_value in modules.items():
            if module_id in settings['modules'] and isinstance(module_value, dict):
                settings['modules'][module_id].update(module_value)
    if incoming_version < 2:
        for module in settings['modules'].values():
            if float(module.get('line_width', 0)) == 0.8:
                module['line_width'] = 1.0
            if int(module.get('tick_size', 0)) == 6:
                module['tick_size'] = 7
            if float(module.get('tick_length', 0)) == 3.0:
                module['tick_length'] = 4.0
            if float(module.get('tick_width', 0)) == 0.6:
                module['tick_width'] = 0.8
    settings['plot_schema_version'] = 2
    return settings


def default_plot_profile_path():
    return config_directory() / 'plotting_default.json'


def load_default_plot_settings(fallback=None):
    path = default_plot_profile_path()
    if not path.exists():
        return merge_plot_settings(fallback)
    try:
        return merge_plot_settings(json.loads(path.read_text(encoding='utf-8')))
    except (OSError, ValueError, json.JSONDecodeError):
        return merge_plot_settings(fallback)


class PlotSettingsDialog(QDialog):
    settings_applied = pyqtSignal(object)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle('绘图设置')
        self.resize(1500, 820)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )
        self.ui_font = QFont('Arial', 11)
        self.bold_font = QFont('Arial', 11)
        self.bold_font.setWeight(QFont.Weight.Bold)
        self._initial = merge_plot_settings(settings)
        self._module_values = copy.deepcopy(self._initial['modules'])
        self._current_module_id = None
        self._color_edits = {}
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(180)
        self._preview_timer.timeout.connect(self._render_preview)
        self._build_ui()
        self._connect_preview_signals()
        self.nav.setCurrentRow(0)
        self._render_preview()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        nav_group = QGroupBox('绘图项目')
        nav_group.setFont(self.bold_font)
        nav_layout = QVBoxLayout(nav_group)
        self.nav = QListWidget()
        self.nav.setFont(self.ui_font)
        self.nav.setFixedWidth(210)
        self.nav.addItem('通用')
        for name in MODULE_NAMES.values():
            self.nav.addItem(name)
        self.nav.currentRowChanged.connect(self._navigation_changed)
        nav_layout.addWidget(self.nav)
        main_layout.addWidget(nav_group, stretch=0)

        settings_group = QGroupBox('绘图设置')
        settings_group.setFont(self.bold_font)
        settings_layout = QVBoxLayout(settings_group)
        self.settings_stack = QStackedWidget()
        settings_layout.addWidget(self.settings_stack)
        self.settings_stack.addWidget(self._build_common_page())
        self.settings_stack.addWidget(self._build_module_page())
        main_layout.addWidget(settings_group, stretch=3)

        preview_group = QGroupBox('单分子晶体管示例图（实时预览）')
        preview_group.setFont(self.bold_font)
        preview_layout = QVBoxLayout(preview_group)
        self.preview_widget = pg.GraphicsLayoutWidget()
        self.preview_widget.setBackground('w')
        self.preview_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        preview_layout.addWidget(self.preview_widget)
        hint = QLabel(
            '九个模块分别使用对应的单分子晶体管合成数据和绘图模板；'
            '中间栏的修改会立即反映到当前图，并用于正式输出。'
            '右侧为放大屏幕预览，文字和曲线显示尺寸会自动放大；'
            '导出文件仍严格使用中间栏标注的物理 pt 数值。'
        )
        hint.setFont(self.ui_font)
        hint.setWordWrap(True)
        preview_layout.addWidget(hint)

        button_layout = QHBoxLayout()
        self.btn_apply = QPushButton('应用')
        self.btn_save_profile = QPushButton('保存方案')
        self.btn_load_profile = QPushButton('加载方案')
        self.btn_set_default = QPushButton('设为默认')
        self.btn_restore = QPushButton('恢复默认')
        self.btn_cancel = QPushButton('取消')
        self.btn_ok = QPushButton('确认')
        for button in (
            self.btn_apply, self.btn_save_profile, self.btn_load_profile,
            self.btn_set_default, self.btn_restore, self.btn_cancel,
            self.btn_ok,
        ):
            button.setFont(self.bold_font)
            button.setMinimumSize(88, 30)
        self.btn_ok.setStyleSheet('color: #AA0000;')
        self.btn_apply.clicked.connect(self._apply_without_close)
        self.btn_save_profile.clicked.connect(self._save_profile)
        self.btn_load_profile.clicked.connect(self._load_profile)
        self.btn_set_default.clicked.connect(self._set_as_default)
        self.btn_restore.clicked.connect(self._restore_defaults)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self._accept_settings)
        button_layout.addWidget(self.btn_apply)
        button_layout.addWidget(self.btn_save_profile)
        button_layout.addWidget(self.btn_load_profile)
        button_layout.addWidget(self.btn_set_default)
        button_layout.addWidget(self.btn_restore)
        button_layout.addStretch()
        button_layout.addWidget(self.btn_cancel)
        button_layout.addWidget(self.btn_ok)
        preview_layout.addLayout(button_layout)
        main_layout.addWidget(preview_group, stretch=5)

    def _scroll_form(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        content = QWidget()
        content.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        form = QFormLayout(content)
        form.setContentsMargins(6, 6, 6, 6)
        form.setSpacing(9)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        scroll.setWidget(content)
        return scroll, form

    def _build_common_page(self):
        scroll, form = self._scroll_form()
        workflow = QGroupBox('自动绘图工作流')
        workflow_form = QFormLayout(workflow)
        self.cb_auto = QCheckBox('测试数据全部保存后自动绘图')
        self.cb_auto.setChecked(self._initial['auto_plot'])
        workflow_form.addRow(self.cb_auto)
        self.cb_partial = QCheckBox('对有效的部分数据绘图（图中标记 PARTIAL）')
        self.cb_partial.setChecked(self._initial['plot_partial'])
        workflow_form.addRow(self.cb_partial)
        self.cb_preview = QCheckBox('绘图完成后自动弹出 PNG 预览')
        self.cb_preview.setChecked(self._initial['show_preview'])
        workflow_form.addRow(self.cb_preview)
        form.addRow(workflow)

        storage = QGroupBox('输出位置')
        storage_form = QFormLayout(storage)
        self.combo_output = QComboBox()
        self.combo_output.addItem('数据目录下的 plots 文件夹', 'plots_subfolder')
        self.combo_output.addItem('指定文件夹', 'custom')
        idx = self.combo_output.findData(self._initial['output_mode'])
        self.combo_output.setCurrentIndex(max(0, idx))
        storage_form.addRow('保存位置：', self.combo_output)
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.edit_output = QLineEdit(str(self._initial['output_folder']))
        btn_browse = QPushButton('浏览')
        btn_browse.clicked.connect(self._browse_output)
        output_layout.addWidget(self.edit_output)
        output_layout.addWidget(btn_browse)
        storage_form.addRow('指定文件夹：', output_row)
        note = QLabel(
            '原始数据仍由九个测试程序分别保存，并未混在一个文件中。'
            '绘图只读取本次测试结果清单；Mapping 的 full IV 将另存到 '
            'plots/mapping/full_iv。'
        )
        note.setWordWrap(True)
        storage_form.addRow(note)
        form.addRow(storage)
        return scroll

    def _build_module_page(self):
        scroll, form = self._scroll_form()
        def group(title):
            box = QGroupBox(title)
            box_form = QFormLayout(box)
            box_form.setFieldGrowthPolicy(
                QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            )
            form.addRow(box)
            return box_form

        data_form = group('数据与分组')
        self.cb_module_enabled = QCheckBox('为该模块生成结束后绘图')
        data_form.addRow(self.cb_module_enabled)

        export_form = group('版式与导出')
        self.combo_size_preset = QComboBox()
        self.combo_size_preset.addItem('Nature 单栏（89 mm）', 'single')
        self.combo_size_preset.addItem('Nature 1.5 栏（120 mm）', 'one_half')
        self.combo_size_preset.addItem('Nature 双栏（183 mm）', 'double')
        self.combo_size_preset.addItem('自定义', 'custom')
        export_form.addRow('宽度预设：', self.combo_size_preset)
        self.spin_width_mm = QDoubleSpinBox()
        self.spin_width_mm.setRange(40.0, 183.0)
        self.spin_width_mm.setDecimals(1)
        self.spin_width_mm.setSuffix(' mm')
        self.spin_height_mm = QDoubleSpinBox()
        self.spin_height_mm.setRange(35.0, 170.0)
        self.spin_height_mm.setDecimals(1)
        self.spin_height_mm.setSuffix(' mm')
        export_form.addRow('图片宽度：', self.spin_width_mm)
        export_form.addRow('图片高度：', self.spin_height_mm)
        self.edit_formats = QLineEdit()
        self.edit_formats.setPlaceholderText('svg, pdf, png')
        export_form.addRow('导出格式：', self.edit_formats)
        self.spin_module_dpi = QSpinBox()
        self.spin_module_dpi.setRange(150, 1200)
        self.spin_module_dpi.setSuffix(' DPI')
        export_form.addRow('PNG 分辨率：', self.spin_module_dpi)

        axis_form = group('坐标轴与刻度')
        self.edit_title = QLineEdit()
        self.edit_x_label = QLineEdit()
        self.edit_y_label = QLineEdit()
        self.combo_x_scale = QComboBox()
        self.combo_y_scale = QComboBox()
        for combo in (self.combo_x_scale, self.combo_y_scale):
            combo.addItem('线性', 'linear')
            combo.addItem('对数', 'log')
            combo.addItem('对称对数', 'symlog')
        axis_form.addRow('X 轴尺度：', self.combo_x_scale)
        axis_form.addRow('Y 轴尺度：', self.combo_y_scale)
        self.edit_x_min = QLineEdit()
        self.edit_x_max = QLineEdit()
        self.edit_y_min = QLineEdit()
        self.edit_y_max = QLineEdit()
        self.edit_x_min.setPlaceholderText('自动')
        self.edit_x_max.setPlaceholderText('自动')
        self.edit_y_min.setPlaceholderText('自动')
        self.edit_y_max.setPlaceholderText('自动')
        axis_form.addRow('X 轴最小值：', self.edit_x_min)
        axis_form.addRow('X 轴最大值：', self.edit_x_max)
        axis_form.addRow('Y 轴最小值：', self.edit_y_min)
        axis_form.addRow('Y 轴最大值：', self.edit_y_max)
        self.cb_minor_ticks = QCheckBox('显示次刻度')
        axis_form.addRow(self.cb_minor_ticks)
        self.combo_tick_direction = QComboBox()
        self.combo_tick_direction.addItem('向外', 'out')
        self.combo_tick_direction.addItem('向内', 'in')
        self.combo_tick_direction.addItem('双向', 'inout')
        axis_form.addRow('刻度方向：', self.combo_tick_direction)
        self.spin_tick_length = QDoubleSpinBox()
        self.spin_tick_length.setRange(1.0, 10.0)
        self.spin_tick_width = QDoubleSpinBox()
        self.spin_tick_width.setRange(0.25, 2.0)
        axis_form.addRow('刻度长度（pt）：', self.spin_tick_length)
        axis_form.addRow('刻度线宽（pt）：', self.spin_tick_width)

        text_form = group('标题与文字')
        text_form.addRow('图标题：', self.edit_title)
        text_form.addRow('X 轴标题：', self.edit_x_label)
        text_form.addRow('Y 轴标题：', self.edit_y_label)
        self.module_font_family = QFontComboBox()
        text_form.addRow('字体：', self.module_font_family)
        self.spin_module_title_size = QSpinBox()
        self.spin_module_title_size.setRange(5, 14)
        self.spin_module_label_size = QSpinBox()
        self.spin_module_label_size.setRange(5, 12)
        self.spin_module_tick_size = QSpinBox()
        self.spin_module_tick_size.setRange(5, 10)
        text_form.addRow('导出标题字号（pt）：', self.spin_module_title_size)
        text_form.addRow('导出坐标轴字号（pt）：', self.spin_module_label_size)
        text_form.addRow('导出刻度字号（pt）：', self.spin_module_tick_size)

        color_form = group('曲线与配色')
        self.module_primary = self._color_row(
            color_form, '主曲线颜色：', 'module_primary', '#1565C0'
        )
        self.module_secondary = self._color_row(
            color_form, '辅助曲线颜色：', 'module_secondary', '#D32F2F'
        )
        self.spin_module_line_width = QDoubleSpinBox()
        self.spin_module_line_width.setRange(0.25, 3.0)
        self.spin_module_line_width.setSingleStep(0.1)
        color_form.addRow('导出曲线线宽（pt）：', self.spin_module_line_width)
        self.combo_marker = QComboBox()
        for label, value in (
            ('无', 'none'), ('圆点', 'o'), ('方块', 's'),
            ('三角', '^'), ('十字', 'x'),
        ):
            self.combo_marker.addItem(label, value)
        color_form.addRow('数据点标记：', self.combo_marker)
        self.combo_colormap = QComboBox()
        self.combo_colormap.addItems(
            ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
             'coolwarm', 'RdBu_r', 'turbo']
        )
        color_form.addRow('Mapping 色图：', self.combo_colormap)

        frame_form = group('图框、网格与图例')
        self.cb_module_grid = QCheckBox('显示网格')
        self.cb_module_legend = QCheckBox('显示图例')
        self.cb_top_spine = QCheckBox('显示上边框')
        self.cb_right_spine = QCheckBox('显示右边框')
        frame_form.addRow(self.cb_module_grid)
        frame_form.addRow(self.cb_module_legend)
        frame_form.addRow(self.cb_top_spine)
        frame_form.addRow(self.cb_right_spine)
        self.combo_legend_location = QComboBox()
        for label, value in (
            ('自动', 'best'), ('左上', 'upper left'),
            ('右上', 'upper right'), ('左下', 'lower left'),
            ('右下', 'lower right'),
        ):
            self.combo_legend_location.addItem(label, value)
        frame_form.addRow('图例位置：', self.combo_legend_location)

        specific_form = group('模块专用')
        self.specific_stack = QStackedWidget()
        specific_form.addRow(self.specific_stack)
        self._specific_pages = {}
        for module_id in MODULE_NAMES:
            page = QWidget()
            page_form = QFormLayout(page)
            self._specific_pages[module_id] = page
            self.specific_stack.addWidget(page)
            if module_id == 'break_junction':
                page_form.addRow(QLabel(
                    '固定上下布局：Current (mA) / Conductance (G₀)'
                ))
            elif module_id == 'iv_curve':
                self.combo_iv_file_mode = QComboBox()
                self.combo_iv_file_mode.addItem('每个数据文件单独成图', 'separate')
                self.combo_iv_file_mode.addItem('每组 IV 合并成图', 'group')
                self.combo_iv_gate_mode = QComboBox()
                self.combo_iv_gate_mode.addItem('每个栅压单独成图（推荐）', 'per_gate')
                self.combo_iv_gate_mode.addItem('不同栅压叠加', 'overlay')
                self.combo_iv_gate_mode.addItem('不同栅压小多图', 'small_multiples')
                self.combo_iv_cycle_mode = QComboBox()
                self.combo_iv_cycle_mode.addItem('每个循环单独成图', 'per_cycle')
                self.combo_iv_cycle_mode.addItem('循环叠加并降低透明度', 'overlay')
                page_form.addRow('文件/分段：', self.combo_iv_file_mode)
                page_form.addRow('栅压系列：', self.combo_iv_gate_mode)
                page_form.addRow('重复循环：', self.combo_iv_cycle_mode)
            elif module_id == 'isd_vg_setvsd':
                self.combo_isd_mode = QComboBox()
                self.combo_isd_mode.addItem('四条曲线叠加', 'overlay')
                self.combo_isd_mode.addItem('四幅无缝拼图', 'stacked')
                self.combo_isd_mode.addItem('等间距纵向偏移', 'offset')
                self.combo_isd_offset_mode = QComboBox()
                self.combo_isd_offset_mode.addItem('自动计算偏移量', 'auto')
                self.combo_isd_offset_mode.addItem('手动指定偏移量', 'manual')
                self.spin_isd_offset = QDoubleSpinBox()
                self.spin_isd_offset.setRange(0, 1e12)
                self.spin_isd_offset.setDecimals(4)
                page_form.addRow('绘图模式：', self.combo_isd_mode)
                page_form.addRow('偏移方式：', self.combo_isd_offset_mode)
                page_form.addRow('手动偏移量：', self.spin_isd_offset)
            elif module_id == 'mapping_scan':
                self.cb_mapping_full_iv = QCheckBox('导出 full 中全部 IV')
                self.cb_mapping_full_iv_summary = QCheckBox('额外导出 IV 汇总图')
                self.spin_savgol_window = QSpinBox()
                self.spin_savgol_window.setRange(3, 101)
                self.spin_savgol_window.setSingleStep(2)
                self.spin_savgol_order = QSpinBox()
                self.spin_savgol_order.setRange(1, 5)
                page_form.addRow(self.cb_mapping_full_iv)
                page_form.addRow(self.cb_mapping_full_iv_summary)
                page_form.addRow('平滑窗口（奇数）：', self.spin_savgol_window)
                page_form.addRow('平滑阶数：', self.spin_savgol_order)
            elif module_id == 'it_step_setgate':
                self.combo_it_bin_method = QComboBox()
                for label, value in (
                    ('Freedman–Diaconis', 'fd'), ('Scott', 'scott'),
                    ('Sturges', 'sturges'), ('手动数量', 'manual_count'),
                    ('手动宽度', 'manual_width'),
                ):
                    self.combo_it_bin_method.addItem(label, value)
                self.spin_it_bins = QSpinBox()
                self.spin_it_bins.setRange(2, 1000)
                self.spin_it_bin_width = QDoubleSpinBox()
                self.spin_it_bin_width.setRange(0, 1e12)
                self.spin_it_bin_width.setDecimals(6)
                self.combo_it_hist_norm = QComboBox()
                self.combo_it_hist_norm.addItem('概率密度', 'density')
                self.combo_it_hist_norm.addItem('计数', 'count')
                self.cb_it_mean = QCheckBox('标出均值')
                self.cb_it_median = QCheckBox('标出中位数')
                self.cb_it_std = QCheckBox('标出 ±1 标准差')
                page_form.addRow('分箱算法：', self.combo_it_bin_method)
                page_form.addRow('手动箱数：', self.spin_it_bins)
                page_form.addRow('手动箱宽：', self.spin_it_bin_width)
                page_form.addRow('纵轴归一化：', self.combo_it_hist_norm)
                page_form.addRow(self.cb_it_mean)
                page_form.addRow(self.cb_it_median)
                page_form.addRow(self.cb_it_std)
            else:
                drive = (
                    'Gate voltage' if module_id in ('gate_switch', 'arbitrary_gate')
                    else 'Bias voltage'
                )
                page_form.addRow(QLabel(
                    f'固定上下布局：上图 {drive}–Time；下图 Current–Time'
                ))
        return scroll

    def _color_row(self, form, label, key, value):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit(value)
        button = QPushButton('选择')
        button.setFixedWidth(60)
        button.clicked.connect(lambda _=False, item=key: self._choose_color(item))
        layout.addWidget(edit)
        layout.addWidget(button)
        form.addRow(label, row)
        self._color_edits[key] = edit
        self._paint_color_edit(edit)
        return edit

    def _choose_color(self, key):
        edit = self._color_edits[key]
        initial = QColor(edit.text()) if QColor(edit.text()).isValid() else QColor('#000000')
        color = QColorDialog.getColor(initial, self, '选择颜色')
        if color.isValid():
            edit.setText(color.name())
            self._paint_color_edit(edit)
            self.update_preview()

    @staticmethod
    def _paint_color_edit(edit):
        color = QColor(edit.text())
        edit.setStyleSheet(
            f'background-color:{color.name()};'
            f'color:{"black" if color.lightness() > 140 else "white"};'
            if color.isValid() else ''
        )

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, '选择绘图保存文件夹', self.edit_output.text()
        )
        if folder:
            self.edit_output.setText(folder)

    def _module_widgets(self):
        widgets = [
            self.cb_module_enabled, self.edit_title, self.edit_x_label,
            self.edit_y_label, self.combo_x_scale, self.combo_y_scale,
            self.edit_x_min, self.edit_x_max, self.edit_y_min,
            self.edit_y_max, self.module_primary, self.module_secondary,
            self.spin_module_line_width, self.combo_marker,
            self.combo_colormap, self.combo_size_preset,
            self.spin_width_mm, self.spin_height_mm, self.edit_formats,
            self.spin_module_dpi, self.module_font_family,
            self.spin_module_title_size, self.spin_module_label_size,
            self.spin_module_tick_size, self.cb_module_grid,
            self.cb_module_legend, self.cb_top_spine, self.cb_right_spine,
            self.cb_minor_ticks, self.combo_tick_direction,
            self.spin_tick_length, self.spin_tick_width,
            self.combo_legend_location,
        ]
        for name in (
            'combo_iv_file_mode', 'combo_iv_gate_mode',
            'combo_iv_cycle_mode', 'combo_isd_mode',
            'combo_isd_offset_mode', 'spin_isd_offset',
            'cb_mapping_full_iv', 'cb_mapping_full_iv_summary',
            'spin_savgol_window', 'spin_savgol_order',
            'combo_it_bin_method', 'spin_it_bins', 'spin_it_bin_width',
            'combo_it_hist_norm', 'cb_it_mean', 'cb_it_median',
            'cb_it_std',
        ):
            if hasattr(self, name):
                widgets.append(getattr(self, name))
        return tuple(widgets)

    def _save_current_module(self):
        if self._current_module_id is None:
            return
        value = self._module_values[self._current_module_id]
        value.update({
            'enabled': self.cb_module_enabled.isChecked(),
            'title': self.edit_title.text(),
            'x_label': self.edit_x_label.text(),
            'y_label': self.edit_y_label.text(),
            'x_scale': self.combo_x_scale.currentData(),
            'current_scale': self.combo_y_scale.currentData(),
            'x_min': self.edit_x_min.text().strip(),
            'x_max': self.edit_x_max.text().strip(),
            'y_min': self.edit_y_min.text().strip(),
            'y_max': self.edit_y_max.text().strip(),
            'primary_color': self.module_primary.text().strip(),
            'secondary_color': self.module_secondary.text().strip(),
            'line_width': self.spin_module_line_width.value(),
            'marker': self.combo_marker.currentData(),
            'colormap': self.combo_colormap.currentText(),
            'size_preset': self.combo_size_preset.currentData(),
            'width_mm': self.spin_width_mm.value(),
            'height_mm': self.spin_height_mm.value(),
            'formats': [
                item.strip().lower()
                for item in self.edit_formats.text().split(',')
                if item.strip()
            ],
            'dpi': self.spin_module_dpi.value(),
            'font_family': self.module_font_family.currentFont().family(),
            'title_size': self.spin_module_title_size.value(),
            'label_size': self.spin_module_label_size.value(),
            'tick_size': self.spin_module_tick_size.value(),
            'grid': self.cb_module_grid.isChecked(),
            'legend': self.cb_module_legend.isChecked(),
            'top_spine': self.cb_top_spine.isChecked(),
            'right_spine': self.cb_right_spine.isChecked(),
            'minor_ticks': self.cb_minor_ticks.isChecked(),
            'tick_direction': self.combo_tick_direction.currentData(),
            'tick_length': self.spin_tick_length.value(),
            'tick_width': self.spin_tick_width.value(),
            'legend_location': self.combo_legend_location.currentData(),
        })
        module_id = self._current_module_id
        if module_id == 'iv_curve':
            value.update({
                'iv_file_mode': self.combo_iv_file_mode.currentData(),
                'iv_gate_mode': self.combo_iv_gate_mode.currentData(),
                'iv_cycle_mode': self.combo_iv_cycle_mode.currentData(),
            })
        elif module_id == 'isd_vg_setvsd':
            value.update({
                'isd_mode': self.combo_isd_mode.currentData(),
                'isd_offset_mode': self.combo_isd_offset_mode.currentData(),
                'isd_offset': self.spin_isd_offset.value(),
            })
        elif module_id == 'mapping_scan':
            value.update({
                'mapping_full_iv': self.cb_mapping_full_iv.isChecked(),
                'mapping_full_iv_summary':
                    self.cb_mapping_full_iv_summary.isChecked(),
                'savgol_window': self.spin_savgol_window.value(),
                'savgol_order': self.spin_savgol_order.value(),
            })
        elif module_id == 'it_step_setgate':
            value.update({
                'it_bin_method': self.combo_it_bin_method.currentData(),
                'it_bins': self.spin_it_bins.value(),
                'it_bin_width': self.spin_it_bin_width.value(),
                'it_hist_norm': self.combo_it_hist_norm.currentData(),
                'it_show_mean': self.cb_it_mean.isChecked(),
                'it_show_median': self.cb_it_median.isChecked(),
                'it_show_std': self.cb_it_std.isChecked(),
            })

    def _load_module(self, module_id):
        value = self._module_values[module_id]
        controls = self._module_widgets()
        for control in controls:
            control.blockSignals(True)
        self.cb_module_enabled.setChecked(bool(value['enabled']))
        self.edit_title.setText(str(value['title']))
        self.edit_x_label.setText(str(value['x_label']))
        self.edit_y_label.setText(str(value['y_label']))
        self.combo_x_scale.setCurrentIndex(
            max(0, self.combo_x_scale.findData(value['x_scale']))
        )
        self.combo_y_scale.setCurrentIndex(
            max(0, self.combo_y_scale.findData(value['current_scale']))
        )
        self.edit_x_min.setText(str(value['x_min']))
        self.edit_x_max.setText(str(value['x_max']))
        self.edit_y_min.setText(str(value['y_min']))
        self.edit_y_max.setText(str(value['y_max']))
        self.module_primary.setText(str(value['primary_color']))
        self.module_secondary.setText(str(value['secondary_color']))
        self.spin_module_line_width.setValue(float(value['line_width']))
        self.combo_marker.setCurrentIndex(
            max(0, self.combo_marker.findData(value['marker']))
        )
        self.combo_colormap.setCurrentText(str(value['colormap']))
        self.combo_size_preset.setCurrentIndex(max(
            0, self.combo_size_preset.findData(value.get('size_preset', 'custom'))
        ))
        self.spin_width_mm.setValue(float(value.get('width_mm', 89.0)))
        self.spin_height_mm.setValue(float(value.get('height_mm', 70.0)))
        self.edit_formats.setText(', '.join(value.get('formats', ['png'])))
        self.spin_module_dpi.setValue(int(value.get('dpi', 300)))
        self.module_font_family.setCurrentFont(
            QFont(value.get('font_family', 'Arial'))
        )
        self.spin_module_title_size.setValue(int(value.get('title_size', 7)))
        self.spin_module_label_size.setValue(int(value.get('label_size', 7)))
        self.spin_module_tick_size.setValue(int(value.get('tick_size', 6)))
        self.cb_module_grid.setChecked(bool(value.get('grid', False)))
        self.cb_module_legend.setChecked(bool(value.get('legend', True)))
        self.cb_top_spine.setChecked(bool(value.get('top_spine', False)))
        self.cb_right_spine.setChecked(bool(value.get('right_spine', False)))
        self.cb_minor_ticks.setChecked(bool(value.get('minor_ticks', False)))
        self.combo_tick_direction.setCurrentIndex(max(
            0, self.combo_tick_direction.findData(
                value.get('tick_direction', 'out')
            )
        ))
        self.spin_tick_length.setValue(float(value.get('tick_length', 3.0)))
        self.spin_tick_width.setValue(float(value.get('tick_width', 0.6)))
        self.combo_legend_location.setCurrentIndex(max(
            0, self.combo_legend_location.findData(
                value.get('legend_location', 'best')
            )
        ))
        if module_id == 'iv_curve':
            for control, key in (
                (self.combo_iv_file_mode, 'iv_file_mode'),
                (self.combo_iv_gate_mode, 'iv_gate_mode'),
                (self.combo_iv_cycle_mode, 'iv_cycle_mode'),
            ):
                control.setCurrentIndex(max(
                    0, control.findData(value.get(key))
                ))
        elif module_id == 'isd_vg_setvsd':
            self.combo_isd_mode.setCurrentIndex(max(
                0, self.combo_isd_mode.findData(value.get('isd_mode'))
            ))
            self.combo_isd_offset_mode.setCurrentIndex(max(
                0, self.combo_isd_offset_mode.findData(
                    value.get('isd_offset_mode')
                )
            ))
            self.spin_isd_offset.setValue(float(value.get('isd_offset', 0.0)))
        elif module_id == 'mapping_scan':
            self.cb_mapping_full_iv.setChecked(
                bool(value.get('mapping_full_iv', False))
            )
            self.cb_mapping_full_iv_summary.setChecked(
                bool(value.get('mapping_full_iv_summary', False))
            )
            self.spin_savgol_window.setValue(
                int(value.get('savgol_window', 11))
            )
            self.spin_savgol_order.setValue(int(value.get('savgol_order', 2)))
        elif module_id == 'it_step_setgate':
            self.combo_it_bin_method.setCurrentIndex(max(
                0, self.combo_it_bin_method.findData(
                    value.get('it_bin_method', 'fd')
                )
            ))
            self.spin_it_bins.setValue(int(value.get('it_bins', 50)))
            self.spin_it_bin_width.setValue(
                float(value.get('it_bin_width', 0.0))
            )
            self.combo_it_hist_norm.setCurrentIndex(max(
                0, self.combo_it_hist_norm.findData(
                    value.get('it_hist_norm', 'density')
                )
            ))
            self.cb_it_mean.setChecked(bool(value.get('it_show_mean', True)))
            self.cb_it_median.setChecked(
                bool(value.get('it_show_median', True))
            )
            self.cb_it_std.setChecked(bool(value.get('it_show_std', True)))
        self.specific_stack.setCurrentIndex(list(MODULE_NAMES).index(module_id))
        self._paint_color_edit(self.module_primary)
        self._paint_color_edit(self.module_secondary)
        for control in controls:
            control.blockSignals(False)

    def _navigation_changed(self, row):
        self._save_current_module()
        if row <= 0:
            self._current_module_id = None
            self.settings_stack.setCurrentIndex(0)
        else:
            self._current_module_id = list(MODULE_NAMES)[row - 1]
            self._load_module(self._current_module_id)
            self.settings_stack.setCurrentIndex(1)
        self.update_preview()

    def _connect_preview_signals(self):
        export_only_controls = {
            self.cb_module_enabled,
            self.combo_size_preset,
            self.spin_width_mm,
            self.spin_height_mm,
            self.edit_formats,
            self.spin_module_dpi,
            self.cb_mapping_full_iv,
            self.cb_mapping_full_iv_summary,
            self.spin_savgol_window,
            self.spin_savgol_order,
        }
        controls = [
            control for control in self._module_widgets()
            if control not in export_only_controls
        ]
        for control in controls:
            if isinstance(control, QLineEdit):
                control.textChanged.connect(self.update_preview)
                control.textChanged.connect(
                    lambda _text, item=control: self._paint_color_edit(item)
                    if item in self._color_edits.values() else None
                )
            elif isinstance(control, QCheckBox):
                control.toggled.connect(self.update_preview)
            elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                control.valueChanged.connect(self.update_preview)
            elif isinstance(control, QFontComboBox):
                control.currentFontChanged.connect(self.update_preview)
            elif isinstance(control, QComboBox):
                control.currentIndexChanged.connect(self.update_preview)
        self.combo_size_preset.currentIndexChanged.connect(
            self._apply_size_preset
        )

    def _apply_size_preset(self, *_args):
        width = {
            'single': 89.0,
            'one_half': 120.0,
            'double': 183.0,
        }.get(self.combo_size_preset.currentData())
        if width is not None:
            self.spin_width_mm.setValue(width)

    @staticmethod
    def _optional_float(text):
        text = str(text).strip()
        return None if not text else float(text)

    def _preview_style(self):
        if self._current_module_id is not None:
            self._save_current_module()
            return self._module_values[self._current_module_id]
        return None

    @staticmethod
    def _transfer_example():
        gate = np.linspace(-1.8, 1.8, 600)
        current = np.full_like(gate, 0.025)
        for position, width, amplitude in zip(
            (-1.25, -0.45, 0.38, 1.18),
            (0.11, 0.14, 0.12, 0.16),
            (0.75, 1.35, 1.0, 0.62),
        ):
            current += amplitude * width ** 2 / (
                (gate - position) ** 2 + width ** 2
            )
        current += 0.018 * np.sin(np.arange(gate.size) * 0.37)
        return gate, current


    def update_preview(self, *_args):
        if hasattr(self, '_preview_timer'):
            self._preview_timer.start()

    def _configure_preview_plot(
        self, plot, module, title='', x='', y='', compact=False
    ):
        font = module.get('font_family', 'Arial')
        export_title_size = int(module.get('title_size', 7))
        export_label_size = int(module.get('label_size', 7))
        export_tick_size = int(module.get('tick_size', 7))
        if compact:
            title_size, label_size, tick_size = 10, 9, 8
        else:
            title_size = max(16, round(export_title_size * 2.2))
            label_size = max(14, round(export_label_size * 1.9))
            tick_size = max(12, round(export_tick_size * 1.7))
        plot.setTitle(
            title, color='#202020', size=f'{title_size}pt',
            **{'font-family': font},
        )
        label_style = {
            'font-family': font,
            'font-size': f'{label_size}pt',
        }
        plot.setLabel('bottom', x, **label_style)
        plot.setLabel('left', y, **label_style)
        plot.showGrid(x=bool(module.get('grid')), y=bool(module.get('grid')),
                      alpha=0.22)
        plot.getViewBox().setMouseEnabled(x=True, y=True)
        plot.showAxis('top', bool(module.get('top_spine', False)))
        plot.showAxis('right', bool(module.get('right_spine', False)))
        plot.setLogMode(
            x=module.get('x_scale') == 'log',
            y=module.get('current_scale') == 'log',
        )
        try:
            x_min = self._optional_float(module.get('x_min', ''))
            x_max = self._optional_float(module.get('x_max', ''))
            y_min = self._optional_float(module.get('y_min', ''))
            y_max = self._optional_float(module.get('y_max', ''))
            current_x, current_y = plot.getViewBox().viewRange()
            if x_min is not None or x_max is not None:
                plot.setXRange(
                    x_min if x_min is not None else current_x[0],
                    x_max if x_max is not None else current_x[1],
                    padding=0,
                )
            if y_min is not None or y_max is not None:
                plot.setYRange(
                    y_min if y_min is not None else current_y[0],
                    y_max if y_max is not None else current_y[1],
                    padding=0,
                )
        except (TypeError, ValueError):
            pass
        tick_width = max(1.0, float(module.get('tick_width', 0.8)) * 1.5)
        direction = module.get('tick_direction', 'out')
        tick_length = round(float(module.get('tick_length', 4.0)) * 1.5)
        if direction == 'in':
            tick_length = -tick_length
        elif direction == 'inout':
            tick_length = -max(2, tick_length // 2)
        axis_pen = pg.mkPen('#707070', width=tick_width)
        for axis_name in ('bottom', 'left', 'top', 'right'):
            axis = plot.getAxis(axis_name)
            axis.enableAutoSIPrefix(False)
            axis.setPen(axis_pen)
            axis.setTickPen(axis_pen)
            axis.setStyle(
                tickLength=tick_length,
                maxTickLevel=2 if module.get('minor_ticks') else 0,
                showValues=axis_name in ('bottom', 'left'),
            )
            axis.setTickFont(QFont(
                module.get('font_family', 'Arial'),
                tick_size,
            ))
        marker = module.get('marker', 'none')
        symbol = None if marker == 'none' else marker
        for item in plot.listDataItems():
            item.setSymbol(symbol)
            if symbol is not None:
                item.setSymbolSize(7)
                line_pen = item.opts.get('pen')
                item.setSymbolPen(line_pen)
                color = (
                    line_pen.color()
                    if hasattr(line_pen, 'color') else line_pen
                )
                item.setSymbolBrush(pg.mkBrush(color))
        if module.get('legend', True):
            named_items = [
                item for item in plot.listDataItems() if item.name()
            ]
            if named_items:
                legend = plot.addLegend()
                for item in named_items:
                    legend.addItem(item, item.name())
                legend.setLabelTextSize(f'{tick_size}pt')
                location = module.get('legend_location', 'best')
                anchors = {
                    'upper left': ((0, 0), (0, 0), (10, 10)),
                    'upper right': ((1, 0), (1, 0), (-10, 10)),
                    'lower left': ((0, 1), (0, 1), (10, -10)),
                    'lower right': ((1, 1), (1, 1), (-10, -10)),
                    'best': ((1, 0), (1, 0), (-10, 10)),
                }
                item_pos, parent_pos, offset = anchors.get(
                    location, anchors['best']
                )
                legend.anchor(item_pos, parent_pos, offset=offset)

    def _render_overview(self):
        t = np.linspace(0, 6, 240)
        gate, transfer = self._transfer_example()
        for index, (module_id, name) in enumerate(MODULE_NAMES.items()):
            plot = self.preview_widget.addPlot(row=index // 3, col=index % 3)
            module = self._module_values[module_id]
            self._configure_preview_plot(plot, module, name, compact=True)
            primary = module.get('primary_color') or '#1565C0'
            secondary = module.get('secondary_color') or '#D32F2F'
            pen1 = pg.mkPen(primary, width=2.6)
            pen2 = pg.mkPen(secondary, width=2.4)
            if module_id == 'break_junction':
                x = np.linspace(0, .5, 240)
                plot.plot(x, np.exp(-7*x), pen=pen1)
            elif module_id == 'iv_curve':
                x = np.linspace(-.2, .2, 240)
                plot.plot(x, np.tanh(13*x), pen=pen1)
                plot.plot(x, np.tanh(13*(x+.018)), pen=pen2)
            elif module_id == 'isd_vg_setvsd':
                plot.plot(gate, transfer, pen=pen1)
            elif module_id == 'mapping_scan':
                x = np.linspace(-1.5, 1.5, 90)
                y = np.linspace(-1, 1, 70)
                xx, yy = np.meshgrid(x, y)
                z = np.cos(5*xx) ** 2 + np.abs(yy)
                image = pg.ImageItem(z.T)
                plot.addItem(image)
                plot.setAspectLocked(False)
            elif module_id == 'it_step_setgate':
                current = .5 + .08*np.sin(18*t) + np.where(t % 1.4 > .7, .2, 0)
                plot.plot(t, current, pen=pen1)
            else:
                drive = np.where(t % 1.5 < .75, .8, -.2)
                plot.plot(t, drive, pen=pen2)
                plot.plot(t, .2 + .6/(1+np.exp(-5*drive)), pen=pen1)

    def _render_preview(self):
        if not hasattr(self, 'preview_widget'):
            return
        self.preview_widget.setUpdatesEnabled(False)
        try:
            self._render_preview_content()
        finally:
            self.preview_widget.setUpdatesEnabled(True)
            self.preview_widget.update()

    def _render_preview_content(self):
        if not hasattr(self, 'preview_widget'):
            return
        self.preview_widget.clear()
        if self._current_module_id is None:
            self._render_overview()
            return
        self._save_current_module()
        module_id = self._current_module_id
        module = self._module_values[module_id]
        primary = module.get('primary_color') or '#1565C0'
        secondary = module.get('secondary_color') or '#D32F2F'
        width = max(3.0, float(module.get('line_width', 1.0)) * 3.0)
        pen1 = pg.mkPen(primary, width=width)
        pen2 = pg.mkPen(secondary, width=width)
        title = module.get('title', '')

        if module_id == 'break_junction':
            v = np.linspace(.005, .5, 320)
            g = np.exp(-7.2*v) * (1 + .08*np.sin(55*v))
            g[v > .20] *= .48
            g[v > .34] *= .30
            current_ma = v*g*7.748
            top = self.preview_widget.addPlot(row=0, col=0)
            bottom = self.preview_widget.addPlot(row=1, col=0)
            top.setXLink(bottom)
            top.plot(v, current_ma, pen=pen1)
            bottom.plot(v, g, pen=pen2)
            self._configure_preview_plot(
                top, module, title, '', 'Current (mA)'
            )
            self._configure_preview_plot(
                bottom, module, '', 'Voltage (V)', 'Conductance (G₀)'
            )
        elif module_id == 'iv_curve':
            v = np.linspace(-.22, .22, 320)
            gates = (-.6, 0, .6)
            colors = _module_palette(module, len(gates))
            gate_mode = module.get('iv_gate_mode', 'per_gate')
            file_mode = module.get('iv_file_mode', 'group')
            cycle_overlay = module.get('iv_cycle_mode') == 'overlay'

            def add_iv_curves(plot, gate_indices, segment='both'):
                for index in gate_indices:
                    gate = gates[index]
                    color = colors[index]
                    cycle_count = 3 if cycle_overlay else 1
                    for cycle in range(cycle_count):
                        alpha = 105 + cycle * 65 if cycle_overlay else 255
                        pen_color = QColor(color)
                        pen_color.setAlpha(alpha)
                        amplitude = (.5 + .25*index) * (1 + .025*cycle)
                        forward = amplitude*np.tanh(
                            12*(v-.01*gate-.002*cycle)
                        )
                        reverse = amplitude*np.tanh(
                            12*(v+.018-.01*gate+.002*cycle)
                        )
                        label = (
                            f'Gate voltage {gate:+.1f} V'
                            + (f', cycle {cycle + 1}' if cycle_overlay else '')
                        )
                        if segment in ('both', 'forward'):
                            plot.plot(
                                v, forward,
                                pen=pg.mkPen(pen_color, width=width),
                                name=label,
                            )
                        if segment in ('both', 'reverse'):
                            plot.plot(
                                v[::-1], reverse[::-1],
                                pen=pg.mkPen(
                                    pen_color, width=width,
                                    style=Qt.PenStyle.DashLine,
                                ),
                            )

            x_label = module.get('x_label') or 'Bias voltage (V)'
            y_label = _label_with_unit(
                module.get('y_label'), 'Current', 'nA'
            )
            if file_mode == 'separate':
                for column, (segment, segment_title) in enumerate((
                    ('forward', 'Forward data file'),
                    ('reverse', 'Reverse data file'),
                )):
                    plot = self.preview_widget.addPlot(row=0, col=column)
                    add_iv_curves(plot, [0], segment)
                    self._configure_preview_plot(
                        plot, module, segment_title, x_label, y_label
                    )
            elif gate_mode == 'small_multiples':
                for index, gate in enumerate(gates):
                    plot = self.preview_widget.addPlot(row=index, col=0)
                    add_iv_curves(plot, [index])
                    self._configure_preview_plot(
                        plot, module,
                        title if index == 0 else f'Gate voltage {gate:+.1f} V',
                        x_label if index == len(gates) - 1 else '',
                        y_label,
                    )
            else:
                plot = self.preview_widget.addPlot()
                gate_indices = range(len(gates)) if gate_mode == 'overlay' else [0]
                add_iv_curves(plot, gate_indices)
                self._configure_preview_plot(
                    plot, module, title, x_label, y_label
                )
        elif module_id == 'isd_vg_setvsd':
            gate, base = self._transfer_example()
            curves = [base*(.76+.16*i) + .04*i for i in range(4)]
            colors = _module_palette(module, len(curves))
            mode = module.get('isd_mode', 'overlay')
            if mode == 'stacked':
                previous = None
                for i, curve in enumerate(curves):
                    plot = self.preview_widget.addPlot(row=i, col=0)
                    if previous is not None:
                        plot.setXLink(previous)
                    plot.plot(
                        gate, curve,
                        pen=pg.mkPen(colors[i], width=width),
                    )
                    self._configure_preview_plot(
                        plot, module, title if i == 0 else '',
                        (module.get('x_label') or 'Gate voltage (V)')
                        if i == 3 else '',
                        f'Current {i+1} (nA)'
                    )
                    previous = plot
            else:
                plot = self.preview_widget.addPlot()
                separation = max(np.ptp(curve) for curve in curves) * 1.2
                for i, curve in enumerate(curves):
                    y = curve + (i*separation if mode == 'offset' else 0)
                    plot.plot(gate, y, pen=pg.mkPen(colors[i], width=width),
                              name=f'Sweep {i+1}')
                self._configure_preview_plot(
                    plot, module, title,
                    module.get('x_label') or 'Gate voltage (V)',
                    _label_with_unit(
                        module.get('y_label'), 'Current', 'nA'
                    ),
                )
        elif module_id == 'mapping_scan':
            plot = self.preview_widget.addPlot()
            gate = np.linspace(-1.6, 1.6, 180)
            bias = np.linspace(-120, 120, 140)
            x, y = np.meshgrid(gate, bias/120)
            threshold = .16+.7*(1-np.abs(((x+.4) % .8)-.4)/.4)
            didv = .06+1.5/(1+np.exp(-(np.abs(y)-threshold)/.045))
            image = pg.ImageItem(didv.T)
            image.setRect(QRectF(gate.min(), bias.min(),
                                 np.ptp(gate), np.ptp(bias)))
            try:
                image.setColorMap(pg.colormap.get(
                    module.get('colormap', 'viridis'), source='matplotlib'
                ))
            except Exception:
                pass
            plot.addItem(image)
            self._configure_preview_plot(
                plot, module, title + ' — Differential conductance preview',
                module.get('x_label') or 'Gate voltage (V)',
                module.get('y_label') or 'Bias voltage (mV)',
            )
        elif module_id == 'it_step_setgate':
            t = np.linspace(0, 12, 1200)
            raw = .62 + np.where(t % 2.8 > 1.45, .22, 0)
            raw += .045*np.sin(2*np.pi*8*t)+.018*np.sin(2*np.pi*37*t)
            left = self.preview_widget.addPlot(row=0, col=0)
            right = self.preview_widget.addPlot(row=0, col=1)
            left.plot(t, raw, pen=pen1)
            counts, edges = np.histogram(
                raw, bins=_histogram_bins(raw, module),
                density=module.get('it_hist_norm') == 'density',
            )
            centers = (edges[:-1] + edges[1:]) / 2
            right.plot(counts, centers, pen=pen2)
            self._configure_preview_plot(
                left, module, title,
                module.get('x_label') or 'Time (s)',
                _label_with_unit(
                    module.get('y_label'), 'Current', 'nA'
                ),
            )
            self._configure_preview_plot(
                right, module, 'Current distribution',
                'Probability density', 'Current (nA)'
            )
        else:
            t = np.linspace(0, 6, 900)
            drive = (
                np.where(t % 1.5 < .75, .8, -.2)
                if 'switch' in module_id
                else np.select([t < 1, t < 2.4, t < 3.2, t < 4.8],
                               [0, .65, -.25, 1], default=.25)
            )
            current = .15 + .9/(1+np.exp(-5*drive))
            current += .025*np.sin(2*np.pi*13*t)
            top = self.preview_widget.addPlot(row=0, col=0)
            bottom = self.preview_widget.addPlot(row=1, col=0)
            top.setXLink(bottom)
            top.plot(t, drive, pen=pen2)
            bottom.plot(t, current, pen=pen1)
            voltage_name = (
                'Gate voltage (V)'
                if module_id in ('gate_switch', 'arbitrary_gate')
                else 'Bias voltage (V)'
            )
            self._configure_preview_plot(top, module, title, '', voltage_name)
            self._configure_preview_plot(
                bottom, module, '', 'Time (s)', 'Current (nA)'
            )

    def _restore_defaults(self):
        defaults = default_plot_settings()
        self._initial = defaults
        self._module_values = copy.deepcopy(defaults['modules'])
        self.cb_auto.setChecked(defaults['auto_plot'])
        self.cb_partial.setChecked(defaults['plot_partial'])
        self.cb_preview.setChecked(defaults['show_preview'])
        self.combo_output.setCurrentIndex(0)
        self.edit_output.clear()
        if self._current_module_id is not None:
            self._load_module(self._current_module_id)
        self.update_preview()

    def settings(self):
        self._save_current_module()
        value = default_plot_settings()
        value.update({
            'auto_plot': self.cb_auto.isChecked(),
            'plot_partial': self.cb_partial.isChecked(),
            'show_preview': self.cb_preview.isChecked(),
            'output_mode': self.combo_output.currentData(),
            'output_folder': self.edit_output.text().strip(),
        })
        value['modules'] = copy.deepcopy(self._module_values)
        for module_id, module in value['modules'].items():
            parsed = {}
            for key in ('x_min', 'x_max', 'y_min', 'y_max'):
                text = str(module.get(key, '')).strip()
                parsed[key] = None if not text else float(text)
            if (
                parsed['x_min'] is not None
                and parsed['x_max'] is not None
                and parsed['x_min'] >= parsed['x_max']
            ):
                raise ValueError(f'{MODULE_NAMES[module_id]}：X 轴最小值必须小于最大值')
            if (
                parsed['y_min'] is not None
                and parsed['y_max'] is not None
                and parsed['y_min'] >= parsed['y_max']
            ):
                raise ValueError(f'{MODULE_NAMES[module_id]}：Y 轴最小值必须小于最大值')
            for key in ('primary_color', 'secondary_color'):
                color = str(module.get(key, '')).strip()
                if color and not QColor(color).isValid():
                    raise ValueError(
                        f'{MODULE_NAMES[module_id]}：颜色 {color} 无效'
                    )
            formats = module.get('formats', [])
            invalid = set(formats) - {'svg', 'pdf', 'png'}
            if not formats or invalid:
                raise ValueError(
                    f'{MODULE_NAMES[module_id]}：导出格式仅支持 svg、pdf、png'
                )
            if module_id == 'mapping_scan':
                window = int(module.get('savgol_window', 11))
                order = int(module.get('savgol_order', 2))
                if window % 2 == 0 or window <= order:
                    raise ValueError('二维 Mapping：平滑窗口必须为奇数且大于阶数')
        return value

    def _apply_without_close(self):
        try:
            value = self.settings()
        except ValueError as exc:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, '绘图设置错误', str(exc))
            return
        self.settings_applied.emit(value)

    def _save_profile(self):
        try:
            value = self.settings()
        except ValueError as exc:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, '绘图设置错误', str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, '保存绘图方案', '', '绘图方案 (*.json)'
        )
        if path:
            Path(path).write_text(
                json.dumps(value, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )

    def _load_profile(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '加载绘图方案', '', '绘图方案 (*.json)'
        )
        if not path:
            return
        try:
            value = merge_plot_settings(json.loads(
                Path(path).read_text(encoding='utf-8')
            ))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, '无法加载绘图方案', str(exc))
            return
        self._initial = value
        self._module_values = copy.deepcopy(value['modules'])
        self.cb_auto.setChecked(bool(value['auto_plot']))
        self.cb_partial.setChecked(bool(value['plot_partial']))
        self.cb_preview.setChecked(bool(value['show_preview']))
        self.combo_output.setCurrentIndex(max(
            0, self.combo_output.findData(value['output_mode'])
        ))
        self.edit_output.setText(value.get('output_folder', ''))
        if self._current_module_id:
            self._load_module(self._current_module_id)
        self.update_preview()

    def _set_as_default(self):
        try:
            value = self.settings()
            path = default_plot_profile_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(value, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
        except (OSError, ValueError) as exc:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, '无法保存默认方案', str(exc))
            return
        self.settings_applied.emit(value)
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(self, '绘图设置', '已保存为默认绘图方案。')

    def _accept_settings(self):
        try:
            self.settings()
        except ValueError as exc:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, '绘图设置错误', str(exc))
            return
        self.accept()


class PlotPreviewDialog(QDialog):
    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle('绘图结果')
        self.resize(1000, 760)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        for path in image_paths:
            label = _FitPixmapLabel(str(path))
            tabs.addTab(label, Path(path).name)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class _FitPixmapLabel(QLabel):
    """Preview an exported figure without introducing scroll bars or cropping."""

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self._source_pixmap = QPixmap(str(path))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored,
        )
        self.setMinimumSize(240, 180)
        self._update_scaled_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self._source_pixmap.isNull() or self.width() <= 0 or self.height() <= 0:
            return
        self.setPixmap(self._source_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))


class PlotManager(QObject):
    plot_finished = pyqtSignal(object, object, object)

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self.settings = merge_plot_settings(settings)
        self.latest_result = None
        self.latest_plot_paths = []
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='post-measurement-plot'
        )

    def update_settings(self, settings):
        self.settings = merge_plot_settings(settings)

    def handle_result(self, result):
        self.latest_result = copy.deepcopy(result)
        # A new measurement must not inherit preview paths from the previous
        # run when automatic plotting is disabled or skipped.
        self.latest_plot_paths = []
        self.plot_result(result, force=False)

    def plot_result(self, result=None, force=True):
        result = copy.deepcopy(result or self.latest_result)
        if not result:
            return False
        module_id = result.get('module_id')
        module_settings = self.settings['modules'].get(module_id, {})
        if not force and not self.settings['auto_plot']:
            return False
        if not module_settings.get('enabled', False):
            return False
        if result.get('status') == 'partial' and not self.settings['plot_partial']:
            return False
        if result.get('status') == 'error' or not result.get('data_files'):
            return False
        settings = copy.deepcopy(self.settings)
        future = self._executor.submit(render_result, result, settings)

        def completed(done):
            try:
                paths = done.result()
                error = None
            except Exception as exc:
                paths = []
                error = str(exc)
            latest_run_id = (
                self.latest_result.get('run_id')
                if self.latest_result else None
            )
            if result.get('run_id') == latest_run_id:
                self.latest_plot_paths = list(paths)
            self.plot_finished.emit(result, paths, error)

        future.add_done_callback(completed)
        return True


def _load_numeric(path):
    data = np.genfromtxt(path, comments='#', skip_header=1)
    if data.size == 0:
        return np.empty((0, 0))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def _output_directory(result, settings):
    if (
        settings['output_mode'] == 'custom'
        and str(settings.get('output_folder', '')).strip()
    ):
        output = Path(settings['output_folder'])
    else:
        paths = [Path(path).resolve() for path in result['data_files']]
        common = Path(os.path.commonpath([str(path.parent) for path in paths]))
        if common.name in ('pos', 'full'):
            common = common.parent
        output = common / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    return output


def _unique_output(folder, stem, suffix):
    candidate = folder / f'{stem}.{suffix}'
    index = 1
    while candidate.exists():
        candidate = folder / f'{stem}_{index:03d}.{suffix}'
        index += 1
    return candidate


def _save_figure(fig, path, dpi):
    temporary = path.with_name(f'.{path.stem}.tmp{path.suffix}')
    with mpl.rc_context({
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }):
        fig.savefig(
            temporary, dpi=dpi, format=path.suffix[1:],
        )
    os.replace(temporary, path)


def _decorate_axis(axis, settings, current_axis=False):
    module = settings['_module_settings']
    if module.get('grid', False):
        axis.grid(True, color='#B0B0B0', alpha=0.45)
    else:
        axis.grid(False)
    axis.set_xscale(module.get('x_scale', 'linear'))
    if current_axis:
        scale = module.get('current_scale', 'linear')
        axis.set_yscale(scale)
    if axis.get_xscale() == 'linear':
        x_formatter = ScalarFormatter(useOffset=False)
        x_formatter.set_scientific(False)
        axis.xaxis.set_major_formatter(x_formatter)
    if axis.get_yscale() == 'linear':
        y_formatter = ScalarFormatter(useOffset=False)
        y_formatter.set_scientific(False)
        axis.yaxis.set_major_formatter(y_formatter)
    x_min = str(module.get('x_min', '')).strip()
    x_max = str(module.get('x_max', '')).strip()
    y_min = str(module.get('y_min', '')).strip()
    y_max = str(module.get('y_max', '')).strip()
    if x_min or x_max:
        axis.set_xlim(
            left=float(x_min) if x_min else None,
            right=float(x_max) if x_max else None,
        )
    if y_min or y_max:
        axis.set_ylim(
            bottom=float(y_min) if y_min else None,
            top=float(y_max) if y_max else None,
        )


def _apply_figure_style(fig, settings, partial=False):
    module = settings['_module_settings']
    font = module.get('font_family', 'Arial')
    width = float(module.get('line_width', 0.8))
    marker = module.get('marker', 'none')
    background = '#FFFFFF'
    fig.patch.set_facecolor(background)
    data_axes = [
        axis for axis in fig.axes if axis.get_label() != '<colorbar>'
    ]
    for axis in fig.axes:
        axis.set_facecolor(background)
        for line in axis.lines:
            line.set_linewidth(width)
            if marker != 'none':
                line.set_marker(marker)
                line.set_markevery(max(1, len(line.get_xdata()) // 40))
                line.set_markersize(4)
        axis.title.set_fontfamily(font)
        axis.title.set_fontsize(module.get('title_size', 7))
        axis.xaxis.label.set_fontfamily(font)
        axis.yaxis.label.set_fontfamily(font)
        axis.xaxis.label.set_fontsize(module.get('label_size', 7))
        axis.yaxis.label.set_fontsize(module.get('label_size', 7))
        axis.tick_params(
            labelsize=module.get('tick_size', 6),
            direction=module.get('tick_direction', 'out'),
            length=module.get('tick_length', 3.0),
            width=module.get('tick_width', 0.6),
        )
        axis.spines['top'].set_visible(bool(module.get('top_spine', False)))
        axis.spines['right'].set_visible(bool(module.get('right_spine', False)))
        if module.get('minor_ticks'):
            axis.minorticks_on()
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_fontfamily(font)
        legend = axis.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontfamily(font)
                text.set_fontsize(module.get('tick_size', 6))
            legend.set_visible(bool(module.get('legend', True)))
    custom_title = str(module.get('title', '')).strip()
    if custom_title:
        if len(data_axes) > 1:
            fig.suptitle(
                custom_title + (' — PARTIAL' if partial else ''),
                family=font,
                fontsize=module.get('title_size', 7),
            )
        elif data_axes:
            data_axes[0].set_title(
                custom_title + (' — PARTIAL' if partial else ''),
                fontfamily=font,
                fontsize=module.get('title_size', 7),
            )


def _new_figure(module, **kwargs):
    size = (
        float(module.get('width_mm', 89.0)) / 25.4,
        float(module.get('height_mm', 70.0)) / 25.4,
    )
    return Figure(figsize=size, layout='constrained', **kwargs)


def _save_all_formats(fig, folder, stem, module):
    outputs = []
    formats = module.get('formats') or ['png']
    for suffix in formats:
        path = _unique_output(folder, stem, suffix)
        _save_figure(fig, path, int(module.get('dpi', 300)))
        outputs.append(str(path))
    return outputs


def _module_palette(module, count):
    def rgb(value, fallback):
        text = str(value or fallback).lstrip('#')
        if len(text) != 6:
            text = fallback.lstrip('#')
        return tuple(int(text[index:index + 2], 16) for index in (0, 2, 4))
    first = rgb(module.get('primary_color'), '#1565C0')
    last = rgb(module.get('secondary_color'), '#D32F2F')
    if count <= 1:
        return ['#%02x%02x%02x' % first]
    colors = []
    for index in range(count):
        fraction = index / (count - 1)
        value = tuple(round(
            first[channel] * (1 - fraction) + last[channel] * fraction
        ) for channel in range(3))
        colors.append('#%02x%02x%02x' % value)
    return colors


def _engineering_current(values):
    """Return scaled current values and an explicit engineering unit."""
    array = np.asarray(values, dtype=float)
    finite = np.abs(array[np.isfinite(array)])
    maximum = float(np.max(finite)) if finite.size else 0.0
    if maximum == 0:
        factor, unit = 1.0, 'A'
    elif maximum < 1e-9:
        factor, unit = 1e12, 'pA'
    elif maximum < 1e-6:
        factor, unit = 1e9, 'nA'
    elif maximum < 1e-3:
        factor, unit = 1e6, 'µA'
    elif maximum < 1:
        factor, unit = 1e3, 'mA'
    else:
        factor, unit = 1.0, 'A'
    return array * factor, unit, factor


def _engineering_voltage(values):
    """Return readable voltage values without scientific offset multipliers."""
    array = np.asarray(values, dtype=float)
    finite = np.abs(array[np.isfinite(array)])
    maximum = float(np.max(finite)) if finite.size else 0.0
    if 0 < maximum < 0.1:
        return array * 1e3, 'mV', 1e3
    return array, 'V', 1.0


def _label_with_unit(configured, fallback, unit):
    base = re.sub(
        r'\s*\([^)]*\)\s*$', '',
        str(configured or fallback).strip(),
    ).strip()
    return f'{base or fallback} ({unit})'


def _iv_segment_number(path):
    match = re.search(r'_(1st|2nd|3rd|4th)_', path.stem)
    if not match:
        return None
    return {'1st': 1, '2nd': 2, '3rd': 3, '4th': 4}[match.group(1)]


def _iv_curve_label(
    path, data, gate_label, cycle_key,
    include_gate=False, include_cycle=False,
):
    segment = _iv_segment_number(path)
    start = float(data[0, 0])
    stop = float(data[-1, 0])
    parts = []
    if include_gate and gate_label != 'No gate sweep':
        parts.append(gate_label)
    if segment is None:
        parts.append(f'{start:g} → {stop:g} V')
    else:
        parts.append(f'Segment {segment}: {start:g} → {stop:g} V')
    if include_cycle and cycle_key != 'cycle_unknown':
        parts.append(f'Cycle {int(cycle_key.split("_")[-1])}')
    return ', '.join(parts)


def _symmetric_norm(values):
    array = np.asarray(values, dtype=float)
    finite = np.abs(array[np.isfinite(array)])
    if not finite.size:
        return None
    limit = float(np.nanpercentile(finite, 99.5))
    if limit <= 0:
        limit = float(np.max(finite))
    if limit <= 0:
        return None
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def _histogram_bins(values, module):
    method = module.get('it_bin_method', 'fd')
    if method == 'manual_count':
        return max(2, int(module.get('it_bins', 50)))
    if method == 'manual_width':
        width = float(module.get('it_bin_width', 0.0))
        if width > 0 and np.ptp(values) > 0:
            return max(2, int(np.ceil(np.ptp(values) / width)))
        return 50
    return method


def _iv_file_keys(path, data):
    gate_match = re.search(r'VgSeq(\d+)_Vg=([-+.\deE]+)V', path.stem)
    cycle_match = re.search(r'cyc(\d+)', path.stem)
    if gate_match:
        gate_key = f'gate_{int(gate_match.group(1)):03d}'
        gate_label = f'Gate voltage {float(gate_match.group(2)):g} V'
    elif data.shape[1] >= 3 and np.isfinite(data[0, 2]):
        gate_key = f'gate_{float(data[0, 2]):+.9g}'
        gate_label = f'Gate voltage {float(data[0, 2]):g} V'
    else:
        gate_key = 'gate_none'
        gate_label = 'No gate sweep'
    cycle_key = (
        f'cycle_{int(cycle_match.group(1)):03d}'
        if cycle_match else 'cycle_unknown'
    )
    return gate_key, gate_label, cycle_key


def render_result(result, settings):
    module_id = result['module_id']
    module = settings['modules'][module_id]
    settings['_module_settings'] = module
    paths = [
        Path(path) for path in result.get('data_files', [])
        if Path(path).exists() and Path(path).suffix.lower() == '.txt'
    ]
    if not paths:
        raise ValueError('本次测试没有可读取的数据文件')
    output = _output_directory(result, settings)
    partial = result.get('status') == 'partial'
    status = ' — PARTIAL' if partial else ''
    stem_base = f"{module_id}_{result.get('run_id', 'latest')}"
    if partial:
        stem_base += '_partial'
    outputs = []

    if module_id == 'mapping_scan':
        full_paths = [
            path for path in paths
            if re.search(r'_full(?:_partial)?$', path.stem)
        ]
        datasets = []
        for path in full_paths:
            data = _load_numeric(path)
            if data.shape[1] >= 3 and len(data) >= 2:
                datasets.append((data[0, 0], data[:, 1], data[:, 2], path))
        if len(datasets) < 2:
            raise ValueError('二维 Mapping 至少需要两个有效 full 数据文件')
        datasets.sort(key=lambda item: item[0])
        min_len = min(len(item[1]) for item in datasets)
        gate = np.array([item[0] for item in datasets])
        bias = datasets[0][1][:min_len]
        current = np.vstack([item[2][:min_len] for item in datasets])
        window = min(int(module.get('savgol_window', 11)), min_len)
        if window % 2 == 0:
            window -= 1
        if window >= 3:
            try:
                from scipy.signal import savgol_filter
                order = min(int(module.get('savgol_order', 2)), window - 1)
                smooth = np.vstack([
                    savgol_filter(row, window, order) for row in current
                ])
            except ImportError:
                kernel = np.ones(window) / window
                smooth = np.vstack([
                    np.convolve(row, kernel, mode='same') for row in current
                ])
        else:
            smooth = current.copy()
        didv = np.gradient(smooth, bias, axis=1)
        mapping_sets = (
            ('raw_current_mapping', current * 1e9,
             'Raw current', 'Current (nA)', True),
            ('smoothed_current_mapping', smooth * 1e9,
             'Smoothed current', 'Current (nA)', True),
            ('differential_conductance_mapping', didv * 1e6,
             'Differential conductance', 'Differential conductance (µS)', True),
        )
        gate_scaled, gate_unit, _gate_factor = _engineering_voltage(gate)
        bias_scaled, bias_unit, _bias_factor = _engineering_voltage(bias)
        for stem, values, plot_title, color_label, centered in mapping_sets:
            fig = _new_figure(module)
            axis = fig.subplots()
            cmap = module.get('colormap', 'RdBu_r')
            if centered and cmap in ('viridis', 'plasma', 'inferno', 'magma'):
                cmap = 'RdBu_r'
            mesh = axis.pcolormesh(
                gate_scaled, bias_scaled, values.T, shading='auto',
                cmap=cmap,
                norm=_symmetric_norm(values) if centered else None,
            )
            axis.set_ylim(
                float(np.min(bias_scaled)), float(np.max(bias_scaled))
            )
            axis.set_xlabel(
                _label_with_unit(module.get('x_label'), 'Gate voltage', gate_unit)
            )
            axis.set_ylabel(
                _label_with_unit(module.get('y_label'), 'Bias voltage', bias_unit)
            )
            axis.set_title(f'{plot_title}{status}')
            colorbar = fig.colorbar(mesh, ax=axis)
            colorbar.set_label(color_label)
            colorbar_formatter = ScalarFormatter(useOffset=False)
            colorbar_formatter.set_scientific(False)
            colorbar.formatter = colorbar_formatter
            colorbar.update_ticks()
            _decorate_axis(axis, settings)
            _apply_figure_style(fig, settings, partial=partial)
            axis.set_title(f'{plot_title}{status}')
            outputs.extend(_save_all_formats(
                fig, output, stem + ('_partial' if partial else ''), module
            ))
        if module.get('mapping_full_iv', True):
            iv_folder = output / 'mapping' / 'full_iv'
            iv_folder.mkdir(parents=True, exist_ok=True)
            _scaled, iv_unit, iv_factor = _engineering_current(
                np.concatenate([item[2] for item in datasets])
            )
            summary_fig = _new_figure(module)
            summary_axis = summary_fig.subplots()
            for gate_value, voltage, current_values, source in datasets:
                fig = _new_figure(module)
                axis = fig.subplots()
                axis.plot(voltage, current_values * iv_factor)
                axis.set_xlabel('Bias voltage (V)')
                axis.set_ylabel(f'Current ({iv_unit})')
                axis.set_title(f'Gate voltage {gate_value:g} V{status}')
                _decorate_axis(axis, settings, current_axis=True)
                _apply_figure_style(fig, settings, partial=partial)
                axis.set_title(f'Gate voltage {gate_value:g} V{status}')
                outputs.extend(_save_all_formats(
                    fig, iv_folder, source.stem, module
                ))
                summary_axis.plot(
                    voltage, current_values * iv_factor,
                    label=f'{gate_value:g} V',
                )
            if module.get('mapping_full_iv_summary'):
                summary_axis.set_xlabel('Bias voltage (V)')
                summary_axis.set_ylabel(f'Current ({iv_unit})')
                summary_axis.set_title(f'Full current–voltage curves{status}')
                summary_axis.legend(loc=module.get('legend_location', 'best'))
                _decorate_axis(summary_axis, settings, current_axis=True)
                _apply_figure_style(summary_fig, settings, partial=partial)
                summary_axis.set_title(f'Full current–voltage curves{status}')
                outputs.extend(_save_all_formats(
                    summary_fig, iv_folder, 'full_iv_summary', module
                ))
        return outputs

    if module_id == 'break_junction':
        data = _load_numeric(paths[0])
        if data.shape[1] < 3:
            raise ValueError('断裂结数据列不足')
        fig = _new_figure(module)
        axes = fig.subplots(2, 1, sharex=True)
        colors = _module_palette(module, 2)
        current_scaled, current_unit, _factor = _engineering_current(data[:, 1])
        voltage_scaled, voltage_unit, _voltage_factor = _engineering_voltage(
            data[:, 0]
        )
        marker = 'o' if len(data) == 1 else None
        axes[0].plot(
            voltage_scaled, current_scaled, color=colors[0], marker=marker,
        )
        axes[0].set_ylabel(f'Current ({current_unit})')
        axes[1].plot(
            voltage_scaled, data[:, 2], color=colors[1], marker=marker,
        )
        axes[1].set_xlabel(f'Voltage ({voltage_unit})')
        axes[1].set_ylabel(r'Conductance ($G_0$)')
        for axis in axes:
            _decorate_axis(axis, settings, current_axis=axis is axes[0])
        fig.suptitle(module.get('title', 'Break junction') + status)
        _apply_figure_style(fig, settings, partial=partial)
        return _save_all_formats(fig, output, stem_base, module)

    if module_id == 'iv_curve':
        valid = []
        for path in paths:
            data = _load_numeric(path)
            if data.shape[1] >= 2:
                gate_key, gate_label, cycle_key = _iv_file_keys(path, data)
                valid.append(
                    (path, data, gate_key, gate_label, cycle_key)
                )
        if not valid:
            raise ValueError('IV 数据列不足')
        _scaled, current_unit, current_factor = _engineering_current(
            np.concatenate([item[1][:, 1] for item in valid])
        )
        if module.get('iv_file_mode') == 'separate':
            groups = [[item] for item in valid]
        else:
            grouped = {}
            gate_mode = module.get('iv_gate_mode', 'per_gate')
            cycle_mode = module.get('iv_cycle_mode', 'per_cycle')
            for item in valid:
                key = (
                    item[2] if gate_mode == 'per_gate' else 'all_gates',
                    item[4] if cycle_mode == 'per_cycle' else 'all_cycles',
                )
                grouped.setdefault(key, []).append(item)
            groups = list(grouped.values())
        for group_index, group_data in enumerate(groups, 1):
            fig = _new_figure(module)
            small_multiples = module.get('iv_gate_mode') == 'small_multiples'
            gate_keys = list(dict.fromkeys(item[2] for item in group_data))
            cycle_keys = list(dict.fromkeys(item[4] for item in group_data))
            if small_multiples:
                axes = np.atleast_1d(fig.subplots(
                    len(gate_keys), 1, sharex=True
                ))
                axes_by_gate = dict(zip(gate_keys, axes))
            else:
                axis = fig.subplots()
                axes = np.array([axis])
                axes_by_gate = {}
            palette = _module_palette(module, max(1, len(gate_keys)))
            gate_colors = dict(zip(gate_keys, palette))
            segment_colors = _module_palette(module, max(1, len(group_data)))
            all_bias = np.concatenate([item[1][:, 0] for item in group_data])
            _bias_scaled, bias_unit, bias_factor = _engineering_voltage(all_bias)
            for index, (path, data, _gate_key, gate_label, cycle_key) in enumerate(
                group_data
            ):
                axis = axes_by_gate.get(_gate_key, axes[0])
                alpha = .48 if module.get('iv_cycle_mode') == 'overlay' else 1
                label = _iv_curve_label(
                    path, data, gate_label, cycle_key,
                    include_gate=len(gate_keys) > 1,
                    include_cycle=len(cycle_keys) > 1,
                )
                segment = _iv_segment_number(path)
                if len(gate_keys) == 1 and len(group_data) > 1:
                    color = segment_colors[index]
                else:
                    color = gate_colors[_gate_key]
                axis.plot(
                    data[:, 0] * bias_factor,
                    data[:, 1] * current_factor, alpha=alpha,
                    label=label,
                    color=color,
                    linestyle=(
                        '--' if segment in (2, 4) or '_reverse_' in path.stem
                        else '-'
                    ),
                )
            for gate_key, axis in zip(gate_keys, axes):
                axis.set_ylabel(_label_with_unit(
                    module.get('y_label'), 'Current', current_unit
                ))
                if small_multiples:
                    gate_item = next(
                        item for item in group_data if item[2] == gate_key
                    )
                    axis.set_title(gate_item[3])
                _decorate_axis(axis, settings, current_axis=True)
                if module.get('legend') and len(group_data) <= 20:
                    location = module.get('legend_location', 'best')
                    if location == 'best' and len(group_data) > 1:
                        axis.legend(
                            loc='upper center', bbox_to_anchor=(0.5, -0.19),
                            ncol=2, frameon=False,
                        )
                    else:
                        axis.legend(loc=location)
            axes[-1].set_xlabel(
                _label_with_unit(module.get('x_label'), 'Bias voltage', bias_unit)
            )
            if not small_multiples:
                axes[0].set_title(module.get('title', '') + status)
            else:
                fig.suptitle(module.get('title', '') + status)
            _apply_figure_style(fig, settings, partial=partial)
            stem = (
                group_data[0][0].stem if len(groups) > 1
                else stem_base
            )
            outputs.extend(_save_all_formats(fig, output, stem, module))
        return outputs

    if module_id == 'isd_vg_setvsd':
        curves = []
        for path in paths:
            data = _load_numeric(path)
            if data.shape[1] >= 3:
                segment_match = re.search(r'_seg(\d+)', path.stem)
                label = (
                    f'Segment {int(segment_match.group(1))}'
                    if segment_match else path.stem
                )
                curves.append((label, data[:, 1], data[:, 2]))
        if not curves:
            raise ValueError('栅压特性数据列不足')
        _scaled, current_unit, current_factor = _engineering_current(
            np.concatenate([item[2] for item in curves])
        )
        _gate_scaled, gate_unit, gate_factor = _engineering_voltage(
            np.concatenate([item[1] for item in curves])
        )
        mode = module.get('isd_mode', 'overlay')
        colors = _module_palette(module, len(curves))
        if mode == 'stacked':
            fig = _new_figure(module)
            axes = fig.subplots(len(curves), 1, sharex=True)
            axes = np.atleast_1d(axes)
            fig.set_layout_engine(None)
            fig.subplots_adjust(hspace=0)
            for index, (label, gate, current) in enumerate(curves):
                axes[index].plot(
                    gate * gate_factor, current * current_factor,
                    label=label, color=colors[index]
                )
                axes[index].set_ylabel(
                    f'Current {index + 1} ({current_unit})'
                )
                _decorate_axis(axes[index], settings, current_axis=True)
                if index < len(axes) - 1:
                    axes[index].tick_params(labelbottom=False)
            axes[-1].set_xlabel(
                _label_with_unit(module.get('x_label'), 'Gate voltage', gate_unit)
            )
        else:
            fig = _new_figure(module)
            axis = fig.subplots()
            spans = [np.ptp(item[2]) for item in curves]
            auto_offset = max(spans or [0]) * 1.15
            offset = (
                float(module.get('isd_offset', 0.0))
                if module.get('isd_offset_mode') == 'manual'
                else auto_offset
            )
            for index, (label, gate, current) in enumerate(curves):
                values = (
                    current * current_factor
                    + (index * offset * current_factor
                       if mode == 'offset' else 0)
                )
                axis.plot(
                    gate * gate_factor, values, label=label, color=colors[index]
                )
            axis.set_xlabel(
                _label_with_unit(module.get('x_label'), 'Gate voltage', gate_unit)
            )
            axis.set_ylabel(_label_with_unit(
                module.get('y_label'), 'Current', current_unit
            ))
            _decorate_axis(axis, settings, current_axis=True)
            if module.get('legend'):
                location = module.get('legend_location', 'best')
                if location == 'best' and len(curves) > 1:
                    axis.legend(
                        loc='upper center', bbox_to_anchor=(0.5, -0.18),
                        ncol=2, frameon=False,
                    )
                else:
                    axis.legend(loc=location, framealpha=0.85)
            axes = [axis]
        if len(axes) > 1:
            fig.suptitle(module.get('title', '') + status)
        _apply_figure_style(fig, settings, partial=partial)
        return _save_all_formats(fig, output, stem_base, module)

    if module_id == 'it_step_setgate':
        for file_index, path in enumerate(paths, 1):
            data = _load_numeric(path)
            if data.shape[1] < 3:
                raise ValueError(f'It 数据列不足：{path.name}')
            time_values = data[:, 0]
            raw = data[:, -2]
            statistics = data[:, -1]
            _scaled, current_unit, current_factor = _engineering_current(
                np.concatenate([raw, statistics])
            )
            raw_scaled = raw * current_factor
            statistics_scaled = statistics * current_factor
            fig = _new_figure(module)
            axes = fig.subplots(1, 2)
            colors = _module_palette(module, 2)
            axes[0].plot(time_values, raw_scaled, color=colors[0])
            axes[0].set_xlabel(module.get('x_label') or 'Time (s)')
            axes[0].set_ylabel(f'Current ({current_unit})')
            axes[0].set_title('Raw current trace')
            density = module.get('it_hist_norm') == 'density'
            axes[1].hist(
                statistics_scaled,
                bins=_histogram_bins(statistics_scaled, module),
                density=density, orientation='horizontal',
                color=colors[1], alpha=.75,
            )
            mean = np.mean(statistics_scaled)
            median = np.median(statistics_scaled)
            std = np.std(statistics_scaled)
            if module.get('it_show_mean'):
                axes[1].axhline(mean, label='Mean')
            if module.get('it_show_median'):
                axes[1].axhline(median, linestyle='--', label='Median')
            if module.get('it_show_std'):
                axes[1].axhspan(
                    mean-std, mean+std, alpha=.12, label='±1 s.d.'
                )
            axes[1].set_xlabel(
                'Probability density' if density else 'Count'
            )
            axes[1].set_ylabel(f'Current ({current_unit})')
            axes[1].set_title('Current distribution')
            if module.get('legend'):
                axes[1].legend(loc=module.get('legend_location', 'best'))
            statistics_text = (
                f'Mean = {mean:.4g} {current_unit}\n'
                f's.d. = {std:.3g} {current_unit}'
            )
            axes[0].text(
                0.02, 0.98, statistics_text, transform=axes[0].transAxes,
                va='top', ha='left', fontsize=module.get('tick_size', 7),
                bbox={
                    'facecolor': 'white', 'edgecolor': '#BBBBBB',
                    'alpha': 0.85,
                },
            )
            for axis in axes:
                _decorate_axis(axis, settings, current_axis=True)
            fig.suptitle(module.get('title', '') + status)
            _apply_figure_style(fig, settings, partial=partial)
            stem = (
                stem_base if len(paths) == 1
                else f'{stem_base}_{file_index:03d}'
            )
            outputs.extend(_save_all_formats(fig, output, stem, module))
        return outputs

    data = _load_numeric(paths[0])
    if data.shape[1] < 3:
        raise ValueError('时间序列数据列不足')
    time_values = data[:, 0]

    fig = _new_figure(module)
    axes = fig.subplots(2, 1, sharex=True)
    if module_id in ('bias_switch', 'arbitrary_bias'):
        voltage_index = 2 if data.shape[1] >= 4 else 1
        current_index = data.shape[1] - 1
        voltage_label = 'Bias voltage (V)'
    elif module_id in ('gate_switch', 'arbitrary_gate'):
        voltage_index = 1
        current_index = 3 if data.shape[1] >= 4 else data.shape[1] - 1
        voltage_label = 'Gate voltage (V)'
    else:
        raise ValueError(f'不支持的绘图模块：{module_id}')
    voltage_scaled, voltage_unit, _voltage_factor = _engineering_voltage(
        data[:, voltage_index]
    )
    axes[0].plot(time_values, voltage_scaled)
    axes[0].set_ylabel(_label_with_unit(
        voltage_label, voltage_label.split(' (')[0], voltage_unit
    ))
    current_scaled, current_unit, _factor = _engineering_current(
        data[:, current_index]
    )
    colors = _module_palette(module, 2)
    axes[0].lines[-1].set_color(colors[1])
    axes[1].plot(time_values, current_scaled, color=colors[0])
    axes[1].set_ylabel(f'Current ({current_unit})')
    axes[1].set_xlabel(module.get('x_label') or 'Time (s)')
    _decorate_axis(axes[0], settings)
    _decorate_axis(axes[1], settings, current_axis=True)
    fig.suptitle(module.get('title', '') + status)
    _apply_figure_style(fig, settings, partial=partial)
    return _save_all_formats(fig, output, stem_base, module)
