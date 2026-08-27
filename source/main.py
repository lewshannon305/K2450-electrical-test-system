import os
import sys
import time

import json
import markdown
import pyvisa
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QDialog,
    QTextEdit,
    QLineEdit,
    QComboBox,
    QCheckBox,
    QRadioButton,
    QGridLayout,
    QSizePolicy,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QFontMetrics

from core.paths import readme_path
from core.hardware_base import validate_2450_idn
from core.instrument_config import InstrumentSettings, UNSELECTED_TEXT
from modules.iv_curve import IVWidget
from modules.isd_vg_setvsd import IsdVgSetVsdWidget
from modules.break_junction import BreakJunctionWidget
from modules.mapping_scan import MappingWidget
from modules.it_step_setgate import ItStepWidget
from modules.arbitrary_bias import ArbitraryBiasWidget
from modules.arbitrary_gate import ArbitraryGateWidget
from core.plotting import (
    PlotManager,
    PlotPreviewDialog,
    PlotSettingsDialog,
    default_plot_settings,
    load_default_plot_settings,
    merge_plot_settings,
    select_preview_paths,
)


def parse_config_modules(config_data):
    if not isinstance(config_data, dict):
        raise ValueError('配置文件根节点必须是JSON对象')
    schema_version = config_data.get('__schema_version__', 1)
    try:
        schema_version = int(schema_version)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'无效配置版本: {schema_version}') from exc
    if schema_version not in (1, 2):
        raise ValueError(f'不支持的配置版本: {schema_version}')
    modules = config_data.get('modules', {}) if schema_version == 2 else config_data
    if not isinstance(modules, dict):
        raise ValueError('配置中的 modules 必须是JSON对象')
    return schema_version, modules


class RunGuard:
    def __init__(self):
        self.active_id = None
        self.active_name = None

    def request_start(self, module_id, module_name):
        if self.active_id is None:
            self.active_id = module_id
            self.active_name = module_name
            return True
        return False

    def finish(self, module_id):
        if self.active_id == module_id:
            self.active_id = None
            self.active_name = None


class WelcomePage(QWidget):
    def __init__(self, instrument_settings, parent=None):
        super().__init__(parent)
        self.instrument_settings = instrument_settings
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        left = QFrame()
        left.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(10)

        # 使用单个 QLabel 且以像素精确控制 line-height，基于 tips 的字体行距按比例计算
        title_font = QFont("Arial", 18)
        title_font.setWeight(QFont.Weight.Bold)

        # 参照 tips 使用的 12pt 字体，按比例计算对应的像素行高
        ref_font_pt = 12
        ref_fm = QFontMetrics(QFont('Arial', ref_font_pt))
        ref_line_px = ref_fm.lineSpacing()
        # 将参考像素行高按字号比例缩放到 title 字号
        desired_line_px = int(ref_line_px * (title_font.pointSize() / ref_font_pt))

        title = QLabel()
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(title_font)
        title.setText(
            f"<div style='font-family: Arial; font-size:18pt; font-weight:700; line-height:{desired_line_px}px; text-align:center;'>"
            "欢迎使用 Keithley2450 电学测试系统"
            "</div>"
        )

        left_layout.addStretch()
        left_layout.addWidget(title)
        left_layout.addWidget(self._build_instrument_group())
        left_layout.addStretch()

        right = QFrame()
        right.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding
        )
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(10)

        tips = QGroupBox("系统操作指南")
        tips_font = QFont("Arial", 12)
        tips_font.setWeight(QFont.Weight.Bold)
        tips.setFont(tips_font)
        tips.setStyleSheet('QGroupBox { font-size: 12pt; font-weight: bold; }')
        tips_layout = QVBoxLayout(tips)
        tips_layout.setContentsMargins(10, 10, 10, 10)
        tips_layout.setSpacing(10)
        
        tips_html = (
            "<p style='margin: 10px 0; font-size: 12pt;'><b>1. 硬件连接：</b>请确保 Keithley 源表已通过 GPIB 妥善连接。</p>"
            "<p style='margin: 10px 0; font-size: 12pt;'><b>2. 初始化：</b>请先在欢迎页点击“扫描设备”，选择并检测仪器连接。</p>"
            "<p style='margin: 10px 0; font-size: 12pt;'><b>3. 参数保存：</b>本系统支持“文件 -> 保存/加载配置”，可一键备份并恢复所有 7 个模块的测试参数。</p>"
            "<p style='margin: 10px 0; font-size: 12pt;'><b>4. 扫描顺序：</b>所有模块的步长输入均强制要求为正，扫描方向由起始终止电压的大小关系自动确定。</p>"
            "<p style='margin: 10px 0; font-size: 12pt;'><b>5. 极限性能警告：</b>高频采样时，请尽量避免后台高负载任务。</p>"
            "<p style='margin: 10px 0; font-size: 12pt;'><b>6. 紧急制动：</b>若测试遇险，请立即点击各界面右下角红色“强制终止”按钮。</p>"
        )
        tips_label = QLabel(tips_html)
        tips_label.setWordWrap(True)
        tips_label.setFont(QFont('Arial', 12))
        tips_layout.addWidget(tips_label)

        right_layout.addStretch()
        right_layout.addWidget(tips)
        right_layout.addStretch()

        layout.addWidget(left, stretch=1)
        layout.addWidget(right, stretch=1)

    def _build_instrument_group(self):
        group = QGroupBox('仪器选择')
        bold_font = QFont('Arial', 12)
        bold_font.setWeight(QFont.Weight.Bold)
        ui_font = QFont('Arial', 12)
        ui_font.setWeight(QFont.Weight.Normal)
        group.setFont(bold_font)

        grid = QGridLayout(group)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setHorizontalSpacing(10)

        self.bias_address = QComboBox()
        self.bias_address.setEditable(True)
        self.bias_address.setFont(ui_font)
        self.bias_address.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self.bias_terminal = QComboBox()
        self.bias_terminal.addItems(['REAR', 'FRONT'])
        self.bias_terminal.setFont(ui_font)

        self.gate_address = QComboBox()
        self.gate_address.setEditable(True)
        self.gate_address.setFont(ui_font)
        self.gate_address.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self.gate_terminal = QComboBox()
        self.gate_terminal.addItems(['REAR', 'FRONT'])
        self.gate_terminal.setFont(ui_font)

        self.bias_status = QLabel('● 未选择')
        self.gate_status = QLabel('● 未选择（可选）')
        self.summary_status = QLabel('尚未选择仪器')
        self.summary_status.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        for label in (self.bias_status, self.gate_status, self.summary_status):
            label.setFont(ui_font)

        for row, text, address, terminal, status in (
            (0, '偏压表:', self.bias_address, self.bias_terminal,
             self.bias_status),
            (1, '栅压表:', self.gate_address, self.gate_terminal,
             self.gate_status),
        ):
            label = QLabel(text)
            label.setFont(ui_font)
            grid.addWidget(label, row, 0)
            grid.addWidget(address, row, 1)
            grid.addWidget(terminal, row, 2)
            grid.addWidget(status, row, 3)

        self.btn_scan = QPushButton('扫描设备')
        self.btn_scan.setFont(bold_font)
        self.btn_scan.setFixedSize(100, 30)
        self.btn_scan.clicked.connect(self.scan_instruments)
        self.btn_detect = QPushButton('检测连接')
        self.btn_detect.setFont(bold_font)
        self.btn_detect.setFixedSize(100, 30)
        self.btn_detect.clicked.connect(self.detect_connections)

        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.btn_scan)
        buttons.addWidget(self.btn_detect)
        buttons.addWidget(self.summary_status)
        buttons.addStretch()
        grid.addLayout(buttons, 2, 0, 1, 4)

        self.bias_address.currentTextChanged.connect(self._selection_changed)
        self.gate_address.currentTextChanged.connect(self._selection_changed)
        self.bias_terminal.currentTextChanged.connect(self._selection_changed)
        self.gate_terminal.currentTextChanged.connect(self._selection_changed)
        self.set_settings(self.instrument_settings)
        return group

    @staticmethod
    def _address_text(combo):
        text = combo.currentText().strip()
        return '' if text == UNSELECTED_TEXT else text

    def _sync_settings(self):
        self.instrument_settings.update(
            self._address_text(self.bias_address),
            self.bias_terminal.currentText(),
            self._address_text(self.gate_address),
            self.gate_terminal.currentText(),
        )

    def _selection_changed(self):
        self._sync_settings()
        bias_selected = bool(self.instrument_settings.bias_address)
        gate_selected = bool(self.instrument_settings.gate_address)
        self.bias_terminal.setEnabled(bias_selected)
        self.gate_terminal.setEnabled(gate_selected)
        self._set_status(
            self.bias_status,
            '未检测' if bias_selected else '未选择',
            '#666666',
        )
        self._set_status(
            self.gate_status,
            '未检测' if gate_selected else '未选择（可选）',
            '#666666',
        )
        self.summary_status.setText(
            '请检测连接' if bias_selected else '尚未选择仪器'
        )
        self.summary_status.setStyleSheet('color: #666666;')

    @staticmethod
    def _set_status(label, text, color, tooltip=''):
        label.setText(f'● {text}')
        label.setStyleSheet(f'color: {color};')
        label.setToolTip(tooltip)

    @staticmethod
    def _replace_combo_items(combo, resources, selected, fallback):
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(UNSELECTED_TEXT)
        combo.addItems(resources)
        target = selected if selected in resources else fallback
        combo.setCurrentText(target or UNSELECTED_TEXT)
        combo.blockSignals(False)

    def scan_instruments(self):
        bias_current = self._address_text(self.bias_address)
        gate_current = self._address_text(self.gate_address)
        try:
            rm = pyvisa.ResourceManager()
            try:
                resources = list(rm.list_resources())
            finally:
                close = getattr(rm, 'close', None)
                if callable(close):
                    close()
        except Exception as exc:
            self.summary_status.setText(f'设备扫描失败: {exc}')
            self.summary_status.setStyleSheet('color: #AA0000;')
            return

        bias_fallback = resources[0] if resources else ''
        gate_fallback = resources[1] if len(resources) > 1 else ''
        self._replace_combo_items(
            self.bias_address, resources, bias_current, bias_fallback
        )
        self._replace_combo_items(
            self.gate_address, resources, gate_current, gate_fallback
        )
        self._selection_changed()
        if not resources:
            self.summary_status.setText('未扫描到可用仪器')

    @staticmethod
    def _detect_one(resource_manager, address):
        instrument = resource_manager.open_resource(address)
        try:
            instrument.timeout = 3000
            return validate_2450_idn(instrument.query('*IDN?'))
        finally:
            close = getattr(instrument, 'close', None)
            if callable(close):
                close()

    def detect_connections(self):
        self._sync_settings()
        bias_ok = False
        gate_ok = False
        rm = None
        try:
            rm = pyvisa.ResourceManager()
            if self.instrument_settings.bias_address:
                try:
                    self._detect_one(rm, self.instrument_settings.bias_address)
                    bias_ok = True
                    self._set_status(self.bias_status, '已连接', '#159A2E')
                except Exception as exc:
                    self._set_status(
                        self.bias_status, '连接失败', '#AA0000', str(exc)
                    )
            else:
                self._set_status(self.bias_status, '未选择', '#666666')

            if self.instrument_settings.gate_address:
                if (
                    self.instrument_settings.gate_address.lower()
                    == self.instrument_settings.bias_address.lower()
                ):
                    self._set_status(self.gate_status, '地址重复', '#AA0000')
                else:
                    try:
                        self._detect_one(
                            rm, self.instrument_settings.gate_address
                        )
                        gate_ok = True
                        self._set_status(
                            self.gate_status, '已连接', '#159A2E'
                        )
                    except Exception as exc:
                        self._set_status(
                            self.gate_status, '连接失败', '#AA0000', str(exc)
                        )
            else:
                self._set_status(
                    self.gate_status, '未选择（可选）', '#666666'
                )
        except Exception as exc:
            self.summary_status.setText(f'连接检测失败: {exc}')
            self.summary_status.setStyleSheet('color: #AA0000;')
            return
        finally:
            if rm is not None:
                close = getattr(rm, 'close', None)
                if callable(close):
                    close()

        if bias_ok and gate_ok:
            text, color = '两台 2450 已连接，可使用全部测试', '#159A2E'
        elif bias_ok:
            text, color = '已连接 1 台，可使用单表测试', '#159A2E'
        else:
            text, color = '偏压表未连接，暂时无法测试', '#AA0000'
        self.summary_status.setText(text)
        self.summary_status.setStyleSheet(f'color: {color};')

    def set_settings(self, settings):
        for combo, value in (
            (self.bias_address, settings.bias_address),
            (self.gate_address, settings.gate_address),
        ):
            combo.blockSignals(True)
            if combo.findText(UNSELECTED_TEXT) < 0:
                combo.addItem(UNSELECTED_TEXT)
            if value and combo.findText(value) < 0:
                combo.addItem(value)
            combo.setCurrentText(value or UNSELECTED_TEXT)
            combo.blockSignals(False)
        self.bias_terminal.setCurrentText(settings.bias_terminal)
        self.gate_terminal.setCurrentText(settings.gate_terminal)
        self._selection_changed()


class PlaceholderPage(QWidget):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label = QLabel(f"{title}\n(待接入模块)")
        font = QFont("Arial", 16)
        font.setWeight(QFont.Weight.Bold)
        label.setFont(font)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Keithley 2450 电学测试系统")

        self.run_guard = RunGuard()
        self.instrument_settings = InstrumentSettings()
        self.plot_settings = load_default_plot_settings(default_plot_settings())
        self.plot_manager = PlotManager(self.plot_settings, self)
        self.plot_manager.plot_finished.connect(self._on_plot_finished)
        self._plot_preview_dialogs = []
        self._build_menu()
        self._build_ui()

    def _build_menu(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("文件")
        action_save = file_menu.addAction("保存配置")
        action_load = file_menu.addAction("加载配置")
        action_save.triggered.connect(self.save_config)
        action_load.triggered.connect(self.load_config)

        self.measure_menu = menu_bar.addMenu("测量")

        self.plot_menu = menu_bar.addMenu("绘图")
        self.action_auto_plot = self.plot_menu.addAction(
            "测试结束后自动绘图"
        )
        self.action_auto_plot.setCheckable(True)
        self.action_auto_plot.setChecked(self.plot_settings['auto_plot'])
        self.action_auto_plot.toggled.connect(self._set_auto_plot)
        self.plot_menu.addSeparator()
        action_plot_settings = self.plot_menu.addAction("绘图设置...")
        action_plot_settings.triggered.connect(self.open_plot_settings)
        action_plot_latest = self.plot_menu.addAction("绘制最近一次测试")
        action_plot_latest.triggered.connect(self.plot_latest_result)
        action_show_latest = self.plot_menu.addAction("查看最近绘图")
        action_show_latest.triggered.connect(self.show_latest_plots)
        action_open_plot_dir = self.plot_menu.addAction("打开最近绘图目录")
        action_open_plot_dir.triggered.connect(self.open_latest_plot_folder)

        help_menu = menu_bar.addMenu("帮助")
        action_readme = help_menu.addAction("Readme.md")
        action_readme.triggered.connect(self.open_readme)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)

        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(6)

        self.stack = QStackedWidget()

        self.page_names = [
            "欢迎",
            "断裂结",
            "循环IV特性扫描",
            "栅压特性扫描",
            "二维Mapping扫描",
            "It特性扫描",
            "任意偏压波形测试",
            "任意栅压波形测试",
        ]

        self.welcome_page = WelcomePage(self.instrument_settings)
        self.stack.addWidget(self.welcome_page)

        for name in self.page_names[1:]:
            if name == "断裂结":
                self.stack.addWidget(BreakJunctionWidget(
                    run_guard=self.run_guard,
                    instrument_settings=self.instrument_settings,
                ))
            elif name == "循环IV特性扫描":
                self.stack.addWidget(IVWidget(
                    run_guard=self.run_guard,
                    instrument_settings=self.instrument_settings,
                ))
            elif name == "栅压特性扫描":
                self.stack.addWidget(IsdVgSetVsdWidget(
                    run_guard=self.run_guard,
                    instrument_settings=self.instrument_settings,
                ))
            elif name == "二维Mapping扫描":
                self.stack.addWidget(MappingWidget(
                    run_guard=self.run_guard,
                    instrument_settings=self.instrument_settings,
                ))
            elif name == "It特性扫描":
                self.stack.addWidget(ItStepWidget(
                    run_guard=self.run_guard,
                    instrument_settings=self.instrument_settings,
                ))
            elif name == "任意偏压波形测试":
                self.stack.addWidget(ArbitraryBiasWidget(
                    run_guard=self.run_guard,
                    instrument_settings=self.instrument_settings,
                ))
            elif name == "任意栅压波形测试":
                self.stack.addWidget(ArbitraryGateWidget(
                    run_guard=self.run_guard,
                    instrument_settings=self.instrument_settings,
                ))
            else:
                self.stack.addWidget(PlaceholderPage(name))

        for idx, name in enumerate(self.page_names):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setMinimumHeight(36)
            if idx == 0:
                btn.setChecked(True)
            self.nav_group.addButton(btn, idx)
            nav_layout.addWidget(btn)
            btn.clicked.connect(lambda checked, i=idx: self.stack.setCurrentIndex(i))

        for idx, name in enumerate(self.page_names[1:], start=1):
            action = self.measure_menu.addAction(name)
            action.triggered.connect(lambda checked, i=idx: self._select_page(i))

        for widget in self._measurement_widgets():
            widget.result_ready.connect(self.plot_manager.handle_result)

        main_layout.addWidget(nav_widget)
        main_layout.addWidget(self.stack, stretch=1)

    def _select_page(self, index):
        self.stack.setCurrentIndex(index)
        button = self.nav_group.button(index)
        if button is not None:
            button.setChecked(True)

    def _measurement_widgets(self):
        widgets = []
        for i in range(1, self.stack.count()):
            widget = self.stack.widget(i)
            if hasattr(widget, 'is_measurement_active'):
                widgets.append(widget)
        return widgets

    def _active_measurement_widgets(self):
        return [
            widget for widget in self._measurement_widgets()
            if widget.is_measurement_active()
        ]

    def closeEvent(self, event):
        active_widgets = self._active_measurement_widgets()
        if not active_widgets:
            event.accept()
            return

        active_name = self.run_guard.active_name or '当前模块'
        reply = QMessageBox.question(
            self,
            '警告',
            f'{active_name} 测量正在运行中，确认要退出吗？\n'
            '确认后将先请求强制停止并等待仪器清理完成。',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            event.ignore()
            return

        for widget in active_widgets:
            widget.request_shutdown_for_close(force=True)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            QApplication.processEvents()
            if not any(widget.is_measurement_active() for widget in active_widgets):
                event.accept()
                return
            time.sleep(0.05)

        self._show_info(
            '正在停止',
            '已发送停止请求，但测量线程仍在清理仪器或等待通信超时。\n'
            '请稍后确认状态稳定后再关闭程序。',
        )
        event.ignore()

    def save_config(self):
        try:
            fp, _ = QFileDialog.getSaveFileName(self, "保存测试配置", "", "JSON Files (*.json)")
            if not fp:
                return

            config_data = {
                '__schema_version__': 2,
                'instruments': self.instrument_settings.to_config(),
                'modules': {},
                'plotting': self.plot_settings,
            }
            # 遍历所有子页面模块
            for i in range(1, self.stack.count()):
                widget = self.stack.widget(i)
                if not hasattr(widget, 'module_id') or not hasattr(widget, 'inputs'):
                    continue
                
                mod_data = {}
                # 遍历通用表单控件
                for key, control in widget.inputs.items():
                    if isinstance(control, QComboBox):
                        mod_data[key] = control.currentText()
                    elif isinstance(control, QLineEdit):
                        mod_data[key] = control.text()
                
                controls = {}
                for attr_name, attr_value in vars(widget).items():
                    if isinstance(attr_value, (QCheckBox, QRadioButton)):
                        controls[attr_name] = attr_value.isChecked()
                if controls:
                    mod_data['__controls__'] = controls
                if hasattr(widget, 'ent_folder') and isinstance(widget.ent_folder, QLineEdit):
                    mod_data['__folder__'] = widget.ent_folder.text()
                if hasattr(widget, 'folder_input') and isinstance(widget.folder_input, QLineEdit):
                    mod_data['__folder__'] = widget.folder_input.text()
                if hasattr(widget, 'waveform'):
                    mod_data['__waveform__'] = [
                        [float(value) for value in row]
                        for row in widget.waveform
                    ]
                if hasattr(widget, 'gate_settings'):
                    if hasattr(widget, 'current_gate_settings'):
                        mod_data['__gate_settings__'] = (
                            widget.current_gate_settings()
                        )
                    else:
                        mod_data['__gate_settings__'] = dict(
                            widget.gate_settings
                        )

                config_data['modules'][widget.module_id] = mod_data

            with open(fp, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)
            self._show_info("成功", f"配置已保存至：\n{fp}")

        except Exception as e:
            self._show_info("错误", f"保存配置失败: {e}")

    def load_config(self):
        try:
            fp, _ = QFileDialog.getOpenFileName(self, "加载测试配置", "", "JSON Files (*.json)")
            if not fp:
                return

            with open(fp, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            schema_version, module_data = parse_config_modules(config_data)

            self.instrument_settings.load_config(
                config_data.get('instruments', {})
            )
            self.welcome_page.set_settings(self.instrument_settings)

            for i in range(1, self.stack.count()):
                widget = self.stack.widget(i)
                if not hasattr(widget, 'module_id') or not hasattr(widget, 'inputs'):
                    continue
                
                mod_data = module_data.get(widget.module_id, {})
                if not mod_data:
                    continue

                for key, control in widget.inputs.items():
                    if key in mod_data:
                        val = mod_data[key]
                        if isinstance(control, QComboBox):
                            idx = control.findText(val)
                            if idx >= 0:
                                control.setCurrentIndex(idx)
                            else:
                                control.setEditText(val)
                        elif isinstance(control, QLineEdit):
                            if key.lower() in {'folder', 'file_folder'} and not str(val).strip():
                                continue
                            control.setText(str(val))

                if '__cb_gate__' in mod_data and hasattr(widget, 'cb_gate'):
                    widget.cb_gate.setChecked(bool(mod_data['__cb_gate__']))
                for attr_name, checked in mod_data.get('__controls__', {}).items():
                    control = getattr(widget, attr_name, None)
                    if isinstance(control, (QCheckBox, QRadioButton)):
                        control.setChecked(bool(checked))
                if str(mod_data.get('__folder__', '')).strip():
                    if hasattr(widget, 'ent_folder'):
                        widget.ent_folder.setText(str(mod_data['__folder__']))
                    if hasattr(widget, 'folder_input'):
                        widget.folder_input.setText(str(mod_data['__folder__']))
                if '__waveform__' in mod_data and hasattr(widget, 'waveform'):
                    waveform = [
                        [float(value) for value in row]
                        for row in mod_data['__waveform__']
                    ]
                    if all(len(row) == 2 and row[1] > 0 for row in waveform):
                        widget.waveform = waveform
                        if hasattr(widget, 'lbl_wave_info'):
                            label = (
                                '当前栅压波形段数'
                                if widget.module_id == 'arbitrary_gate'
                                else '当前偏压波形段数'
                            )
                            widget.lbl_wave_info.setText(
                                f'{label}: {len(waveform)}'
                            )
                if (
                    '__gate_settings__' in mod_data
                    and hasattr(widget, 'gate_settings')
                ):
                    gate_settings = mod_data['__gate_settings__']
                    if isinstance(gate_settings, dict):
                        widget.gate_settings.update(gate_settings)
                        if hasattr(
                            widget, 'apply_gate_settings_to_main_controls'
                        ):
                            widget.apply_gate_settings_to_main_controls()
                        if hasattr(widget, '_update_gate_summary'):
                            widget._update_gate_summary()

            self.plot_settings = merge_plot_settings(
                config_data.get('plotting')
            )
            self.plot_manager.update_settings(self.plot_settings)
            self.action_auto_plot.blockSignals(True)
            self.action_auto_plot.setChecked(
                self.plot_settings['auto_plot']
            )
            self.action_auto_plot.blockSignals(False)

            self._show_info("成功", "所有模块参数配置已恢复！")

        except Exception as e:
            self._show_info("错误", f"加载配置失败: {e}")

    def open_readme(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Readme.md")
        dialog.resize(900, 700)
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowType.WindowMinMaxButtonsHint)
        layout = QVBoxLayout(dialog)
        
        from PyQt6.QtWidgets import QTextBrowser
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        md_file = readme_path()
        if md_file.exists():
            try:
                with md_file.open("r", encoding="utf-8") as f:
                    md_text = f.read()
                html_content = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'codehilite'])
                full_html = f"""
                <html>
                <head>
                    <style>
                        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; font-size: 14px; line-height: 1.6; padding: 20px; }}
                        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                        h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 8px; }}
                        h3 {{ color: #7f8c8d; }}
                        code {{ background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: 'Consolas', monospace; }}
                        pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                        pre code {{ background-color: transparent; padding: 0; }}
                        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #3498db; color: white; }}
                        tr:nth-child(even) {{ background-color: #f2f2f2; }}
                        a {{ color: #3498db; text-decoration: none; }}
                        a:hover {{ text-decoration: underline; }}
                        blockquote {{ border-left: 4px solid #3498db; padding-left: 15px; color: #7f8c8d; margin: 10px 0; }}
                        ul, ol {{ padding-left: 30px; }}
                    </style>
                </head>
                <body>
                {html_content}
                </body>
                </html>
                """
                browser.setHtml(full_html)
            except Exception as e:
                browser.setPlainText(f"无法渲染文件: {e}")
        else:
            browser.setPlainText("未找到 Readme.md 文件。")
            
        layout.addWidget(browser)
        dialog.show()

    def _show_info(self, title, text):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.NoIcon)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def _set_auto_plot(self, checked):
        self.plot_settings['auto_plot'] = bool(checked)
        self.plot_manager.update_settings(self.plot_settings)
        self.statusBar().showMessage(
            '测试结束后自动绘图已开启'
            if checked else '测试结束后自动绘图已关闭',
            4000,
        )

    def open_plot_settings(self):
        dialog = PlotSettingsDialog(self.plot_settings, self)
        dialog.settings_applied.connect(self._apply_plot_settings)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            settings = dialog.settings()
        except Exception as exc:
            self._show_info('绘图设置错误', str(exc))
            return
        self._apply_plot_settings(settings)
        self.statusBar().showMessage('绘图设置已应用', 4000)

    def _apply_plot_settings(self, settings):
        self.plot_settings = merge_plot_settings(settings)
        self.plot_manager.update_settings(self.plot_settings)
        self.action_auto_plot.blockSignals(True)
        self.action_auto_plot.setChecked(self.plot_settings['auto_plot'])
        self.action_auto_plot.blockSignals(False)
        self.statusBar().showMessage('绘图设置已应用', 4000)

    def plot_latest_result(self):
        if not self.plot_manager.latest_result:
            self._show_info('绘图', '当前还没有可绘制的测试结果。')
            return
        if not self.plot_manager.plot_result(force=True):
            self._show_info(
                '绘图',
                '最近一次测试没有有效数据，或该模块已在绘图设置中禁用。',
            )
            return
        self.statusBar().showMessage('正在绘制最近一次测试...', 5000)

    def show_latest_plots(self, paths=None, module_id=None):
        if paths is None:
            paths = self.plot_manager.latest_plot_paths
        if module_id is None and self.plot_manager.latest_result:
            module_id = self.plot_manager.latest_result.get('module_id')
        paths = select_preview_paths(paths, module_id)
        if not paths:
            self._show_info('绘图', '当前没有可预览的 PNG 绘图。')
            return
        dialog = PlotPreviewDialog(paths, self)
        self._plot_preview_dialogs.append(dialog)
        dialog.finished.connect(
            lambda _result, item=dialog: self._plot_preview_dialogs.remove(item)
            if item in self._plot_preview_dialogs else None
        )
        dialog.show()

    def open_latest_plot_folder(self):
        paths = self.plot_manager.latest_plot_paths
        if not paths:
            self._show_info('绘图', '当前没有绘图输出目录。')
            return
        folder = os.path.dirname(paths[0])
        try:
            os.startfile(folder)
        except Exception as exc:
            self._show_info('无法打开目录', str(exc))

    def _on_plot_finished(self, result, paths, error):
        module_name = result.get('module_name', result.get('module_id', '测试'))
        if error:
            self.statusBar().showMessage(
                f'{module_name} 自动绘图失败: {error}', 12000
            )
            return
        self.statusBar().showMessage(
            f'{module_name} 绘图完成，共生成 {len(paths)} 个文件', 8000
        )
        if self.plot_settings.get('show_preview') and any(
            str(path).lower().endswith('.png') for path in paths
        ):
            self.show_latest_plots(paths, result.get('module_id'))


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
