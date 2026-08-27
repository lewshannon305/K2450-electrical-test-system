import os
import math
import queue
import re
import threading
import time
import numpy as np
import pyvisa

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QGroupBox,
    QTextEdit,
    QScrollArea,
    QFileDialog,
    QMessageBox,
    QSizePolicy,
    QComboBox,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QRadioButton,
    QButtonGroup,
    QPlainTextEdit,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont

import pyqtgraph as pg

from core.app_base import BaseAppWidget
from core.paths import default_data_directory
from core.hardware_base import (
    allocate_unique_path,
    atomic_text_writer,
    assert_no_scpi_errors,
    clear_scpi_status,
    configure_current_autozero,
    fast_shutdown_zero_2450,
    generate_exact_ramp_levels,
    reliable_output_off,
    required_float_query,
    validate_2450_idn,
    validate_current_range_limit,
    validate_distinct_addresses,
    validate_nplc,
    validate_positive_step,
    validate_program_step_plan,
    validate_source_voltage,
    validate_step_divides_interval,
    validate_terminal,
    validate_voltage_range,
    validate_voltage_within_range,
    verify_current_configuration,
    write_result_metadata,
)
from core.instrument_config import InstrumentSettings
from core.utils import G0, NoScrollComboBox, _0, _1, _2, _3, configure_pyqtgraph


class GateLeakageError(RuntimeError):
    pass


def default_iv_gate_settings():
    return {
        'mode': 'single',
        'single_target': 0.0,
        'step_start': 0.0,
        'step_end': 1.0,
        'test_step': 0.2,
        'custom_text': '0, 0.5, 1.0',
        'gate_voltage_range': 20.0,
        'gate_nplc': 1.0,
        'gate_ilimit': 1e-9,
        'gate_leakage_limit': 1e-9,
        'gate_ramp_step': 0.1,
        'gate_step_delay': 0.5,
        'gate_settle': 20.0,
        'gate_group_wait': 0.0,
    }


def parse_custom_gate_values(text):
    tokens = [
        token for token in re.split(r'[\s,;，；]+', str(text).strip())
        if token
    ]
    if not tokens:
        raise ValueError('自定义栅压序列不能为空')
    values = []
    for index, token in enumerate(tokens, 1):
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(
                f'自定义栅压第{index}项不是有效数字：{token}'
            ) from exc
        if not math.isfinite(value):
            raise ValueError(f'自定义栅压第{index}项必须为有限数')
        values.append(value)
    return values


def build_gate_targets(settings):
    mode = settings['mode']
    if mode == 'single':
        targets = [float(settings['single_target'])]
    elif mode == 'step':
        start = float(settings['step_start'])
        end = float(settings['step_end'])
        step = validate_positive_step(
            settings['test_step'], 'Vg测试步长'
        )
        count = validate_step_divides_interval(
            start, end, step, 'Vg测试步进'
        )
        direction = 1.0 if end > start else -1.0
        targets = [
            start + direction * step * index
            for index in range(count + 1)
        ]
        targets[-1] = end
    elif mode == 'custom':
        targets = parse_custom_gate_values(settings['custom_text'])
    else:
        raise ValueError(f'未知栅压模式：{mode}')

    voltage_range = validate_voltage_range(
        settings['gate_voltage_range'], '栅压量程'
    )
    for index, target in enumerate(targets, 1):
        validate_source_voltage(target, f'第{index}个Vg')
        validate_voltage_within_range(
            target, voltage_range, f'第{index}个Vg'
        )
    return [float(value) for value in targets]


def merge_gate_settings_into_iv_params(params, gate_settings, gate_targets):
    """Add gate settings without overwriting the independent IV scan mode."""
    merged = dict(params)
    merged['gate_mode'] = gate_settings['mode']
    for key, value in gate_settings.items():
        if key not in {'mode', 'gate_targets'}:
            merged[key] = value
    merged['gate_targets'] = list(gate_targets)
    return merged


class IVGateSequenceDialog(QDialog):
    def __init__(
        self, settings, cycles, parent=None, ui_font=None, bold_font=None
    ):
        super().__init__(parent)
        self.setWindowTitle('自定义栅压序列编辑器')
        self.resize(1200, 620)
        self.setMinimumSize(1000, 500)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )
        self.ui_font = ui_font or QFont('Arial', 12)
        self.bold_font = bold_font or QFont('Arial', 12)
        self.bold_font.setWeight(QFont.Weight.Bold)
        self.setFont(self.ui_font)
        self.settings = dict(default_iv_gate_settings())
        self.settings.update(settings or {})
        self.cycles = int(cycles)

        main_layout = QVBoxLayout(self)
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(12)
        self.mode_group = QButtonGroup(self)

        def make_card(title, mode):
            card = QGroupBox(title)
            card.setFont(self.bold_font)
            layout = QVBoxLayout(card)
            radio = QRadioButton(f'使用{title}')
            radio.setFont(self.bold_font)
            radio.setProperty('gate_mode', mode)
            radio.toggled.connect(self._refresh)
            self.mode_group.addButton(radio)
            layout.addWidget(radio)
            return card, layout, radio

        self.single_card, single_layout, self.rb_single = make_card(
            '单个栅压', 'single'
        )
        single_grid = QGridLayout()
        single_label = QLabel('目标 Vg (V):')
        single_label.setFont(self.ui_font)
        self.single_target = QLineEdit()
        self.single_target.setFont(self.ui_font)
        single_grid.addWidget(single_label, 0, 0)
        single_grid.addWidget(self.single_target, 0, 1)
        single_layout.addLayout(single_grid)
        single_layout.addWidget(QLabel('最终序列:', font=self.bold_font))
        self.single_preview = QPlainTextEdit()
        self.single_preview.setReadOnly(True)
        self.single_preview.setFont(self.ui_font)
        single_layout.addWidget(self.single_preview, stretch=1)
        columns_layout.addWidget(self.single_card, stretch=1)

        self.step_card, step_layout, self.rb_step = make_card(
            '等步长栅压', 'step'
        )
        step_grid = QGridLayout()
        self.step_start = QLineEdit()
        self.step_end = QLineEdit()
        self.test_step = QLineEdit()
        for row, (text, control) in enumerate((
            ('起点 (V):', self.step_start),
            ('终点 (V):', self.step_end),
            ('测试步长 (V):', self.test_step),
        )):
            label = QLabel(text)
            label.setFont(self.ui_font)
            control.setFont(self.ui_font)
            step_grid.addWidget(label, row, 0)
            step_grid.addWidget(control, row, 1)
        step_layout.addLayout(step_grid)
        step_layout.addWidget(QLabel('最终序列:', font=self.bold_font))
        self.step_preview = QPlainTextEdit()
        self.step_preview.setReadOnly(True)
        self.step_preview.setFont(self.ui_font)
        step_layout.addWidget(self.step_preview, stretch=1)
        columns_layout.addWidget(self.step_card, stretch=1)

        self.custom_card, custom_layout, self.rb_custom = make_card(
            '自定义栅压', 'custom'
        )
        hint = QLabel(
            '支持从Excel粘贴，或使用逗号、分号、空格和换行；'
            '顺序及重复值原样保留。'
        )
        hint.setWordWrap(True)
        hint.setFont(self.ui_font)
        custom_layout.addWidget(hint)
        self.custom_text = QPlainTextEdit()
        self.custom_text.setFont(self.ui_font)
        self.custom_text.setPlaceholderText('例如：-5, -3, 0, 1, 5')
        self.custom_text.setMaximumHeight(125)
        custom_layout.addWidget(self.custom_text)
        custom_layout.addWidget(QLabel('最终序列:', font=self.bold_font))
        self.custom_preview = QPlainTextEdit()
        self.custom_preview.setReadOnly(True)
        self.custom_preview.setFont(self.ui_font)
        custom_layout.addWidget(self.custom_preview, stretch=1)
        columns_layout.addWidget(self.custom_card, stretch=1)
        main_layout.addLayout(columns_layout, stretch=1)

        self.error_label = QLabel('')
        self.error_label.setWordWrap(True)
        self.error_label.setFont(self.ui_font)
        self.error_label.setStyleSheet('color: #AA0000;')
        main_layout.addWidget(self.error_label)

        button_layout = QHBoxLayout()
        btn_cancel = QPushButton('取消')
        btn_cancel.setFont(self.bold_font)
        btn_cancel.setFixedSize(100, 30)
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton('确认')
        btn_ok.setFont(self.bold_font)
        btn_ok.setFixedSize(100, 30)
        btn_ok.setStyleSheet('color: #AA0000;')
        btn_ok.clicked.connect(self._accept_settings)
        button_layout.addStretch()
        button_layout.addWidget(btn_cancel)
        button_layout.addWidget(btn_ok)
        main_layout.addLayout(button_layout)

        self._load_settings()
        for control in (
            self.single_target, self.step_start, self.step_end,
            self.test_step,
        ):
            control.textChanged.connect(self._refresh)
        self.custom_text.textChanged.connect(self._refresh)
        self._refresh()

    def _load_settings(self):
        mode_button = {
            'single': self.rb_single,
            'step': self.rb_step,
            'custom': self.rb_custom,
        }.get(self.settings.get('mode'), self.rb_single)
        mode_button.setChecked(True)
        self.single_target.setText(str(self.settings['single_target']))
        self.step_start.setText(str(self.settings['step_start']))
        self.step_end.setText(str(self.settings['step_end']))
        self.test_step.setText(str(self.settings['test_step']))
        self.custom_text.setPlainText(str(self.settings['custom_text']))

    def _selected_mode(self):
        for button in (self.rb_single, self.rb_step, self.rb_custom):
            if button.isChecked():
                return button.property('gate_mode')
        return 'single'

    def _collect(self):
        mode = self._selected_mode()

        def read_float(control, key, required):
            text = control.text().strip()
            try:
                return float(text)
            except (TypeError, ValueError):
                if required:
                    raise ValueError(f'{key}必须是有效数字')
                return float(self.settings[key])

        values = dict(self.settings)
        values.update({
            'mode': mode,
            'single_target': read_float(
                self.single_target, '单个目标Vg', mode == 'single'
            ),
            'step_start': read_float(
                self.step_start, '步进起点', mode == 'step'
            ),
            'step_end': read_float(
                self.step_end, '步进终点', mode == 'step'
            ),
            'test_step': read_float(
                self.test_step, 'Vg测试步长', mode == 'step'
            ),
            'custom_text': self.custom_text.toPlainText().strip(),
        })
        return values

    def _validate(self):
        values = self._collect()
        targets = build_gate_targets(values)
        return values, targets

    def _refresh(self):
        selected = self._selected_mode()
        cards = {
            'single': self.single_card,
            'step': self.step_card,
            'custom': self.custom_card,
        }
        for mode, card in cards.items():
            card.setStyleSheet(
                'QGroupBox { border: 2px solid #2F75B5; '
                'border-radius: 4px; margin-top: 8px; }'
                if mode == selected else ''
            )

        common = dict(self.settings)
        previews = (
            ('single', self.single_preview, {
                'single_target': self.single_target.text().strip(),
            }),
            ('step', self.step_preview, {
                'step_start': self.step_start.text().strip(),
                'step_end': self.step_end.text().strip(),
                'test_step': self.test_step.text().strip(),
            }),
            ('custom', self.custom_preview, {
                'custom_text': self.custom_text.toPlainText().strip(),
            }),
        )
        active_error = ''
        for mode, preview, updates in previews:
            values = dict(common)
            values['mode'] = mode
            values.update(updates)
            try:
                targets = build_gate_targets(values)
                shown = ', '.join(f'{value:g}' for value in targets)
                preview.setStyleSheet('')
                preview.setPlainText(
                    f'[{shown}]\n\n'
                    f'Vg数量：{len(targets)}\n'
                    f'每个Vg循环：{self.cycles}\n'
                    f'总IV轮数：{len(targets) * self.cycles}'
                )
            except Exception as exc:
                preview.setStyleSheet('color: #AA0000;')
                preview.setPlainText(f'参数错误：{exc}')
                if mode == selected:
                    active_error = str(exc)
        self.error_label.setText(active_error)

    def _accept_settings(self):
        try:
            values, targets = self._validate()
        except Exception as exc:
            QMessageBox.warning(self, '栅压参数错误', str(exc))
            return
        values['gate_targets'] = targets
        self.settings = values
        self.accept()


class IV_Measurement:
    def __init__(self, params, update_queue, alarm_queue, stop_event, force_stop_event):
        self.params = params
        self.keithley = None
        self.gate_keithley = None
        self.mode = params['mode']
        self.segments = self._generate_voltage_segments()
        self.update_queue = update_queue
        self.alarm_queue = alarm_queue
        self.stop_event = stop_event
        self.force_stop_event = force_stop_event
        self.actual_vg = None
        self._cycle_voltage = {index: [] for index in range(4)}
        self._cycle_current = {index: [] for index in range(4)}
        self._cycle_gate_current = {index: [] for index in range(4)}

    def _interruptible_sleep(self, duration_s, stage_msg=None):
        if duration_s <= 0:
            return True
        if stage_msg:
            self.update_queue.put(('stage', stage_msg))
        steps = int(duration_s / 0.1)
        for _ in range(steps):
            if self.stop_event.is_set() or self.force_stop_event.is_set():
                return False
            time.sleep(0.1)
        rem = duration_s - steps * 0.1
        if rem > 0:
            time.sleep(rem)
        return not (self.stop_event.is_set() or self.force_stop_event.is_set())

    def _ramp_voltage(self, inst, target_v, step_abs, step_delay_s, is_zeroing=False):
        current_v = required_float_query(inst, ':SOUR:VOLT?', '源电压回读')
            
        if abs(target_v - current_v) < 1e-9:
            return True

        step_abs = abs(step_abs)
        if step_abs == 0:
            step_abs = 0.01

        direction = 1 if target_v > current_v else -1
        steps = int(round(abs(target_v - current_v) / step_abs))
        if steps == 0:
            steps = 1

        self.update_queue.put(('stage', "安全归零中..." if is_zeroing else "电压爬坡中..."))

        for i in range(1, steps + 1):
            if self.force_stop_event.is_set():
                return False
            if not is_zeroing and self.stop_event.is_set():
                return False

            v = current_v + direction * i * step_abs
            if direction == 1 and v > target_v:
                v = target_v
            elif direction == -1 and v < target_v:
                v = target_v

            inst.write(f':SOUR:VOLT {v}')
            time.sleep(step_delay_s)

            reading = required_float_query(inst, ':READ?', '爬坡电流读数')

            if is_zeroing:
                self.update_queue.put(('zeroing', v, reading))
            else:
                self.update_queue.put(('ramp', v, reading))

            if v == target_v:
                break
                
        return True

    def _generate_voltage_segments(self):
        start = self.params['v_start']
        end = self.params['v_end']
        step_abs = abs(self.params['v_step'])
        segments = []

        if self.mode == 'single':
            num_points = int(round(abs(end - start) / step_abs)) + 1
            voltages = np.linspace(start, end, num_points)
            segments.append((_0, voltages))

        elif self.mode == 'bidirectional':
            num_points_f = int(round(abs(end - start) / step_abs)) + 1
            forward = np.linspace(start, end, num_points_f)
            num_points_r = int(round(abs(end - start) / step_abs))
            reverse = np.linspace(end, start, num_points_r + 1)[_1:]
            segments.append((_0, forward))
            segments.append((_1, reverse))

        elif self.mode == 'hysteresis':
            num0 = int(round(abs(start) / step_abs)) + 1 if start != 0 else 1
            seg0 = np.linspace(0, start, num0)

            num1 = int(round(abs(end - start) / step_abs)) + 1
            seg1 = np.linspace(start, end, num1)

            num2 = int(round(abs(end - start) / step_abs))
            seg2 = np.linspace(end, start, num2 + 1)[_1:]

            num3 = int(round(abs(start) / step_abs))
            seg3 = np.linspace(start, 0, num3 + 1)[_1:] if num3 > 0 else np.zeros(0)

            segments.append((_0, seg0))
            segments.append((_1, seg1))
            segments.append((_2, seg2))
            segments.append((_3, seg3))

        return segments

    def connect(self):
        try:
            validate_distinct_addresses(
                self.params['address'],
                self.params.get('gate_address', ''),
                self.params.get('gate_enabled', False),
            )
            rm = pyvisa.ResourceManager()
            self.keithley = rm.open_resource(self.params['address'])
            self.keithley.timeout = 5000
            idn = validate_2450_idn(self.keithley.query('*IDN?'))
            self.alarm_queue.put(f"仪器已连接: {idn}")
            if self.params.get('gate_enabled', False):
                self.gate_keithley = rm.open_resource(
                    self.params['gate_address']
                )
                self.gate_keithley.timeout = 10000
                gate_idn = validate_2450_idn(
                    self.gate_keithley.query('*IDN?')
                )
                self.alarm_queue.put(f"栅表已连接: {gate_idn}")
        except Exception as e:
            self.alarm_queue.put(f"连接失败: {repr(e)}")
            raise

    def setup(self):
        k = self.keithley
        validate_nplc(self.params['nplc'])
        validate_terminal(self.params['terminal'])
        validate_source_voltage(self.params['v_start'], 'IV起始电压')
        validate_source_voltage(self.params['v_end'], 'IV终止电压')
        validate_positive_step(self.params['v_step'], 'IV步长')
        validate_program_step_plan('iv', self.params)
        validate_current_range_limit(
            self.params['current_range'], self.params['i_limit'], '偏压'
        )
        k.write('*RST')
        clear_scpi_status(k)
        k.write(':ABORt')
        k.write(':SOUR:FUNC VOLT')
        k.write(':SENS:FUNC "CURR"')
        k.write('SENS:CURR:RSEN OFF')
        k.write(f":ROUT:TERM {self.params['terminal']}")
        k.write(':SENS:CURR:AVER OFF')
        k.write(':SOUR:VOLT:READ:BACK ON')
        k.write(f':SENS:CURR:NPLC {self.params["nplc"]}')
        configure_current_autozero(k, 'continuous')

        r_val = self.params['current_range']
        if isinstance(r_val, str) and r_val.upper() == 'AUTO':
            k.write(':SENS:CURR:RANG:AUTO ON')
        else:
            k.write(':SENS:CURR:RANG:AUTO OFF')
            k.write(f':SENS:CURR:RANG {r_val}')

        k.write(f':SOUR:VOLT:ILIM {self.params["i_limit"]}')
        k.write(':SOUR:VOLT 0')
        time.sleep(0.05)
        verify_current_configuration(
            k,
            nplc=self.params['nplc'],
            current_range=self.params['current_range'],
            current_limit=self.params['i_limit'],
            terminal=self.params['terminal'],
            autozero_mode='continuous',
            label='IV源表',
        )
        if self.params.get('gate_enabled', False):
            g = self.gate_keithley
            validate_nplc(self.params['gate_nplc'], '栅表NPLC')
            validate_terminal(self.params['gate_terminal'], '栅表端口')
            validate_voltage_range(
                self.params['gate_voltage_range'], '栅压量程'
            )
            validate_positive_step(
                self.params['gate_ramp_step'], '栅压爬坡步长'
            )
            validate_current_range_limit(
                'AUTO', self.params['gate_ilimit'], '栅极'
            )
            for index, target in enumerate(self.params['gate_targets'], 1):
                validate_voltage_within_range(
                    target, self.params['gate_voltage_range'],
                    f'第{index}个Vg',
                )
            g.write('*RST')
            clear_scpi_status(g)
            g.write(':ABORt')
            g.write(':SOUR:FUNC VOLT')
            g.write(':SENS:FUNC "CURR"')
            g.write(':SENS:CURR:RSEN OFF')
            g.write(f":ROUT:TERM {self.params['gate_terminal']}")
            g.write(':SENS:CURR:AVER OFF')
            g.write(':SOUR:VOLT:READ:BACK ON')
            g.write(f":SENS:CURR:NPLC {self.params['gate_nplc']}")
            configure_current_autozero(g, 'continuous')
            g.write(':SENS:CURR:RANG:AUTO ON')
            g.write(f":SOUR:VOLT:ILIM {self.params['gate_ilimit']}")
            g.write(
                f":SOUR:VOLT:RANG {self.params['gate_voltage_range']}"
            )
            g.write(':SOUR:VOLT 0')
            time.sleep(0.05)
            verify_current_configuration(
                g,
                nplc=self.params['gate_nplc'],
                current_range='AUTO',
                current_limit=self.params['gate_ilimit'],
                terminal=self.params['gate_terminal'],
                autozero_mode='continuous',
                label='IV栅表',
            )
        if self.gate_keithley is not None:
            assert_no_scpi_errors(self.gate_keithley, 'IV栅表初始化')

    def ramp_to_start(self):
        if self.mode in ['single', 'bidirectional']:
            v_start = self.params['v_start']
            step_abs = abs(self.params['v_step'])
            return self._ramp_voltage(self.keithley, v_start, step_abs, 0.01, is_zeroing=False)
        return True

    def safe_ramp_to_zero(self, turn_off=True, fast=False):
        if self.keithley is None:
            return
        if self.force_stop_event.is_set():
            reliable_output_off(self.keithley, 'IV源表')
            return
        step_abs = abs(self.params['v_step'])
        if fast:
            self.update_queue.put(('stage', '安全归零中...'))
            report = fast_shutdown_zero_2450(
                self.keithley,
                step_abs,
                label='IV源表',
                force_event=self.force_stop_event,
            )
            if report['status'] == 'complete':
                self.alarm_queue.put(
                    f"归零完成，用时 {report['elapsed_s']:.1f} s"
                )
                self.alarm_queue.put('输出已关闭')
            else:
                self.alarm_queue.put(
                    '安全归零失败，已执行紧急关断：'
                    + ' | '.join(report['errors'])
                )
            return
        
        self._ramp_voltage(self.keithley, 0.0, step_abs, 0.01, is_zeroing=True)
        
        if not self.force_stop_event.is_set():
            self.keithley.write(':SOUR:VOLT 0')
            if turn_off:
                self.keithley.write(':OUTP OFF')

    def _query_gate_current(self, context):
        current = required_float_query(
            self.gate_keithley, ':READ?', context
        )
        return current

    def _check_gate_current(self, current):
        if abs(current) > self.params['gate_leakage_limit']:
            raise GateLeakageError(
                f'栅电流超过保护阈值：{current:.6e} A > '
                f'{self.params["gate_leakage_limit"]:.6e} A'
            )

    def _ramp_gate(self, target):
        current = required_float_query(
            self.gate_keithley, ':SOUR:VOLT?', '栅压起点回读'
        )
        levels = generate_exact_ramp_levels(
            current, target, self.params['gate_ramp_step']
        )
        self.update_queue.put(('stage', '栅压爬坡中...'))
        for voltage in levels:
            if self.stop_event.is_set() or self.force_stop_event.is_set():
                return False
            self.gate_keithley.write(f':SOUR:VOLT {voltage:.12g}')
            if not self._interruptible_sleep(
                self.params['gate_step_delay'], '栅压爬坡中...'
            ):
                return False
            gate_current = self._query_gate_current('栅压爬坡Ig读数')
            self.update_queue.put(
                ('gate_ramp', voltage, gate_current)
            )
            self._check_gate_current(gate_current)

        actual = required_float_query(
            self.gate_keithley, ':SOUR:VOLT?', '目标栅压回读'
        )
        tolerance = max(
            1e-9, abs(self.params['gate_ramp_step']) * 1e-6
        )
        if not math.isclose(
            actual, target, rel_tol=1e-8, abs_tol=tolerance
        ):
            raise RuntimeError(
                f'栅压未到达目标：请求 {target:g} V，'
                f'回读 {actual:g} V'
            )
        if not self._interruptible_sleep(
            self.params['gate_settle'],
            f'栅压到位等待 ({self.params["gate_settle"]:g}s)',
        ):
            return False
        gate_current = self._query_gate_current('栅压到位Ig读数')
        self._check_gate_current(gate_current)
        self.actual_vg = actual
        self.update_queue.put(('gate_ready', actual, gate_current))
        return True

    def _confirm_bias_zero(self):
        actual = required_float_query(
            self.keithley, ':SOUR:VOLT?', '切换Vg前偏压回读'
        )
        tolerance = max(1e-9, abs(self.params['v_step']) * 1e-6)
        if not math.isclose(actual, 0.0, rel_tol=0.0, abs_tol=tolerance):
            raise RuntimeError(
                f'切换Vg前偏压未归零：回读 {actual:g} V'
            )

    def _reset_cycle_data(self):
        self._cycle_voltage = {index: [] for index in range(4)}
        self._cycle_current = {index: [] for index in range(4)}
        self._cycle_gate_current = {index: [] for index in range(4)}

    def _cycle_snapshot(self):
        return {
            'voltage': {
                index: np.asarray(values, dtype=float).copy()
                for index, values in self._cycle_voltage.items()
            },
            'current': {
                index: np.asarray(values, dtype=float).copy()
                for index, values in self._cycle_current.items()
            },
            'gate_current': {
                index: np.asarray(values, dtype=float).copy()
                for index, values in self._cycle_gate_current.items()
            },
            'counts': {
                index: len(values)
                for index, values in self._cycle_voltage.items()
            },
        }

    def measure_loop(self):
        required_float_query(self.keithley, ':READ?', 'IV预热读数')

        for seg_idx, voltages in self.segments:
            for v in voltages:
                if self.stop_event.is_set() or self.force_stop_event.is_set():
                    return
                self.keithley.write(f':SOUR:VOLT {v}')
                
                if not self._interruptible_sleep(self.params['settle_time'], "测量中..."):
                    return
                    
                try:
                    curr = required_float_query(
                        self.keithley, ':READ?', 'IV正式电流读数'
                    )
                    gate_current = None
                    gate_error = None
                    if self.params.get('gate_enabled', False):
                        try:
                            gate_current = self._query_gate_current(
                                'IV正式同步Ig读数'
                            )
                        except Exception as exc:
                            gate_current = float('nan')
                            gate_error = exc
                    self._cycle_voltage[seg_idx].append(float(v))
                    self._cycle_current[seg_idx].append(float(curr))
                    if self.params.get('gate_enabled', False):
                        self._cycle_gate_current[seg_idx].append(
                            float(gate_current)
                        )
                        self.update_queue.put((
                            seg_idx, v, curr, self.actual_vg, gate_current
                        ))
                        if gate_error is None:
                            self._check_gate_current(gate_current)
                    else:
                        self.update_queue.put((seg_idx, v, curr))
                    if gate_error is not None:
                        raise gate_error
                except GateLeakageError as exc:
                    self.alarm_queue.put(f"栅电流保护触发: {exc}")
                    raise
                except Exception as exc:
                    self.alarm_queue.put(f"读数失败: {exc}")
                    raise

    def run(self):
        try:
            self.connect()
            self.setup()
            self.keithley.write(':OUTP ON')
            if self.params.get('gate_enabled', False):
                self.gate_keithley.write(':OUTP ON')
            time.sleep(0.01)

            cycles = self.params['cycles']
            gate_targets = (
                self.params['gate_targets']
                if self.params.get('gate_enabled', False)
                else [None]
            )
            total_gate = len(gate_targets)

            for gate_index, requested_vg in enumerate(gate_targets, 1):
                if self.stop_event.is_set() or self.force_stop_event.is_set():
                    break
                if self.params.get('gate_enabled', False):
                    self._confirm_bias_zero()
                    if not self._ramp_gate(requested_vg):
                        break
                else:
                    self.actual_vg = None

                for cycle in range(1, cycles + 1):
                    if (
                        self.stop_event.is_set()
                        or self.force_stop_event.is_set()
                    ):
                        break
                    self._reset_cycle_data()
                    self.update_queue.put((
                        'cycle_start', cycle, cycles, gate_index,
                        total_gate, requested_vg, self.actual_vg,
                    ))

                    cycle_status = 'complete'
                    cycle_error = None
                    try:
                        if self.mode in ['single', 'bidirectional']:
                            if self.ramp_to_start():
                                self.measure_loop()
                            more_work = (
                                cycle < cycles or gate_index < total_gate
                            )
                            if more_work:
                                self.safe_ramp_to_zero(
                                    turn_off=False, fast=False
                                )
                        elif self.mode == 'hysteresis':
                            self.measure_loop()
                    except Exception as exc:
                        cycle_status = 'partial'
                        cycle_error = exc
                        raise
                    finally:
                        if (
                            cycle_status == 'complete'
                            and (
                                self.stop_event.is_set()
                                or self.force_stop_event.is_set()
                            )
                        ):
                            cycle_status = 'partial'
                            cycle_error = '用户停止或强制终止'
                        self.update_queue.put((
                            'cycle_done', cycle, cycle_status, cycle_error,
                            gate_index, total_gate, requested_vg,
                            self.actual_vg, self._cycle_snapshot(),
                        ))

                if self.stop_event.is_set() or self.force_stop_event.is_set():
                    break
                if gate_index < total_gate:
                    if self.mode == 'hysteresis':
                        self._confirm_bias_zero()
                    if not self._interruptible_sleep(
                        self.params['gate_group_wait'],
                        f'Vg组间等待 ({self.params["gate_group_wait"]:g}s)',
                    ):
                        break

        except Exception as e:
            self.alarm_queue.put(f"测量异常: {e}")
        finally:
            try:
                if self.force_stop_event.is_set():
                    reliable_output_off(self.keithley, 'IV偏压表')
                    reliable_output_off(self.gate_keithley, 'IV栅表')
                else:
                    if self.keithley is None:
                        bias_report = None
                    else:
                        self.update_queue.put((
                            'stage',
                            '偏压归零中...'
                            if self.gate_keithley is not None
                            else '安全归零中...',
                        ))
                        bias_report = fast_shutdown_zero_2450(
                            self.keithley,
                            abs(self.params['v_step']),
                            label='IV偏压表',
                            force_event=self.force_stop_event,
                        )
                    gate_report = None
                    if self.gate_keithley is not None:
                        self.update_queue.put(('stage', '栅压归零中...'))
                        gate_report = fast_shutdown_zero_2450(
                            self.gate_keithley,
                            self.params['gate_ramp_step'],
                            label='IV栅表',
                            force_event=self.force_stop_event,
                        )
                    reports = [
                        report for report in (bias_report, gate_report)
                        if report is not None
                    ]
                    if reports and all(
                        report['status'] == 'complete'
                        for report in reports
                    ):
                        elapsed = sum(
                            report['elapsed_s'] for report in reports
                        )
                        self.alarm_queue.put(
                            f'归零完成，用时 {elapsed:.1f} s'
                        )
                        self.alarm_queue.put('输出已关闭')
                    elif reports:
                        self.alarm_queue.put(
                            '安全归零失败，已执行紧急关断：'
                            + ' | '.join(
                                error
                                for report in reports
                                for error in report['errors']
                            )
                        )
            except Exception as e:
                if not self.force_stop_event.is_set():
                    self.alarm_queue.put(f"安全归零异常: {e}")
            bias_confirmed, bias_failures = reliable_output_off(
                self.keithley, 'IV偏压表'
            )
            gate_confirmed = True
            gate_failures = []
            if self.gate_keithley is not None:
                gate_confirmed, gate_failures = reliable_output_off(
                    self.gate_keithley, 'IV栅表'
                )
            if not bias_confirmed or not gate_confirmed:
                self.alarm_queue.put(
                    '严重警告：无法确认IV源表输出已关闭，请立即从仪器面板确认。 '
                    + ' | '.join(bias_failures + gate_failures)
                )

            if self.keithley:
                try:
                    self.keithley.close()
                except Exception:
                    pass
            if self.gate_keithley:
                try:
                    self.gate_keithley.close()
                except Exception:
                    pass

            self.update_queue.put(None)


class IVWidget(BaseAppWidget):
    def __init__(self, run_guard=None, instrument_settings=None, parent=None):
        super().__init__(run_guard, parent)
        self.module_id = 'iv_curve'
        self.module_name = '循环IV特性扫描'
        self.instrument_settings = instrument_settings or InstrumentSettings(
            bias_address='GPIB0::1::INSTR',
            gate_address='GPIB0::2::INSTR',
        )
        
        self.ui_font = QFont("Arial", 12)
        self.ui_font.setWeight(QFont.Weight.Normal)
        self.bold_font = QFont("Arial", 12)
        self.bold_font.setWeight(QFont.Weight.Bold)
        self.setFont(self.ui_font)

        self.capacity = 100000
        self.data_count = {_0: 0, _1: 0, _2: 0, _3: 0}
        self.v_data = {
            _0: np.zeros(self.capacity), _1: np.zeros(self.capacity),
            _2: np.zeros(self.capacity), _3: np.zeros(self.capacity)
        }
        self.i_data = {
            _0: np.zeros(self.capacity), _1: np.zeros(self.capacity),
            _2: np.zeros(self.capacity), _3: np.zeros(self.capacity)
        }
        self.ig_data = {
            _0: np.full(self.capacity, np.nan),
            _1: np.full(self.capacity, np.nan),
            _2: np.full(self.capacity, np.nan),
            _3: np.full(self.capacity, np.nan),
        }
        self.gate_settings = default_iv_gate_settings()
        self.points_changed = False

        self.init_ui()
        self._update_gate_summary()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=3)

        self.graph_widget = pg.GraphicsLayoutWidget()
        left_layout.addWidget(self.graph_widget)

        self.plot_iv = self.graph_widget.addPlot(title="I-V Curve")
        self.plot_iv.setTitle("I-V Curve", size="12pt")

        label_style = {'color': '#000', 'font-size': '12pt'}
        self.plot_iv.setLabel('left', text='Current', units='A', **label_style)
        self.plot_iv.setLabel('bottom', text='Voltage', units='V', **label_style)

        self.plot_iv.getAxis('left').setTickFont(self.ui_font)
        self.plot_iv.getAxis('bottom').setTickFont(self.ui_font)
        self.plot_iv.showGrid(x=True, y=True, alpha=0.3)
        self.plot_iv.setClipToView(True)
        self.plot_iv.setDownsampling(auto=True, mode='subsample')
        self.legend = self.plot_iv.addLegend(offset=(10, 10), labelTextSize='12pt')

        self.curves = {}
        colors = {0: 'r', 1: 'b', 2: 'g', 3: 'm'}
        for i in range(4):
            c = self.plot_iv.plot(pen=pg.mkPen(colors[i], width=1.5))
            c.hide()
            self.curves[i] = c

        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, stretch=2)

        status_group = QGroupBox("实时状态显示")
        status_group.setFont(self.bold_font)
        status_group.setFixedHeight(135)
        status_layout = QGridLayout(status_group)

        status_layout.setColumnStretch(1, 1)
        status_layout.setColumnStretch(3, 1)
        status_layout.setHorizontalSpacing(10)

        self.status_labels = {}
        status_items = [
            ("偏压 Vsd (V):", "bias_v", 0, 0),
            ("电阻 (Ω):", "R", 1, 2),
            ("栅压 Vg (V):", "gate_v", 1, 0),
            ("电导 (G₀):", "G", 0, 2),
            ("偏置电流 Isd (A):", "bias_i", 2, 0),
            ("当前轮次:", "cycle", 2, 2),
            ("栅电流 Ig (A):", "gate_i", 3, 0),
            ("系统状态:", "stage", 3, 2),
        ]
        for text, key, row, col in status_items:
            lbl = QLabel(text)
            lbl.setFont(self.ui_font)
            lbl.setStyleSheet("font-weight: normal;")
            status_layout.addWidget(lbl, row, col, alignment=Qt.AlignmentFlag.AlignLeft)

            val = QLabel("-")
            val.setFont(self.bold_font)
            val.setStyleSheet("color: #0055A4;")
            status_layout.addWidget(val, row, col + 1, alignment=Qt.AlignmentFlag.AlignLeft)
            self.status_labels[key] = val
        right_layout.addWidget(status_group)

        param_group = QGroupBox("测量参数")
        param_group.setFont(self.bold_font)
        param_layout = QVBoxLayout(param_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_content = QWidget()
        scroll_content.setFont(self.ui_font)
        scroll_content_layout = QVBoxLayout(scroll_content)
        scroll_content_layout.setContentsMargins(0, 0, 0, 0)
        self.inputs = {}

        self.cb_gate = QCheckBox('启用栅表（勾选展示栅压参数）')
        self.cb_gate.setFont(self.ui_font)
        self.cb_gate.setStyleSheet('font-weight: normal;')
        self.cb_gate.stateChanged.connect(self.toggle_gate_controls)
        scroll_content_layout.addWidget(self.cb_gate)

        self.gate_group = QGroupBox('栅压参数')
        self.gate_group.setFont(self.bold_font)
        gate_grid = QGridLayout(self.gate_group)
        gate_grid.setColumnStretch(0, 0)
        gate_grid.setColumnStretch(1, 0)
        gate_grid.setColumnStretch(2, 1)
        gate_grid.setColumnStretch(3, 1)
        self.btn_gate_settings = QPushButton('设置栅压序列...')
        self.btn_gate_settings.setFont(self.bold_font)
        self.btn_gate_settings.setStyleSheet('color: #B35A00;')
        self.btn_gate_settings.clicked.connect(self.open_gate_settings)
        gate_grid.addWidget(self.btn_gate_settings, 0, 0, 1, 2)
        self.lbl_gate_summary = QLabel('未启用栅表')
        self.lbl_gate_summary.setWordWrap(False)
        self.lbl_gate_summary.setFont(self.ui_font)
        self.lbl_gate_summary.setStyleSheet(
            'color: #005500; font-weight: normal;'
        )
        self.lbl_gate_summary.setAlignment(
            Qt.AlignmentFlag.AlignRight
            | Qt.AlignmentFlag.AlignVCenter
        )
        gate_grid.addWidget(
            self.lbl_gate_summary, 0, 2, 1, 2,
            alignment=Qt.AlignmentFlag.AlignRight
            | Qt.AlignmentFlag.AlignVCenter,
        )

        def add_gate_parameter(row, column, text, key, default):
            label = QLabel(text)
            label.setFont(self.ui_font)
            label.setStyleSheet('font-weight: normal;')
            control = QLineEdit(str(default))
            control.setFont(self.ui_font)
            gate_grid.addWidget(label, row, column)
            gate_grid.addWidget(control, row, column + 1)
            self.inputs[key] = control

        defaults = default_iv_gate_settings()
        add_gate_parameter(
            2, 0, '栅压量程 (V):',
            'gate_voltage_range', defaults['gate_voltage_range']
        )
        add_gate_parameter(
            2, 2, '栅表 NPLC:',
            'gate_nplc', defaults['gate_nplc']
        )
        add_gate_parameter(
            3, 0, '栅极限流 (A):',
            'gate_ilimit', defaults['gate_ilimit']
        )
        add_gate_parameter(
            3, 2, '栅电流保护阈值 (A):',
            'gate_leakage_limit', defaults['gate_leakage_limit']
        )
        add_gate_parameter(
            4, 0, '栅压爬坡/归零步长 (V):',
            'gate_ramp_step', defaults['gate_ramp_step']
        )
        add_gate_parameter(
            4, 2, '栅压单步等待 (s):',
            'gate_step_delay', defaults['gate_step_delay']
        )
        add_gate_parameter(
            5, 0, '栅压到位等待 (s):',
            'gate_settle', defaults['gate_settle']
        )
        add_gate_parameter(
            5, 2, 'Vg组间等待 (s):',
            'gate_group_wait', defaults['gate_group_wait']
        )
        scroll_content_layout.addWidget(self.gate_group)
        self.gate_group.setVisible(False)

        ctrl_group = QGroupBox("控制参数")
        ctrl_group.setFont(self.bold_font)
        ctrl_grid = QGridLayout(ctrl_group)

        ctrl_items_left = [
            ("起始电压 (V):", "v_start", "-1.0"),
            ("终止电压 (V):", "v_end", "1.0"),
            ("步长 (V) (正):", "v_step", "0.02"),
            ("电流限制 (A):", "i_limit", "1.05e-6"),
            ("稳定时间 (s):", "settle_time", "0.0")
        ]

        ctrl_items_right = [
            ("NPLC:", "nplc", "1.0"),
            ("电流量程 (A 或 AUTO):", "current_range", "1e-6"),
            ("循环次数 (Cycles):", "cycles", "1"),
            ("扫描模式:", "mode", ""),
            ("", "", "")
        ]

        for r in range(5):
            lbl_l_txt, key_l, def_l = ctrl_items_left[r]
            lbl_l = QLabel(lbl_l_txt)
            lbl_l.setFont(self.ui_font)
            ent_l = QLineEdit(def_l)
            ent_l.setFont(self.ui_font)
            ctrl_grid.addWidget(lbl_l, r, 0)
            ctrl_grid.addWidget(ent_l, r, 1)
            self.inputs[key_l] = ent_l

            lbl_r_txt, key_r, def_r = ctrl_items_right[r]
            if lbl_r_txt:
                lbl_r = QLabel(lbl_r_txt)
                lbl_r.setFont(self.ui_font)
                ctrl_grid.addWidget(lbl_r, r, 2)
                if key_r == "mode":
                    self.mode_combo = NoScrollComboBox()
                    self.mode_combo.setFont(self.ui_font)
                    self.mode_combo.addItems(['single', 'bidirectional', 'hysteresis'])
                    ctrl_grid.addWidget(self.mode_combo, r, 3)
                    self.inputs['mode'] = self.mode_combo
                else:
                    ent_r = QLineEdit(def_r)
                    ent_r.setFont(self.ui_font)
                    ctrl_grid.addWidget(ent_r, r, 3)
                    self.inputs[key_r] = ent_r

        scroll_content_layout.addWidget(ctrl_group)

        path_group = QGroupBox("文件保存路径")
        path_group.setFont(self.bold_font)
        path_grid = QGridLayout(path_group)

        lbl_prefix = QLabel("文件名前缀 (后缀自动追加模式、电压、循环信息)：")
        lbl_prefix.setFont(self.ui_font)
        path_grid.addWidget(lbl_prefix, 0, 0, 1, 2)

        ent_prefix = QLineEdit("IV")
        ent_prefix.setFont(self.ui_font)
        self.inputs['file_prefix'] = ent_prefix
        path_grid.addWidget(ent_prefix, 1, 0, 1, 2)

        lbl_folder = QLabel("保存文件夹：")
        lbl_folder.setFont(self.ui_font)
        path_grid.addWidget(lbl_folder, 2, 0, 1, 2)

        folder_hbox = QHBoxLayout()
        folder_hbox.setContentsMargins(0, 0, 0, 0)
        self.folder_input = QLineEdit(default_data_directory("IV"))
        self.folder_input.setFont(self.ui_font)
        folder_hbox.addWidget(self.folder_input)

        btn_browse = QPushButton("浏览")
        btn_browse.setFont(self.ui_font)
        btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(self.browse_folder)
        folder_hbox.addWidget(btn_browse)

        path_grid.addLayout(folder_hbox, 3, 0, 1, 2)
        scroll_content_layout.addWidget(path_group)

        scroll_content_layout.addStretch()
        scroll.setWidget(scroll_content)
        param_layout.addWidget(scroll)
        right_layout.addWidget(param_group, stretch=1)

        log_group = QGroupBox("日志信息")
        log_group.setFont(self.bold_font)
        log_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(5, 5, 5, 5)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(self.ui_font)
        self.log_text.setStyleSheet("background-color: #FFF0F0; color: #333333;")
        self.log_text.setFixedHeight(60)
        log_layout.addWidget(self.log_text)

        btn_clear_log = QPushButton("清除信息")
        btn_clear_log.setFont(self.bold_font)
        btn_clear_log.setFixedWidth(100)
        btn_clear_log.setFixedHeight(30)
        btn_clear_log.clicked.connect(self.clear_log)
        log_layout.addWidget(btn_clear_log, alignment=Qt.AlignmentFlag.AlignCenter)

        right_layout.addWidget(log_group, stretch=0)

        btn_widget = QWidget()
        btn_layout = QHBoxLayout(btn_widget)
        btn_layout.setContentsMargins(0, 10, 0, 10)

        self.start_btn = QPushButton("开始")
        self.start_btn.setFixedSize(100, 30)
        self.start_btn.setFont(self.bold_font)
        self.start_btn.clicked.connect(self.start_measurement)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setFixedSize(100, 30)
        self.stop_btn.setFont(self.bold_font)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_measurement)

        self.force_stop_btn = QPushButton("强制终止")
        self.force_stop_btn.setFixedSize(100, 30)
        self.force_stop_btn.setFont(self.bold_font)
        self.force_stop_btn.setStyleSheet("color: #AA0000;")
        self.force_stop_btn.clicked.connect(self.force_stop_measurement)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.force_stop_btn)

        right_layout.addWidget(btn_widget)

    def open_gate_settings(self):
        try:
            cycles = int(self.inputs['cycles'].text())
        except Exception:
            cycles = 1
        try:
            settings = self.current_gate_settings()
        except Exception as exc:
            QMessageBox.warning(
                self, '栅表参数错误',
                f'请先修正主界面的栅表参数：{exc}'
            )
            return
        dialog = IVGateSequenceDialog(
            settings,
            cycles,
            self,
            self.ui_font,
            self.bold_font,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            for key in (
                'mode', 'single_target', 'step_start', 'step_end',
                'test_step', 'custom_text',
            ):
                self.gate_settings[key] = dialog.settings[key]
            self._update_gate_summary()

    def current_gate_settings(self):
        settings = dict(default_iv_gate_settings())
        settings.update(self.gate_settings)
        settings.pop('gate_address', None)
        settings.pop('gate_terminal', None)
        for key in (
            'gate_voltage_range', 'gate_nplc', 'gate_ilimit',
            'gate_leakage_limit', 'gate_ramp_step', 'gate_step_delay',
            'gate_settle', 'gate_group_wait',
        ):
            settings[key] = float(self.inputs[key].text().strip())
        return settings

    def apply_gate_settings_to_main_controls(self):
        settings = dict(default_iv_gate_settings())
        settings.update(self.gate_settings)
        for key in (
            'gate_voltage_range', 'gate_nplc', 'gate_ilimit',
            'gate_leakage_limit', 'gate_ramp_step', 'gate_step_delay',
            'gate_settle', 'gate_group_wait',
        ):
            self.inputs[key].setText(str(settings[key]))

    def toggle_gate_controls(self):
        enabled = self.cb_gate.isChecked()
        self.gate_group.setVisible(enabled)
        self._update_gate_summary()

    def _update_gate_summary(self):
        if not self.cb_gate.isChecked():
            self.lbl_gate_summary.setText('未启用栅表')
            return
        try:
            targets = build_gate_targets(self.current_gate_settings())
            mode_text = {
                'single': '单个栅压',
                'step': '等步长栅压',
                'custom': '自定义序列',
            }.get(self.gate_settings['mode'], '栅压序列')
            self.lbl_gate_summary.setText(
                f'{mode_text}，共{len(targets)}个Vg'
            )
        except Exception as exc:
            self.lbl_gate_summary.setText(f'栅压设置无效：{exc}')

    # ---------------- 补充的三大缺失的 UI 辅助方法 ----------------
    def browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, "选择保存文件夹")
        if d:
            self.folder_input.setText(d)

    def log_info(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def clear_log(self):
        self.log_text.clear()
    # -----------------------------------------------------------

    def start_measurement(self):
        if self.measure_running:
            return
        if not self.request_start(self.module_id, self.module_name):
            return
        self.reset_status_display()

        self.clear_log()

        try:
            p = {k: v.currentText().strip() if isinstance(v, QComboBox) else v.text().strip() for k, v in self.inputs.items()}
            gate_enabled = self.cb_gate.isChecked()
            instrument = self.instrument_settings.snapshot(
                require_gate=gate_enabled
            )
            params = {
                'address': instrument['bias_address'],
                'v_start': float(p['v_start']),
                'v_end': float(p['v_end']),
                'v_step': float(p['v_step']),
                'i_limit': float(p['i_limit']),
                'settle_time': float(p['settle_time']),
                'nplc': float(p['nplc']),
                'terminal': instrument['bias_terminal'],
                'cycles': int(p['cycles'])
            }
            if params['v_step'] <= 0:
                raise ValueError("步长必须为正值")
            if params['v_start'] == params['v_end']:
                raise ValueError("起止电压不能相等")
            if params['cycles'] <= 0:
                raise ValueError("循环次数必须大于 0")
            if params['settle_time'] < 0:
                raise ValueError("稳定时间不能为负值")
            if params['nplc'] <= 0:
                raise ValueError("NPLC 必须大于 0")
            if params['i_limit'] <= 0:
                raise ValueError("电流限制必须大于 0")

            r_val = p['current_range']
            params['current_range'] = r_val if r_val.upper() == 'AUTO' else float(r_val)
            if not isinstance(params['current_range'], str) and params['current_range'] <= 0:
                raise ValueError("电流量程必须大于 0 或为 AUTO")
            params['mode'] = p['mode']
            params['file_prefix'] = p['file_prefix']
            params['folder'] = self.folder_input.text().strip()
            validate_program_step_plan('iv', params)
            params['gate_enabled'] = gate_enabled
            if params['gate_enabled']:
                gate = self.current_gate_settings()
                gate['gate_address'] = instrument['gate_address']
                gate['gate_terminal'] = instrument['gate_terminal']
                gate_targets = build_gate_targets(gate)
                validate_distinct_addresses(
                    params['address'], gate['gate_address'], True
                )
                validate_nplc(gate['gate_nplc'], '栅表NPLC')
                validate_terminal(gate['gate_terminal'], '栅表端口')
                validate_positive_step(
                    gate['gate_ramp_step'], '栅压爬坡步长'
                )
                validate_current_range_limit(
                    'AUTO', gate['gate_ilimit'], '栅极'
                )
                if gate['gate_leakage_limit'] <= 0:
                    raise ValueError('栅电流保护阈值必须大于0')
                for key, label in (
                    ('gate_step_delay', '栅压单步等待'),
                    ('gate_settle', '栅压到位等待'),
                    ('gate_group_wait', 'Vg组间等待'),
                ):
                    if gate[key] < 0:
                        raise ValueError(f'{label}不能为负值')
                params = merge_gate_settings_into_iv_params(
                    params, gate, gate_targets
                )
            else:
                params['gate_targets'] = []

            os.makedirs(params['folder'], exist_ok=True)
            test_file = os.path.join(params['folder'], '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            points_per_iv = sum(
                len(values)
                for _segment, values
                in IV_Measurement(
                    params, queue.Queue(), queue.Queue(),
                    threading.Event(), threading.Event(),
                ).segments
            )
            vg_count = len(params['gate_targets']) or 1
            self.log_info(
                f'测量规模：{vg_count}个Vg × {params["cycles"]}轮 × '
                f'{points_per_iv}点 = '
                f'{vg_count * params["cycles"] * points_per_iv}个正式IV点'
            )
        except Exception as e:
            self.log_info(f"启动失败(参数或路径错误): {e}")
            self.show_parameter_error(e)
            self.mark_measurement_finished(self.module_id)
            return

        for i in range(4):
            self.data_count[i] = 0
            self.v_data[i].fill(0)
            self.i_data[i].fill(0)
            self.ig_data[i].fill(np.nan)
            self.curves[i].setData([], [])
            self.curves[i].hide()

        self.legend.clear()
        mode = params['mode']
        if mode == 'single':
            self.curves[_0].setPen(pg.mkPen('r', width=1.5))
            self.curves[_0].show()
        elif mode == 'bidirectional':
            self.curves[_0].setPen(pg.mkPen('r', width=1.5))
            self.curves[_0].show()
            self.curves[_1].setPen(pg.mkPen('b', width=1.5))
            self.curves[_1].show()
            self.legend.addItem(self.curves[_0], 'Forward')
            self.legend.addItem(self.curves[_1], 'Reverse')
        elif mode == 'hysteresis':
            colors = {0: 'g', 1: 'r', 2: 'b', 3: 'm'}
            labels = {0: '0→V_start', 1: 'V_start→V_end', 2: 'V_end→V_start', 3: 'V_start→0'}
            for i in range(4):
                self.curves[i].setPen(pg.mkPen(colors[i], width=1.5))
                self.curves[i].show()
                self.legend.addItem(self.curves[i], labels[i])

        while not self.update_queue.empty():
            self.update_queue.get()
        while not self.alarm_queue.empty():
            self.alarm_queue.get()

        self.stop_event.clear()
        self.force_stop_event.clear()
        self.measure_running = True
        self.active_params = dict(params)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.force_stop_btn.setEnabled(True)

        self.start_worker(
            target=self._worker_thread,
            args=(params,),
            name=f'{self.module_id}-worker',
        )

    def _worker_thread(self, params):
        meas = IV_Measurement(params, self.update_queue, self.alarm_queue, self.stop_event, self.force_stop_event)
        meas.run()

    def stop_measurement(self):
        self.stop_event.set()
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("归零中...")
        self.log_info("已触发停止，正在极速安全归零...")

    def force_stop_measurement(self):
        if not self.measure_running:
            self._reset_buttons()
            return

        self.force_stop_event.set()
        self.stop_event.set()
        self.force_stop_btn.setEnabled(False)
        self.force_stop_btn.setText("强制终止中...")
        self.stop_btn.setEnabled(False)
        self.log_info("执行强制终止，切断输出...")
        QTimer.singleShot(500, self._reset_buttons)

    def _reset_buttons(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("停止")
        self.force_stop_btn.setEnabled(False)
        self.force_stop_btn.setText("强制终止")

    def poll_queue(self):
        while not self.alarm_queue.empty():
            message = self.alarm_queue.get()
            self.note_status_from_message(message)
            if '严重警告' in message:
                self.raise_persistent_safety_alarm(message)
            else:
                self.log_info(message)

        count = 0
        while count < 500 and not self.update_queue.empty():
            msg = self.update_queue.get_nowait()
            if msg is None:
                self._reset_buttons()
                self.log_info("流程结束。")
                self.show_final_status()
                self.mark_measurement_finished(self.module_id)
                break

            msg_type = msg[_0]

            if msg_type == 'cycle_start':
                actual_vg = msg[6]
                cycle_text = f"{msg[_1]} / {msg[_2]}"
                self.status_labels['cycle'].setText(cycle_text)
                self.status_labels['gate_v'].setText(
                    '-' if actual_vg is None else f'{actual_vg:.6f}'
                )
                self.status_labels['stage'].setText("扫描准备中...")
                for i in range(4):
                    self.data_count[i] = 0
                    self.v_data[i].fill(0)
                    self.i_data[i].fill(0)
                    self.ig_data[i].fill(np.nan)
                self.points_changed = True
                continue

            if msg_type == 'cycle_done':
                current_cycle = msg[_1]
                status = msg[_2] if len(msg) > 2 else 'complete'
                error = msg[_3] if len(msg) > 3 else None
                self.note_result_status(status, error)
                gate_index, gate_total = msg[4], msg[5]
                requested_vg, actual_vg = msg[6], msg[7]
                snapshot = msg[8]
                counts = snapshot['counts']
                self.submit_save(
                    self.save_data,
                    self.folder_input.text().strip(),
                    self.inputs['file_prefix'].text().strip() or 'IV',
                    self.inputs['mode'].currentText(),
                    float(self.inputs['v_start'].text()),
                    float(self.inputs['v_end'].text()),
                    abs(float(self.inputs['v_step'].text())),
                    snapshot['voltage'],
                    snapshot['current'],
                    snapshot['gate_current'],
                    counts,
                    current_cycle,
                    status,
                    error,
                    gate_enabled=bool(
                        getattr(self, 'active_params', {}).get(
                            'gate_enabled', False
                        )
                    ),
                    gate_index=gate_index,
                    gate_total=gate_total,
                    requested_vg=requested_vg,
                    actual_vg=actual_vg,
                    gate_metadata=dict(
                        getattr(self, 'active_params', {})
                    ),
                    stopped_at_local=time.strftime('%Y-%m-%d %H:%M:%S'),
                )
                continue
                
            if msg_type == 'stage':
                self.status_labels['stage'].setText(msg[_1])
                continue

            if msg_type in ('gate_ramp', 'gate_ready'):
                self.status_labels['gate_v'].setText(f'{msg[1]:.6f}')
                self.status_labels['gate_i'].setText(f'{msg[2]:.6e}')
                self.status_labels['stage'].setText(
                    '栅压爬坡中...'
                    if msg_type == 'gate_ramp'
                    else '栅压已到位'
                )
                continue

            v, i = msg[_1], msg[_2]
            self.status_labels['bias_v'].setText(f"{v:.6f}")
            self.status_labels['bias_i'].setText(f"{i:.6e}")
            
            if v != 0:
                self.status_labels['G'].setText(f"{(i/v)/G0:.6e}")
                self.status_labels['R'].setText(f"{v/i:.2e}" if i != 0 else "inf")
            else:
                self.status_labels['G'].setText("0")
                self.status_labels['R'].setText("inf")

            if msg_type == 'ramp':
                self.status_labels['stage'].setText("爬坡缓冲中...")
            elif msg_type == 'zeroing':
                self.status_labels['stage'].setText("安全归零中...")
            elif isinstance(msg_type, int) and msg_type in (_0, _1, _2, _3):
                self.status_labels['stage'].setText("测量中...")
                if len(msg) >= 5:
                    self.status_labels['gate_v'].setText(f'{msg[3]:.6f}')
                    self.status_labels['gate_i'].setText(f'{msg[4]:.6e}')
                idx = self.data_count[msg_type]
                if idx >= self.capacity:
                    new_cap = self.capacity * 2
                    for seg in range(4):
                        new_v = np.zeros(new_cap)
                        new_i = np.zeros(new_cap)
                        new_ig = np.full(new_cap, np.nan)
                        new_v[:self.capacity] = self.v_data[seg]
                        new_i[:self.capacity] = self.i_data[seg]
                        new_ig[:self.capacity] = self.ig_data[seg]
                        self.v_data[seg] = new_v
                        self.i_data[seg] = new_i
                        self.ig_data[seg] = new_ig
                    self.capacity = new_cap

                self.v_data[msg_type][idx] = v
                self.i_data[msg_type][idx] = i
                if len(msg) >= 5:
                    self.ig_data[msg_type][idx] = msg[4]
                self.data_count[msg_type] += 1
                self.points_changed = True

            count += 1

    def update_plot(self):
        if self.points_changed:
            for i in range(4):
                c = self.data_count[i]
                if c > 0:
                    self.curves[i].setData(self.v_data[i][:c], self.i_data[i][:c])
                else:
                    self.curves[i].setData([], [])
            self.points_changed = False

    def save_data(
        self,
        folder,
        prefix,
        mode,
        v_start,
        v_end,
        v_step,
        voltage_data,
        current_data,
        gate_current_data,
        counts,
        current_cycle=None,
        status='complete',
        error=None,
        gate_enabled=False,
        gate_index=1,
        gate_total=1,
        requested_vg=None,
        actual_vg=None,
        gate_metadata=None,
        stopped_at_local=None,
    ):
        if all(c == 0 for c in counts.values()):
            return

        saved_paths = []
        try:
            s_start = f"{v_start:g}V"
            s_end = f"{v_end:g}V"
            s_step = f"{v_step:g}V"
            s_0 = "0V"
            gate_metadata = gate_metadata or {}
            save_prefix = prefix
            metadata_extra = {
                'gate_enabled': bool(gate_enabled),
                'iv_mode': mode,
                'cycle': current_cycle,
            }
            if gate_enabled:
                save_prefix = (
                    f'{prefix}_VgSeq{int(gate_index):03d}_'
                    f'Vg={float(requested_vg):g}V'
                )
                metadata_extra.update({
                    'gate_mode': gate_metadata.get('gate_mode'),
                    'gate_targets': gate_metadata.get('gate_targets', []),
                    'gate_sequence_index': int(gate_index),
                    'gate_sequence_total': int(gate_total),
                    'requested_vg': float(requested_vg),
                    'actual_vg': float(actual_vg),
                    'gate_address': gate_metadata.get('gate_address'),
                    'gate_terminal': gate_metadata.get('gate_terminal'),
                    'gate_voltage_range': gate_metadata.get(
                        'gate_voltage_range'
                    ),
                    'gate_nplc': gate_metadata.get('gate_nplc'),
                    'gate_ilimit': gate_metadata.get('gate_ilimit'),
                    'gate_leakage_limit': gate_metadata.get(
                        'gate_leakage_limit'
                    ),
                    'gate_ramp_step': gate_metadata.get('gate_ramp_step'),
                    'gate_step_delay': gate_metadata.get('gate_step_delay'),
                    'gate_settle': gate_metadata.get('gate_settle'),
                    'gate_group_wait': gate_metadata.get(
                        'gate_group_wait'
                    ),
                })

            if status == 'complete':
                s_cyc = f"cyc{current_cycle}" if current_cycle else "partial"
            else:
                s_cyc = f"cyc{current_cycle}_partial"

            def write_segment(path, segment):
                with atomic_text_writer(path) as out:
                    if gate_enabled:
                        out.write(
                            '# Voltage (V)\tCurrent (A)\t'
                            'GateVoltage (V)\tGateCurrent (A)\n'
                        )
                        for voltage, current, gate_current in zip(
                            voltage_data[segment],
                            current_data[segment],
                            gate_current_data[segment],
                        ):
                            out.write(
                                f'{voltage:.6f}\t{current:.6e}\t'
                                f'{float(actual_vg):.6f}\t'
                                f'{gate_current:.6e}\n'
                            )
                    else:
                        out.write('# Voltage (V)\tCurrent (A)\n')
                        for voltage, current in zip(
                            voltage_data[segment], current_data[segment]
                        ):
                            out.write(
                                f'{voltage:.6f}\t{current:.6e}\n'
                            )
                write_result_metadata(
                    path,
                    status=status,
                    point_count=counts[segment],
                    error=error,
                    extra=metadata_extra,
                    stopped_at_local=stopped_at_local,
                )

            if mode == 'single':
                f = allocate_unique_path(
                    folder, f"{save_prefix}_single_{s_start}_{s_end}_step{s_step}_{s_cyc}.txt")
                write_segment(f, _0)
                saved_paths.append(str(f))
                self.post_log(f"第 {current_cycle} 轮数据安全落盘至: {f}")

            elif mode == 'bidirectional':
                f_fwd = allocate_unique_path(
                    folder, f"{save_prefix}_forward_{s_start}_{s_end}_step{s_step}_{s_cyc}.txt")
                write_segment(f_fwd, _0)
                saved_paths.append(str(f_fwd))
                f_rev = allocate_unique_path(
                    folder, f"{save_prefix}_reverse_{s_end}_{s_start}_step{s_step}_{s_cyc}.txt")
                write_segment(f_rev, _1)
                saved_paths.append(str(f_rev))
                self.post_log(f"第 {current_cycle} 轮双向数据安全落盘。")

            elif mode == 'hysteresis':
                names = {
                    0: f"{save_prefix}_hysteresis_1st_{s_0}_{s_start}_step{s_step}_{s_cyc}.txt",
                    1: f"{save_prefix}_hysteresis_2nd_{s_start}_{s_end}_step{s_step}_{s_cyc}.txt",
                    2: f"{save_prefix}_hysteresis_3rd_{s_end}_{s_start}_step{s_step}_{s_cyc}.txt",
                    3: f"{save_prefix}_hysteresis_4th_{s_start}_{s_0}_step{s_step}_{s_cyc}.txt",
                }
                for idx in range(4):
                    c = counts[idx]
                    if c == 0:
                        continue
                    fname = names[idx]
                    full_path = allocate_unique_path(folder, fname)
                    write_segment(full_path, idx)
                    saved_paths.append(str(full_path))
                self.post_log(f"第 {current_cycle} 轮四段回滞数据安全保存。")
            return {
                'paths': saved_paths, 'status': status, 'error': error,
            }

        except Exception as exc:
            self.post_log(f"保存失败: {exc}")
            return {'paths': saved_paths, 'status': 'error', 'error': exc}

    def closeEvent(self, event):
        if self.measure_running:
            reply = QMessageBox.question(
                self,
                '警告',
                '测量正在运行中，确认要退出吗？',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.force_stop_event.set()
                self.stop_event.set()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
