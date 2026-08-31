import json
import os
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QFrame, QGroupBox, QLabel

from core.hardware_base import (
    InstrumentConfigurationError,
    GateCurrentLimitError,
    MeasurementReadError,
    allocate_unique_path,
    assert_no_scpi_errors,
    atomic_text_writer,
    check_gate_current_limit,
    configure_gate_meter,
    fast_shutdown_zero_2450,
    generate_exact_ramp_levels,
    reliable_output_off,
    shutdown_report_confirmed,
    required_float_query,
    validate_2450_idn,
    validate_current_range_limit,
    validate_gate_current_limit,
    validate_distinct_addresses,
    validate_nplc,
    validate_positive_step,
    validate_program_step_plan,
    validate_source_voltage,
    validate_step_divides_interval,
    validate_terminal,
    validate_voltage_range,
    validate_gate_voltage_within_range,
    validate_voltage_within_range,
    verify_current_configuration,
    write_result_metadata,
)
from modules.it_step_setgate import (
    ItMeasurement,
    ItStepWidget,
    _8,
    _fit_line_harmonics,
    build_it_voltage_targets,
    parse_it_voltage_sequence,
)
from modules.isd_vg_setvsd import IsdVgMeasurement
from modules.iv_curve import (
    IV_Measurement,
    IVWidget,
    build_gate_targets,
    default_iv_gate_settings,
    merge_gate_settings_into_iv_params,
    parse_custom_iv_values,
    parse_custom_gate_values,
)
from modules.mapping_scan import MappingMeasurement
from modules.break_junction import BreakMeasurement
from modules.arbitrary_bias import ArbMeasurement, ArbitraryBiasWidget
from modules.arbitrary_gate import ArbitraryGateWidget, GateArbMeasurement
from main import MainWindow, WelcomePage, parse_config_modules
from core.app_base import BaseAppWidget
from core.instrument_config import InstrumentSettings
from core.paths import resource_path
from core.time_acquisition import (
    InternalSegmentCollector, RealtimeSampler, timing_metadata,
)
from core.ui_builder import create_status_group


class FakeInstrument:
    def __init__(self, responses=None, write_failures=None):
        self.responses = dict(responses or {})
        self.write_failures = set(write_failures or ())
        self.writes = []

    def write(self, command):
        self.writes.append(command)
        if command in self.write_failures:
            raise OSError(f'write failed: {command}')

    def query(self, command):
        response = self.responses.get(command)
        if isinstance(response, list):
            if not response:
                raise OSError(f'no response left for {command}')
            return response.pop(0)
        if isinstance(response, Exception):
            raise response
        if response is None:
            raise OSError(f'unexpected query: {command}')
        return response


class FakeZeroInstrument:
    def __init__(self, voltage=0.0, fail_command=None, current_limit=1.05e-6):
        self.voltage = float(voltage)
        self.output = 1
        self.current_limit = float(current_limit)
        self.timeout = 5000
        self.fail_command = fail_command
        self.writes = []
        self.pending_endpoint = None

    def write(self, command):
        self.writes.append(command)
        if self.fail_command and command.startswith(self.fail_command):
            raise OSError(f'write failed: {command}')
        upper = command.upper()
        if upper.startswith(':SOUR:SWE:VOLT:LIN '):
            arguments = command.split(' ', 1)[1].split(',')
            self.pending_endpoint = float(arguments[1])
        elif upper == '*WAI' and self.pending_endpoint is not None:
            self.voltage = self.pending_endpoint
            self.pending_endpoint = None
        elif upper.startswith(':SOUR:VOLT '):
            self.voltage = float(command.split()[-1])
        elif upper == ':OUTP OFF':
            self.output = 0

    def query(self, command):
        upper = command.upper()
        if upper == ':SOUR:VOLT?':
            return str(self.voltage)
        if upper == ':OUTP?':
            return str(self.output)
        if upper == ':SOUR:VOLT:ILIM?':
            return str(self.current_limit)
        if upper == '*OPC?':
            return '1'
        if upper == ':TRIG:BLOC:LIST?':
            return (
                ' 6) MEASURE_DIGITIZE      BUFFER: zero\n'
                ' 8) SOURCE_OUTPUT         OUTPUT: OFF\n'
            )
        if upper == ':SYST:ERR?':
            return '0,"No error"'
        raise OSError(f'unexpected query: {command}')


class FakeGateInstrument:
    def __init__(self, voltage=0.0, currents=None):
        self.voltage = float(voltage)
        self.currents = list(currents or [0.0])
        self.writes = []

    def write(self, command):
        self.writes.append(command)
        if command.upper().startswith(':SOUR:VOLT '):
            self.voltage = float(command.split()[-1])

    def query(self, command):
        if command == ':SOUR:VOLT?':
            return str(self.voltage)
        if command == ':READ?':
            value = self.currents.pop(0) if len(self.currents) > 1 else self.currents[0]
            if isinstance(value, Exception):
                raise value
            return str(value)
        raise OSError(f'unexpected query: {command}')


class HardwareReliabilityTests(unittest.TestCase):
    def test_model_and_address_guards(self):
        self.assertIn('2450', validate_2450_idn('KEITHLEY INSTRUMENTS,MODEL 2450,1,1'))
        with self.assertRaises(InstrumentConfigurationError):
            validate_2450_idn('OTHER,MODEL 2400,1,1')
        with self.assertRaises(ValueError):
            validate_distinct_addresses('GPIB0::2::INSTR', 'gpib0::2::instr', True)

    def test_range_limit_and_nplc_guards(self):
        self.assertEqual(
            validate_current_range_limit(1e-6, 1.05e-6),
            (1e-6, 1.05e-6),
        )
        with self.assertRaises(ValueError):
            validate_current_range_limit(1e-6, 1.051e-6)
        with self.assertRaises(ValueError):
            validate_current_range_limit(5e-6, 1e-6)
        self.assertEqual(validate_nplc(0.01), 0.01)
        with self.assertRaises(ValueError):
            validate_nplc(0.001)

    def test_gate_meter_uses_fixed_10_na_range_and_readback(self):
        instrument = FakeInstrument({
            ':SENS:AZER:ONCE;*OPC?': '1',
            ':SENS:CURR:NPLC?': '0.01',
            ':SENS:CURR:RANG:AUTO?': '0',
            ':SENS:CURR:RANG?': '1e-8',
            ':SOUR:VOLT:ILIM?': '1e-9',
            ':ROUT:TERM?': 'REAR',
            ':SENS:CURR:AZER?': '0',
            ':SOUR:VOLT:RANG?': '20',
            ':SYST:ERR?': '0,"No error"',
        })
        settings = configure_gate_meter(
            instrument,
            voltage_range=20,
            current_limit=1e-9,
            nplc=0.01,
            terminal='REAR',
            autozero_mode='block_once',
        )
        self.assertEqual(settings['current_range_A'], 1e-8)
        self.assertEqual(settings['voltage_range_V'], 20)
        self.assertIn(':SENS:CURR:RANG:AUTO OFF', instrument.writes)
        self.assertIn(':SENS:CURR:RANG 1e-08', instrument.writes)
        self.assertIn(':SOUR:VOLT:ILIM 1e-09', instrument.writes)
        self.assertIn(':SOUR:VOLT:RANG 20', instrument.writes)

    def test_gate_limit_rejects_over_range_and_trips_at_equal_value(self):
        self.assertEqual(validate_gate_current_limit(1e-9), 1e-9)
        with self.assertRaises(ValueError):
            validate_gate_current_limit(10.5001e-9)
        with self.assertRaises(GateCurrentLimitError):
            check_gate_current_limit(-1e-9, 1e-9)
        self.assertEqual(validate_gate_voltage_within_range(20, 20), (20, 20))
        with self.assertRaises(ValueError):
            validate_gate_voltage_within_range(20.01, 20)

    def test_realtime_sampler_uses_gate_limit_for_software_stop(self):
        sampler = RealtimeSampler(
            FakeInstrument({':READ?': '2e-6'}),
            FakeInstrument({':READ?': '-1e-9'}),
            0,
            gate_current_limit=1e-9,
        )
        with self.assertRaises(GateCurrentLimitError):
            sampler.sample()


class FakeResourceManager:
    def __init__(self, resources=(), instruments=None):
        self.resources = tuple(resources)
        self.instruments = dict(instruments or {})
        self.closed = False

    def list_resources(self):
        return self.resources

    def open_resource(self, address):
        value = self.instruments[address]
        if isinstance(value, Exception):
            raise value
        return value

    def close(self):
        self.closed = True


class GlobalInstrumentSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_single_and_dual_meter_requirements(self):
        settings = InstrumentSettings(
            bias_address='GPIB0::1::INSTR',
            bias_terminal='REAR',
        )
        self.assertEqual(
            settings.snapshot(require_gate=False)['bias_address'],
            'GPIB0::1::INSTR',
        )
        with self.assertRaisesRegex(ValueError, '目前仅检测到 1 台'):
            settings.snapshot(require_gate=True)

        settings.gate_address = 'GPIB0::2::INSTR'
        self.assertEqual(
            settings.snapshot(require_gate=True)['gate_address'],
            'GPIB0::2::INSTR',
        )
        settings.gate_address = 'gpib0::1::instr'
        with self.assertRaises(ValueError):
            settings.snapshot(require_gate=True)

    def test_scan_zero_one_and_two_resources(self):
        settings = InstrumentSettings()
        page = WelcomePage(settings)

        with patch('main.pyvisa.ResourceManager', return_value=FakeResourceManager()):
            page.scan_instruments()
        self.assertEqual(settings.bias_address, '')
        self.assertEqual(settings.gate_address, '')
        self.assertEqual(page.summary_status.text(), '未扫描到可用仪器')

        one = 'GPIB0::8::INSTR'
        with patch(
            'main.pyvisa.ResourceManager',
            return_value=FakeResourceManager([one]),
        ):
            page.scan_instruments()
        self.assertEqual(settings.bias_address, one)
        self.assertEqual(settings.gate_address, '')
        self.assertEqual(page.summary_status.text(), '已扫描到 1 台设备')

        two = 'GPIB0::9::INSTR'
        with patch(
            'main.pyvisa.ResourceManager',
            return_value=FakeResourceManager([one, two]),
        ):
            page.scan_instruments()
        self.assertEqual(settings.bias_address, one)
        self.assertEqual(settings.gate_address, two)
        self.assertEqual(page.summary_status.text(), '已扫描到 2 台设备')

    def test_scan_status_reports_progress_and_missing_visa(self):
        settings = InstrumentSettings()
        page = WelcomePage(settings)
        progress = {}

        def missing_backend():
            progress['text'] = page.summary_status.text()
            progress['button_enabled'] = page.btn_scan.isEnabled()
            raise ValueError(
                'Could not locate a VISA implementation. '
                'Install either the IVI binary or pyvisa-py.'
            )

        with patch('main.pyvisa.ResourceManager', side_effect=missing_backend):
            page.scan_instruments()

        self.assertEqual(progress['text'], '正在扫描设备…')
        self.assertFalse(progress['button_enabled'])
        self.assertEqual(
            page.summary_status.text(), '扫描失败：未安装 VISA 驱动'
        )
        self.assertIn(
            'Could not locate a VISA implementation',
            page.summary_status.toolTip(),
        )
        self.assertTrue(page.btn_scan.isEnabled())

    def test_scanning_long_addresses_keeps_welcome_columns_equal(self):
        settings = InstrumentSettings()
        page = WelcomePage(settings)
        page.resize(1200, 700)
        page.show()
        self.app.processEvents()
        try:
            frames = page.findChildren(
                QFrame, options=Qt.FindChildOption.FindDirectChildrenOnly
            )
            self.assertEqual(len(frames), 2)
            before = [frame.width() for frame in frames]

            resources = [
                'TCPIP0::192.168.100.250::inst0::INSTR',
                'GPIB0::12345678901234567890::INSTR',
            ]
            with patch(
                'main.pyvisa.ResourceManager',
                return_value=FakeResourceManager(resources),
            ):
                page.scan_instruments()
            self.app.processEvents()

            after = [frame.width() for frame in frames]
            self.assertEqual(before, after)
            self.assertEqual(after[0], after[1])
        finally:
            page.close()

    def test_scan_and_detect_buttons_align_with_selector_edges(self):
        page = WelcomePage(InstrumentSettings())
        page.resize(1200, 700)
        page.show()
        self.app.processEvents()
        try:
            self.assertEqual(
                page.btn_scan.geometry().right(),
                page.bias_address.geometry().right(),
            )
            self.assertEqual(
                page.btn_detect.geometry().left(),
                page.bias_terminal.geometry().left(),
            )
        finally:
            page.close()

    def test_detection_statuses_for_one_two_and_failed_meter(self):
        idn = 'KEITHLEY INSTRUMENTS,MODEL 2450,1,1'
        bias = 'GPIB0::1::INSTR'
        gate = 'GPIB0::2::INSTR'
        settings = InstrumentSettings(bias_address=bias)
        page = WelcomePage(settings)
        one_rm = FakeResourceManager(
            [bias], {bias: FakeInstrument({'*IDN?': idn})}
        )
        with patch('main.pyvisa.ResourceManager', return_value=one_rm):
            page.detect_connections()
        self.assertEqual(
            page.summary_status.text(), '已连接 1 台，可使用单表测试'
        )

        settings.gate_address = gate
        page.set_settings(settings)
        two_rm = FakeResourceManager(
            [bias, gate],
            {
                bias: FakeInstrument({'*IDN?': idn}),
                gate: FakeInstrument({'*IDN?': idn}),
            },
        )
        with patch('main.pyvisa.ResourceManager', return_value=two_rm):
            page.detect_connections()
        self.assertEqual(
            page.summary_status.text(), '两台 2450 已连接，可使用全部测试'
        )

        bad_rm = FakeResourceManager(
            [bias, gate],
            {
                bias: FakeInstrument({'*IDN?': idn}),
                gate: FakeInstrument({'*IDN?': OSError('read failed')}),
            },
        )
        with patch('main.pyvisa.ResourceManager', return_value=bad_rm):
            page.detect_connections()
        self.assertIn('连接失败', page.gate_status.text())
        self.assertEqual(
            page.summary_status.text(), '已连接 1 台，可使用单表测试'
        )

    def test_main_window_shares_one_configuration_without_address_inputs(self):
        window = MainWindow()
        try:
            self.assertTrue(resource_path('assets', 'app_icon.ico').is_file())
            self.assertFalse(window.windowIcon().isNull())
            widgets = window._measurement_widgets()
            self.assertEqual(len(widgets), 7)
            for widget in widgets:
                self.assertIs(
                    widget.instrument_settings, window.instrument_settings
                )
                self.assertFalse({
                    'RESOURCE_NAME', 'TERMINAL', 'address', 'terminal',
                    'BIAS_ADDR', 'GATE_ADDR', 'BIAS_TERM', 'GATE_TERM',
                    'bias_addr', 'gate_addr', 'bias_term', 'gate_term',
                    'bias_address', 'gate_address',
                    'bias_terminal', 'gate_terminal',
                } & set(widget.inputs))
            self.assertEqual(
                [action.text() for action in window.menuBar().actions()],
                ['文件', '测量', '绘图', '帮助'],
            )
            self.assertEqual(
                [window.nav_group.button(i).text() for i in range(8)],
                [
                    '首页', '断裂结', '循环IV特性扫描', '栅压特性扫描',
                    '二维Mapping扫描', 'It特性扫描',
                    '任意偏压波形测试', '任意栅压波形测试',
                ],
            )
        finally:
            window.close()

    def test_shared_root_resolves_each_module_subfolder_and_updates_preview(self):
        window = MainWindow()
        try:
            with tempfile.TemporaryDirectory() as folder:
                window.data_settings.set_root(folder)
                expected = (
                    'Break', 'IV', 'Isd_Vg', 'Mapping', 'It',
                    'Arbitrary_Bias', 'Arbitrary_Gate',
                )
                for widget, subfolder in zip(
                    window._measurement_widgets(), expected
                ):
                    self.assertIs(widget.data_settings, window.data_settings)
                    self.assertEqual(
                        Path(widget.resolved_output_folder()),
                        Path(folder) / subfolder,
                    )
                    self.assertTrue(
                        widget.combined_output_input.text().startswith(
                            subfolder + '/'
                        )
                    )

                first = window.stack.widget(1)
                absolute = Path(folder) / 'manual' / 'sample-a'
                first.set_combined_output_path(
                    str(absolute / 'Break2.txt')
                )
                self.assertEqual(
                    Path(first.resolved_output_folder()), absolute
                )
                self.assertEqual(
                    first.inputs['FILENAME'].text(), 'Break2.txt'
                )
        finally:
            window.close()

    def test_current_configuration_round_trip(self):
        window = MainWindow()
        try:
            with tempfile.TemporaryDirectory() as folder:
                config_path = Path(folder) / 'current.json'
                data_root = Path(folder) / 'data'
                break_widget = window.stack.widget(1)
                it_widget = window.stack.widget(5)
                window.welcome_page.data_root_input.setText(str(data_root))
                break_widget.set_combined_output_path('Break/audit.txt')
                it_widget.rb_meas_time.setChecked(True)

                with (
                    patch(
                        'main.QFileDialog.getSaveFileName',
                        return_value=(str(config_path), 'JSON Files (*.json)'),
                    ),
                    patch.object(window, '_show_info') as save_message,
                ):
                    window.save_config()
                self.assertTrue(config_path.is_file())
                saved = json.loads(config_path.read_text(encoding='utf-8'))
                self.assertEqual(saved['__schema_version__'], 4)
                self.assertEqual(saved['storage']['root'], str(data_root))
                self.assertFalse(any(
                    call.args and call.args[0] == '错误'
                    for call in save_message.call_args_list
                ))

                window.welcome_page.data_root_input.setText('C:/changed')
                break_widget.set_combined_output_path('Break/changed.txt')
                it_widget.rb_meas_points.setChecked(True)
                with (
                    patch(
                        'main.QFileDialog.getOpenFileName',
                        return_value=(str(config_path), 'JSON Files (*.json)'),
                    ),
                    patch.object(window, '_show_info') as load_message,
                ):
                    window.load_config()
                QApplication.processEvents()

                self.assertEqual(Path(window.data_settings.root), data_root)
                self.assertEqual(
                    break_widget.combined_output_input.text(),
                    'Break/audit.txt',
                )
                self.assertTrue(it_widget.rb_meas_time.isChecked())
                self.assertFalse(any(
                    call.args and call.args[0] == '错误'
                    for call in load_message.call_args_list
                ))
        finally:
            window.close()

    def test_all_status_panels_use_equal_left_and_right_halves(self):
        window = MainWindow()
        try:
            window.resize(1600, 900)
            window.show()
            QApplication.processEvents()
            for index in range(1, 8):
                window.stack.setCurrentIndex(index)
                QApplication.processEvents()
                widget = window.stack.widget(index)
                panes = {
                    label.parentWidget()
                    for label in widget.status_labels.values()
                }
                self.assertEqual(len(panes), 2)
                widths = [pane.width() for pane in panes]
                self.assertLessEqual(abs(widths[0] - widths[1]), 1)
                for pane in panes:
                    labels = pane.findChildren(
                        QLabel, options=(
                            Qt.FindChildOption.FindDirectChildrenOnly
                        )
                    )
                    colons = [label for label in labels if label.text() == ':']
                    self.assertTrue(colons)
                    self.assertEqual(len({label.x() for label in colons}), 1)
                    names = [
                        label for label in labels
                        if label.text() not in {':', '-'}
                    ]
                    self.assertTrue(all(
                        label.alignment() & Qt.AlignmentFlag.AlignLeft
                        for label in names
                    ))
        finally:
            window.close()

    def test_dynamic_parameter_panels_repaint_after_layout_settles(self):
        window = MainWindow()
        try:
            window.show()
            QApplication.processEvents()
            for index in (2, 5, 6):
                widget = window.stack.widget(index)
                window.stack.setCurrentWidget(widget)
                QApplication.processEvents()
                widget.cb_gate.setChecked(True)
                self.assertFalse(widget.parameter_panel.updatesEnabled())
                QApplication.processEvents()
                self.assertTrue(widget.parameter_panel.updatesEnabled())
                widget.cb_gate.setChecked(False)
                widget.cb_gate.setChecked(True)
                widget.cb_gate.setChecked(False)
                self.assertFalse(widget.parameter_panel.updatesEnabled())
                QApplication.processEvents()
                self.assertTrue(widget.parameter_panel.updatesEnabled())
        finally:
            window.close()

    def test_approved_status_and_output_hint_texts_are_exact(self):
        window = MainWindow()
        try:
            widgets = window._measurement_widgets()
            self.assertEqual(
                Path(window.data_settings.root),
                Path('C:/Users/Public/Documents/K2450_Data'),
            )
            self.assertIn('progress', widgets[2].status_labels)
            self.assertEqual(widgets[2].status_labels['progress'].text(), '-')
            self.assertEqual(
                widgets[4].lbl_measurement_mode.text(), '测量模式:'
            )
            self.assertEqual(
                [widget.output_hint_label.text() for widget in widgets],
                [
                    '',
                    '后缀自动追加：扫描模式、电压、循环信息',
                    '后缀自动追加：_seg1 至 _seg4',
                    '后缀自动追加：栅压、偏压信息',
                    '后缀自动追加：时长/点数、栅压、偏压信息',
                    '',
                    '',
                ],
            )
            self.assertNotIn('插入', widgets[2].output_hint_label.text())
            self.assertNotIn('.txt', widgets[2].output_hint_label.text())
            for widget in widgets:
                self.assertTrue(any(
                    label.text() == '保存路径 (根目录下)：'
                    for label in widget.findChildren(QLabel)
                ))
            self.assertNotIn('SETTLE_TIME', widgets[0].inputs)
            self.assertNotIn('settle_time', widgets[1].inputs)
            expected_time_labels = (
                (widgets[1], {
                    '栅压单步延时 (s):', '栅压到位等待 (s):',
                    '栅压组间等待 (s):',
                }),
                (widgets[2], {
                    '偏压到位等待 (s):', '栅压稳定时间 (s):',
                }),
                (widgets[4], {
                    '栅压到位等待 (s):', '偏压到位等待 (s):',
                    '偏压归零后等待 (s):', '测量后保持 (s):',
                }),
                (widgets[5], {
                    '栅压单步延时 (s):', '栅压到位等待 (s):',
                    '跃变缓冲时延 (s):',
                }),
                (widgets[6], {
                    '偏压单步延时 (s):', '偏压到位等待 (s):',
                    '跃变缓冲时延 (s):',
                }),
            )
            for widget, expected in expected_time_labels:
                visible_text = {
                    label.text() for label in widget.findChildren(QLabel)
                }
                self.assertTrue(expected <= visible_text)
            self.assertEqual(widgets[5].inputs['b_range'].currentData(), 1e-6)
            self.assertNotIn('g_range', widgets[6].inputs)
            self.assertEqual(widgets[6].inputs['g_voltage_range'].text(), '20')
            self.assertEqual(widgets[6].inputs['b_range'].currentData(), 1e-6)
            gate_fields = (
                (widgets[1], 'gate_voltage_range', 'gate_ilimit'),
                (widgets[2], 'Gate_VOLT_RANGE', 'Gate_I_LIMIT'),
                (widgets[3], 'gate_v_range', 'gate_i_limit'),
                (widgets[4], 'g_voltage_range', 'g_ilimit'),
                (widgets[5], 'g_voltage_range', 'g_ilimit'),
                (widgets[6], 'g_voltage_range', 'g_ilimit'),
            )
            for widget, range_key, limit_key in gate_fields:
                self.assertEqual(float(widget.inputs[range_key].text()), 20.0)
                self.assertEqual(float(widget.inputs[limit_key].text()), 1e-9)
                for obsolete in (
                    'gate_leakage_limit', 'Ig_THRESHOLD',
                    'ig_threshold', 'g_range',
                ):
                    self.assertNotIn(obsolete, widget.inputs)
            it_widget = widgets[4]
            window.resize(1600, 900)
            window.stack.setCurrentWidget(it_widget)
            window.show()
            QApplication.processEvents()
            self.assertEqual(
                it_widget.rb_sample_realtime.x(), it_widget.rb_meas_time.x()
            )
        finally:
            window.close()

    def test_bundled_configs_use_only_global_instrument_section(self):
        forbidden = {
            'RESOURCE_NAME', 'TERMINAL', 'address', 'terminal',
            'BIAS_ADDR', 'GATE_ADDR', 'BIAS_TERM', 'GATE_TERM',
            'bias_addr', 'gate_addr', 'bias_term', 'gate_term',
            'bias_address', 'gate_address',
            'bias_terminal', 'gate_terminal',
        }
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        for path in config_dir.glob('*.json'):
            if path.name == 'plotting_default.json':
                continue
            value = json.loads(path.read_text(encoding='utf-8'))
            self.assertIn('instruments', value, path.name)
            for module in value['modules'].values():
                self.assertFalse(forbidden & set(module), path.name)
                gate_settings = module.get('__gate_settings__', {})
                self.assertFalse(forbidden & set(gate_settings), path.name)
            self.assertNotIn(
                'SETTLE_TIME', value['modules']['break_junction'], path.name
            )
            self.assertNotIn(
                'settle_time', value['modules']['iv_curve'], path.name
            )
            self.assertNotIn(
                'gate_leakage_limit', value['modules']['iv_curve'], path.name
            )
            self.assertNotIn(
                'gate_leakage_limit',
                value['modules']['iv_curve']['__gate_settings__'], path.name
            )
            self.assertNotIn(
                'Ig_THRESHOLD', value['modules']['isd_vg_setvsd'], path.name
            )
            self.assertNotIn(
                'ig_threshold', value['modules']['mapping_scan'], path.name
            )
            self.assertNotIn(
                'g_range', value['modules']['arbitrary_gate'], path.name
            )


class FastZeroingTests(unittest.TestCase):
    def test_exact_levels_positive_negative_and_residual(self):
        positive = generate_exact_ramp_levels(0.25, 0.0, 0.1)
        negative = generate_exact_ramp_levels(-0.25, 0.0, 0.1)
        np.testing.assert_allclose(positive, [0.15, 0.05, 0.0])
        np.testing.assert_allclose(negative, [-0.15, -0.05, 0.0])
        self.assertEqual(generate_exact_ramp_levels(0.0, 0.0, 0.1), [])
        for levels, start in ((positive, 0.25), (negative, -0.25)):
            previous = start
            for level in levels:
                self.assertLessEqual(abs(level - previous), 0.1 + 1e-12)
                previous = level
            self.assertEqual(levels[-1], 0.0)

    def test_internal_sweep_chunk_counts_and_final_confirmation(self):
        for steps, expected_chunks in (
            (250, 1),
            (251, 2),
            (1000, 4),
            (2501, 11),
        ):
            with self.subTest(steps=steps):
                instrument = FakeZeroInstrument(steps * 0.0001)
                report = fast_shutdown_zero_2450(instrument, 0.0001)
                self.assertEqual(report['status'], 'complete')
                self.assertEqual(report['step_count'], steps)
                self.assertEqual(report['chunk_count'], expected_chunks)
                self.assertEqual(report['zero_readback'], 0.0)
                self.assertTrue(report['output_off_confirmed'])
                self.assertEqual(instrument.output, 0)
                self.assertEqual(instrument.voltage, 0.0)
                makes = [
                    command for command in instrument.writes
                    if command.startswith(':TRAC:MAKE ')
                ]
                deletes = [
                    command for command in instrument.writes
                    if command.startswith(':TRAC:DEL ')
                ]
                self.assertEqual(len(makes), 1)
                self.assertEqual(len(deletes), 1)
                self.assertEqual(
                    makes[0].split('"')[1], deletes[0].split('"')[1]
                )
                self.assertIn(':SOUR:VOLT:DEL 0', instrument.writes)
                self.assertIn(':SOUR:VOLT:DEL:AUTO OFF', instrument.writes)
                self.assertIn(':TRIG:BLOC:NOP 8', instrument.writes)
                self.assertIn(':SENS:CURR:RANG 1e-06', instrument.writes)
                self.assertFalse(any(
                    command.startswith(':SOUR:VOLT:ILIM ')
                    for command in instrument.writes
                ))

    def test_single_exact_step_uses_confirmed_direct_change(self):
        instrument = FakeZeroInstrument(0.05)
        report = fast_shutdown_zero_2450(instrument, 0.05)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['step_count'], 1)
        self.assertEqual(report['chunk_count'], 0)
        self.assertEqual(instrument.voltage, 0.0)
        self.assertFalse(any(
            command.startswith(':SOUR:SWE:VOLT:LIN ')
            for command in instrument.writes
        ))

    def test_sub_minimum_gate_limit_is_not_changed_by_range_command(self):
        instrument = FakeZeroInstrument(0.1, current_limit=1e-9)
        report = fast_shutdown_zero_2450(instrument, 0.001)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(instrument.current_limit, 1e-9)
        self.assertFalse(any(
            command.startswith(':SENS:CURR:RANG ')
            for command in instrument.writes
        ))

    def test_fast_zero_temporarily_extends_and_restores_visa_timeout(self):
        instrument = FakeZeroInstrument(0.1)
        report = fast_shutdown_zero_2450(instrument, 0.001)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(instrument.timeout, 5000)

    def test_sweep_failure_still_zeros_and_turns_output_off(self):
        instrument = FakeZeroInstrument(0.1, fail_command=':INIT')
        report = fast_shutdown_zero_2450(instrument, 0.001)
        self.assertEqual(report['status'], 'emergency_off')
        self.assertTrue(report['output_off_confirmed'])
        self.assertEqual(instrument.voltage, 0.0)
        self.assertEqual(instrument.output, 0)
        self.assertIn(':ABOR', instrument.writes)
        self.assertIn(':OUTP OFF', instrument.writes)

    def test_force_stop_uses_direct_emergency_sequence(self):
        event = threading.Event()
        event.set()
        instrument = FakeZeroInstrument(0.1)
        report = fast_shutdown_zero_2450(
            instrument, 0.001, force_event=event
        )
        self.assertEqual(report['status'], 'emergency_off')
        self.assertEqual(
            instrument.writes[:3],
            [':ABOR', ':SOUR:VOLT 0', ':OUTP OFF'],
        )
        self.assertFalse(any(
            command.startswith(':SOUR:SWE:') for command in instrument.writes
        ))

    def test_confirmed_emergency_shutdown_is_reported_as_safe(self):
        event = threading.Event()
        event.set()
        instrument = FakeZeroInstrument(0.05)
        report = fast_shutdown_zero_2450(
            instrument, 0.001, force_event=event
        )
        self.assertTrue(shutdown_report_confirmed(report))

        messages = queue.Queue()
        worker = ArbMeasurement(
            {'b_ramp_step': 0.001, 'gate_enabled': False},
            messages, threading.Event(), event,
        )
        worker.bias_k = FakeZeroInstrument(0.05)
        worker.safe_zeroing()
        queued = []
        while not messages.empty():
            queued.append(messages.get_nowait())
        self.assertTrue(any(
            item[0] == 'log' and '已紧急归零并关闭输出' in item[1]
            for item in queued
        ))
        self.assertFalse(any(
            item[0] == 'log' and '安全归零失败' in item[1]
            for item in queued
        ))
        self.assertIn(('ramp_b', 0.0, 0.0), queued)

    def test_cleanup_waits_are_not_truncated(self):
        worker = IsdVgMeasurement(
            {}, queue.Queue(), threading.Event(), threading.Event()
        )
        for duration in (0.01, 0.05, 0.1, 0.5):
            with self.subTest(duration=duration):
                started = time.monotonic()
                self.assertTrue(worker._zeroing_sleep(duration))
                self.assertGreaterEqual(
                    time.monotonic() - started, duration * 0.95
                )

    def test_isdvg_finished_payload_keeps_formal_bias_after_zeroing(self):
        bias = FakeInstrument({
            '*IDN?': 'KEITHLEY INSTRUMENTS,MODEL 2450,BIAS,1',
        })
        gate = FakeInstrument({
            '*IDN?': 'KEITHLEY INSTRUMENTS,MODEL 2450,GATE,1',
        })

        class ResourceManager:
            def open_resource(self, address):
                return bias if address.endswith('::1::INSTR') else gate

            def close(self):
                pass

        preset = {
            'BIAS_ADDR': 'GPIB0::1::INSTR',
            'GATE_ADDR': 'GPIB0::2::INSTR',
            'BIAS_TERM': 'REAR',
            'GATE_TERM': 'REAR',
            'Bias_target': 0.05,
            'Bias_step': 0.05,
            'Bias_RANGE': 1e-6,
            'Bias_I_LIMIT': 1.05e-6,
            'Bias_NPLC': 0.1,
            'Bias_Delay': 0.0,
            'Gate_VOLT_RANGE': 20.0,
            'Gate_I_LIMIT': 1e-9,
            'Vg_1st': 0.05,
            'Vg_2nd': -0.05,
            'Vg_step': 0.05,
            'SETTLE_TIME': 0.0,
        }
        messages = queue.Queue()
        worker = IsdVgMeasurement(
            preset, messages, threading.Event(), threading.Event()
        )

        def read_value(instrument, _command, _label):
            return 5e-8 if instrument is bias else 1e-12

        with (
            patch('modules.isd_vg_setvsd.pyvisa.ResourceManager',
                  return_value=ResourceManager()),
            patch('modules.isd_vg_setvsd.clear_scpi_status'),
            patch('modules.isd_vg_setvsd.configure_current_autozero'),
            patch('modules.isd_vg_setvsd.verify_current_configuration'),
            patch('modules.isd_vg_setvsd.configure_gate_meter',
                  return_value={}),
            patch('modules.isd_vg_setvsd.required_float_query',
                  side_effect=read_value),
            patch('modules.isd_vg_setvsd.reliable_output_off',
                  return_value=(True, [])),
        ):
            worker.run()

        queued = []
        while not messages.empty():
            queued.append(messages.get_nowait())
        finished = next(item for item in queued if item[0] == 'finished')
        self.assertEqual(finished[1][4], 0.05)
        self.assertTrue(finished[1][5])

    def test_it_intergroup_point_keeps_001_second_delay(self):
        instrument = FakeInstrument({
            'SOUR:VOLT?': '0',
            'READ?': '1e-9',
        })
        worker = ItMeasurement(
            {}, queue.Queue(), threading.Event(), threading.Event()
        )
        with patch('modules.it_step_setgate.time.sleep') as sleep_mock:
            self.assertTrue(worker._ramp_voltage(
                instrument, 0.001, 0.001, 0.01, is_gate=False
            ))
        sleep_mock.assert_called_once_with(0.01)

    def test_dual_meter_final_zeroing_preserves_program_order(self):
        cases = [
            (
                'modules.it_step_setgate.fast_shutdown_zero_2450',
                ItMeasurement,
                {'gate_enabled': True, 'b_ramp_step': 0.001,
                 'g_ramp_step': 0.05},
                ('bias_keithley', 'gate_keithley'),
                ['偏压表', '栅压表'],
            ),
            (
                'modules.arbitrary_bias.fast_shutdown_zero_2450',
                ArbMeasurement,
                {'gate_enabled': True, 'b_ramp_step': 0.001,
                 'g_ramp_step': 0.05},
                ('bias_k', 'gate_k'),
                ['偏压表', '栅压表'],
            ),
            (
                'modules.arbitrary_gate.fast_shutdown_zero_2450',
                GateArbMeasurement,
                {'b_ramp_step': 0.001, 'g_ramp_step': 0.05},
                ('bias_k', 'gate_k'),
                ['栅压表', '偏压表'],
            ),
        ]

        for target, cls, params, attributes, expected in cases:
            with self.subTest(cls=cls.__name__):
                worker = cls(
                    params, queue.Queue(), threading.Event(), threading.Event()
                )
                setattr(worker, attributes[0], object())
                setattr(worker, attributes[1], object())
                labels = []

                def fake_fast(_instrument, _step, **kwargs):
                    labels.append(kwargs['label'])
                    return {
                        'status': 'complete',
                        'errors': [],
                        'elapsed_s': 0.0,
                    }

                with patch(target, side_effect=fake_fast):
                    worker.safe_zeroing()
                self.assertEqual(labels, expected)

    def test_mapping_final_zeroing_is_bias_then_gate(self):
        worker = MappingMeasurement(
            {'bias_step_up': 0.001, 'vg_step': 0.05},
            queue.Queue(), queue.Queue(),
            threading.Event(), threading.Event(),
        )
        worker.smu_b = object()
        worker.smu_g = object()
        labels = []

        def fake_fast(_instrument, _step, **kwargs):
            labels.append(kwargs['label'])
            return {'status': 'complete', 'errors': [], 'elapsed_s': 0.0}

        with patch(
            'modules.mapping_scan.fast_shutdown_zero_2450',
            side_effect=fake_fast,
        ):
            worker.safe_ramp_to_zero()
        self.assertEqual(labels, ['Mapping偏压表', 'Mapping栅压表'])


class IVMultiGateTests(unittest.TestCase):
    def gate_settings(self, **changes):
        settings = default_iv_gate_settings()
        settings.update({
            'gate_voltage_range': 20.0,
            'gate_step_delay': 0.0,
            'gate_settle': 0.0,
            'gate_group_wait': 0.0,
        })
        settings.update(changes)
        return settings

    def worker_params(self, **changes):
        params = {
            'mode': 'single',
            'v_start': 0.0,
            'v_end': 0.0,
            'v_step': 0.1,
            'gate_enabled': True,
            'gate_ramp_step': 0.5,
            'gate_step_delay': 0.0,
            'gate_settle': 0.0,
            'gate_ilimit': 1e-9,
        }
        params.update(changes)
        return params

    def test_single_step_reverse_and_custom_sequences(self):
        self.assertEqual(
            build_gate_targets(self.gate_settings(
                mode='single', single_target=-2
            )),
            [-2.0],
        )
        self.assertEqual(
            build_gate_targets(self.gate_settings(
                mode='step', step_start=-1, step_end=1,
                test_step=0.5,
            )),
            [-1.0, -0.5, 0.0, 0.5, 1.0],
        )
        self.assertEqual(
            build_gate_targets(self.gate_settings(
                mode='step', step_start=1, step_end=-1,
                test_step=0.5,
            )),
            [1.0, 0.5, 0.0, -0.5, -1.0],
        )
        self.assertEqual(
            parse_custom_gate_values('0, 1；0\n-1  0'),
            [0.0, 1.0, 0.0, -1.0, 0.0],
        )
        self.assertEqual(
            build_gate_targets(self.gate_settings(
                mode='custom', custom_text='0, 0.05, 0.1, 0.05'
            )),
            [0.0, 0.05, 0.1, 0.05],
        )

    def test_gate_mode_does_not_overwrite_iv_mode(self):
        gate = self.gate_settings(
            mode='custom', custom_text='0,0.05,0.1,0.05'
        )
        targets = build_gate_targets(gate)
        merged = merge_gate_settings_into_iv_params(
            {'mode': 'hysteresis'}, gate, targets
        )
        self.assertEqual(merged['mode'], 'hysteresis')
        self.assertEqual(merged['gate_mode'], 'custom')
        self.assertEqual(merged['gate_targets'], targets)

    def test_invalid_gate_sequences_are_rejected(self):
        with self.assertRaisesRegex(ValueError, '不能整除'):
            build_gate_targets(self.gate_settings(
                mode='step', step_start=0, step_end=1,
                test_step=0.3,
            ))
        with self.assertRaisesRegex(ValueError, '不能为空'):
            build_gate_targets(self.gate_settings(
                mode='custom', custom_text=' '
            ))
        with self.assertRaisesRegex(ValueError, '有效数字'):
            build_gate_targets(self.gate_settings(
                mode='custom', custom_text='0, abc, 1'
            ))
        with self.assertRaisesRegex(ValueError, '有限数'):
            build_gate_targets(self.gate_settings(
                mode='custom', custom_text='0, nan'
            ))
        with self.assertRaises(ValueError):
            build_gate_targets(self.gate_settings(
                mode='custom', custom_text='0, 25'
            ))

    def test_gate_ramp_moves_directly_from_current_to_next_target(self):
        worker = IV_Measurement(
            self.worker_params(),
            queue.Queue(), queue.Queue(),
            threading.Event(), threading.Event(),
        )
        worker.gate_keithley = FakeGateInstrument(
            voltage=1.0, currents=[1e-12, 1e-12, 1e-12]
        )
        self.assertTrue(worker._ramp_gate(2.0))
        source_levels = [
            float(command.split()[-1])
            for command in worker.gate_keithley.writes
            if command.startswith(':SOUR:VOLT ')
        ]
        self.assertEqual(source_levels, [1.5, 2.0])
        self.assertEqual(worker.actual_vg, 2.0)

    def test_disabled_gate_keeps_three_item_iv_messages(self):
        params = self.worker_params(
            gate_enabled=False, v_start=0.0, v_end=0.0
        )
        updates = queue.Queue()
        worker = IV_Measurement(
            params, updates, queue.Queue(),
            threading.Event(), threading.Event(),
        )
        worker.keithley = FakeInstrument({
            ':READ?': ['1e-9', '2e-9'],
        })
        worker._reset_cycle_data()
        worker.measure_loop()
        messages = []
        while not updates.empty():
            messages.append(updates.get())
        formal = [message for message in messages if isinstance(message[0], int)]
        self.assertEqual(len(formal), 1)
        self.assertEqual(len(formal[0]), 3)
        self.assertEqual(worker._cycle_snapshot()['counts'][0], 1)

    def test_ig_failure_keeps_isd_and_records_nan(self):
        worker = IV_Measurement(
            self.worker_params(),
            queue.Queue(), queue.Queue(),
            threading.Event(), threading.Event(),
        )
        worker.keithley = FakeInstrument({
            ':READ?': ['1e-9', '1e-7'],
        })
        worker.gate_keithley = FakeGateInstrument(
            currents=[OSError('Ig timeout')]
        )
        worker.actual_vg = 0.05
        worker._reset_cycle_data()
        with self.assertRaises(MeasurementReadError):
            worker.measure_loop()
        snapshot = worker._cycle_snapshot()
        self.assertEqual(snapshot['counts'][0], 1)
        self.assertEqual(snapshot['current'][0][0], 1e-7)
        self.assertTrue(np.isnan(snapshot['gate_current'][0][0]))

    def test_leakage_trip_keeps_triggering_reading(self):
        worker = IV_Measurement(
            self.worker_params(),
            queue.Queue(), queue.Queue(),
            threading.Event(), threading.Event(),
        )
        worker.keithley = FakeInstrument({
            ':READ?': ['1e-9', '1e-7'],
        })
        worker.gate_keithley = FakeGateInstrument(currents=[2e-9])
        worker.actual_vg = 0.05
        worker._reset_cycle_data()
        with self.assertRaises(GateCurrentLimitError):
            worker.measure_loop()
        snapshot = worker._cycle_snapshot()
        self.assertEqual(snapshot['counts'][0], 1)
        self.assertEqual(snapshot['gate_current'][0][0], 2e-9)

    def test_gate_enabled_raw_has_four_columns_and_sequence_name(self):
        class SaveHarness:
            def post_log(self, _message):
                pass

        with tempfile.TemporaryDirectory() as folder:
            arrays = {
                0: np.array([0.0, 0.1]),
                1: np.array([]),
                2: np.array([]),
                3: np.array([]),
            }
            IVWidget.save_data(
                SaveHarness(), folder, 'IV', 'single',
                0.0, 0.1, 0.1,
                arrays,
                {**arrays, 0: np.array([0.0, 1e-7])},
                {**arrays, 0: np.array([1e-12, 2e-12])},
                {0: 2, 1: 0, 2: 0, 3: 0},
                1, 'complete', None,
                gate_enabled=True,
                gate_index=4,
                gate_total=4,
                requested_vg=0.05,
                actual_vg=0.05,
                gate_metadata={
                    **self.gate_settings(
                        mode='custom',
                        custom_text='0,0.05,0.1,0.05',
                    ),
                    'gate_mode': 'custom',
                    'gate_targets': [0, 0.05, 0.1, 0.05],
                },
            )
            paths = list(Path(folder).glob('*.txt'))
            self.assertEqual(len(paths), 1)
            self.assertIn('VgSeq004_Vg=0.05V', paths[0].name)
            lines = paths[0].read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(lines[1].split('\t')), 4)
            metadata_path = (
                paths[0].parent / 'metadata'
                / f'{paths[0].stem}_meta.json'
            )
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            self.assertEqual(metadata['gate_sequence_index'], 4)
            self.assertEqual(metadata['gate_mode'], 'custom')
            self.assertEqual(
                metadata['gate_targets'], [0, 0.05, 0.1, 0.05]
            )


class IVScanModeTests(unittest.TestCase):
    def test_custom_voltage_parser_accepts_dense_text_and_keeps_duplicates(self):
        text = '-1,-0.5 0；0.5，1\t0.5\n0'
        self.assertEqual(
            parse_custom_iv_values(text),
            [-1.0, -0.5, 0.0, 0.5, 1.0, 0.5, 0.0],
        )
        with self.assertRaisesRegex(ValueError, '第2项'):
            parse_custom_iv_values('0, wrong, 1')
        with self.assertRaisesRegex(ValueError, '不能为空'):
            parse_custom_iv_values('  ')

    def test_four_radio_modes_show_matching_parameter_page(self):
        app = QApplication.instance() or QApplication([])
        widget = IVWidget()
        try:
            widget.resize(1920, 800)
            widget.show()
            app.processEvents()
            self.assertFalse(hasattr(widget, 'mode_combo'))
            group_titles = {
                group.title() for group in widget.findChildren(QGroupBox)
            }
            self.assertIn('扫描方式与范围', group_titles)
            self.assertIn('采集与保护参数', group_titles)
            self.assertNotIn('控制参数', group_titles)
            self.assertEqual(
                widget.inputs['v_start'].mapToGlobal(QPoint()).x(),
                widget.inputs['i_limit'].mapToGlobal(QPoint()).x(),
            )
            self.assertEqual(
                widget.inputs['v_end'].mapToGlobal(QPoint()).x(),
                widget.inputs['nplc'].mapToGlobal(QPoint()).x(),
            )
            self.assertEqual(
                widget.rb_iv_bidirectional.mapToGlobal(QPoint()).x(),
                widget.lbl_v_end.mapToGlobal(QPoint()).x(),
            )
            self.assertEqual(
                widget.rb_iv_custom.mapToGlobal(QPoint()).x(),
                widget.lbl_v_end.mapToGlobal(QPoint()).x(),
            )
            expected = {
                'single': widget.rb_iv_single,
                'bidirectional': widget.rb_iv_bidirectional,
                'hysteresis': widget.rb_iv_hysteresis,
                'custom': widget.rb_iv_custom,
            }
            self.assertEqual(widget.rb_iv_custom.text(), 'Custom 自定义')
            for mode, button in expected.items():
                acquisition_group = next(
                    group for group in widget.findChildren(QGroupBox)
                    if group.title() == '采集与保护参数'
                )
                before_y = acquisition_group.mapToGlobal(QPoint()).y()
                button.setChecked(True)
                app.processEvents()
                self.assertEqual(widget.selected_iv_mode(), mode)
                self.assertEqual(
                    widget.iv_mode_stack.currentIndex(),
                    1 if mode == 'custom' else 0,
                )
                self.assertEqual(
                    acquisition_group.mapToGlobal(QPoint()).y(), before_y
                )
            widget.set_iv_mode('hysteresis')
            self.assertEqual(widget.lbl_v_start.text(), '目标电压一 (V):')
            self.assertEqual(widget.lbl_v_end.text(), '目标电压二 (V):')
            widget.custom_iv_text = ','.join(str(i / 10) for i in range(40))
            widget._update_custom_iv_summary()
            self.assertIn('40', widget.lbl_custom_iv_summary.text())
        finally:
            widget.close()

    def test_custom_measurement_uses_only_entered_points_in_order(self):
        values = [-0.2, 0.0, 0.3, 0.3, -0.1]
        worker = IV_Measurement(
            {
                'mode': 'custom',
                'custom_voltages': values,
                'v_start': values[0],
                'v_end': values[-1],
                'v_step': 0.01,
                'gate_enabled': False,
            },
            queue.Queue(), queue.Queue(),
            threading.Event(), threading.Event(),
        )
        worker.keithley = FakeInstrument({
            ':READ?': ['0'] + ['1e-9'] * len(values),
        })
        worker._reset_cycle_data()
        worker.measure_loop()
        source_values = [
            float(command.split()[-1])
            for command in worker.keithley.writes
            if command.startswith(':SOUR:VOLT ')
        ]
        self.assertEqual(source_values, values)
        np.testing.assert_allclose(
            worker._cycle_snapshot()['voltage'][0], values
        )

    def test_custom_sequence_saves_one_file_and_metadata(self):
        class SaveHarness:
            def post_log(self, _message):
                pass

        values = np.array([-0.2, 0.0, 0.3, 0.3, -0.1])
        empty = np.array([])
        with tempfile.TemporaryDirectory() as folder:
            result = IVWidget.save_data(
                SaveHarness(), folder, 'IV', 'custom',
                values[0], values[-1], 0.01,
                {0: values, 1: empty, 2: empty, 3: empty},
                {0: values * 1e-6, 1: empty, 2: empty, 3: empty},
                {0: empty, 1: empty, 2: empty, 3: empty},
                {0: len(values), 1: 0, 2: 0, 3: 0},
                1, 'complete', None,
                gate_metadata={
                    'custom_voltage_text': '-0.2, 0, 0.3, 0.3, -0.1',
                    'custom_voltages': values.tolist(),
                },
            )
            self.assertEqual(len(result['paths']), 1)
            path = Path(result['paths'][0])
            self.assertIn('custom_5points', path.name)
            metadata = json.loads(
                (path.parent / 'metadata' / f'{path.stem}_meta.json')
                .read_text(encoding='utf-8')
            )
            self.assertEqual(metadata['custom_voltages'], values.tolist())


class StepDivisibilityTests(unittest.TestCase):
    def test_exact_steps_are_accepted_and_inexact_steps_are_rejected(self):
        self.assertEqual(validate_step_divides_interval(-5, 5, 0.05), 200)
        self.assertEqual(validate_step_divides_interval(0, -0.3, 0.1), 3)
        with self.assertRaisesRegex(ValueError, '不能整除'):
            validate_step_divides_interval(0, 1, 0.3, '测试步进')

    def test_all_bundled_profiles_have_exact_step_plans(self):
        root = Path(__file__).resolve().parents[1]
        for config_path in sorted((root / 'configs').glob('*.json')):
            modules = json.loads(config_path.read_text(encoding='utf-8'))['modules']

            def numeric(section):
                result = {}
                for key, value in section.items():
                    if key.startswith('__'):
                        continue
                    try:
                        result[key] = float(value)
                    except (TypeError, ValueError):
                        result[key] = value
                return result

            iv = numeric(modules['iv_curve'])
            validate_program_step_plan('iv', iv)

            isd = numeric(modules['isd_vg_setvsd'])
            validate_program_step_plan('isd_vg', isd)

            mapping = numeric(modules['mapping_scan'])
            validate_program_step_plan('mapping', mapping)

            it_raw = modules['it_step_setgate']
            controls = it_raw.get('__controls__', {})
            it = {
                'gate_enabled': bool(controls.get('cb_gate', False)),
                'g_mode': 'single' if controls.get('rb_g_single', True) else 'step',
                'b_mode': 'single' if controls.get('rb_b_single', True) else 'step',
                'g_target': float(it_raw['g_target_s']),
                'g_start': float(it_raw['g_start']),
                'g_end': float(it_raw['g_end']),
                'g_test_step': float(it_raw['g_test_step']),
                'g_ramp_step': float(it_raw['g_ramp_step']),
                'b_target': float(it_raw['b_target_s']),
                'b_start': float(it_raw['b_start']),
                'b_end': float(it_raw['b_end']),
                'b_test_step': float(it_raw['b_test_step']),
                'b_ramp_step': float(it_raw['b_ramp_step']),
            }
            validate_program_step_plan('it', it)

            arb_bias = numeric(modules['arbitrary_bias'])
            arb_bias['gate_enabled'] = bool(
                modules['arbitrary_bias'].get('__controls__', {}).get(
                    'cb_gate', False
                )
            )
            arb_bias['waveform'] = modules['arbitrary_bias']['__waveform__']
            validate_program_step_plan('arbitrary_bias', arb_bias)

            arb_gate = numeric(modules['arbitrary_gate'])
            arb_gate['waveform'] = modules['arbitrary_gate']['__waveform__']
            validate_program_step_plan('arbitrary_gate', arb_gate)

            break_params = numeric(modules['break_junction'])
            validate_program_step_plan('break_junction', break_params)
        self.assertEqual(validate_source_voltage(210), 210.0)
        with self.assertRaises(ValueError):
            validate_source_voltage(210.1)
        with self.assertRaises(ValueError):
            validate_positive_step(0)
        self.assertEqual(validate_terminal('FRON'), 'FRONT')
        self.assertEqual(validate_voltage_range(20), 20.0)
        with self.assertRaises(ValueError):
            validate_voltage_range(10)
        self.assertEqual(
            validate_voltage_within_range(5, 20),
            (5.0, 20.0),
        )
        with self.assertRaises(ValueError):
            validate_voltage_within_range(5, 2)

    def test_required_read_never_fabricates_zero(self):
        instrument = FakeInstrument({':READ?': OSError('timeout')})
        with self.assertRaises(MeasurementReadError):
            required_float_query(instrument, ':READ?', '正式读数')

    def test_any_scpi_error_fails_configuration(self):
        instrument = FakeInstrument({
            ':SYST:ERR?': ['5077,"bad limit"', '0,"No error"'],
        })
        with self.assertRaises(InstrumentConfigurationError):
            assert_no_scpi_errors(instrument)

    def test_configuration_readback(self):
        instrument = FakeInstrument({
            ':SENS:CURR:NPLC?': '0.01',
            ':SENS:CURR:RANG:AUTO?': '0',
            ':SENS:CURR:RANG?': '1e-6',
            ':SOUR:VOLT:ILIM?': '1.05e-6',
            ':ROUT:TERM?': 'REAR',
            ':SENS:CURR:AZER?': '0',
            ':SYST:ERR?': '0,"No error"',
        })
        actual = verify_current_configuration(
            instrument,
            nplc=0.01,
            current_range=1e-6,
            current_limit=1.05e-6,
            terminal='REAR',
            autozero_mode='block_once',
        )
        self.assertEqual(actual['current_range_A'], 1e-6)

    def test_front_terminal_abbreviation_is_accepted(self):
        instrument = FakeInstrument({
            ':SENS:CURR:NPLC?': '1',
            ':SENS:CURR:RANG:AUTO?': '1',
            ':SOUR:VOLT:ILIM?': '1e-9',
            ':ROUT:TERM?': 'FRON',
            ':SENS:CURR:AZER?': '1',
            ':SYST:ERR?': '0,"No error"',
        })
        actual = verify_current_configuration(
            instrument,
            nplc=1,
            current_range='AUTO',
            current_limit=1e-9,
            terminal='FRONT',
            autozero_mode='continuous',
        )
        self.assertEqual(actual['terminal'], 'FRON')

    def test_shutdown_attempts_off_after_earlier_failure(self):
        instrument = FakeInstrument(
            {':OUTP?': '0'},
            write_failures={':ABOR', ':SOUR:VOLT 0'},
        )
        confirmed, failures = reliable_output_off(instrument)
        self.assertTrue(confirmed)
        self.assertIn(':OUTP OFF', instrument.writes)
        self.assertEqual(len(failures), 2)


class FileAndRawDataTests(unittest.TestCase):
    def test_unique_atomic_result_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            first = allocate_unique_path(folder, 'result.txt')
            with atomic_text_writer(first) as stream:
                stream.write('raw\n')
            second = allocate_unique_path(folder, 'result.txt')
            self.assertEqual(second.name, 'result_backup001.txt')
            meta = write_result_metadata(
                first,
                status='partial',
                point_count=1,
                error=TimeoutError('timeout'),
                stopped_at_local='2026-07-24 12:34:56',
            )
            self.assertEqual(meta.parent.name, 'metadata')
            self.assertEqual(meta.parent.parent, Path(folder))
            payload = json.loads(meta.read_text(encoding='utf-8'))
            self.assertEqual(payload['status'], 'partial')
            self.assertEqual(payload['point_count'], 1)
            self.assertEqual(payload['error_type'], 'TimeoutError')
            self.assertEqual(
                payload['stopped_at_local'], '2026-07-24 12:34:56'
            )

    def test_partial_backup_number_precedes_partial_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            complete = allocate_unique_path(folder, 'result.txt')
            with atomic_text_writer(complete) as stream:
                stream.write('complete')
            partial = allocate_unique_path(folder, 'result_partial.txt')
            self.assertEqual(partial.name, 'result_backup001_partial.txt')

    def test_line_filter_refuses_out_of_band_fit_and_keeps_raw(self):
        times = np.arange(100, dtype=float) / 80.0
        raw = np.sin(2 * np.pi * 3 * times)
        filtered, metadata = _fit_line_harmonics(times, raw)
        self.assertFalse(metadata['enabled'])
        np.testing.assert_array_equal(filtered, raw)

class ItBufferTests(unittest.TestCase):
    def test_block_done_status_index_is_available(self):
        self.assertEqual(_8, 8)

    @staticmethod
    def _setup_responses(nplc=0.05):
        return {
            ':SENS:CURR:NPLC?': str(nplc),
            ':SENS:CURR:RANG:AUTO?': '0',
            ':SENS:CURR:RANG?': '1e-6',
            ':SOUR:VOLT:ILIM?': '1.05e-6',
            ':ROUT:TERM?': 'REAR',
            ':SENS:CURR:AZER?': '0',
            ':SYST:ERR?': '0,"No error"',
        }

    def test_single_step_and_custom_modes_validate_only_active_fields(self):
        base = {
            'sample_nplc': 0.05,
            'bias_terminal': 'REAR',
            'b_ramp_step': 0.01,
            'b_range': 1e-6,
            'b_ilimit': 1.05e-6,
            'gate_enabled': False,
        }
        for mode_values in (
            {'b_mode': 'single', 'b_target': 0.1},
            {
                'b_mode': 'step',
                'b_start': 0.0,
                'b_end': 0.1,
                'b_test_step': 0.01,
            },
            {
                'b_mode': 'custom',
                'b_targets': [0.0, 0.1, -0.1, 0.1],
            },
        ):
            params = dict(base)
            params.update(mode_values)
            measurement = ItMeasurement(
                params, queue.Queue(), threading.Event(), threading.Event()
            )
            measurement.bias_keithley = FakeInstrument(
                self._setup_responses()
            )
            measurement.setup()

    def test_custom_sequence_parser_keeps_order_and_duplicates(self):
        self.assertEqual(
            parse_it_voltage_sequence(
                '0, 0.2；-0.1  0.2\n0', '自定义偏压'
            ),
            [0.0, 0.2, -0.1, 0.2, 0.0],
        )
        with self.assertRaisesRegex(ValueError, '第2项'):
            parse_it_voltage_sequence('0, bad, 1', '自定义栅压')
        with self.assertRaisesRegex(ValueError, '不能为空'):
            parse_it_voltage_sequence('  ', '自定义偏压')

    def test_three_by_three_voltage_modes_build_expected_targets(self):
        mode_params = {
            'single': {
                'target': 0.2,
                'expected': [0.2],
            },
            'step': {
                'start': -0.1,
                'end': 0.1,
                'test_step': 0.1,
                'expected': [-0.1, 0.0, 0.1],
            },
            'custom': {
                'targets': [0.2, -0.1, 0.2],
                'expected': [0.2, -0.1, 0.2],
            },
        }
        for gate_mode, gate_values in mode_params.items():
            for bias_mode, bias_values in mode_params.items():
                with self.subTest(gate=gate_mode, bias=bias_mode):
                    params = {
                        'g_mode': gate_mode,
                        'b_mode': bias_mode,
                    }
                    for key, value in gate_values.items():
                        if key != 'expected':
                            params[f'g_{key}'] = value
                    for key, value in bias_values.items():
                        if key != 'expected':
                            params[f'b_{key}'] = value
                    self.assertEqual(
                        build_it_voltage_targets(params, 'g'),
                        gate_values['expected'],
                    )
                    self.assertEqual(
                        build_it_voltage_targets(params, 'b'),
                        bias_values['expected'],
                    )

    def test_custom_step_plan_checks_every_transition_and_zeroing(self):
        valid = {
            'gate_enabled': True,
            'g_mode': 'custom',
            'g_targets': [0.0, 0.2, -0.1, 0.2],
            'g_ramp_step': 0.1,
            'b_mode': 'custom',
            'b_targets': [0.1, -0.1, 0.1],
            'b_ramp_step': 0.1,
        }
        validate_program_step_plan('it', valid)
        invalid = dict(valid, b_targets=[0.1, 0.25])
        with self.assertRaisesRegex(ValueError, '不能整除'):
            validate_program_step_plan('it', invalid)

    def test_it_page_has_three_modes_for_gate_and_bias(self):
        app = QApplication.instance() or QApplication([])
        widget = ItStepWidget()
        try:
            self.assertTrue(widget.rb_g_single.isChecked())
            self.assertTrue(widget.rb_b_single.isChecked())
            widget.cb_gate.setChecked(True)
            widget.rb_g_custom.click()
            widget.rb_b_custom.click()
            app.processEvents()
            self.assertFalse(widget.wg_g_custom.isHidden())
            self.assertTrue(widget.wg_g_single.isHidden())
            self.assertTrue(widget.wg_g_step.isHidden())
            self.assertFalse(widget.wg_b_custom.isHidden())
            self.assertTrue(widget.wg_b_single.isHidden())
            self.assertTrue(widget.wg_b_step.isHidden())
            self.assertIn('3 点', widget.lbl_g_custom_summary.text())
            self.assertIn('3 点', widget.lbl_b_custom_summary.text())
            self.assertIn('= 9 个It采集块', widget.lbl_combination_summary.text())
            widget.cb_gate.setChecked(False)
            self.assertIn(
                '= 3 个It采集块', widget.lbl_combination_summary.text()
            )
        finally:
            widget.close()

    def test_it_page_can_toggle_every_measurement_and_voltage_mode(self):
        app = QApplication.instance() or QApplication([])
        widget = ItStepWidget()
        try:
            widget.cb_gate.setChecked(True)
            for button, visible_widget in (
                (widget.rb_g_single, widget.wg_g_single),
                (widget.rb_g_step, widget.wg_g_step),
                (widget.rb_g_custom, widget.wg_g_custom),
            ):
                button.click()
                app.processEvents()
                self.assertFalse(visible_widget.isHidden())
            for button, visible_widget in (
                (widget.rb_b_single, widget.wg_b_single),
                (widget.rb_b_step, widget.wg_b_step),
                (widget.rb_b_custom, widget.wg_b_custom),
            ):
                button.click()
                app.processEvents()
                self.assertFalse(visible_widget.isHidden())
            widget.rb_meas_time.click()
            self.assertTrue(widget.inputs['duration'].isEnabled())
            self.assertFalse(widget.inputs['num_points'].isEnabled())
            widget.rb_meas_points.click()
            self.assertTrue(widget.inputs['num_points'].isEnabled())
            self.assertFalse(widget.inputs['duration'].isEnabled())
        finally:
            widget.close()

    def test_status_group_keeps_uneven_columns_in_equal_visible_halves(self):
        app = QApplication.instance() or QApplication([])
        font = QFont('Arial', 12)
        group, labels = create_status_group(
            [
                ('左一:', 'left_1', 0, 0), ('右一:', 'right_1', 0, 2),
                ('左二较长:', 'left_2', 1, 0), ('右二:', 'right_2', 1, 2),
                ('左三更长一些:', 'left_3', 2, 0), ('右三:', 'right_3', 2, 2),
                ('左四:', 'left_4', 3, 0),
            ],
            font,
            font,
        )
        try:
            group.resize(600, 135)
            group.show()
            app.processEvents()
            midpoint = group.width() / 2
            self.assertLess(labels['left_1'].mapTo(group, QPoint()).x(), midpoint)
            self.assertGreaterEqual(
                labels['right_1'].mapTo(group, QPoint()).x(), midpoint
            )
            self.assertTrue(labels['left_4'].isVisible())
            self.assertTrue(labels['right_3'].isVisible())
        finally:
            group.close()

    def test_custom_custom_run_uses_gate_major_cartesian_order(self):
        with tempfile.TemporaryDirectory() as folder:
            params = {
                'gate_enabled': True,
                'g_mode': 'custom',
                'g_targets': [0.0, 0.1],
                'g_ramp_step': 0.1,
                'g_step_delay': 0.0,
                'g_settle': 0.0,
                'g_post_zero_wait': 0.0,
                'b_mode': 'custom',
                'b_targets': [0.2, -0.1, 0.2],
                'b_ramp_step': 0.1,
                'b_step_delay': 0.0,
                'b_settle': 0.0,
                'b_post_wait': 0.0,
                'meas_mode': 'points',
                'num_points': 2,
                'prefix': 'ItCustom',
                'output_folder': folder,
            }
            messages = queue.Queue()
            measurement = ItMeasurement(
                params, messages, threading.Event(), threading.Event()
            )
            measurement.bias_keithley = FakeInstrument()
            measurement.gate_keithley = FakeInstrument()
            result = (
                np.array([0.0, 0.1]),
                np.array([1e-6, 2e-6]),
                np.array([1e-6, 2e-6]),
                {'aborted': False},
            )
            with (
                patch.object(measurement, 'connect'),
                patch.object(measurement, 'setup'),
                patch.object(measurement, '_ramp_voltage', return_value=True),
                patch.object(
                    measurement, '_interruptible_sleep', return_value=True
                ),
                patch.object(measurement, '_measure_it', return_value=result),
                patch.object(measurement, 'safe_zeroing'),
                patch(
                    'modules.it_step_setgate.reliable_output_off',
                    return_value=(True, []),
                ),
            ):
                measurement.run()
            completed = []
            while not messages.empty():
                message = messages.get_nowait()
                if message[0] == 'block_done':
                    completed.append(message)
            self.assertEqual(
                [(message[1], message[2]) for message in completed],
                [
                    (0.0, 0.2), (0.0, -0.1), (0.0, 0.2),
                    (0.1, 0.2), (0.1, -0.1), (0.1, 0.2),
                ],
            )
            self.assertEqual(
                [message[10]['combination_index'] for message in completed],
                list(range(1, 7)),
            )
            self.assertTrue(all(
                '_G' in Path(message[7]).name
                and '_B' in Path(message[7]).name
                for message in completed
            ))

    def test_it_save_writes_sequence_metadata(self):
        class SaveTarget:
            def post_log(self, _message):
                pass

        with tempfile.TemporaryDirectory() as folder:
            reserved = allocate_unique_path(folder, 'custom_it.txt')
            sequence_metadata = {
                'gate_mode': 'custom',
                'bias_mode': 'custom',
                'gate_sequence_index': 2,
                'bias_sequence_index': 3,
                'combination_index': 6,
                'combination_count': 6,
            }
            result = ItStepWidget.save_data(
                SaveTarget(),
                0.1,
                0.2,
                np.array([0.0, 0.1]),
                np.array([1e-6, 2e-6]),
                np.array([1e-6, 2e-6]),
                {'enabled': False},
                reserved,
                sequence_metadata,
                gate_enabled=True,
            )
            self.assertEqual(result['status'], 'complete')
            path = Path(result['paths'][0])
            metadata_path = (
                path.parent / 'metadata' / f'{path.stem}_meta.json'
            )
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            self.assertEqual(metadata['gate_mode'], 'custom')
            self.assertEqual(metadata['bias_sequence_index'], 3)
            self.assertEqual(metadata['combination_count'], 6)

    def test_fast_buffer_creation_does_not_delete_unknown_buffer(self):
        params = {
            'meas_mode': 'points',
            'num_points': 25,
            'duration': 1.0,
            'sample_nplc': 0.05,
        }
        measurement = ItMeasurement(
            params, queue.Queue(), threading.Event(), threading.Event()
        )
        measurement.bias_keithley = FakeInstrument({
            ':SYST:ERR?': '0,"No error"',
        })
        measurement._prepare_fast_buffer()
        self.assertFalse(any(
            command.startswith(':TRAC:DEL')
            for command in measurement.bias_keithley.writes
        ))
        self.assertTrue(any(
            command.startswith(':TRAC:MAKE')
            for command in measurement.bias_keithley.writes
        ))
        first_name = measurement._buffer_name
        measurement._delete_fast_buffer()
        measurement._prepare_fast_buffer()
        self.assertNotEqual(first_name, measurement._buffer_name)
        self.assertEqual(
            sum(command.startswith(':TRAC:MAKE')
                for command in measurement.bias_keithley.writes),
            2,
        )

    def test_force_stop_during_transfer_is_reported_as_truncated(self):
        measurement = ItMeasurement(
            {'transfer_chunk_points': 1000},
            queue.Queue(),
            threading.Event(),
            threading.Event(),
        )
        measurement._buffer_name = 'test_buffer'
        measurement.bias_keithley = FakeInstrument({
            ':TRAC:ACT? "test_buffer"': '2000',
        })
        measurement.force_stop_event.set()
        times, currents, truncated = measurement._read_fast_buffer()
        self.assertTrue(truncated)
        self.assertEqual(len(times), 0)
        self.assertEqual(len(currents), 0)

    def test_realtime_keeps_point_and_duration_end_conditions(self):
        class DeterministicSampler:
            def __init__(self):
                self.index = 0

            def sample(self, _label):
                timestamp = self.index * 0.0006
                self.index += 1
                return timestamp, 1e-6

        base = {
            'gate_enabled': False,
            'gate_monitor_interval': 1.0,
            'plot_interval': 2,
            'line_filter_mode': 'off',
            'line_frequency': 50.0,
            'line_search_hz': 0.5,
            'max_odd_harmonic': 5,
            'sample_nplc': 0.05,
            'autozero_mode': 'block_once',
            'num_points': 3,
            'duration': 0.001,
        }
        for mode in ('points', 'time'):
            params = dict(base, meas_mode=mode)
            measurement = ItMeasurement(
                params, queue.Queue(), threading.Event(), threading.Event()
            )
            measurement.bias_keithley = FakeInstrument({':READ?': '1e-6'})
            with (
                patch.object(measurement, '_single_autozero'),
                patch(
                    'modules.it_step_setgate.RealtimeSampler',
                    return_value=DeterministicSampler(),
                ),
            ):
                times, raw, filtered, metadata = (
                    measurement._measure_it_realtime(0.0, 0.1)
                )
            self.assertEqual(len(times), 3)
            np.testing.assert_array_equal(raw, filtered)
            self.assertEqual(metadata['acquisition_mode'], 'realtime')


class MappingMonitorTests(unittest.TestCase):
    def test_missing_and_expired_gate_readings_stop_measurement(self):
        measurement = MappingMeasurement(
            {'gate_i_limit': 1e-9},
            queue.Queue(),
            queue.Queue(),
            threading.Event(),
            threading.Event(),
        )
        with self.assertRaises(RuntimeError):
            measurement._gate_snapshot()
        measurement.latest_ig = 1e-12
        measurement.latest_ig_time = time.monotonic() - 2.0
        with self.assertRaises(RuntimeError):
            measurement._gate_snapshot()

    def test_monitor_thread_can_be_joined(self):
        stop_event = threading.Event()
        measurement = MappingMeasurement(
            {'gate_i_limit': 1e-9},
            queue.Queue(),
            queue.Queue(),
            stop_event,
            threading.Event(),
        )
        measurement.smu_g = FakeInstrument({':MEAS:CURR?': '1e-12'})
        measurement.gate_monitor = threading.Thread(
            target=measurement.gate_monitor_thread,
            daemon=False,
        )
        measurement.gate_monitor.start()
        deadline = time.monotonic() + 1.0
        while measurement.latest_ig_time is None and time.monotonic() < deadline:
            time.sleep(0.005)
        measurement._stop_gate_monitor()
        self.assertFalse(measurement.gate_monitor.is_alive())


class ArbitraryGateSamplingTests(unittest.TestCase):
    def _worker(self, gate_response='2e-10', duration=0.01):
        params = {
            'b_target': 0.1, 'b_ramp_step': 0.001,
            'b_step_delay': 0.0, 'b_settle': 0.0,
            'g_ramp_step': 0.05, 'waveform': [[0.2, duration]],
            'switch_settle': 0.0, 'cycles': 1, 'plot_interval': 2,
            'acquisition_mode': 'realtime', 'sample_nplc': 0.05,
            'gate_monitor_interval': 0.001, 'filename': 'gate.txt',
            'g_ilimit': 1e-9,
        }
        worker = GateArbMeasurement(
            params, queue.Queue(), threading.Event(), threading.Event()
        )
        worker.bias_k = FakeInstrument({':READ?': '1e-6'})
        worker.gate_k = FakeInstrument({':READ?': gate_response})
        return worker

    def _run(self, worker):
        with (
            patch.object(worker, 'connect'),
            patch.object(worker, 'setup'),
            patch.object(worker, '_ramp_voltage', return_value=True),
            patch.object(worker, '_sleep', return_value=True),
            patch.object(worker, 'safe_zeroing'),
            patch('modules.arbitrary_gate.reliable_output_off', return_value=(True, [])),
        ):
            worker.run()
        messages = []
        while not worker.update_queue.empty():
            messages.append(worker.update_queue.get_nowait())
        return messages

    def test_realtime_samples_isd_and_only_monitors_ig(self):
        messages = self._run(self._worker())
        completed = next(msg for msg in messages if msg[0] == 'block_done')
        _kind, _vb, times, gate_v, isd, _name, status, _error, metadata = completed
        self.assertEqual(status, 'complete')
        self.assertGreater(len(times), 0)
        self.assertEqual({len(times), len(gate_v), len(isd)}, {len(times)})
        self.assertTrue(all(value == 1e-6 for value in isd))
        self.assertTrue(any(msg[0] == 'gate_leakage' for msg in messages))
        self.assertEqual(metadata['acquisition_mode'], 'realtime')

    def test_gate_monitor_failure_stops_without_formal_ig_data(self):
        messages = self._run(self._worker(OSError('gate read failed')))
        self.assertFalse(any(msg[0] == 'block_done' for msg in messages))
        self.assertTrue(any(
            msg[0] == 'log' and 'gate read failed' in msg[1]
            for msg in messages
        ))
        self.assertEqual(messages[-1][0], 'finished')

    def test_stop_and_force_stop_save_aligned_partial_data(self):
        for event_name in ('stop_event', 'force_stop_event'):
            with self.subTest(event=event_name):
                worker = self._worker(duration=0.05)

                def read_current(instrument, _command, _label):
                    if instrument is worker.bias_k:
                        getattr(worker, event_name).set()
                        return 1e-6
                    return 2e-10

                with patch(
                    'core.time_acquisition.required_float_query',
                    side_effect=read_current,
                ):
                    messages = self._run(worker)
                completed = next(msg for msg in messages if msg[0] == 'block_done')
                _kind, _vb, times, gate_v, isd, _name, status, _error, _meta = completed
                self.assertEqual(status, 'partial')
                self.assertGreater(len(times), 0)
                self.assertEqual({len(times), len(gate_v), len(isd)}, {len(times)})

    def test_four_column_result_is_saved(self):
        class SaveTarget:
            def post_log(self, _message):
                pass

        with tempfile.TemporaryDirectory() as folder:
            result = ArbitraryGateWidget.save_data(
                SaveTarget(), 0.1, [0.0, 0.5], [0.2, -0.2],
                [1e-6, 2e-6], 'gate.txt', folder,
            )
            self.assertEqual(result['status'], 'complete')
            path = Path(result['paths'][0])
            header = path.read_text(encoding='utf-8').splitlines()[0]
            self.assertEqual(
                header,
                'Time(s)\tGateVoltage(V)\tBiasVoltage(V)\tBiasCurrent(A)',
            )
            self.assertEqual(np.loadtxt(path, skiprows=1).shape, (2, 4))


class UnifiedTimeSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_three_pages_share_sampling_controls_and_defaults(self):
        widgets = [ItStepWidget(), ArbitraryBiasWidget(), ArbitraryGateWidget()]
        try:
            for widget in widgets:
                self.assertTrue(widget.rb_sample_triggered.isChecked())
                self.assertEqual(widget.inputs['sample_nplc'].text(), '0.05')
                self.assertEqual(widget.inputs['plot_interval'].text(), '50')
                self.assertEqual(widget.inputs['gate_monitor_interval'].text(), '1.0')
                self.assertFalse(widget.inputs['plot_interval'].isEnabled())
                self.assertNotIn('b_nplc', widget.inputs)
                self.assertNotIn('g_nplc', widget.inputs)
            self.assertFalse(hasattr(widgets[2], 'plot_ig'))
            widgets[2].rb_sample_realtime.setChecked(True)
            self.assertTrue(widgets[2].inputs['plot_interval'].isEnabled())
            self.assertTrue(widgets[2].inputs['gate_monitor_interval'].isEnabled())
        finally:
            for widget in widgets:
                widget.close()

    def test_it_uses_shared_mode_parameters_and_mutually_exclusive_duration(self):
        widget = ItStepWidget()
        try:
            self.assertNotIn('meas_mode', widget.inputs)
            self.assertTrue(widget.rb_meas_points.isChecked())
            self.assertTrue(widget.inputs['num_points'].isEnabled())
            self.assertFalse(widget.inputs['duration'].isEnabled())
            widget.rb_meas_time.setChecked(True)
            self.assertFalse(widget.inputs['num_points'].isEnabled())
            self.assertTrue(widget.inputs['duration'].isEnabled())
            for key in (
                'g_ramp_step', 'g_ilimit', 'g_settle',
                'g_voltage_range',
                'g_post_zero_wait', 'b_ramp_step', 'b_range',
                'b_ilimit', 'b_settle', 'b_post_wait',
            ):
                self.assertIn(key, widget.inputs)
            self.assertFalse(any(
                key.endswith(('_s', '_st', '_c'))
                and key.startswith(('g_ramp', 'b_ramp', 'g_settle', 'b_settle'))
                for key in widget.inputs
            ))
            self.assertFalse(hasattr(widget, 'c_mode'))
            self.assertTrue(widget.lbl_gate_monitor_note.isHidden())
        finally:
            widget.close()

    def test_current_range_selection_updates_editable_limit_to_105_percent(self):
        window = MainWindow()
        try:
            pairs = (
                (2, 'current_range', 'i_limit'),
                (3, 'Bias_RANGE', 'Bias_I_LIMIT'),
                (4, 'bias_range', 'bias_i_limit'),
                (5, 'b_range', 'b_ilimit'),
                (6, 'b_range', 'b_ilimit'),
                (7, 'b_range', 'b_ilimit'),
            )
            for page, range_key, limit_key in pairs:
                widget = window.stack.widget(page)
                combo = widget.inputs[range_key]
                index = combo.findData(1e-3)
                self.assertGreaterEqual(index, 0)
                combo.setCurrentIndex(index)
                combo.activated.emit(index)
                self.assertAlmostEqual(
                    float(widget.inputs[limit_key].text()), 1.05e-3
                )
                widget.inputs[limit_key].setText('0.0009')
                self.assertEqual(widget.inputs[limit_key].text(), '0.0009')
        finally:
            window.close()

    def test_bias_current_limit_is_immediately_below_its_range(self):
        window = MainWindow()
        try:
            pairs = (
                (2, 'current_range', 'i_limit'),
                (3, 'Bias_RANGE', 'Bias_I_LIMIT'),
                (4, 'bias_range', 'bias_i_limit'),
                (5, 'b_range', 'b_ilimit'),
                (6, 'b_range', 'b_ilimit'),
                (7, 'b_range', 'b_ilimit'),
            )
            for page, range_key, limit_key in pairs:
                widget = window.stack.widget(page)
                range_control = widget.inputs[range_key]
                limit_control = widget.inputs[limit_key]
                layout = range_control.parentWidget().layout()
                self.assertIs(layout, limit_control.parentWidget().layout())
                range_row, range_col, _rs, _cs = layout.getItemPosition(
                    layout.indexOf(range_control)
                )
                limit_row, limit_col, _rs, _cs = layout.getItemPosition(
                    layout.indexOf(limit_control)
                )
                self.assertEqual(limit_col, range_col)
                self.assertEqual(limit_row, range_row + 1)
        finally:
            window.close()

    def test_gate_parameter_columns_are_contiguous_and_pixel_aligned(self):
        window = MainWindow()
        try:
            window.resize(1600, 1000)
            window.show()
            iv = window.stack.widget(2)
            it = window.stack.widget(5)
            arbitrary_bias = window.stack.widget(6)
            for widget in (iv, it, arbitrary_bias):
                window.stack.setCurrentWidget(widget)
                widget.cb_gate.setChecked(True)
                QApplication.processEvents()
                QApplication.processEvents()

            iv_layout = iv.inputs['gate_voltage_range'].parentWidget().layout()
            expected_iv_positions = {
                'gate_voltage_range': (2, 1),
                'gate_ilimit': (3, 1),
                'gate_nplc': (4, 1),
                'gate_ramp_step': (5, 1),
                'gate_step_delay': (2, 3),
                'gate_settle': (3, 3),
                'gate_group_wait': (4, 3),
            }
            for key, expected in expected_iv_positions.items():
                index = iv_layout.indexOf(iv.inputs[key])
                row, column, _row_span, _column_span = (
                    iv_layout.getItemPosition(index)
                )
                self.assertEqual((row, column), expected)

            for target_key, reference_key in (
                ('g_target_s', 'g_voltage_range'),
                ('b_target_s', 'b_ramp_step'),
            ):
                target = it.inputs[target_key]
                reference = it.inputs[reference_key]
                self.assertEqual(target.x(), reference.x())
                self.assertEqual(target.width(), reference.width())

            aligned_pairs = (
                (iv, 'gate_voltage_range', 'gate_step_delay'),
                (window.stack.widget(3), 'Vg_1st', 'Gate_VOLT_RANGE'),
                (window.stack.widget(4), 'vg_start', 'gate_v_range'),
                (it, 'g_voltage_range', 'g_settle'),
                (arbitrary_bias, 'g_target', 'g_voltage_range'),
                (window.stack.widget(7), 'cycles', 'g_voltage_range'),
            )
            for widget, left_key, right_key in aligned_pairs:
                left = widget.inputs[left_key]
                right = widget.inputs[right_key]
                self.assertEqual(left.width(), right.width())
        finally:
            window.close()

    def test_bundled_configs_use_new_sampling_schema(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        expected_it_ranges = {
            'default.json': 1e-6,
            '5T.json': 1e-5,
            '9T.json': 1e-6,
        }
        for filename in ('default.json', '5T.json', '9T.json'):
            payload = json.loads((config_dir / filename).read_text(encoding='utf-8'))
            modules = payload['modules']
            for module_id in ('it_step_setgate', 'arbitrary_bias', 'arbitrary_gate'):
                data = modules[module_id]
                self.assertEqual(data['acquisition_mode'], 'triggered')
                self.assertEqual(str(data['sample_nplc']), '0.05')
                self.assertEqual(str(data['gate_monitor_interval']), '1.0')
                for old_key in ('b_nplc', 'g_nplc', 'b_nplc_s', 'b_nplc_st',
                                'g_nplc_s', 'g_nplc_st', 'autozero_mode'):
                    self.assertNotIn(old_key, data)
            it_data = modules['it_step_setgate']
            self.assertEqual(
                float(it_data['b_range']), expected_it_ranges[filename]
            )
            self.assertAlmostEqual(
                float(it_data['b_ilimit']),
                expected_it_ranges[filename] * 1.05,
            )
            self.assertTrue(it_data['custom_gate_text'].strip())
            self.assertTrue(it_data['custom_bias_text'].strip())
            for key in (
                'g_ramp_step', 'g_ilimit', 'g_settle',
                'g_post_zero_wait', 'b_ramp_step', 'b_range',
                'b_ilimit', 'b_settle', 'b_post_wait',
            ):
                self.assertIn(key, it_data)
            for old_key in (
                'g_ramp_step_s', 'g_ramp_step_st', 'g_ramp_step_c',
                'b_ramp_step_s', 'b_ramp_step_st', 'b_ramp_step_c',
                'g_step_delay_s', 'b_step_delay_s', 'meas_mode',
                'line_filter_mode',
            ):
                self.assertNotIn(old_key, it_data)
            self.assertTrue(it_data['__controls__']['rb_meas_points'])
            self.assertFalse(
                it_data['__controls__']['rb_g_custom']
            )
            if filename == '9T.json':
                it_plot = payload['plotting']['modules']['it_step_setgate']
                self.assertEqual(it_plot['title'], 'Current-Time Characteristics')
                self.assertEqual(it_plot['width_mm'], 183.0)
                self.assertEqual(it_plot['it_line_filter_mode'], 'off')
            self.assertFalse(
                it_data['__controls__']['rb_b_custom']
            )

    def test_internal_segments_preserve_voltage_assignment(self):
        instrument = FakeInstrument({':SYST:ERR?': '0,"No error"'})

        def query(command):
            if command == ':TRIG:STAT?':
                return 'IDLE'
            if command.startswith(':TRAC:ACT?'):
                return '2'
            if command.startswith(':TRAC:DATA?'):
                return '1e-6,0.0,2e-6,0.01'
            if command == ':SYST:ERR?':
                return '0,"No error"'
            raise OSError(command)

        instrument.query = query
        collector = InternalSegmentCollector(
            instrument, queue.Queue(), threading.Event(), threading.Event(),
            0.05,
        )
        try:
            collector.acquire_segment(0.1, 0.001)
            collector.acquire_segment(-0.2, 0.001)
            times, voltages, currents = collector.read_all()
            self.assertEqual(len(times), 4)
            np.testing.assert_allclose(voltages, [0.1, 0.1, -0.2, -0.2])
            np.testing.assert_allclose(currents, [1e-6, 2e-6, 1e-6, 2e-6])
            self.assertGreaterEqual(times[2], times[1])
            self.assertGreater(timing_metadata(times)['sample_rate_hz'], 0)
            segments = collector.transition_metadata()
            self.assertIsNone(segments[0]['transition_from_previous_s'])
            self.assertGreaterEqual(
                segments[1]['transition_from_previous_s'], 0.0
            )
        finally:
            collector.cleanup()

    def test_internal_segment_rejects_buffer_capacity_overflow(self):
        collector = InternalSegmentCollector(
            FakeInstrument(), queue.Queue(), threading.Event(),
            threading.Event(), 0.05,
        )
        with self.assertRaisesRegex(ValueError, '单缓冲上限'):
            collector.acquire_segment(0.1, 2000.0)


class ConfigurationSchemaTests(unittest.TestCase):
    def test_all_bundled_profiles_use_current_schema(self):
        config_dir = Path(__file__).resolve().parents[1] / 'configs'
        for name in ('default.json', '5T.json', '9T.json'):
            payload = json.loads(
                (config_dir / name).read_text(encoding='utf-8')
            )
            version, modules = parse_config_modules(payload)
            self.assertEqual(version, 4)
            self.assertIn('it_step_setgate', modules)
            controls = modules['it_step_setgate']['__controls__']
            self.assertTrue(controls['rb_b_single'])
            self.assertFalse(controls['rb_b_step'])
            self.assertFalse(controls['rb_b_custom'])
            self.assertFalse(controls['rb_g_custom'])

    def test_noncurrent_schemas_are_rejected(self):
        for payload in (
            {'__schema_version__': 2, 'modules': {}},
            {'__schema_version__': 3, 'modules': {}},
            {'iv_curve': {'v_start': '0', 'v_end': '1'}},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    parse_config_modules(payload)


class SaveQueueRaceTests(unittest.TestCase):
    def test_stale_drained_signal_does_not_release_guard(self):
        class Guard:
            def __init__(self):
                self.finished = []

            def finish(self, module_id):
                self.finished.append(module_id)

        class Dummy:
            pass

        dummy = Dummy()
        dummy._save_lock = threading.Lock()
        dummy._save_futures = {object()}
        dummy._deferred_finish_module_id = 'it_step_setgate'
        dummy.run_guard = Guard()
        BaseAppWidget._complete_deferred_finish(dummy)
        self.assertEqual(dummy._deferred_finish_module_id, 'it_step_setgate')
        self.assertEqual(dummy.run_guard.finished, [])


class BreakJunctionStatusTests(unittest.TestCase):
    def _measurement(self):
        return BreakMeasurement(
            {},
            {},
            queue.Queue(),
            queue.Queue(),
            queue.Queue(),
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

    def test_internal_terminal_condition_is_complete(self):
        measurement = self._measurement()
        measurement.normal_completion_reason = 'voltage_limit_reached'
        measurement.stop_event.set()
        self.assertFalse(measurement._measurement_was_aborted())

    def test_user_stop_is_partial(self):
        measurement = self._measurement()
        measurement.user_stop_event.set()
        measurement.stop_event.set()
        self.assertTrue(measurement._measurement_was_aborted())


class StatusDisplayTests(unittest.TestCase):
    class Label:
        def __init__(self, text='-'):
            self.value = text
            self.word_wrap = False
            self.minimum_height = 0
            self.style = ''

        def setText(self, text):
            self.value = str(text)

        def text(self):
            return self.value

        def setWordWrap(self, enabled):
            self.word_wrap = bool(enabled)

        def setMinimumHeight(self, height):
            self.minimum_height = int(height)

        def setStyleSheet(self, style):
            self.style = str(style)

    class Harness:
        reset_status_display = BaseAppWidget.reset_status_display
        note_result_status = BaseAppWidget.note_result_status
        show_final_status = BaseAppWidget.show_final_status
        note_status_from_message = BaseAppWidget.note_status_from_message
        _render_persistent_safety_alarm = (
            BaseAppWidget._render_persistent_safety_alarm
        )

        def __init__(self):
            self.status_labels = {
                'voltage': StatusDisplayTests.Label('0.123'),
                'current': StatusDisplayTests.Label('4.56e-7'),
                'stage': StatusDisplayTests.Label('上次测试完成'),
            }
            self._display_result_status = 'complete'
            self._display_result_error = None
            self._safety_alarm_active = False
            self.measure_running = True
            self.force_stop_event = threading.Event()

    def test_start_clears_stale_values(self):
        display = self.Harness()
        display.reset_status_display()
        self.assertEqual(display.status_labels['voltage'].text(), '-')
        self.assertEqual(display.status_labels['current'].text(), '-')
        self.assertEqual(
            display.status_labels['stage'].text(), '仪器初始化中...'
        )

    def test_final_status_preserves_most_severe_result(self):
        display = self.Harness()
        display.note_result_status('partial', '用户停止或强制终止')
        display.note_result_status('complete')
        display.show_final_status()
        self.assertEqual(display.status_labels['stage'].text(), '用户停止')

        display.force_stop_event.set()
        display.show_final_status()
        self.assertEqual(display.status_labels['stage'].text(), '强制终止')

    def test_failure_message_cannot_finish_as_success(self):
        display = self.Harness()
        display.note_status_from_message('仪器连接失败: timeout')
        display.show_final_status()
        self.assertEqual(display.status_labels['stage'].text(), '错误中断')

    def test_safety_alarm_wraps_and_overrides_final_status(self):
        display = self.Harness()
        display._safety_alarm_active = True
        display.show_final_status('complete')
        stage = display.status_labels['stage']
        self.assertIn('输出状态未确认', stage.text())
        self.assertTrue(stage.word_wrap)
        self.assertGreaterEqual(stage.minimum_height, 42)


if __name__ == '__main__':
    unittest.main()
