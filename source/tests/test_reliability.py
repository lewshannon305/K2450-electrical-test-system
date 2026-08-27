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
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QFrame

from core.hardware_base import (
    InstrumentConfigurationError,
    MeasurementReadError,
    allocate_unique_path,
    assert_no_scpi_errors,
    atomic_text_writer,
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
from modules.it_step_setgate import ItMeasurement, ItStepWidget, _8, _fit_line_harmonics
from modules.isd_vg_setvsd import IsdVgMeasurement
from modules.iv_curve import (
    IV_Measurement,
    IVWidget,
    GateLeakageError,
    build_gate_targets,
    default_iv_gate_settings,
    merge_gate_settings_into_iv_params,
    parse_custom_gate_values,
)
from modules.mapping_scan import MappingMeasurement
from modules.break_junction import BreakMeasurement
from modules.arbitrary_bias import ArbMeasurement, ArbitraryBiasWidget
from modules.arbitrary_gate import ArbitraryGateWidget, GateArbMeasurement
from main import MainWindow, WelcomePage, parse_config_modules
from core.app_base import BaseAppWidget
from core.instrument_config import InstrumentSettings
from core.time_acquisition import InternalSegmentCollector, timing_metadata


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
                    '欢迎', '断裂结', '循环IV特性扫描', '栅压特性扫描',
                    '二维Mapping扫描', 'It特性扫描',
                    '任意偏压波形测试', '任意栅压波形测试',
                ],
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
        config_dir = Path(__file__).resolve().parents[2] / 'configs'
        for path in config_dir.glob('*.json'):
            if path.name == 'plotting_default.json':
                continue
            value = json.loads(path.read_text(encoding='utf-8'))
            self.assertIn('instruments', value, path.name)
            for module in value['modules'].values():
                self.assertFalse(forbidden & set(module), path.name)
                gate_settings = module.get('__gate_settings__', {})
                self.assertFalse(forbidden & set(gate_settings), path.name)


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

    def test_it_intergroup_point_keeps_001_second_delay(self):
        instrument = FakeInstrument({
            'SOUR:VOLT?': '0',
            'READ?': '1e-9',
        })
        worker = ItMeasurement(
            {}, queue.Queue(), threading.Event(), threading.Event()
        )
        started = time.monotonic()
        self.assertTrue(worker._ramp_voltage(
            instrument, 0.001, 0.001, 0.01, is_gate=False
        ))
        self.assertGreaterEqual(time.monotonic() - started, 0.0095)

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
            'settle_time': 0.0,
            'gate_enabled': True,
            'gate_ramp_step': 0.5,
            'gate_step_delay': 0.0,
            'gate_settle': 0.0,
            'gate_leakage_limit': 1e-9,
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
        with self.assertRaises(GateLeakageError):
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


class StepDivisibilityTests(unittest.TestCase):
    def test_exact_steps_are_accepted_and_inexact_steps_are_rejected(self):
        self.assertEqual(validate_step_divides_interval(-5, 5, 0.05), 200)
        self.assertEqual(validate_step_divides_interval(0, -0.3, 0.1), 3)
        with self.assertRaisesRegex(ValueError, '不能整除'):
            validate_step_divides_interval(0, 1, 0.3, '测试步进')

    def test_all_bundled_profiles_have_exact_step_plans(self):
        root = Path(__file__).resolve().parents[2]
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
                'g_ramp_step': float(
                    it_raw['g_ramp_step_s']
                    if controls.get('rb_g_single', True)
                    else it_raw['g_ramp_step_st']
                ),
                'b_target': float(it_raw['b_target_s']),
                'b_start': float(it_raw['b_start']),
                'b_end': float(it_raw['b_end']),
                'b_test_step': float(it_raw['b_test_step']),
                'b_ramp_step': float(
                    it_raw['b_ramp_step_s']
                    if controls.get('rb_b_single', True)
                    else it_raw['b_ramp_step_st']
                ),
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
            self.assertEqual(second.name, 'result_001.txt')
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

    def test_single_and_step_modes_validate_only_active_fields(self):
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
            {'ig_threshold': 1e-9},
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
            {'ig_threshold': 0.0},
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
    def _worker(self, gate_response='2e-9', duration=0.01):
        params = {
            'b_target': 0.1, 'b_ramp_step': 0.001,
            'b_step_delay': 0.0, 'b_settle': 0.0,
            'g_ramp_step': 0.05, 'waveform': [[0.2, duration]],
            'switch_settle': 0.0, 'cycles': 1, 'plot_interval': 2,
            'acquisition_mode': 'realtime', 'sample_nplc': 0.05,
            'gate_monitor_interval': 0.001, 'filename': 'gate.txt',
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
                    return 2e-9

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

    def test_bundled_configs_use_new_sampling_schema(self):
        config_dir = Path(__file__).resolve().parents[2] / 'configs'
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


class ConfigurationCompatibilityTests(unittest.TestCase):
    def test_all_legacy_profiles_and_v2_wrapper_parse(self):
        config_dir = Path(__file__).resolve().parents[2] / 'configs'
        for name in ('default.json', '5T.json', '9T.json'):
            payload = json.loads(
                (config_dir / name).read_text(encoding='utf-8')
            )
            version, modules = parse_config_modules(payload)
            self.assertEqual(version, 2)
            self.assertIn('it_step_setgate', modules)
            controls = modules['it_step_setgate']['__controls__']
            self.assertTrue(controls['rb_b_single'])
            self.assertFalse(controls['rb_b_step'])

        version, modules = parse_config_modules({
            '__schema_version__': 2,
            'modules': {
                'it_step_setgate': {
                    '__controls__': {'cb_gate': True},
                    '__folder__': 'results',
                    '__waveform__': [[0.1, 1.0]],
                }
            },
        })
        self.assertEqual(version, 2)
        self.assertTrue(
            modules['it_step_setgate']['__controls__']['cb_gate']
        )

    def test_future_schema_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_config_modules({'__schema_version__': 3, 'modules': {}})

    def test_unversioned_legacy_profile_still_parses(self):
        version, modules = parse_config_modules({
            'iv_curve': {'v_start': '0', 'v_end': '1'}
        })
        self.assertEqual(version, 1)
        self.assertIn('iv_curve', modules)


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
