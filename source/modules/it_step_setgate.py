import os
import json
import time
from pathlib import Path
import numpy as np
import pyvisa
from scipy.optimize import minimize

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
    QCheckBox,
    QRadioButton,
    QButtonGroup,
    QSizePolicy,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont

import pyqtgraph as pg

from core.app_base import BaseAppWidget
from core.paths import default_data_directory
from core.hardware_base import (
    allocate_unique_path,
    assert_no_scpi_errors,
    atomic_text_writer,
    clear_scpi_status,
    configure_current_autozero,
    fast_shutdown_zero_2450,
    reliable_output_off,
    release_path_reservation,
    required_float_query,
    validate_2450_idn,
    validate_current_range_limit,
    validate_distinct_addresses,
    validate_nplc,
    validate_positive_step,
    validate_program_step_plan,
    validate_source_voltage,
    validate_terminal,
    verify_current_configuration,
    write_result_metadata,
)
from core.utils import (
    NoScrollComboBox,
    _0, _1, _2, _3, _4, _5, _6, _7, _8,
    configure_pyqtgraph,
)


MAX_BUFFER_POINTS = 1_000_000


class PartialItAcquisitionError(RuntimeError):
    def __init__(self, cause, times, currents):
        super().__init__(str(cause))
        self.cause = cause
        self.times = np.asarray(times, dtype=float)
        self.currents = np.asarray(currents, dtype=float)


def _fit_line_harmonics(times, currents, nominal_freq=50.0, search_hz=0.5, max_harmonic=5):
    """Remove fitted, slowly drifting odd line harmonics without reducing the sample rate."""
    t = np.asarray(times, dtype=float)
    y = np.asarray(currents, dtype=float)
    if len(t) < 20 or len(y) != len(t) or t[-1] <= t[0]:
        return y.copy(), {
            'enabled': False,
            'reason': '数据点过少，未执行工频拟合',
            'line_frequency_hz': None,
            'raw_std_A': float(np.std(y)) if len(y) else 0.0,
            'filtered_std_A': float(np.std(y)) if len(y) else 0.0,
        }

    duration = t[-1] - t[0]
    if duration < 0.5:
        return y.copy(), {
            'enabled': False,
            'reason': '采集时长小于0.5 s，为避免过拟合未执行工频扣除',
            'line_frequency_hz': None,
            'raw_std_A': float(np.std(y)),
            'filtered_std_A': float(np.std(y)),
        }
    centered_t = t - t[0]
    phase_t = centered_t - duration / 2.0
    sample_rate = 1.0 / np.median(np.diff(t))
    requested_harmonic_orders = list(
        range(1, max(1, int(max_harmonic)) + 1, 2)
    )
    harmonic_orders = [
        order
        for order in requested_harmonic_orders
        if order * nominal_freq < sample_rate * 0.45
    ]
    if not harmonic_orders:
        return y.copy(), {
            'enabled': False,
            'reason': '采样率不足，50 Hz已超出安全拟合带宽',
            'line_frequency_hz': None,
            'raw_std_A': float(np.std(y)),
            'filtered_std_A': float(np.std(y)),
        }
    search_hz = max(0.0, float(search_hz))
    nominal_freq = float(nominal_freq)

    # A fine grid is useful for long acquisitions, while the point cap keeps
    # fitting time bounded for very long It traces.
    if search_hz == 0:
        candidates = np.array([nominal_freq])
    else:
        grid_step = max(0.002, min(0.02, 1.0 / (duration * 25.0)))
        grid_count = min(1001, max(3, int(round(2 * search_hz / grid_step)) + 1))
        candidates = np.linspace(nominal_freq - search_hz, nominal_freq + search_hz, grid_count)

    if len(t) > 20_000:
        search_start = (len(t) - 20_000) // 2
        search_stop = search_start + 20_000
        search_t = centered_t[search_start:search_stop]
        search_phase_t = phase_t[search_start:search_stop]
        search_y = y[search_start:search_stop]
    else:
        search_t = centered_t
        search_phase_t = phase_t
        search_y = y

    best = None
    for line_freq in candidates:
        columns = [np.ones_like(search_t), search_t]
        for order in harmonic_orders:
            phase = 2 * np.pi * order * line_freq * search_t
            columns.extend((np.sin(phase), np.cos(phase)))
        design = np.column_stack(columns)
        coefficients, _, _, _ = np.linalg.lstsq(design, search_y, rcond=None)
        residual = search_y - design @ coefficients
        residual_power = float(np.mean(residual * residual))
        if best is None or residual_power < best[0]:
            best = residual_power, float(line_freq)

    _, line_freq = best

    # Grid frequency is not constant over a multi-second acquisition.  A
    # fixed-frequency fit accumulates phase error and leaves a conspicuous
    # beat envelope.  Refine the midpoint frequency together with a small
    # linear frequency drift (a quadratic phase term).
    objective_scale = max(float(np.var(search_y)), np.finfo(float).tiny)
    max_drift_hz_per_s = min(0.02, max(0.002, 2.0 * search_hz / duration))

    def drift_objective(parameters):
        trial_freq, trial_drift = parameters
        base_phase = 2 * np.pi * (
            trial_freq * search_phase_t
            + 0.5 * trial_drift * search_phase_t * search_phase_t
        )
        drift_columns = [np.ones_like(search_t), search_t]
        for order in harmonic_orders:
            drift_columns.extend((
                np.sin(order * base_phase),
                np.cos(order * base_phase),
            ))
        drift_design = np.column_stack(drift_columns)
        drift_coefficients, _, _, _ = np.linalg.lstsq(
            drift_design,
            search_y,
            rcond=None,
        )
        drift_residual = search_y - drift_design @ drift_coefficients
        return float(np.mean(drift_residual * drift_residual) / objective_scale)

    fixed_objective = drift_objective((line_freq, 0.0))
    drift_result = minimize(
        drift_objective,
        np.array([line_freq, 0.0]),
        method='Powell',
        bounds=(
            (nominal_freq - search_hz, nominal_freq + search_hz),
            (-max_drift_hz_per_s, max_drift_hz_per_s),
        ),
        options={'xtol': 1e-9, 'ftol': 1e-11, 'maxiter': 80},
    )
    drift_improvement = (
        1.0 - float(drift_result.fun) / fixed_objective
        if fixed_objective > 0 and np.isfinite(drift_result.fun)
        else 0.0
    )
    drift_away_from_bound = (
        abs(float(drift_result.x[1])) < 0.95 * max_drift_hz_per_s
        if np.all(np.isfinite(drift_result.x))
        else False
    )
    drift_fit_accepted = (
        drift_result.success
        and np.all(np.isfinite(drift_result.x))
        and drift_result.fun < fixed_objective
        and drift_improvement >= 0.05
        and drift_away_from_bound
    )
    if drift_fit_accepted:
        line_freq, line_drift = (float(value) for value in drift_result.x)
    else:
        line_drift = 0.0

    columns = [np.ones_like(centered_t), centered_t]
    base_phase = 2 * np.pi * (
        line_freq * phase_t + 0.5 * line_drift * phase_t * phase_t
    )
    for order in harmonic_orders:
        phase = order * base_phase
        columns.extend((np.sin(phase), np.cos(phase)))
    design = np.column_stack(columns)
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    # Preserve the fitted DC level and linear drift. Only the periodic terms
    # are subtracted from the raw trace.
    baseline = design[:, :2] @ coefficients[:2]
    filtered = baseline + residual
    amplitudes = {}
    for index, order in enumerate(harmonic_orders):
        sin_coef = coefficients[2 + index * 2]
        cos_coef = coefficients[3 + index * 2]
        amplitudes[str(order)] = float(np.hypot(sin_coef, cos_coef))

    raw_std = float(np.std(y))
    filtered_std = float(np.std(filtered))
    return filtered, {
        'enabled': True,
        'method': 'drifting_odd_harmonic_least_squares',
        'line_frequency_hz': line_freq,
        'line_frequency_start_hz': line_freq - line_drift * duration / 2.0,
        'line_frequency_end_hz': line_freq + line_drift * duration / 2.0,
        'line_frequency_drift_hz_per_s': line_drift,
        'line_frequency_drift_fit_accepted': drift_fit_accepted,
        'line_frequency_drift_fit_improvement': drift_improvement,
        'nominal_frequency_hz': nominal_freq,
        'search_half_width_hz': search_hz,
        'max_odd_harmonic': int(max_harmonic),
        'requested_harmonic_orders': requested_harmonic_orders,
        'fitted_harmonic_orders': harmonic_orders,
        'excluded_harmonics_reason': (
            '超过有效奈奎斯特安全带宽，未强制拟合'
            if len(harmonic_orders) < len(requested_harmonic_orders)
            else None
        ),
        'harmonic_peak_amplitudes_A': amplitudes,
        'raw_std_A': raw_std,
        'filtered_std_A': filtered_std,
        'variance_removed_fraction': (
            1.0 - (filtered_std * filtered_std) / (raw_std * raw_std)
            if raw_std > 0 else 0.0
        ),
    }


def _diagnose_periodic_variance_burst(times, currents):
    """Detect a repeating noise-power burst without modifying the data."""
    t = np.asarray(times, dtype=float)
    y = np.asarray(currents, dtype=float)
    if len(t) < 200 or t[-1] - t[0] < 1.5:
        return {
            'detected': False,
            'reason': '数据长度不足以判断约0.5 s周期噪声',
            'period_s': None,
            'correlation': 0.0,
        }

    sample_rate = 1.0 / np.median(np.diff(t))
    centered = y - np.median(y)
    window = max(3, int(round(sample_rate * 0.01)))
    power_envelope = np.convolve(
        centered * centered,
        np.ones(window, dtype=float) / window,
        mode='same',
    )
    power_envelope -= np.mean(power_envelope)
    fft_size = 1 << (2 * len(power_envelope) - 1).bit_length()
    spectrum = np.fft.rfft(power_envelope, n=fft_size)
    autocorrelation = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size)
    autocorrelation = autocorrelation[:len(power_envelope)]

    low = max(1, int(round(sample_rate * 0.40)))
    high = min(len(autocorrelation), int(round(sample_rate * 0.60)) + 1)
    if high <= low or autocorrelation[0] <= 0:
        return {
            'detected': False,
            'reason': '无法计算周期噪声相关性',
            'period_s': None,
            'correlation': 0.0,
        }

    lag = low + int(np.argmax(autocorrelation[low:high]))
    correlation = float(autocorrelation[lag] / autocorrelation[0])
    period = float(lag / sample_rate)
    return {
        'detected': bool(0.45 <= period <= 0.55 and correlation >= 0.65),
        'period_s': period,
        'correlation': correlation,
        'search_range_s': [0.40, 0.60],
        'detection_threshold': 0.65,
    }


class ItMeasurement:
    def __init__(self, params, update_queue, stop_event, force_stop_event):
        self.params = params
        self.update_queue = update_queue
        self.stop_event = stop_event
        self.force_stop_event = force_stop_event
        self.bias_keithley = None
        self.gate_keithley = None
        self._buffer_created = False
        self._buffer_name = None

    def connect(self):
        try:
            validate_distinct_addresses(
                self.params['bias_address'],
                self.params['gate_address'],
                self.params['gate_enabled'],
            )
            rm = pyvisa.ResourceManager()
            self.bias_keithley = rm.open_resource(self.params['bias_address'])
            self.bias_keithley.timeout = 60000
            self.bias_keithley.chunk_size = 4 * 1024 * 1024
            idn_b = validate_2450_idn(self.bias_keithley.query('*IDN?'))
            self.update_queue.put(('log', f"偏压表连接成功: {idn_b}"))

            if self.params['gate_enabled']:
                self.gate_keithley = rm.open_resource(
                    self.params['gate_address'])
                self.gate_keithley.timeout = 10000
                idn_g = validate_2450_idn(self.gate_keithley.query('*IDN?'))
                self.update_queue.put(('log', f"栅压表连接成功: {idn_g}"))
        except Exception as exc:
            self.update_queue.put(('log', f"仪器连接失败: {repr(exc)}"))
            raise

    def setup(self):
        try:
            validate_nplc(self.params['b_nplc'], '偏压NPLC')
            validate_terminal(self.params['bias_terminal'], '偏压端子')
            validate_positive_step(self.params['b_ramp_step'], '偏压爬坡步长')
            if self.params['b_mode'] == 'single':
                validate_source_voltage(self.params['b_target'], '偏压目标')
            else:
                validate_source_voltage(self.params['b_start'], '偏压起点')
                validate_source_voltage(self.params['b_end'], '偏压终点')
                validate_positive_step(
                    self.params['b_test_step'], '偏压测试步长'
                )
            validate_current_range_limit(
                self.params['b_range'], self.params['b_ilimit'], '偏压'
            )
            if self.params['gate_enabled']:
                validate_nplc(self.params['g_nplc'], '栅压NPLC')
                validate_terminal(self.params['gate_terminal'], '栅压端子')
                validate_positive_step(self.params['g_ramp_step'], '栅压爬坡步长')
                if self.params['g_mode'] == 'single':
                    validate_source_voltage(self.params['g_target'], '栅压目标')
                else:
                    validate_source_voltage(self.params['g_start'], '栅压起点')
                    validate_source_voltage(self.params['g_end'], '栅压终点')
                    validate_positive_step(
                        self.params['g_test_step'], '栅压测试步长'
                    )
                validate_current_range_limit('AUTO', self.params['g_ilimit'], '栅极')
            validate_program_step_plan('it', self.params)
            k_b = self.bias_keithley
            k_b.write('*RST')
            clear_scpi_status(k_b)
            k_b.write(':ABORt')
            k_b.write(':SOUR:FUNC VOLT')
            k_b.write(':SENS:FUNC "CURR"')
            k_b.write('SENS:CURR:RSEN OFF')
            k_b.write(f":ROUT:TERM {self.params['bias_terminal']}")
            k_b.write(':SENS:CURR:AZER OFF')
            k_b.write(':SENS:CURR:AVER OFF')
            # Constant-bias It does not need a source readback on every point.
            # Disabling it is essential for the 2450 internal-buffer rate.
            k_b.write(':SOUR:VOLT:READ:BACK OFF')
            k_b.write(f":SENS:CURR:NPLC {self.params['b_nplc']}")
            r_val = self.params['b_range']
            if str(r_val).upper() == 'AUTO':
                k_b.write(':SENS:CURR:RANG:AUTO ON')
            else:
                k_b.write(':SENS:CURR:RANG:AUTO OFF')
                k_b.write(f":SENS:CURR:RANG {r_val}")
            k_b.write(f":SOUR:VOLT:ILIM {self.params['b_ilimit']}")
            k_b.write(':SOUR:VOLT 0')

            if self.params['gate_enabled']:
                k_g = self.gate_keithley
                k_g.write('*RST')
                clear_scpi_status(k_g)
                k_g.write(':ABORt')
                k_g.write(':SOUR:FUNC VOLT')
                k_g.write(':SENS:FUNC "CURR"')
                k_g.write('SENS:CURR:RSEN OFF')
                k_g.write(f":ROUT:TERM {self.params['gate_terminal']}")
                k_g.write(':SENS:CURR:AZER OFF')
                k_g.write(':SENS:CURR:AVER OFF')
                k_g.write(':SOUR:VOLT:READ:BACK ON')
                k_g.write(f":SENS:CURR:NPLC {self.params['g_nplc']}")
                k_g.write(':SENS:CURR:RANG:AUTO ON')
                k_g.write(f":SOUR:VOLT:ILIM {self.params['g_ilimit']}")
                k_g.write(':SOUR:VOLT 0')

            if not self._interruptible_sleep(0.05, "仪器初始化中"):
                return
            verify_current_configuration(
                k_b,
                nplc=self.params['b_nplc'],
                current_range=self.params['b_range'],
                current_limit=self.params['b_ilimit'],
                terminal=self.params['bias_terminal'],
                autozero_mode='block_once',
                label='偏压表',
            )
            if self.params['gate_enabled']:
                verify_current_configuration(
                    k_g,
                    nplc=self.params['g_nplc'],
                    current_range='AUTO',
                    current_limit=self.params['g_ilimit'],
                    terminal=self.params['gate_terminal'],
                    autozero_mode='block_once',
                    label='栅压表',
                )
        except Exception as exc:
            self.update_queue.put(('log', f"仪器初始化错误: {exc}"))
            raise

    def _interruptible_sleep(self, duration_s, stage_msg):
        if duration_s <= 0:
            return True
        self.update_queue.put(('stage', stage_msg))
        steps = int(duration_s / 0.1)
        for _ in range(steps):
            if self.stop_event.is_set() or self.force_stop_event.is_set():
                return False
            time.sleep(0.1)
        rem = duration_s - steps * 0.1
        if rem > 0:
            if self.stop_event.is_set() or self.force_stop_event.is_set():
                return False
            time.sleep(rem)
        return not (self.stop_event.is_set() or self.force_stop_event.is_set())

    def _ramp_voltage(self, inst, target_v, step_abs, step_delay_s, is_gate=False, is_zeroing=False):
        current_v = required_float_query(inst, 'SOUR:VOLT?', '源电压回读')
        if abs(target_v - current_v) < 1e-9:
            return True

        step_abs = abs(step_abs)
        if step_abs == 0:
            step_abs = 0.001

        direction = 1 if target_v > current_v else -1
        steps = int(round(abs(target_v - current_v) / step_abs))
        if steps == 0:
            steps = 1

        self.update_queue.put(('stage', f"{'栅压' if is_gate else '偏压'}爬坡中"))

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
            if is_zeroing:
                if self.force_stop_event.is_set():
                    return False
                steps = int(step_delay_s / 0.1)
                for _ in range(steps):
                    if self.force_stop_event.is_set():
                        return False
                    time.sleep(0.1)
            else:
                if not self._interruptible_sleep(step_delay_s, "电压调整中"):
                    return False

            reading = required_float_query(inst, 'READ?', '爬坡电流读数')

            if is_gate:
                self.update_queue.put(('ramp_g', v, reading))
            else:
                self.update_queue.put(('ramp_b', v, reading))

            if v == target_v:
                break
        return True

    def _single_autozero(self):
        if self.params['autozero_mode'] != 'block_once':
            return
        self.update_queue.put(('stage', '块间自动调零'))
        configure_current_autozero(self.bias_keithley, 'block_once')
        if self.params['gate_enabled'] and self.gate_keithley:
            configure_current_autozero(self.gate_keithley, 'block_once')

    @staticmethod
    def _estimate_rate(nplc):
        # Empirical 2450 timing with source readback disabled on a 50 Hz grid.
        # The small fixed term accounts for conversion and trigger overhead.
        return 1.0 / (float(nplc) / 50.0 + 0.00028)

    def _prepare_fast_buffer(self):
        if self._buffer_created:
            raise RuntimeError('上一个It高速缓冲区尚未清理')
        self._buffer_name = (
            f'it_{time.monotonic_ns() & 0xFFFFFFFFFFFF:x}'
        )
        if self.params['meas_mode'] == 'points':
            expected_points = int(self.params['num_points'])
        else:
            expected_points = int(
                np.ceil(self.params['duration'] * self._estimate_rate(self.params['b_nplc']) * 1.20)
            ) + 100

        if expected_points > MAX_BUFFER_POINTS:
            raise ValueError(
                f'当前高速采集预计需要 {expected_points} 点，超过2450单缓冲上限 '
                f'{MAX_BUFFER_POINTS} 点；请缩短时长或改用点数模式分段测量。'
            )

        self.bias_keithley.write(
            f':TRAC:MAKE "{self._buffer_name}", {max(100, expected_points)}'
        )
        self._buffer_created = True

        if self.params['meas_mode'] == 'points':
            self.bias_keithley.write(
                f':TRIG:LOAD "SimpleLoop", {int(self.params["num_points"])}, 0, "{self._buffer_name}"'
            )
            expected_duration = (
                int(self.params['num_points']) / self._estimate_rate(self.params['b_nplc'])
            )
        else:
            self.bias_keithley.write(
                f':TRIG:LOAD "DurationLoop", {self.params["duration"]}, 0, "{self._buffer_name}"'
            )
            expected_duration = self.params['duration']
        assert_no_scpi_errors(self.bias_keithley, 'It高速缓冲与触发配置')
        return expected_duration

    def _delete_fast_buffer(self):
        if not self._buffer_created or not self._buffer_name:
            return
        self.bias_keithley.write(f':TRAC:DEL "{self._buffer_name}"')
        self._buffer_created = False
        self._buffer_name = None
        assert_no_scpi_errors(self.bias_keithley, 'It高速缓冲清理')

    def _wait_for_internal_acquisition(self, expected_duration):
        self.bias_keithley.write(':INIT')
        start = time.perf_counter()
        quiet_wait = max(0.0, expected_duration * 0.85)

        # Do not poll the instrument during most of the acquisition. This keeps
        # GPIB traffic from disturbing the internal trigger timing while still
        # allowing an immediate ABOR when the user presses Stop.
        while time.perf_counter() - start < quiet_wait:
            if self.stop_event.is_set() or self.force_stop_event.is_set():
                self.bias_keithley.write(':ABOR')
                return time.perf_counter() - start, True
            time.sleep(min(0.05, max(0.0, quiet_wait - (time.perf_counter() - start))))

        while True:
            if self.stop_event.is_set() or self.force_stop_event.is_set():
                self.bias_keithley.write(':ABOR')
                return time.perf_counter() - start, True
            state = self.bias_keithley.query(':TRIG:STAT?').strip().upper()
            if state.startswith(('IDLE', 'ABORTED')):
                return time.perf_counter() - start, False
            if state.startswith('FAILED'):
                raise RuntimeError(f'2450内部触发模型失败: {state}')
            time.sleep(0.1)

    def _read_fast_buffer(self, acquisition_aborted=False):
        actual = int(float(self.bias_keithley.query(
            f':TRAC:ACT? "{self._buffer_name}"'
        )))
        if actual <= 0:
            return (
                np.array([], dtype=float),
                np.array([], dtype=float),
                False,
            )

        all_times = []
        all_currents = []
        transfer_aborted = False
        chunk_points = max(1000, int(self.params['transfer_chunk_points']))
        for start_index in range(1, actual + 1, chunk_points):
            if self.force_stop_event.is_set():
                transfer_aborted = True
                break
            if self.stop_event.is_set() and not acquisition_aborted:
                transfer_aborted = True
                break
            end_index = min(actual, start_index + chunk_points - 1)
            response = self.bias_keithley.query(
                f':TRAC:DATA? {start_index}, {end_index}, '
                f'"{self._buffer_name}", READ, REL'
            )
            values = np.fromstring(response.replace(';', ','), sep=',')
            expected_values = (end_index - start_index + 1) * 2
            if len(values) != expected_values:
                raise RuntimeError(
                    f'高速缓冲数据长度异常：期望 {expected_values}，实际 {len(values)}'
                )
            all_currents.append(values[0::2])
            all_times.append(values[1::2])

        if not all_times:
            return (
                np.array([], dtype=float),
                np.array([], dtype=float),
                transfer_aborted,
            )
        return (
            np.concatenate(all_times),
            np.concatenate(all_currents),
            transfer_aborted,
        )

    def safe_zeroing(self):
        started = time.perf_counter()
        reports = []
        try:
            if self.bias_keithley:
                self.update_queue.put(('stage', '偏压归零中...'))
                reports.append(fast_shutdown_zero_2450(
                    self.bias_keithley,
                    self.params['b_ramp_step'],
                    label='偏压表',
                    force_event=self.force_stop_event,
                ))
            if self.params['gate_enabled'] and self.gate_keithley:
                self.update_queue.put(('stage', '栅压归零中...'))
                reports.append(fast_shutdown_zero_2450(
                    self.gate_keithley,
                    self.params['g_ramp_step'],
                    label='栅压表',
                    force_event=self.force_stop_event,
                ))
            if not reports:
                return
            elapsed = time.perf_counter() - started
            if all(report['status'] == 'complete' for report in reports):
                self.update_queue.put(('log', f'归零完成，用时 {elapsed:.1f} s'))
                self.update_queue.put(('log', '输出已关闭'))
            else:
                details = '; '.join(
                    ', '.join(report['errors']) for report in reports
                    if report['errors']
                )
                self.update_queue.put((
                    'log',
                    f'安全归零失败，已执行紧急关断：{details or "状态确认失败"}',
                ))
        except Exception as exc:
            self.update_queue.put((
                'log',
                f'安全归零失败，已执行紧急关断：{exc}',
            ))

    def _measure_it(self, vg, vb):
        self._single_autozero()
        expected_duration = self._prepare_fast_buffer()
        self.update_queue.put((
            'stage',
            f'2450内部高速采集（预计 {expected_duration:.2f}s）',
        ))
        times = np.array([], dtype=float)
        raw_currents = np.array([], dtype=float)
        try:
            elapsed, acquisition_aborted = (
                self._wait_for_internal_acquisition(expected_duration)
            )
            self.update_queue.put(('stage', '读取2450内部缓冲'))
            times, raw_currents, transfer_aborted = self._read_fast_buffer(
                acquisition_aborted=acquisition_aborted
            )
        except Exception as exc:
            try:
                self.bias_keithley.write(':ABOR')
            except Exception:
                pass
            try:
                times, raw_currents, _ = self._read_fast_buffer(
                    acquisition_aborted=True
                )
            except Exception:
                times = np.array([], dtype=float)
                raw_currents = np.array([], dtype=float)
            try:
                self._delete_fast_buffer()
            except Exception:
                pass
            raise PartialItAcquisitionError(exc, times, raw_currents) from exc
        try:
            self._delete_fast_buffer()
        except Exception as exc:
            raise PartialItAcquisitionError(
                exc, times, raw_currents
            ) from exc

        aborted = acquisition_aborted or transfer_aborted

        if len(times) == 0:
            return times, raw_currents, raw_currents.copy(), {
                'enabled': False,
                'reason': '没有有效读数',
                'acquisition_elapsed_s': elapsed,
                'aborted': aborted,
            }

        if self.params['line_filter_mode'] == 'harmonic_fit':
            filtered_currents, filter_meta = _fit_line_harmonics(
                times,
                raw_currents,
                nominal_freq=self.params['line_frequency'],
                search_hz=self.params['line_search_hz'],
                max_harmonic=self.params['max_odd_harmonic'],
            )
        else:
            filtered_currents = raw_currents.copy()
            filter_meta = {
                'enabled': False,
                'reason': '用户关闭工频处理',
                'raw_std_A': float(np.std(raw_currents)),
                'filtered_std_A': float(np.std(raw_currents)),
            }
        filter_meta['acquisition_elapsed_s'] = elapsed
        filter_meta['aborted'] = aborted
        filter_meta['points'] = int(len(times))
        filter_meta['sample_rate_hz'] = (
            float((len(times) - 1) / (times[-1] - times[0]))
            if len(times) > 1 and times[-1] > times[0] else 0.0
        )
        filter_meta['acquisition_engine'] = 'keithley_2450_internal_trigger_buffer'
        filter_meta['nplc'] = float(self.params['b_nplc'])
        filter_meta['configured_current_range_A'] = self.params['b_range']
        filter_meta['source_readback'] = False
        filter_meta['autozero_mode'] = self.params['autozero_mode']
        filter_meta['periodic_variance_burst'] = _diagnose_periodic_variance_burst(
            times,
            raw_currents,
        )
        filter_meta['periodic_variance_burst']['scope'] = (
            '限定搜索0.40–0.60 s周期的Raw噪声功率诊断，不是通用噪声检测'
        )
        if filter_meta['periodic_variance_burst']['detected']:
            burst = filter_meta['periodic_variance_burst']
            self.update_queue.put((
                'log',
                f"检测到 {burst['period_s']:.4f}s 周期噪声突发"
                f"（相关性 {burst['correlation']:.2f}）。"
                "建议将偏压NPLC设为0.05；Raw数据未被插值或删除。",
            ))

        batch_size = max(1000, int(self.params['transfer_chunk_points']))
        for index in range(0, len(times), batch_size):
            end = min(len(times), index + batch_size)
            self.update_queue.put((
                'data_batch',
                vg,
                vb,
                times[index:end].tolist(),
                raw_currents[index:end].tolist(),
                filtered_currents[index:end].tolist(),
            ))
        return times, raw_currents, filtered_currents, filter_meta

    def run(self):
        try:
            self.connect()
            self.setup()

            vg_list = [0.0]
            if self.params['gate_enabled']:
                if self.params['g_mode'] == 'single':
                    vg_list = [self.params['g_target']]
                else:
                    gs, ge, gstep = self.params['g_start'], self.params['g_end'], self.params['g_test_step']
                    if gstep > 0:
                        num = int(round(abs(ge - gs) / gstep)) + 1
                        vg_list = np.linspace(gs, ge, num).tolist()

            vb_list = []
            if self.params['b_mode'] == 'single':
                vb_list = [self.params['b_target']]
            else:
                bs, be, bstep = self.params['b_start'], self.params['b_end'], self.params['b_test_step']
                if bstep > 0:
                    num = int(round(abs(be - bs) / bstep)) + 1
                    vb_list = np.linspace(bs, be, num).tolist()

            if self.params['gate_enabled']:
                self.gate_keithley.write(':OUTP ON')
            self.bias_keithley.write(':OUTP ON')
            if not self._interruptible_sleep(0.5, '全局输出等待缓冲 (0.5s)'):
                return

            for i_g, vg in enumerate(vg_list):
                if self.stop_event.is_set() or self.force_stop_event.is_set():
                    break

                if self.params['gate_enabled']:
                    if not self._ramp_voltage(
                        self.gate_keithley,
                        vg,
                        self.params['g_ramp_step'],
                        self.params['g_step_delay'],
                        is_gate=True,
                    ):
                        break
                    if not self._interruptible_sleep(
                        self.params['g_settle'],
                        f"栅压到位等待 ({self.params['g_settle']}s)",
                    ):
                        break

                for vb in vb_list:
                    if self.stop_event.is_set() or self.force_stop_event.is_set():
                        break

                    self.update_queue.put(('clear_plot', None))

                    if not self._ramp_voltage(
                        self.bias_keithley,
                        vb,
                        self.params['b_ramp_step'],
                        self.params['b_step_delay'],
                        is_gate=False,
                    ):
                        break
                    if not self._interruptible_sleep(
                        self.params['b_settle'],
                        f"偏压到位等待 ({self.params['b_settle']}s)",
                    ):
                        break

                    duration_str = f"{int(self.params['num_points'])}points" if self.params['meas_mode'] == 'points' else f"{self.params['duration']}s"
                    fname = (
                        f"{self.params['prefix']}_"
                        f"{duration_str}"
                        f"_Vg={vg:.3f}V_Vb={vb:.3f}V.txt"
                    )
                    reserved_path = allocate_unique_path(
                        self.params['output_folder'], fname
                    )
                    try:
                        times, raw_currents, filtered_currents, filter_meta = self._measure_it(vg, vb)
                        result_status = (
                            'partial' if filter_meta.get('aborted') else 'complete'
                        )
                        result_error = (
                            '用户停止或强制终止'
                            if result_status == 'partial'
                            else None
                        )
                    except PartialItAcquisitionError as exc:
                        times = exc.times
                        raw_currents = exc.currents
                        filtered_currents = raw_currents.copy()
                        filter_meta = {
                            'enabled': False,
                            'reason': '采集异常，未执行滤波',
                        }
                        result_status = 'partial'
                        result_error = exc.cause
                    except Exception:
                        release_path_reservation(reserved_path)
                        raise
                    self.update_queue.put(
                        (
                            'block_done',
                            vg,
                            vb,
                            times,
                            raw_currents,
                            filtered_currents,
                            filter_meta,
                            reserved_path,
                            result_status,
                            result_error,
                        )
                    )
                    if result_status != 'complete' and result_error != '用户停止或强制终止':
                        raise RuntimeError(result_error)

                    if not self._interruptible_sleep(
                        self.params['b_post_wait'],
                        f"测量后保持 ({self.params['b_post_wait']}s)",
                    ):
                        break

                if self.stop_event.is_set() or self.force_stop_event.is_set():
                    break
                if not self._ramp_voltage(
                    self.bias_keithley,
                    0.0,
                    self.params['b_ramp_step'],
                    self.params['b_step_delay'],
                    is_gate=False,
                ):
                    break

                if i_g < len(vg_list) - 1:
                    if not self._interruptible_sleep(
                        self.params['g_post_zero_wait'],
                        f"偏压归零后等待 ({self.params['g_post_zero_wait']}s)",
                    ):
                        break

        except Exception as exc:
            self.update_queue.put(('log', f"测量中断或出错: {exc}"))
        finally:
            cleanup_errors = []
            try:
                self.safe_zeroing()
            except Exception as exc:
                cleanup_errors.append(f'安全归零失败: {exc}')
            if self.bias_keithley is not None and self._buffer_created:
                try:
                    self._delete_fast_buffer()
                except Exception as exc:
                    cleanup_errors.append(f'高速缓冲区清理失败: {exc}')
            bias_confirmed, bias_failures = reliable_output_off(
                self.bias_keithley, '偏压表'
            )
            cleanup_errors.extend(bias_failures)
            gate_confirmed = True
            if self.gate_keithley:
                gate_confirmed, gate_failures = reliable_output_off(
                    self.gate_keithley, '栅压表'
                )
                cleanup_errors.extend(gate_failures)
            if not bias_confirmed or not gate_confirmed:
                self.update_queue.put((
                    'critical',
                    '无法确认源表输出已经关闭，请立即从仪器面板确认。'
                    + (' ' + ' | '.join(cleanup_errors) if cleanup_errors else ''),
                ))
            if self.bias_keithley:
                try:
                    self.bias_keithley.close()
                except Exception:
                    pass
            if self.gate_keithley:
                try:
                    self.gate_keithley.close()
                except Exception:
                    pass

            self.update_queue.put(('finished', None))


class ItStepWidget(BaseAppWidget):
    def __init__(self, run_guard=None, parent=None):
        configure_pyqtgraph(use_opengl=False)
        super().__init__(run_guard=run_guard, parent=parent)

        self.module_id = 'it_step_setgate'
        self.module_name = 'It特性扫描'

        self.ui_font = QFont('Arial', 12)
        self.ui_font.setWeight(QFont.Weight.Normal)
        self.bold_font = QFont('Arial', 12)
        self.bold_font.setWeight(QFont.Weight.Bold)
        self.setFont(self.ui_font)

        self.capacity = 1000000
        self.data_count = 0
        self.time_data = np.zeros(self.capacity)
        self.raw_curr_data = np.zeros(self.capacity)
        self.filtered_curr_data = np.zeros(self.capacity)
        self.points_changed = False
        self.current_folder = ''
        self.active_line_filter_mode = 'off'

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=3)

        self.graph_widget = pg.GraphicsLayoutWidget()
        left_layout.addWidget(self.graph_widget)

        self.plot_it = self.graph_widget.addPlot(title='Current vs Time')
        label_style = {'color': '#000', 'font-size': '12pt'}
        self.plot_it.setLabel('left', text='Current', units='A', **label_style)
        self.plot_it.setLabel('bottom', text='Time', units='s', **label_style)
        self.plot_it.getAxis('left').setTickFont(self.ui_font)
        self.plot_it.getAxis('bottom').setTickFont(self.ui_font)
        self.plot_it.showGrid(x=True, y=True, alpha=0.3)
        self.plot_it.setDownsampling(auto=True, mode='peak')
        self.plot_it.setClipToView(True)
        self.plot_it.addLegend()
        self.curve_raw = self.plot_it.plot(
            pen=pg.mkPen('b', width=1.5),
            name='Raw',
        )
        self.curve_it = self.plot_it.plot(
            pen=pg.mkPen('b', width=1.5),
            name='辅助工频扣除',
        )
        self.curve_it.setVisible(False)

        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, stretch=2)

        status_group = QGroupBox('实时状态显示')
        status_group.setFont(self.bold_font)
        status_group.setFixedHeight(135)
        status_layout = QGridLayout(status_group)
        status_layout.setColumnStretch(1, 1)
        status_layout.setColumnStretch(3, 1)
        status_layout.setHorizontalSpacing(10)

        self.status_labels = {}
        status_items = [
            ('偏压 Vsd (V):', 'bias_v', 0, 0), ('用时 (s):', 'time', 0, 2),
            ('栅压 Vg (V):', 'gate_v', 1, 0), ('已采点数:', 'count', 1, 2),
            ('偏置电流 Isd (A):', 'bias_i', 2, 0), ('采样率 (Hz):', 'rate', 2, 2),
            ('栅电流 Ig (A):', 'gate_i', 3, 0), ('系统状态:', 'stage', 3, 2),
        ]
        for text, key, row, col in status_items:
            lbl = QLabel(text)
            lbl.setFont(self.ui_font)
            lbl.setStyleSheet('font-weight: normal;')
            status_layout.addWidget(
                lbl, row, col, alignment=Qt.AlignmentFlag.AlignLeft)

            val = QLabel('-')
            val.setFont(self.bold_font)
            val.setStyleSheet('color: #0055A4;')
            status_layout.addWidget(
                val, row, col + 1, alignment=Qt.AlignmentFlag.AlignLeft)
            self.status_labels[key] = val

        right_layout.addWidget(status_group)

        param_group = QGroupBox('测量参数')
        param_group.setFont(self.bold_font)
        param_layout = QVBoxLayout(param_group)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_content = QWidget()
        scroll_content.setFont(self.ui_font)
        box_vbox = QVBoxLayout(scroll_content)
        box_vbox.setContentsMargins(0, 0, 0, 0)
        self.inputs = {}

        addr_box = QGroupBox('仪器地址')
        addr_box.setFont(self.bold_font)
        addr_grid = QGridLayout(addr_box)
        addr_grid.setColumnStretch(1, 1)
        addr_grid.setColumnStretch(3, 1)
        addr_grid.setHorizontalSpacing(10)

        lbl_baddr = QLabel('Bias 表地址:')
        lbl_baddr.setFont(self.ui_font)
        lbl_baddr.setStyleSheet('font-weight: normal;')
        addr_grid.addWidget(lbl_baddr, 0, 0)
        self.inputs['bias_address'] = NoScrollComboBox()
        self.inputs['bias_address'].setEditable(True)
        self.inputs['bias_address'].addItem('GPIB0::1::INSTR')
        self.inputs['bias_address'].setFont(self.ui_font)
        self.inputs['bias_address'].setStyleSheet('font-weight: normal;')
        addr_grid.addWidget(self.inputs['bias_address'], 0, 1)

        lbl_gaddr = QLabel('Gate 表地址:')
        lbl_gaddr.setFont(self.ui_font)
        lbl_gaddr.setStyleSheet('font-weight: normal;')
        addr_grid.addWidget(lbl_gaddr, 0, 2)
        self.inputs['gate_address'] = NoScrollComboBox()
        self.inputs['gate_address'].setEditable(True)
        self.inputs['gate_address'].addItem('GPIB0::2::INSTR')
        self.inputs['gate_address'].setFont(self.ui_font)
        self.inputs['gate_address'].setStyleSheet('font-weight: normal;')
        addr_grid.addWidget(self.inputs['gate_address'], 0, 3)

        lbl_bterm = QLabel('Bias 表端口:')
        lbl_bterm.setFont(self.ui_font)
        lbl_bterm.setStyleSheet('font-weight: normal;')
        addr_grid.addWidget(lbl_bterm, 1, 0)
        self.inputs['bias_terminal'] = NoScrollComboBox()
        self.inputs['bias_terminal'].addItems(['REAR', 'FRONT'])
        self.inputs['bias_terminal'].setFont(self.ui_font)
        self.inputs['bias_terminal'].setStyleSheet('font-weight: normal;')
        addr_grid.addWidget(self.inputs['bias_terminal'], 1, 1)

        lbl_gterm = QLabel('Gate 表端口:')
        lbl_gterm.setFont(self.ui_font)
        lbl_gterm.setStyleSheet('font-weight: normal;')
        addr_grid.addWidget(lbl_gterm, 1, 2)
        self.inputs['gate_terminal'] = NoScrollComboBox()
        self.inputs['gate_terminal'].addItems(['REAR', 'FRONT'])
        self.inputs['gate_terminal'].setFont(self.ui_font)
        self.inputs['gate_terminal'].setStyleSheet('font-weight: normal;')
        addr_grid.addWidget(self.inputs['gate_terminal'], 1, 3)

        btn_scan = QPushButton('扫描设备')
        btn_scan.setFont(self.bold_font)
        btn_scan.setFixedSize(100, 30)
        btn_scan.clicked.connect(self.scan_instruments)
        btn_scan.setStyleSheet('font-weight: normal;')
        addr_grid.addWidget(btn_scan, 2, 0, 1, 4,
                            alignment=Qt.AlignmentFlag.AlignCenter)
        box_vbox.addWidget(addr_box)

        meas_box = QGroupBox('测量设置')
        meas_box.setFont(self.bold_font)
        meas_grid = QGridLayout(meas_box)

        lbl_mode = QLabel('测量模式:')
        lbl_mode.setFont(self.ui_font)
        lbl_mode.setStyleSheet('font-weight: normal;')
        meas_grid.addWidget(lbl_mode, 0, 0)
        self.c_mode = NoScrollComboBox()
        self.c_mode.addItems(['points', 'time'])
        self.c_mode.setFont(self.ui_font)
        self.c_mode.setStyleSheet('font-weight: normal;')
        self.inputs['meas_mode'] = self.c_mode
        meas_grid.addWidget(self.c_mode, 0, 1, 1, 3)

        lbl_pts = QLabel('总点数 (模式=points):')
        lbl_pts.setFont(self.ui_font)
        lbl_pts.setStyleSheet('font-weight: normal;')
        meas_grid.addWidget(lbl_pts, 1, 0)
        self.inputs['num_points'] = QLineEdit('1000')
        self.inputs['num_points'].setFont(self.ui_font)
        self.inputs['num_points'].setStyleSheet('font-weight: normal;')
        meas_grid.addWidget(self.inputs['num_points'], 1, 1)

        lbl_dur = QLabel('时长 (s) (模式=time):')
        lbl_dur.setFont(self.ui_font)
        lbl_dur.setStyleSheet('font-weight: normal;')
        meas_grid.addWidget(lbl_dur, 1, 2)
        self.inputs['duration'] = QLineEdit('10.0')
        self.inputs['duration'].setFont(self.ui_font)
        self.inputs['duration'].setStyleSheet('font-weight: normal;')
        meas_grid.addWidget(self.inputs['duration'], 1, 3)

        self.cb_plot = QCheckBox('采集完成后绘图（采集中不访问仪表，保证连续高速采样）')
        self.cb_plot.setFont(self.ui_font)
        self.cb_plot.setStyleSheet('font-weight: normal;')
        self.cb_plot.setChecked(True)
        meas_grid.addWidget(self.cb_plot, 2, 0, 1, 4)

        lbl_int = QLabel('缓冲读取/绘图批大小（点）:')
        lbl_int.setFont(self.ui_font)
        lbl_int.setStyleSheet('font-weight: normal;')
        meas_grid.addWidget(lbl_int, 3, 0)
        self.inputs['plot_interval'] = QLineEdit('1000')
        self.inputs['plot_interval'].setFont(self.ui_font)
        self.inputs['plot_interval'].setStyleSheet('font-weight: normal;')
        meas_grid.addWidget(self.inputs['plot_interval'], 3, 1, 1, 3)

        lbl_az = QLabel('自动调零:')
        lbl_az.setFont(self.ui_font)
        lbl_az.setStyleSheet('font-weight: normal;')
        meas_grid.addWidget(lbl_az, 4, 0)
        self.inputs['autozero_mode'] = NoScrollComboBox()
        self.inputs['autozero_mode'].addItems(['block_once', 'off'])
        self.inputs['autozero_mode'].setFont(self.ui_font)
        self.inputs['autozero_mode'].setStyleSheet('font-weight: normal;')
        meas_grid.addWidget(self.inputs['autozero_mode'], 4, 1)

        lbl_filter = QLabel('工频处理:')
        lbl_filter.setFont(self.ui_font)
        lbl_filter.setStyleSheet('font-weight: normal;')
        meas_grid.addWidget(lbl_filter, 4, 2)
        self.inputs['line_filter_mode'] = NoScrollComboBox()
        self.inputs['line_filter_mode'].addItems(['off', 'harmonic_fit'])
        self.inputs['line_filter_mode'].setFont(self.ui_font)
        self.inputs['line_filter_mode'].setStyleSheet('font-weight: normal;')
        meas_grid.addWidget(self.inputs['line_filter_mode'], 4, 3)

        lbl_line = QLabel('工频中心 (Hz):')
        lbl_line.setFont(self.ui_font)
        lbl_line.setStyleSheet('font-weight: normal;')
        meas_grid.addWidget(lbl_line, 5, 0)
        self.inputs['line_frequency'] = QLineEdit('50.0')
        self.inputs['line_frequency'].setFont(self.ui_font)
        self.inputs['line_frequency'].setStyleSheet('font-weight: normal;')
        meas_grid.addWidget(self.inputs['line_frequency'], 5, 1)

        lbl_search = QLabel('频率搜索 ±Hz:')
        lbl_search.setFont(self.ui_font)
        lbl_search.setStyleSheet('font-weight: normal;')
        meas_grid.addWidget(lbl_search, 5, 2)
        self.inputs['line_search_hz'] = QLineEdit('0.5')
        self.inputs['line_search_hz'].setFont(self.ui_font)
        self.inputs['line_search_hz'].setStyleSheet('font-weight: normal;')
        meas_grid.addWidget(self.inputs['line_search_hz'], 5, 3)

        lbl_harmonic = QLabel('最高奇次谐波:')
        lbl_harmonic.setFont(self.ui_font)
        lbl_harmonic.setStyleSheet('font-weight: normal;')
        meas_grid.addWidget(lbl_harmonic, 6, 0)
        self.inputs['max_odd_harmonic'] = QLineEdit('5')
        self.inputs['max_odd_harmonic'].setFont(self.ui_font)
        self.inputs['max_odd_harmonic'].setStyleSheet('font-weight: normal;')
        meas_grid.addWidget(self.inputs['max_odd_harmonic'], 6, 1)

        lbl_hint = QLabel('原始电流和工频处理后电流会同时保存')
        lbl_hint.setFont(self.ui_font)
        lbl_hint.setStyleSheet('font-weight: normal; color: #555555;')
        meas_grid.addWidget(lbl_hint, 6, 2, 1, 2)

        box_vbox.addWidget(meas_box)

        self.cb_gate = QCheckBox('启用栅表 (勾选展示栅压参数)')
        self.cb_gate.setFont(self.ui_font)
        self.cb_gate.setStyleSheet('font-weight: normal;')
        self.cb_gate.stateChanged.connect(self.toggle_gate)
        box_vbox.addWidget(self.cb_gate)

        self.gate_box = QGroupBox('栅压参数')
        self.gate_box.setFont(self.bold_font)
        gate_vbox = QVBoxLayout(self.gate_box)

        g_radio_hbox = QHBoxLayout()
        self.rb_g_single = QRadioButton('单栅压模式')
        self.rb_g_single.setChecked(True)
        self.rb_g_single.setFont(self.ui_font)
        self.rb_g_single.setStyleSheet('font-weight: normal;')
        self.rb_g_step = QRadioButton('步进栅压模式')
        self.rb_g_step.setFont(self.ui_font)
        self.rb_g_step.setStyleSheet('font-weight: normal;')
        self.bg_g_mode = QButtonGroup()
        self.bg_g_mode.addButton(self.rb_g_single)
        self.bg_g_mode.addButton(self.rb_g_step)
        g_radio_hbox.addWidget(self.rb_g_single)
        g_radio_hbox.addWidget(self.rb_g_step)
        g_radio_hbox.addStretch()
        gate_vbox.addLayout(g_radio_hbox)

        self.wg_g_single = QWidget()
        gg_s = QGridLayout(self.wg_g_single)
        gg_s.setContentsMargins(0, 0, 0, 0)

        def add_param(grid, row, col, label, key, default):
            lbl = QLabel(label)
            lbl.setFont(self.ui_font)
            lbl.setStyleSheet('font-weight: normal;')
            ent = QLineEdit(default)
            ent.setFont(self.ui_font)
            ent.setStyleSheet('font-weight: normal;')
            grid.addWidget(lbl, row, col)
            grid.addWidget(ent, row, col + 1)
            self.inputs[key] = ent

        add_param(gg_s, 0, 0, '目标栅压 (V):', 'g_target_s', '0.5')
        add_param(gg_s, 0, 2, '栅压 NPLC:', 'g_nplc_s', '1.0')
        add_param(gg_s, 1, 0, '爬坡/归零步长 (V) (正):', 'g_ramp_step_s', '0.1')
        add_param(gg_s, 1, 2, '栅压单步延时 (s):', 'g_step_delay_s', '0.5')
        add_param(gg_s, 2, 0, '栅电流限值 (A):', 'g_ilimit_s', '1e-9')
        add_param(gg_s, 2, 2, '栅压到位等待 (s):', 'g_settle_s', '20.0')
        gate_vbox.addWidget(self.wg_g_single)

        self.wg_g_step = QWidget()
        gg_st = QGridLayout(self.wg_g_step)
        gg_st.setContentsMargins(0, 0, 0, 0)
        add_param(gg_st, 0, 0, '起始栅压 (V):', 'g_start', '0.0')
        add_param(gg_st, 0, 2, '栅压 NPLC:', 'g_nplc_st', '1.0')
        add_param(gg_st, 1, 0, '终止栅压 (V):', 'g_end', '1.0')
        add_param(gg_st, 1, 2, '栅压单步延时 (s):', 'g_step_delay_st', '0.5')
        add_param(gg_st, 2, 0, '栅压测试步长 (V) (正):', 'g_test_step', '0.2')
        add_param(gg_st, 2, 2, '栅压到位等待 (s):', 'g_settle_st', '20.0')
        add_param(gg_st, 3, 0, '爬坡/归零步长 (V) (正):', 'g_ramp_step_st', '0.1')
        add_param(gg_st, 3, 2, '偏压归零后等待 (s):', 'g_post_zero_wait', '2.0')
        add_param(gg_st, 4, 0, '栅电流限值 (A):', 'g_ilimit_st', '1e-9')
        gate_vbox.addWidget(self.wg_g_step)

        self.wg_g_step.setVisible(False)
        self.bg_g_mode.buttonClicked.connect(self.toggle_g_mode)
        box_vbox.addWidget(self.gate_box)
        self.gate_box.setVisible(False)

        bias_box = QGroupBox('偏压参数')
        bias_box.setFont(self.bold_font)
        bias_vbox = QVBoxLayout(bias_box)

        b_radio_hbox = QHBoxLayout()
        self.rb_b_single = QRadioButton('单偏压模式')
        self.rb_b_single.setChecked(True)
        self.rb_b_single.setFont(self.ui_font)
        self.rb_b_single.setStyleSheet('font-weight: normal;')
        self.rb_b_step = QRadioButton('步进偏压模式')
        self.rb_b_step.setFont(self.ui_font)
        self.rb_b_step.setStyleSheet('font-weight: normal;')
        self.bg_b_mode = QButtonGroup()
        self.bg_b_mode.addButton(self.rb_b_single)
        self.bg_b_mode.addButton(self.rb_b_step)
        b_radio_hbox.addWidget(self.rb_b_single)
        b_radio_hbox.addWidget(self.rb_b_step)
        b_radio_hbox.addStretch()
        bias_vbox.addLayout(b_radio_hbox)

        self.wg_b_single = QWidget()
        gb_s = QGridLayout(self.wg_b_single)
        gb_s.setContentsMargins(0, 0, 0, 0)
        add_param(gb_s, 0, 0, '目标偏压 (V):', 'b_target_s', '0.1')
        add_param(gb_s, 0, 2, '偏压 NPLC (高速低噪推荐0.05):', 'b_nplc_s', '0.05')
        add_param(gb_s, 1, 0, '爬坡/归零步长 (V) (正):', 'b_ramp_step_s', '0.001')
        add_param(gb_s, 1, 2, '偏压单步延时 (s):', 'b_step_delay_s', '0.01')
        add_param(gb_s, 2, 0, '偏压固定电流量程 (A):', 'b_range_s', '1e-6')
        add_param(gb_s, 2, 2, '偏压到位等待 (s):', 'b_settle_s', '2.0')
        add_param(gb_s, 3, 0, '偏压电流限制 (A):', 'b_ilimit_s', '1.05e-6')
        add_param(gb_s, 3, 2, '测量后保持时间 (s):', 'b_post_wait_s', '5.0')
        bias_vbox.addWidget(self.wg_b_single)

        self.wg_b_step = QWidget()
        gb_st = QGridLayout(self.wg_b_step)
        gb_st.setContentsMargins(0, 0, 0, 0)
        add_param(gb_st, 0, 0, '起始偏压 (V):', 'b_start', '0.1')
        add_param(gb_st, 0, 2, '偏压电流限制 (A):', 'b_ilimit_st', '1.05e-6')
        add_param(gb_st, 1, 0, '终止偏压 (V):', 'b_end', '0.3')
        add_param(gb_st, 1, 2, '偏压 NPLC (高速低噪推荐0.05):', 'b_nplc_st', '0.05')
        add_param(gb_st, 2, 0, '偏压测试步长 (V) (正):', 'b_test_step', '0.05')
        add_param(gb_st, 2, 2, '偏压单步延时 (s):', 'b_step_delay_st', '0.01')
        add_param(gb_st, 3, 0, '爬坡/归零步长 (V) (正):', 'b_ramp_step_st', '0.001')
        add_param(gb_st, 3, 2, '偏压到位等待 (s):', 'b_settle_st', '2.0')
        add_param(gb_st, 4, 0, '偏压固定电流量程 (A):', 'b_range_st', '1e-6')
        add_param(gb_st, 4, 2, '测量后保持时间 (s):', 'b_post_wait_st', '5.0')
        bias_vbox.addWidget(self.wg_b_step)

        self.wg_b_step.setVisible(False)
        self.bg_b_mode.buttonClicked.connect(self.toggle_b_mode)
        box_vbox.addWidget(bias_box)

        path_box = QGroupBox('文件保存路径')
        path_box.setFont(self.bold_font)
        path_grid = QGridLayout(path_box)

        lbl_pf = QLabel('文件名前缀 (后缀自动追加 _时长/点数_Vg=X_Vb=X.txt):')
        lbl_pf.setFont(self.ui_font)
        lbl_pf.setStyleSheet('font-weight: normal;')
        path_grid.addWidget(lbl_pf, 0, 0, 1, 2)

        self.inputs['filename_prefix'] = QLineEdit('It1')
        self.inputs['filename_prefix'].setFont(self.ui_font)
        self.inputs['filename_prefix'].setStyleSheet('font-weight: normal;')
        path_grid.addWidget(self.inputs['filename_prefix'], 1, 0, 1, 2)

        lbl_fd = QLabel('保存文件夹:')
        lbl_fd.setFont(self.ui_font)
        lbl_fd.setStyleSheet('font-weight: normal;')
        path_grid.addWidget(lbl_fd, 2, 0, 1, 2)

        fhbox = QHBoxLayout()
        fhbox.setContentsMargins(0, 0, 0, 0)
        self.ent_folder = QLineEdit(default_data_directory("It_Step_SetGate"))
        self.ent_folder.setFont(self.ui_font)
        self.ent_folder.setStyleSheet('font-weight: normal;')
        fhbox.addWidget(self.ent_folder)
        btn_br = QPushButton('浏览')
        btn_br.setFont(self.ui_font)
        btn_br.setStyleSheet('font-weight: normal;')
        btn_br.clicked.connect(self.browse_folder)
        fhbox.addWidget(btn_br)
        path_grid.addLayout(fhbox, 3, 0, 1, 2)
        box_vbox.addWidget(path_box)
        box_vbox.addStretch()

        scroll.setWidget(scroll_content)
        param_layout.addWidget(scroll)
        right_layout.addWidget(param_group, stretch=1)

        log_group = QGroupBox('日志信息')
        log_group.setFont(self.bold_font)
        log_group.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        log_vbox = QVBoxLayout(log_group)
        log_vbox.setContentsMargins(5, 5, 5, 5)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(self.ui_font)
        self.log_text.setStyleSheet(
            'background-color: #FFF0F0; color: #333333;')
        self.log_text.setFont(self.ui_font)  # standardized

        self.log_text.setFixedHeight(60)
        log_vbox.addWidget(self.log_text)
        btn_clr = QPushButton('清除信息')
        btn_clr.setFont(self.bold_font)
        btn_clr.setFixedWidth(100)
        btn_clr.setFixedHeight(30)
        btn_clr.clicked.connect(self.clear_log)
        log_vbox.addWidget(btn_clr, alignment=Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(log_group, stretch=0)

        btn_area = QWidget()
        btn_hbox = QHBoxLayout(btn_area)
        btn_hbox.setContentsMargins(0, 10, 0, 10)
        self.btn_start = QPushButton('开始')
        self.btn_start.setFixedSize(100, 30)
        self.btn_start.setFont(self.bold_font)
        self.btn_start.clicked.connect(self.start_measurement)
        self.btn_stop = QPushButton('停止')
        self.btn_stop.setFixedSize(100, 30)
        self.btn_stop.setFont(self.bold_font)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_measurement)
        self.btn_force = QPushButton('强制终止')
        self.btn_force.setFixedSize(100, 30)
        self.btn_force.setFont(self.bold_font)
        self.btn_force.setStyleSheet('color: #AA0000; font-weight: bold;')
        self.btn_force.clicked.connect(self.force_stop_measurement)

        btn_hbox.addWidget(self.btn_start)
        btn_hbox.addStretch()
        btn_hbox.addWidget(self.btn_stop)
        btn_hbox.addStretch()
        btn_hbox.addWidget(self.btn_force)
        right_layout.addWidget(btn_area)

    def toggle_gate(self):
        self.gate_box.setVisible(self.cb_gate.isChecked())

    def toggle_g_mode(self):
        is_single = self.rb_g_single.isChecked()
        self.wg_g_single.setVisible(is_single)
        self.wg_g_step.setVisible(not is_single)

    def toggle_b_mode(self):
        is_single = self.rb_b_single.isChecked()
        self.wg_b_single.setVisible(is_single)
        self.wg_b_step.setVisible(not is_single)

    def scan_instruments(self):
        self.log_info('扫描设备中...')
        QApplication.processEvents()
        try:
            rm = pyvisa.ResourceManager()
            res = rm.list_resources()
            self.inputs['bias_address'].clear()
            self.inputs['gate_address'].clear()
            if res:
                self.inputs['bias_address'].addItems(res)
                self.inputs['gate_address'].addItems(res)
                if len(res) >= 2:
                    self.inputs['gate_address'].setCurrentIndex(1)
                self.log_info(f"找到 {len(res)} 个设备。")
            else:
                self.log_info('未找到设备。')
        except Exception as exc:
            self.log_info(f"扫描失败: {exc}")

    def browse_folder(self):
        directory = QFileDialog.getExistingDirectory(self, '选择保存文件夹')
        if directory:
            self.ent_folder.setText(directory)

    def log_info(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum())

    def clear_log(self):
        self.log_text.clear()

    def start_measurement(self):
        if self.measure_running:
            return
        if not self.request_start(self.module_id, self.module_name):
            return
        self.reset_status_display()
        self.clear_log()

        p = {}
        try:
            for key, widget in self.inputs.items():
                if isinstance(widget, NoScrollComboBox):
                    p[key] = widget.currentText().strip()
                else:
                    p[key] = widget.text().strip()

            preset = {
                'gate_enabled': self.cb_gate.isChecked(),
                'meas_mode': p['meas_mode'],
                'num_points': float(p['num_points']),
                'duration': float(p['duration']),
                'plot_interval': int(p['plot_interval']),
                'transfer_chunk_points': int(p['plot_interval']),
                'autozero_mode': p['autozero_mode'],
                'line_filter_mode': p['line_filter_mode'],
                'line_frequency': float(p['line_frequency']),
                'line_search_hz': float(p['line_search_hz']),
                'max_odd_harmonic': int(p['max_odd_harmonic']),
                'bias_address': p['bias_address'],
                'gate_address': p['gate_address'],
                'bias_terminal': p['bias_terminal'],
                'gate_terminal': p['gate_terminal'],
                'prefix': p['filename_prefix'],
            }

            preset['g_mode'] = 'single' if self.rb_g_single.isChecked() else 'step'
            if preset['g_mode'] == 'single':
                preset['g_target'] = float(p['g_target_s'])
                preset['g_nplc'] = float(p['g_nplc_s'])
                preset['g_ramp_step'] = float(p['g_ramp_step_s'])
                if preset['g_ramp_step'] <= 0:
                    raise ValueError(f'栅压爬坡步长必须为正值，当前值: {p["g_ramp_step_s"]}')
                preset['g_step_delay'] = float(p['g_step_delay_s'])
                preset['g_ilimit'] = float(p['g_ilimit_s'])
                preset['g_settle'] = float(p['g_settle_s'])
            else:
                preset['g_start'] = float(p['g_start'])
                preset['g_end'] = float(p['g_end'])
                preset['g_test_step'] = float(p['g_test_step'])
                if preset['g_test_step'] <= 0:
                    raise ValueError(f'栅压测试步长必须为正值，当前值: {p["g_test_step"]}')
                preset['g_nplc'] = float(p['g_nplc_st'])
                preset['g_ramp_step'] = float(p['g_ramp_step_st'])
                if preset['g_ramp_step'] <= 0:
                    raise ValueError(f'栅压爬坡步长必须为正值，当前值: {p["g_ramp_step_st"]}')
                preset['g_step_delay'] = float(p['g_step_delay_st'])
                preset['g_ilimit'] = float(p['g_ilimit_st'])
                preset['g_settle'] = float(p['g_settle_st'])
                preset['g_post_zero_wait'] = float(p['g_post_zero_wait'])

            preset['b_mode'] = 'single' if self.rb_b_single.isChecked() else 'step'
            if preset['b_mode'] == 'single':
                preset['b_target'] = float(p['b_target_s'])
                preset['b_nplc'] = float(p['b_nplc_s'])
                preset['b_ramp_step'] = float(p['b_ramp_step_s'])
                if preset['b_ramp_step'] <= 0:
                    raise ValueError(f'偏压爬坡步长必须为正值，当前值: {p["b_ramp_step_s"]}')
                preset['b_step_delay'] = float(p['b_step_delay_s'])
                preset['b_range'] = p['b_range_s']
                preset['b_ilimit'] = float(p['b_ilimit_s'])
                preset['b_settle'] = float(p['b_settle_s'])
                preset['b_post_wait'] = float(p['b_post_wait_s'])
            else:
                preset['b_start'] = float(p['b_start'])
                preset['b_end'] = float(p['b_end'])
                preset['b_test_step'] = float(p['b_test_step'])
                if preset['b_test_step'] <= 0:
                    raise ValueError(f'偏压测试步长必须为正值，当前值: {p["b_test_step"]}')
                preset['b_nplc'] = float(p['b_nplc_st'])
                preset['b_ramp_step'] = float(p['b_ramp_step_st'])
                if preset['b_ramp_step'] <= 0:
                    raise ValueError(f'偏压爬坡步长必须为正值，当前值: {p["b_ramp_step_st"]}')
                preset['b_step_delay'] = float(p['b_step_delay_st'])
                preset['b_range'] = p['b_range_st']
                preset['b_ilimit'] = float(p['b_ilimit_st'])
                preset['b_settle'] = float(p['b_settle_st'])
                preset['b_post_wait'] = float(p['b_post_wait_st'])

            if preset['meas_mode'] == 'points':
                if preset['num_points'] <= 0 or int(preset['num_points']) != preset['num_points']:
                    raise ValueError('采样点数必须为正整数')
            elif preset['duration'] <= 0:
                raise ValueError('采样时长必须大于 0')
            if preset['plot_interval'] <= 0:
                raise ValueError('界面刷新间隔必须为正整数')
            if preset['autozero_mode'] not in ('block_once', 'off'):
                raise ValueError('自动调零模式必须为 block_once 或 off')
            if preset['line_filter_mode'] not in ('harmonic_fit', 'off'):
                raise ValueError('工频处理模式必须为 harmonic_fit 或 off')
            if preset['line_frequency'] <= 0:
                raise ValueError('工频中心必须大于 0')
            if preset['line_search_hz'] < 0:
                raise ValueError('工频搜索宽度不能为负值')
            if (
                preset['max_odd_harmonic'] <= 0
                or preset['max_odd_harmonic'] % 2 == 0
            ):
                raise ValueError('最高奇次谐波必须为正奇数，例如 1、3、5')
            for key, label in [
                ('g_nplc', '栅压 NPLC'),
                ('b_nplc', '偏压 NPLC'),
                ('g_ilimit', '栅电流限值'),
                ('b_ilimit', '偏压电流限制'),
            ]:
                if preset[key] <= 0:
                    raise ValueError(f'{label}必须大于 0')
            for key, label in [
                ('g_step_delay', '栅压单步延时'),
                ('g_settle', '栅压到位等待'),
                ('b_step_delay', '偏压单步延时'),
                ('b_settle', '偏压到位等待'),
                ('b_post_wait', '测量后保持'),
            ]:
                if preset[key] < 0:
                    raise ValueError(f'{label}不能为负值')
            if 'g_post_zero_wait' in preset and preset['g_post_zero_wait'] < 0:
                raise ValueError('偏压归零后等待不能为负值')
            for key, label in [('b_range', '偏压电流量程')]:
                range_val = str(preset[key]).strip()
                if range_val.upper() == 'AUTO':
                    raise ValueError('高速 It 必须使用固定电流量程，不能使用 AUTO')
                if float(range_val) <= 0:
                    raise ValueError(f'{label}必须大于 0')
            validate_program_step_plan('it', preset)

            folder = self.ent_folder.text().strip()
            self.current_folder = folder
            preset['output_folder'] = folder
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, '.test'), 'w') as file_obj:
                file_obj.write('1')
            os.remove(os.path.join(folder, '.test'))

        except Exception as exc:
            self.log_info(f"参数错误: {exc}")
            self.show_parameter_error(exc)
            self.mark_measurement_finished(self.module_id)
            return

        self.data_count = 0
        self.active_line_filter_mode = preset['line_filter_mode']
        self.time_data.fill(0)
        self.raw_curr_data.fill(0)
        self.filtered_curr_data.fill(0)
        if self.active_line_filter_mode == 'harmonic_fit':
            self.curve_raw.setPen(pg.mkPen((145, 145, 145), width=1.0))
            self.curve_it.setVisible(True)
        else:
            self.curve_raw.setPen(pg.mkPen('b', width=1.5))
            self.curve_it.setVisible(False)
        self.curve_raw.setData([], [])
        self.curve_it.setData([], [])

        while not self.update_queue.empty():
            self.update_queue.get()
        self.stop_event.clear()
        self.force_stop_event.clear()
        self.measure_running = True

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_force.setEnabled(True)
        self.start_worker(
            target=self._worker,
            args=(preset,),
            name=f'{self.module_id}-worker',
        )

    def _worker(self, preset):
        ItMeasurement(
            preset,
            update_queue=self.update_queue,
            stop_event=self.stop_event,
            force_stop_event=self.force_stop_event,
        ).run()

    def stop_measurement(self):
        self.stop_event.set()
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText('安全归零中...')
        self.log_info('已触发停止，正在执行安全归零...')

    def force_stop_measurement(self):
        if not self.measure_running:
            self._reset_btns()
            return
        self.force_stop_event.set()
        self.stop_event.set()
        self.btn_force.setEnabled(False)
        self.btn_force.setText('强行切断中...')
        self.btn_stop.setEnabled(False)
        self.log_info('强制终止已触发！瞬间切断输出！')
        QTimer.singleShot(500, self._reset_btns)

    def _reset_btns(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText('停止')
        self.btn_force.setEnabled(True)
        self.btn_force.setText('强制终止')

    def poll_queue(self):
        c = 0
        while c < 500 and not self.update_queue.empty():
            msg = self.update_queue.get_nowait()
            if msg is None:
                continue

            msg_type = msg[_0]

            if msg_type == 'finished':
                self._reset_btns()
                self.show_final_status()
                self.measure_running = False
                self.mark_measurement_finished(self.module_id)
                break

            if msg_type == 'log':
                self.note_status_from_message(msg[_1])
                self.log_info(msg[_1])

            elif msg_type == 'critical':
                self.raise_persistent_safety_alarm(
                    '【严重警告】' + msg[_1]
                )

            elif msg_type == 'stage':
                self.status_labels['stage'].setText(msg[_1])

            elif msg_type == 'clear_plot':
                self.data_count = 0
                self.curve_raw.setData([], [])
                self.curve_it.setData([], [])

            elif msg_type == 'ramp_b':
                self.status_labels['bias_v'].setText(f"{msg[_1]:.6f}")
                self.status_labels['bias_i'].setText(f"{msg[_2]:.6e}")

            elif msg_type == 'ramp_g':
                self.status_labels['gate_v'].setText(f"{msg[_1]:.6f}")
                self.status_labels['gate_i'].setText(f"{msg[_2]:.6e}")

            elif msg_type == 'gate_leakage':
                self.status_labels['gate_i'].setText(f"{msg[_1]:.6e}")

            elif msg_type == 'data_batch':
                target_vg, target_vb = msg[_1], msg[_2]
                batch_t, batch_raw, batch_filtered = msg[_3], msg[_4], msg[_5]

                if batch_t:
                    self.status_labels['gate_v'].setText(f"{target_vg:.6f}")
                    self.status_labels['bias_v'].setText(f"{target_vb:.6f}")
                    self.status_labels['bias_i'].setText(f"{batch_raw[-1]:.6e}")
                    self.status_labels['time'].setText(f"{batch_t[-1]:.3f}")

                    new_count = len(batch_t)
                    self.status_labels['count'].setText(
                        str(self.data_count + new_count))

                    needed = self.data_count + new_count
                    if needed > self.capacity:
                        new_capacity = max(needed, self.capacity * 2)
                        self.time_data = np.pad(
                            self.time_data, (0, new_capacity - self.capacity))
                        self.raw_curr_data = np.pad(
                            self.raw_curr_data, (0, new_capacity - self.capacity))
                        self.filtered_curr_data = np.pad(
                            self.filtered_curr_data, (0, new_capacity - self.capacity))
                        self.capacity = new_capacity
                        self.log_info(f'绘图缓存已自动扩容至 {new_capacity} 点')
                    self.time_data[self.data_count:needed] = batch_t
                    self.raw_curr_data[self.data_count:needed] = batch_raw
                    self.filtered_curr_data[self.data_count:needed] = batch_filtered
                    self.data_count = needed
                    self.points_changed = True

            elif msg_type == 'block_done':
                vg, vb = msg[_1], msg[_2]
                times, raw_currents, filtered_currents = msg[_3], msg[_4], msg[_5]
                filter_meta, fname = msg[_6], msg[_7]
                result_status = msg[_8] if len(msg) > 8 else 'complete'
                result_error = msg[9] if len(msg) > 9 else None
                self.note_result_status(result_status, result_error)
                rate = (
                    (len(times) - 1) / (times[-1] - times[0])
                    if len(times) > 1 and times[-1] > times[0] else 0
                )
                self.status_labels['rate'].setText(f"{rate:.2f}")
                self.submit_save(
                    self.save_data,
                    vg,
                    vb,
                    times,
                    raw_currents,
                    filtered_currents,
                    filter_meta,
                    fname,
                    status=result_status,
                    error=result_error,
                    gate_enabled=self.cb_gate.isChecked(),
                    stopped_at_local=time.strftime('%Y-%m-%d %H:%M:%S'),
                )
                if filter_meta.get('enabled'):
                    drift_text = ''
                    if 'line_frequency_drift_hz_per_s' in filter_meta:
                        drift_text = (
                            f"（{filter_meta['line_frequency_start_hz']:.4f} → "
                            f"{filter_meta['line_frequency_end_hz']:.4f} Hz）；"
                        )
                    self.log_info(
                        f"工频拟合 {filter_meta['line_frequency_hz']:.4f} Hz"
                        f"{drift_text}"
                        f"RMS {filter_meta['raw_std_A']:.3e} → "
                        f"{filter_meta['filtered_std_A']:.3e} A"
                    )

                if self.cb_plot.isChecked() and self.data_count > 0:
                    self.curve_raw.setData(
                        self.time_data[:self.data_count],
                        self.raw_curr_data[:self.data_count],
                    )
                    self.curve_it.setData(
                        self.time_data[:self.data_count],
                        self.filtered_curr_data[:self.data_count],
                    )

            c += 1

    def update_plot(self):
        if self.points_changed and self.data_count > 0:
            if self.cb_plot.isChecked():
                self.curve_raw.setData(
                    self.time_data[:self.data_count],
                    self.raw_curr_data[:self.data_count],
                )
                self.curve_it.setData(
                    self.time_data[:self.data_count],
                    self.filtered_curr_data[:self.data_count],
                )
            self.points_changed = False

    def save_data(
        self,
        vg,
        vb,
        times,
        raw_currents,
        filtered_currents,
        filter_meta,
        filename,
        status='complete',
        error=None,
        gate_enabled=False,
        stopped_at_local=None,
    ):
        reserved_path = Path(filename)
        requested_name = reserved_path.name
        if status != 'complete':
            release_path_reservation(reserved_path)
            stem, suffix = os.path.splitext(requested_name)
            requested_name = f'{stem}_partial{suffix}'
            filepath = allocate_unique_path(
                reserved_path.parent, requested_name
            )
        else:
            filepath = reserved_path
        try:
            with atomic_text_writer(filepath) as file_obj:
                if gate_enabled:
                    file_obj.write(
                        'Time(s)\tGateVoltage(V)\tBiasVoltage(V)\t'
                        'BiasCurrentRaw(A)\tBiasCurrentFiltered(A)\n'
                    )
                    for t, raw, filtered in zip(times, raw_currents, filtered_currents):
                        file_obj.write(
                            f"{t:.9f}\t{vg:.6f}\t{vb:.6f}\t"
                            f"{raw:.9e}\t{filtered:.9e}\n"
                        )
                else:
                    file_obj.write(
                        'Time(s)\tBiasVoltage(V)\t'
                        'BiasCurrentRaw(A)\tBiasCurrentFiltered(A)\n'
                    )
                    for t, raw, filtered in zip(times, raw_currents, filtered_currents):
                        file_obj.write(
                            f"{t:.9f}\t{vb:.6f}\t"
                            f"{raw:.9e}\t{filtered:.9e}\n"
                        )

            metadata = {
                'gate_voltage_V': float(vg),
                'bias_voltage_V': float(vb),
                'raw_data_column': 'BiasCurrentRaw(A)',
                'filtered_data_column': 'BiasCurrentFiltered(A)',
                'filter': filter_meta,
            }
            write_result_metadata(
                filepath,
                status=status,
                point_count=len(times),
                error=error,
                extra=metadata,
                stopped_at_local=stopped_at_local,
            )
            self.post_log(f"数据已保存: {filepath}")
            return {
                'paths': [str(filepath)], 'status': status, 'error': error,
            }
        except Exception as exc:
            self.post_log(f"保存文件失败: {exc}")
            return {'paths': [], 'status': 'error', 'error': exc}

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
