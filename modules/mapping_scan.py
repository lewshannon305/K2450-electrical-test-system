import os
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
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont

import pyqtgraph as pg

from core.app_base import BaseAppWidget
from core.hardware_base import (
    allocate_unique_path,
    atomic_text_writer,
    clear_scpi_status,
    configure_current_autozero,
    fast_shutdown_zero_2450,
    reliable_output_off,
    required_float_query,
    validate_2450_idn,
    validate_current_range_limit,
    validate_distinct_addresses,
    validate_nplc,
    validate_positive_step,
    validate_program_step_plan,
    validate_source_voltage,
    validate_terminal,
    validate_voltage_range,
    validate_voltage_within_range,
    verify_current_configuration,
    write_result_metadata,
)
from core.utils import NoScrollComboBox, G0, _0, _1, configure_pyqtgraph

class MappingMeasurement:
    GATE_RAMP_DELAY = 0.5

    def __init__(self, preset, update_queue, alarm_queue, stop_event, force_stop_event):
        self.preset = preset
        self.update_queue = update_queue
        self.alarm_queue = alarm_queue
        self.stop_event = stop_event
        self.force_stop_event = force_stop_event
        self.smu_b = None
        self.smu_g = None
        self.current_vg = 0.0
        self.current_bias = 0.0
        self.gate_io_lock = threading.Lock()
        self.gate_monitor_stop = threading.Event()
        self.gate_monitor = None
        self.latest_ig = None
        self.latest_ig_time = None
        self.active_phase = None
        self.active_vg = None
        self.active_records = []
        self.saved_paths = []

    def _interruptible_sleep(self, duration):
        if duration <= 0:
            return False
        steps = int(duration / 0.05)
        for _ in range(steps):
            if self.force_stop_event.is_set() or self.stop_event.is_set():
                return True
            time.sleep(0.05)
        rem = duration - steps * 0.05
        if rem > 0:
            if self.force_stop_event.is_set() or self.stop_event.is_set():
                return True
            time.sleep(rem)
        return False

    def _generate_ramp(self, start, stop, step):
        if step <= 0:
            raise ValueError(f"步长必须为正值")
        if start <= stop:
            return np.arange(start, stop + step / 2, step)
        return np.arange(start, stop - step / 2, -step)

    def connect(self):
        try:
            validate_distinct_addresses(
                self.preset['bias_addr'], self.preset['gate_addr'], True
            )
            rm = pyvisa.ResourceManager()
            self.smu_b = rm.open_resource(self.preset['bias_addr'])
            self.smu_b.timeout = 5000
            bias_idn = validate_2450_idn(self.smu_b.query('*IDN?'))
            self.alarm_queue.put(f"偏压表连接成功: {bias_idn}")

            self.smu_g = rm.open_resource(self.preset['gate_addr'])
            self.smu_g.timeout = 5000
            gate_idn = validate_2450_idn(self.smu_g.query('*IDN?'))
            self.alarm_queue.put(f"栅压表连接成功: {gate_idn}")
        except Exception as exc:
            self.alarm_queue.put(f"仪器连接失败: {repr(exc)}")
            raise

    def setup(self):
        try:
            validate_nplc(self.preset['bias_nplc'], '偏压NPLC')
            validate_terminal(self.preset['bias_term'], '偏压端子')
            validate_terminal(self.preset['gate_term'], '栅压端子')
            validate_source_voltage(self.preset['bias_min'], 'Mapping偏压下限')
            validate_source_voltage(self.preset['bias_max'], 'Mapping偏压上限')
            validate_source_voltage(self.preset['vg_start'], 'Mapping栅压起点')
            validate_source_voltage(self.preset['vg_stop'], 'Mapping栅压终点')
            validate_positive_step(self.preset['bias_step_up'], '上升偏压步长')
            validate_positive_step(self.preset['bias_step_full'], '全程偏压步长')
            validate_positive_step(self.preset['vg_step'], '栅压步长')
            validate_program_step_plan('mapping', self.preset)
            validate_voltage_range(self.preset['gate_v_range'], '栅压量程')
            for voltage in self._generate_ramp(
                self.preset['vg_start'],
                self.preset['vg_stop'],
                self.preset['vg_step'],
            ):
                validate_voltage_within_range(
                    voltage,
                    self.preset['gate_v_range'],
                    'Mapping栅压点',
                )
            validate_current_range_limit(
                self.preset['bias_range'], self.preset['bias_i_limit'], '偏压'
            )
            validate_current_range_limit(
                'AUTO', self.preset['gate_i_limit'], '栅极'
            )
            self.smu_b.write('*RST')
            clear_scpi_status(self.smu_b)
            self.smu_b.write(f":ROUT:TERM {self.preset['bias_term']}")
            self.smu_b.write(':SOUR:FUNC VOLT')
            self.smu_b.write(':SOUR:VOLT 0')
            self.smu_b.write(':SENS:FUNC "CURR"')
            if str(self.preset['bias_range']).upper() == 'AUTO':
                self.smu_b.write(':SENS:CURR:RANG:AUTO ON')
            else:
                self.smu_b.write(':SENS:CURR:RANG:AUTO OFF')
                self.smu_b.write(
                    f":SENS:CURR:RANG {self.preset['bias_range']}")
            self.smu_b.write(f":SOUR:VOLT:ILIM {self.preset['bias_i_limit']}")
            self.smu_b.write(f":SENS:CURR:NPLC {self.preset['bias_nplc']}")
            configure_current_autozero(self.smu_b, 'continuous')

            self.smu_g.write('*RST')
            clear_scpi_status(self.smu_g)
            self.smu_g.write(f":ROUT:TERM {self.preset['gate_term']}")
            self.smu_g.write(':SOUR:FUNC VOLT')
            self.smu_g.write(':SOUR:VOLT 0')
            self.smu_g.write(f":SOUR:VOLT:RANG {self.preset['gate_v_range']}")
            self.smu_g.write(f":SOUR:VOLT:ILIM {self.preset['gate_i_limit']}")
            self.smu_g.write(':SENS:FUNC "CURR"')
            self.smu_g.write(':SENS:CURR:NPLC 1')
            self.smu_g.write(':SENS:CURR:RANG:AUTO ON')
            configure_current_autozero(self.smu_g, 'continuous')
            verify_current_configuration(
                self.smu_b,
                nplc=self.preset['bias_nplc'],
                current_range=self.preset['bias_range'],
                current_limit=self.preset['bias_i_limit'],
                terminal=self.preset['bias_term'],
                autozero_mode='continuous',
                label='Mapping偏压表',
            )
            verify_current_configuration(
                self.smu_g,
                nplc=1,
                current_range='AUTO',
                current_limit=self.preset['gate_i_limit'],
                terminal=self.preset['gate_term'],
                autozero_mode='continuous',
                label='Mapping栅压表',
            )
        except Exception as exc:
            self.alarm_queue.put(f"仪器初始化错误: {exc}")
            raise

    def _safe_float_query(self, instr, cmd):
        return required_float_query(instr, cmd, f'Mapping查询 {cmd}')

    def gate_monitor_thread(self):
        while (
            not self.gate_monitor_stop.is_set()
            and not self.stop_event.is_set()
            and not self.force_stop_event.is_set()
        ):
            if self.smu_g is None:
                break
            try:
                with self.gate_io_lock:
                    ig = self._safe_float_query(self.smu_g, ':MEAS:CURR?')
                self.latest_ig = ig
                self.latest_ig_time = time.monotonic()
                if (
                    self.preset['ig_threshold'] > 0
                    and abs(ig) > self.preset['ig_threshold']
                ):
                    self.alarm_queue.put(f"漏电保护触发！Ig = {ig:.2e} A")
                    self.stop_event.set()
                    break
            except Exception as exc:
                self.alarm_queue.put(f'栅电流监视失败，已触发安全停止: {exc}')
                self.stop_event.set()
                break
            self.gate_monitor_stop.wait(1.0)

    def _gate_snapshot(self):
        if self.latest_ig is None or self.latest_ig_time is None:
            raise RuntimeError('尚未取得有效栅电流，停止Mapping扫描')
        age = time.monotonic() - self.latest_ig_time
        if age > 1.5:
            raise RuntimeError(f'栅电流数据已过期 {age:.3f}s，停止Mapping扫描')
        return self.latest_ig, age

    def _stop_gate_monitor(self):
        self.gate_monitor_stop.set()
        if self.gate_monitor and self.gate_monitor.is_alive():
            self.gate_monitor.join(timeout=3.0)
        if self.gate_monitor and self.gate_monitor.is_alive():
            self.alarm_queue.put(
                '严重警告：Mapping栅电流监视线程未能退出，'
                '不得关闭VISA资源，请从仪器面板确认输出。'
            )
            return False
        return True

    def _gate_write(self, command):
        with self.gate_io_lock:
            self.smu_g.write(command)

    def _save_partial_records(self, error):
        if not self.active_records or self.active_phase not in ('pos', 'full'):
            return
        folder = (
            self.preset['pos_dir']
            if self.active_phase == 'pos'
            else self.preset['full_dir']
        )
        name = (
            f"{self.preset['prefix']}_Vg={self.active_vg:.3f}V_"
            f"{self.active_phase}_partial.txt"
        )
        path = allocate_unique_path(folder, name)
        with atomic_text_writer(path) as stream:
            stream.write(
                '# Vg (V)\tBias_voltage (V)\tBias_current (A)\t'
                'Gate_current (A)\tGateCurrentAge(s)\n'
            )
            for vg, vb, ib, ig, age in self.active_records:
                stream.write(
                    f'{vg:.6f}\t{vb:.6f}\t{ib:.6e}\t{ig:.6e}\t{age:.6f}\n'
                )
        write_result_metadata(
            path,
            status='partial',
            point_count=len(self.active_records),
            error=error,
        )
        self.saved_paths.append(str(path))
        self.alarm_queue.put(f'Mapping部分数据已保存: {path}')

    def safe_ramp_to_zero(self):
        if self.smu_b is None or self.smu_g is None:
            return
        if self.force_stop_event.is_set():
            self.force_cutoff()
            return
        started = time.monotonic()
        self.update_queue.put(('stage', '偏压归零中...'))
        bias_report = fast_shutdown_zero_2450(
            self.smu_b,
            self.preset['bias_step_up'],
            label='Mapping偏压表',
            force_event=self.force_stop_event,
        )
        self.update_queue.put(('stage', '栅压归零中...'))
        gate_report = fast_shutdown_zero_2450(
            self.smu_g,
            self.preset['vg_step'],
            label='Mapping栅压表',
            force_event=self.force_stop_event,
        )
        reports = (bias_report, gate_report)
        failed = [r for r in reports if r['status'] != 'complete']
        if failed:
            self.alarm_queue.put(
                '安全归零失败，已执行紧急关断：'
                + ' | '.join(
                    error
                    for report in failed
                    for error in report['errors']
                )
            )
        else:
            self.alarm_queue.put(
                f'归零完成，用时 {time.monotonic() - started:.1f} s'
            )
            self.alarm_queue.put('输出已关闭')

    def force_cutoff(self):
        reliable_output_off(self.smu_b, 'Mapping偏压表')
        reliable_output_off(self.smu_g, 'Mapping栅压表')

    def measure_loop(self):
        try:
            if self.preset['bias_max'] <= self.preset['bias_min']:
                raise ValueError(
                    f"偏压最大值 ({self.preset['bias_max']}V) 必须大于最小值 ({self.preset['bias_min']}V)")

            vg_list = self._generate_ramp(
                self.preset['vg_start'], self.preset['vg_stop'], self.preset['vg_step'])
            bias_up_ramp = self._generate_ramp(
                0, self.preset['bias_max'], self.preset['bias_step_up'])
            bias_full_ramp = self._generate_ramp(
                self.preset['bias_max'], self.preset['bias_min'], self.preset['bias_step_full']
            )
        except ValueError as exc:
            self.alarm_queue.put(f"参数错误: {exc}")
            return

        total_vg = len(vg_list)
        pos_dir, full_dir = self.preset['pos_dir'], self.preset['full_dir']

        self._gate_write(':OUTP ON')
        self._interruptible_sleep(0.01)

        first_vg = vg_list[_0]
        if abs(first_vg) > 1e-9:
            self.update_queue.put(('stage', '栅压初始爬坡'))
            vg_init_ramp = self._generate_ramp(
                0, first_vg, min(self.preset['vg_step'], 0.1))
            if len(vg_init_ramp) == 0 or abs(vg_init_ramp[-1] - first_vg) > 1e-9:
                vg_init_ramp = np.append(vg_init_ramp, first_vg)

            for vg in vg_init_ramp:
                if self.stop_event.is_set() or self.force_stop_event.is_set():
                    return
                self._gate_write(f':SOUR:VOLT {vg}')
                self.current_vg = vg
                if self._interruptible_sleep(self.GATE_RAMP_DELAY):
                    return
                self.update_queue.put(
                    ('data', (vg, 0, 0, 0, 0, 0, 0, total_vg, 'ramp')))
        else:
            self._gate_write(f':SOUR:VOLT {first_vg}')
            self.current_vg = first_vg

        self.latest_ig = None
        self.latest_ig_time = None
        self.gate_monitor_stop.clear()
        self.smu_g.timeout = min(int(self.smu_g.timeout), 1200)
        self.gate_monitor = threading.Thread(
            target=self.gate_monitor_thread,
            name='mapping-gate-monitor',
            daemon=False,
        )
        self.gate_monitor.start()
        deadline = time.monotonic() + 1.5
        while self.latest_ig_time is None and time.monotonic() < deadline:
            if self.stop_event.is_set() or self.force_stop_event.is_set():
                break
            time.sleep(0.01)
        self._gate_snapshot()

        for idx_vg, vg in enumerate(vg_list):
            if self.stop_event.is_set() or self.force_stop_event.is_set():
                break

            self._gate_write(f':SOUR:VOLT {vg}')
            self.current_vg = vg
            self.update_queue.put(
                ('data', (vg, self.current_bias, 0, 0,
                 0, 0, idx_vg + 1, total_vg, 'waiting'))
            )

            if idx_vg == 0:
                self.update_queue.put(('stage', '初始等待中'))
                wait_rem = self.preset['wait_init']
                while wait_rem > 0 and not self.stop_event.is_set() and not self.force_stop_event.is_set():
                    w = min(1.0, wait_rem)
                    time.sleep(w)
                    wait_rem -= w
                    self.update_queue.put(('stage', f'初始等待中 ({wait_rem:.1f}s)'))
                if self.force_stop_event.is_set():
                    break

                if self.stop_event.is_set() or self.force_stop_event.is_set():
                    break
                self.smu_b.write(':OUTP ON')
                self.update_queue.put(('stage', '偏压输出开启'))
                self._interruptible_sleep(0.01)
            else:
                self.update_queue.put(('stage', '栅压等待中'))
                wait_rem = self.preset['gate_settle']
                while wait_rem > 0 and not self.stop_event.is_set() and not self.force_stop_event.is_set():
                    w = min(1.0, wait_rem)
                    time.sleep(w)
                    wait_rem -= w
                    self.update_queue.put(('stage', f'栅压等待中 ({wait_rem:.1f}s)'))
                if self.force_stop_event.is_set():
                    break

            if self.stop_event.is_set() or self.force_stop_event.is_set():
                break

            self.update_queue.put(('new_vg_cycle', 0))

            self.update_queue.put(('stage', '上升扫描'))
            vb_up, ib_up, ig_up, ig_age_up = [], [], [], []
            self.active_phase = 'pos'
            self.active_vg = vg
            self.active_records = []
            for vb in bias_up_ramp:
                if self.stop_event.is_set() or self.force_stop_event.is_set():
                    break
                self.smu_b.write(f':SOUR:VOLT {vb}')
                self.current_bias = vb
                if self._interruptible_sleep(self.preset['bias_settle']):
                    break

                ib = self._safe_float_query(self.smu_b, ':MEAS:CURR?')
                ig, ig_age = self._gate_snapshot()

                cond = ib / vb if vb != 0 else 0
                res = vb / ib if ib != 0 else float('inf')

                self.update_queue.put(
                    ('data', (vg, vb, ib, ig, cond / G0,
                     res, idx_vg + 1, total_vg, 'up'))
                )
                vb_up.append(vb)
                ib_up.append(ib)
                ig_up.append(ig)
                ig_age_up.append(ig_age)
                self.active_records.append((vg, vb, ib, ig, ig_age))

            if self.stop_event.is_set() or self.force_stop_event.is_set():
                break

            try:
                fname = allocate_unique_path(
                    pos_dir, f"{self.preset['prefix']}_Vg={vg:.3f}V_pos.txt")
                with atomic_text_writer(fname) as file_obj:
                    file_obj.write(
                        '# Vg (V)\tBias_voltage (V)\tBias_current (A)\tGate_current (A)\tGateCurrentAge(s)\n')
                    for i in range(len(vb_up)):
                        file_obj.write(
                            f"{vg:.6f}\t{vb_up[i]:.6f}\t{ib_up[i]:.6e}\t{ig_up[i]:.6e}\t{ig_age_up[i]:.6f}\n")
                write_result_metadata(
                    fname, status='complete', point_count=len(vb_up)
                )
                self.saved_paths.append(str(fname))
                self.active_records = []
            except Exception as exc:
                self.alarm_queue.put(f"保存上升阶段数据失败: {exc}")

            if self.preset['wait_after_up'] > 0:
                self.update_queue.put(('stage', '上升后等待'))
                wait_rem = self.preset['wait_after_up']
                while wait_rem > 0 and not self.stop_event.is_set() and not self.force_stop_event.is_set():
                    w = min(1.0, wait_rem)
                    time.sleep(w)
                    wait_rem -= w
                    self.update_queue.put(('stage', f'上升后等待 ({wait_rem:.1f}s)'))
                if self.force_stop_event.is_set():
                    break
            if self.stop_event.is_set() or self.force_stop_event.is_set():
                break

            self.update_queue.put(('stage', '全程扫描'))
            vb_full, ib_full, ig_full, ig_age_full = [], [], [], []
            self.active_phase = 'full'
            self.active_vg = vg
            self.active_records = []
            for vb in bias_full_ramp:
                if self.stop_event.is_set() or self.force_stop_event.is_set():
                    break
                self.smu_b.write(f':SOUR:VOLT {vb}')
                self.current_bias = vb
                if self._interruptible_sleep(self.preset['bias_settle']):
                    break

                ib = self._safe_float_query(self.smu_b, ':MEAS:CURR?')
                ig, ig_age = self._gate_snapshot()

                cond = ib / vb if vb != 0 else 0
                res = vb / ib if ib != 0 else float('inf')

                self.update_queue.put(
                    ('data', (vg, vb, ib, ig, cond / G0,
                     res, idx_vg + 1, total_vg, 'full'))
                )
                vb_full.append(vb)
                ib_full.append(ib)
                ig_full.append(ig)
                ig_age_full.append(ig_age)
                self.active_records.append((vg, vb, ib, ig, ig_age))

            if self.stop_event.is_set() or self.force_stop_event.is_set():
                break

            try:
                fname = allocate_unique_path(
                    full_dir, f"{self.preset['prefix']}_Vg={vg:.3f}V_full.txt")
                with atomic_text_writer(fname) as file_obj:
                    file_obj.write(
                        '# Vg (V)\tBias_voltage (V)\tBias_current (A)\tGate_current (A)\tGateCurrentAge(s)\n')
                    for i in range(len(vb_full)):
                        file_obj.write(
                            f"{vg:.6f}\t{vb_full[i]:.6f}\t{ib_full[i]:.6e}\t{ig_full[i]:.6e}\t{ig_age_full[i]:.6f}\n")
                write_result_metadata(
                    fname, status='complete', point_count=len(vb_full)
                )
                self.saved_paths.append(str(fname))
                self.active_records = []
            except Exception as exc:
                self.alarm_queue.put(f"保存全程数据失败: {exc}")

            self.update_queue.put(('stage', '轮次偏压归零'))
            bias_return_ramp = self._generate_ramp(
                self.current_bias, 0, self.preset['bias_step_up'])
            if len(bias_return_ramp) == 0 or abs(bias_return_ramp[-1]) > 1e-9:
                bias_return_ramp = np.append(bias_return_ramp, 0)

            for vb in bias_return_ramp:
                if self.stop_event.is_set() or self.force_stop_event.is_set():
                    break
                self.smu_b.write(f':SOUR:VOLT {vb}')
                self.current_bias = vb
                if self.force_stop_event.is_set():
                    break
                if self._interruptible_sleep(self.preset['bias_settle']):
                    break

                ib_zero = self._safe_float_query(self.smu_b, ':MEAS:CURR?')

                self.update_queue.put(
                    ('data', (vg, vb, ib_zero, 0, 0, 0,
                     idx_vg + 1, total_vg, 'zeroing'))
                )

            if idx_vg < total_vg - 1 and not self.stop_event.is_set() and not self.force_stop_event.is_set():
                wait_rem = 2.0
                while wait_rem > 0 and not self.stop_event.is_set() and not self.force_stop_event.is_set():
                    w = min(1.0, wait_rem)
                    time.sleep(w)
                    wait_rem -= w
                    self.update_queue.put(('stage', f'偏压归零后等待 ({wait_rem:.1f}s)'))
                if self.force_stop_event.is_set():
                    break

    def run(self):
        result_status = 'complete'
        result_error = None
        try:
            self.connect()
            self.setup()
            self.measure_loop()
            if self.stop_event.is_set() or self.force_stop_event.is_set():
                result_status = 'partial'
                result_error = '用户停止或强制终止'
            if self.active_records:
                result_status = 'partial'
                result_error = (
                    '用户停止或强制终止'
                    if self.stop_event.is_set() or self.force_stop_event.is_set()
                    else '扫描未完整结束'
                )
                self._save_partial_records(
                    result_error
                )
        except Exception as exc:
            result_status = 'error'
            result_error = str(exc)
            self.alarm_queue.put(f"异常中断: {exc}")
            try:
                self._save_partial_records(exc)
            except Exception as save_exc:
                self.alarm_queue.put(f'Mapping部分数据保存失败: {save_exc}')
        finally:
            monitor_stopped = self._stop_gate_monitor()
            try:
                if self.force_stop_event.is_set():
                    self.force_cutoff()
                else:
                    self.safe_ramp_to_zero()
            except Exception as exc:
                self.alarm_queue.put(f'安全归零异常: {exc}')
            bias_confirmed, bias_failures = reliable_output_off(
                self.smu_b, 'Mapping偏压表'
            )
            if monitor_stopped:
                gate_confirmed, gate_failures = reliable_output_off(
                    self.smu_g, 'Mapping栅压表'
                )
            else:
                gate_confirmed = False
                gate_failures = ['栅电流监视线程仍占用VISA资源']
            if not bias_confirmed or not gate_confirmed:
                self.alarm_queue.put(
                    '严重警告：无法确认Mapping源表输出已关闭，请立即从仪器面板确认。 '
                    + ' | '.join(bias_failures + gate_failures)
                )

            if self.smu_b:
                try:
                    self.smu_b.close()
                except Exception:
                    pass
            if self.smu_g and monitor_stopped:
                try:
                    self.smu_g.close()
                except Exception:
                    pass

            self.update_queue.put((
                'finished',
                {
                    'paths': list(self.saved_paths),
                    'status': result_status,
                    'error': result_error,
                },
            ))


class MappingWidget(BaseAppWidget):
    def __init__(self, run_guard=None, parent=None):
        configure_pyqtgraph(use_opengl=True)
        super().__init__(run_guard=run_guard, parent=parent)

        self.module_id = 'mapping_scan'
        self.module_name = '二维Mapping扫描'

        self.ui_font = QFont('Arial', 12)
        self.ui_font.setWeight(QFont.Weight.Normal)
        self.bold_font = QFont('Arial', 12)
        self.bold_font.setWeight(QFont.Weight.Bold)
        self.setFont(self.ui_font)

        self.capacity = 10000
        self.up_x = np.zeros(self.capacity)
        self.up_y = np.zeros(self.capacity)
        self.full_x = np.zeros(self.capacity)
        self.full_y = np.zeros(self.capacity)
        self.up_count = 0
        self.full_count = 0
        self.points_changed = False

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=3)

        self.graph_widget = pg.GraphicsLayoutWidget()
        left_layout.addWidget(self.graph_widget)
        label_style = {'color': '#000', 'font-size': '12pt'}

        self.plot_up = self.graph_widget.addPlot(title='Ramp-up Phase')
        self.plot_up.setLabel('left', text='Bias Current',
                              units='A', **label_style)
        self.plot_up.setLabel('bottom', text='Bias Voltage',
                              units='V', **label_style)
        self.plot_up.getAxis('left').setTickFont(self.ui_font)
        self.plot_up.getAxis('bottom').setTickFont(self.ui_font)
        self.plot_up.showGrid(x=True, y=True, alpha=0.3)
        self.plot_up.setClipToView(True)
        self.plot_up.setDownsampling(auto=True, mode='peak')
        self.curve_up = self.plot_up.plot(pen=pg.mkPen('b', width=1.5))

        self.graph_widget.nextRow()

        self.plot_full = self.graph_widget.addPlot(title='Full Scan Phase')
        self.plot_full.setLabel(
            'left', text='Bias Current', units='A', **label_style)
        self.plot_full.setLabel(
            'bottom', text='Bias Voltage', units='V', **label_style)
        self.plot_full.getAxis('left').setTickFont(self.ui_font)
        self.plot_full.getAxis('bottom').setTickFont(self.ui_font)
        self.plot_full.showGrid(x=True, y=True, alpha=0.3)
        self.plot_full.setClipToView(True)
        self.plot_full.setDownsampling(auto=True, mode='peak')
        self.curve_full = self.plot_full.plot(pen=pg.mkPen('r', width=1.5))

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
            ('偏压 Vsd (V):', 'vb', 0, 0), ('电导 (G₀):', 'cond', 0, 2),
            ('栅压 Vg (V):', 'vg', 1, 0), ('电阻 (Ω):', 'res', 1, 2),
            ('偏置电流 Isd (A):', 'ib', 2, 0), ('进度:', 'progress', 2, 2),
            ('栅电流 Ig (A):', 'ig', 3, 0), ('系统状态:', 'stage', 3, 2),
        ]
        for text, key, row, col in status_items:
            lbl = QLabel(text)
            lbl.setFont(self.ui_font)
            status_layout.addWidget(
                lbl, row, col, alignment=Qt.AlignmentFlag.AlignLeft)

            val = QLabel('-')
            val.setFont(self.ui_font)
            val.setStyleSheet('color: #0055A4; font-weight: bold;')
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

        lbl_ba = QLabel('Bias 表地址:')
        lbl_ba.setFont(self.ui_font)
        addr_grid.addWidget(lbl_ba, 0, 0)
        self.c_b_addr = NoScrollComboBox()
        self.c_b_addr.setFont(self.ui_font)
        self.c_b_addr.setEditable(True)
        self.c_b_addr.addItem('GPIB0::1::INSTR')
        self.c_b_addr.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.inputs['bias_addr'] = self.c_b_addr
        addr_grid.addWidget(self.c_b_addr, 0, 1)

        lbl_ga = QLabel('Gate 表地址:')
        lbl_ga.setFont(self.ui_font)
        addr_grid.addWidget(lbl_ga, 0, 2)
        self.c_g_addr = NoScrollComboBox()
        self.c_g_addr.setFont(self.ui_font)
        self.c_g_addr.setEditable(True)
        self.c_g_addr.addItem('GPIB0::2::INSTR')
        self.c_g_addr.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.inputs['gate_addr'] = self.c_g_addr
        addr_grid.addWidget(self.c_g_addr, 0, 3)

        lbl_bt = QLabel('Bias 表端口:')
        lbl_bt.setFont(self.ui_font)
        addr_grid.addWidget(lbl_bt, 1, 0)
        self.c_b_term = NoScrollComboBox()
        self.c_b_term.setFont(self.ui_font)
        self.c_b_term.addItems(['REAR', 'FRONT'])
        self.c_b_term.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.inputs['bias_term'] = self.c_b_term
        addr_grid.addWidget(self.c_b_term, 1, 1)

        lbl_gt = QLabel('Gate 表端口:')
        lbl_gt.setFont(self.ui_font)
        addr_grid.addWidget(lbl_gt, 1, 2)
        self.c_g_term = NoScrollComboBox()
        self.c_g_term.setFont(self.ui_font)
        self.c_g_term.addItems(['REAR', 'FRONT'])
        self.c_g_term.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.inputs['gate_term'] = self.c_g_term
        addr_grid.addWidget(self.c_g_term, 1, 3)

        btn_scan = QPushButton('扫描设备')
        btn_scan.setFont(self.bold_font)
        btn_scan.setFixedSize(100, 30)
        btn_scan.clicked.connect(self.scan_instruments)
        addr_grid.addWidget(btn_scan, 2, 0, 1, 4,
                            alignment=Qt.AlignmentFlag.AlignCenter)
        box_vbox.addWidget(addr_box)

        gate_box = QGroupBox('栅压参数')
        gate_box.setFont(self.bold_font)
        gate_grid = QGridLayout(gate_box)
        gate_items = [
            ('起始栅压 (V):', 'vg_start', '-5.0'), ('终止栅压 (V):', 'vg_stop', '5.0'),
            ('栅压步长 (V) (正):', 'vg_step', '0.05'), ('栅压量程 (V):', 'gate_v_range', '20'),
            ('栅极限流 (A):', 'gate_i_limit',
             '1e-9'), ('栅极报警阈值 (A):', 'ig_threshold', '1e-9'),
        ]
        for i, (label, key, def_val) in enumerate(gate_items):
            r = i % 3
            c = (i // 3) * 2
            lbl = QLabel(label)
            lbl.setFont(self.ui_font)
            gate_grid.addWidget(lbl, r, c)
            ent = QLineEdit(def_val)
            ent.setFont(self.ui_font)
            self.inputs[key] = ent
            gate_grid.addWidget(ent, r, c + 1)
        box_vbox.addWidget(gate_box)

        bias_box = QGroupBox('偏压参数')
        bias_box.setFont(self.bold_font)
        bias_grid = QGridLayout(bias_box)
        bias_items = [
            ('偏压最大值 (V):', 'bias_max', '0.02'), ('偏压最小值 (V):', 'bias_min', '-0.02'),
            ('全程步长 (V) (正):', 'bias_step_full',
             '0.0001'), ('上升步长 (V) (正):', 'bias_step_up', '0.001'),
            ('偏压 NPLC:', 'bias_nplc', '1'), ('电流量程 (A / AUTO):', 'bias_range', '1e-6'),
            ('偏压限流 (A):', 'bias_i_limit', '1.05e-6'),
        ]
        for i, (label, key, def_val) in enumerate(bias_items):
            r = i % 4
            c = (i // 4) * 2
            lbl = QLabel(label)
            lbl.setFont(self.ui_font)
            bias_grid.addWidget(lbl, r, c)
            ent = QLineEdit(def_val)
            ent.setFont(self.ui_font)
            self.inputs[key] = ent
            bias_grid.addWidget(ent, r, c + 1)
        box_vbox.addWidget(bias_box)

        time_box = QGroupBox('时间设定')
        time_box.setFont(self.bold_font)
        time_grid = QGridLayout(time_box)
        time_items = [
            ('初始等待 (s):', 'wait_init', '600'), ('上升后等待 (s):', 'wait_after_up', '5'),
            ('栅压等待 (s):', 'gate_settle', '20'), ('偏压等待 (s):', 'bias_settle', '0'),
        ]
        for i, (label, key, def_val) in enumerate(time_items):
            r = i % 2
            c = (i // 2) * 2
            lbl = QLabel(label)
            lbl.setFont(self.ui_font)
            time_grid.addWidget(lbl, r, c)
            ent = QLineEdit(def_val)
            ent.setFont(self.ui_font)
            self.inputs[key] = ent
            time_grid.addWidget(ent, r, c + 1)
        box_vbox.addWidget(time_box)

        path_box = QGroupBox('文件保存路径')
        path_box.setFont(self.bold_font)
        path_grid = QGridLayout(path_box)

        lbl_pf = QLabel('文件名前缀 (后缀自动追加栅压、偏压信息):')
        lbl_pf.setFont(self.ui_font)
        path_grid.addWidget(lbl_pf, 0, 0, 1, 2)
        self.ent_prefix = QLineEdit('Mapping')
        self.ent_prefix.setFont(self.ui_font)
        self.inputs['prefix'] = self.ent_prefix
        path_grid.addWidget(self.ent_prefix, 1, 0, 1, 2)

        lbl_fd = QLabel('保存文件夹:')
        lbl_fd.setFont(self.ui_font)
        path_grid.addWidget(lbl_fd, 2, 0, 1, 2)
        fhbox = QHBoxLayout()
        fhbox.setContentsMargins(0, 0, 0, 0)
        self.ent_folder = QLineEdit(r"C:\lxr\data\202603\Au_test\Mapping")
        self.ent_folder.setFont(self.ui_font)
        fhbox.addWidget(self.ent_folder)
        btn_br = QPushButton('浏览')
        btn_br.setFont(self.ui_font)
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
        self.btn_force.setStyleSheet('color: #AA0000;')
        self.btn_force.clicked.connect(self.force_stop_measurement)

        btn_hbox.addWidget(self.btn_start)
        btn_hbox.addStretch()
        btn_hbox.addWidget(self.btn_stop)
        btn_hbox.addStretch()
        btn_hbox.addWidget(self.btn_force)
        right_layout.addWidget(btn_area)

    def scan_instruments(self):
        self.log_info('扫描设备中...')
        QApplication.processEvents()
        try:
            rm = pyvisa.ResourceManager()
            res = rm.list_resources()
            self.c_b_addr.clear()
            self.c_g_addr.clear()
            if res:
                self.c_b_addr.addItems(res)
                self.c_g_addr.addItems(res)
                if len(res) >= 2:
                    self.c_g_addr.setCurrentIndex(1)
                self.log_info(f"扫描完成，找到 {len(res)} 个设备。")
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

        preset = {}
        try:
            for key, widget in self.inputs.items():
                if isinstance(widget, NoScrollComboBox):
                    preset[key] = widget.currentText().strip()
                elif isinstance(widget, QLineEdit):
                    txt = widget.text().strip()
                    if key in ['prefix']:
                        preset[key] = txt
                    else:
                        preset[key] = txt if txt.upper(
                        ) == 'AUTO' else float(txt)

            preset['pos_dir'] = os.path.join(self.ent_folder.text(), 'pos')
            preset['full_dir'] = os.path.join(self.ent_folder.text(), 'full')
            if preset['vg_step'] <= 0:
                raise ValueError('栅压步长必须大于 0')
            if preset['gate_v_range'] <= 0:
                raise ValueError('栅压量程必须大于 0')
            if preset['gate_i_limit'] <= 0:
                raise ValueError('栅极限流必须大于 0')
            if preset['ig_threshold'] < 0:
                raise ValueError('栅极报警阈值不能为负值')
            if preset['bias_max'] <= preset['bias_min']:
                raise ValueError('偏压最大值必须大于偏压最小值')
            if preset['bias_step_full'] <= 0:
                raise ValueError('全程扫描偏压步长必须大于 0')
            if preset['bias_step_up'] <= 0:
                raise ValueError('上升扫描偏压步长必须大于 0')
            if preset['bias_nplc'] <= 0:
                raise ValueError('偏压 NPLC 必须大于 0')
            if not isinstance(preset['bias_range'], str) and preset['bias_range'] <= 0:
                raise ValueError('电流量程必须大于 0 或为 AUTO')
            if preset['bias_i_limit'] <= 0:
                raise ValueError('偏压限流必须大于 0')
            for key, label in [
                ('wait_init', '初始等待'),
                ('wait_after_up', '上升后等待'),
                ('gate_settle', '栅压等待'),
                ('bias_settle', '偏压等待'),
            ]:
                if preset[key] < 0:
                    raise ValueError(f'{label}不能为负值')
            validate_program_step_plan('mapping', preset)
            os.makedirs(preset['pos_dir'], exist_ok=True)
            os.makedirs(preset['full_dir'], exist_ok=True)

            with open(os.path.join(preset['full_dir'], '.test'), 'w') as file_obj:
                file_obj.write('1')
            os.remove(os.path.join(preset['full_dir'], '.test'))
        except Exception as exc:
            self.log_info(f"参数格式或路径错误: {exc}")
            self.show_parameter_error(exc)
            self.mark_measurement_finished(self.module_id)
            return

        self.up_count = 0
        self.full_count = 0
        self.up_x.fill(0)
        self.up_y.fill(0)
        self.full_x.fill(0)
        self.full_y.fill(0)
        self.curve_up.setData([], [])
        self.curve_full.setData([], [])

        while not self.update_queue.empty():
            self.update_queue.get()
        while not self.alarm_queue.empty():
            self.alarm_queue.get()

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
        MappingMeasurement(
            preset,
            update_queue=self.update_queue,
            alarm_queue=self.alarm_queue,
            stop_event=self.stop_event,
            force_stop_event=self.force_stop_event,
        ).run()

    def stop_measurement(self):
        self.stop_event.set()
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText('安全降压中...')
        self.log_info('停止已触发，执行步进归零...')

    def force_stop_measurement(self):
        if not self.measure_running:
            self._reset_btns()
            return
        self.force_stop_event.set()
        self.stop_event.set()
        self.btn_force.setEnabled(False)
        self.btn_force.setText('强制切断中...')
        self.btn_stop.setEnabled(False)
        self.log_info('强制终止已触发，切断物理输出')
        QTimer.singleShot(500, self._reset_btns)

    def _reset_btns(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText('停止')
        self.btn_force.setEnabled(True)
        self.btn_force.setText('强制终止')

    def poll_queue(self):
        while not self.alarm_queue.empty():
            message = self.alarm_queue.get()
            self.note_status_from_message(message)
            if '严重警告' in message:
                self.raise_persistent_safety_alarm(message)
            else:
                self.log_info(message)

        c = 0
        while c < 500 and not self.update_queue.empty():
            msg = self.update_queue.get_nowait()

            if isinstance(msg, tuple) and msg[_0] == 'finished':
                if len(msg) > 1 and isinstance(msg[1], dict):
                    result = msg[1]
                    self.note_result_status(
                        result.get('status', 'complete'), result.get('error')
                    )
                    self.record_saved_result(
                        result.get('paths', []),
                        status=result.get('status', 'complete'),
                        error=result.get('error'),
                    )
                self._reset_btns()
                self.log_info('测试结束。')
                self.show_final_status()
                self.measure_running = False
                self.mark_measurement_finished(self.module_id)

                break

            if isinstance(msg, tuple) and msg[_0] == 'stage':
                self.status_labels['stage'].setText(msg[_1])
                continue

            if isinstance(msg, tuple) and msg[_0] == 'new_vg_cycle':
                self.up_count = 0
                self.full_count = 0
                self.curve_up.setData([], [])
                self.curve_full.setData([], [])
                continue

            if isinstance(msg, tuple) and msg[_0] == 'data':
                _, payload = msg
                vg, vb, ib, ig, cond, res, cycle, total, stage = payload
                self.status_labels['vg'].setText(f"{vg:.6f}")
                self.status_labels['vb'].setText(f"{vb:.6f}")
                self.status_labels['ib'].setText(f"{ib:.2e}")
                self.status_labels['ig'].setText(f"{ig:.2e}")
                self.status_labels['cond'].setText(f"{cond:.6e}")
                self.status_labels['res'].setText(
                    f"{res:.2e}" if res != float('inf') else 'inf')
                if total > 0:
                    self.status_labels['progress'].setText(f"{cycle}/{total}")

                if stage == 'up':
                    if self.up_count >= self.capacity:
                        self._expand_plot_capacity()
                    self.up_x[self.up_count] = vb
                    self.up_y[self.up_count] = ib
                    self.up_count += 1
                    self.points_changed = True
                elif stage == 'full':
                    if self.full_count >= self.capacity:
                        self._expand_plot_capacity()
                    self.full_x[self.full_count] = vb
                    self.full_y[self.full_count] = ib
                    self.full_count += 1
                    self.points_changed = True
            c += 1

    def _expand_plot_capacity(self):
        old_capacity = self.capacity
        self.capacity *= 2
        extra = self.capacity - old_capacity
        self.up_x = np.pad(self.up_x, (0, extra))
        self.up_y = np.pad(self.up_y, (0, extra))
        self.full_x = np.pad(self.full_x, (0, extra))
        self.full_y = np.pad(self.full_y, (0, extra))
        self.log_info(f'绘图缓存已自动扩容至 {self.capacity} 点')

    def update_plot(self):
        if self.points_changed:
            if self.up_count > 0:
                self.curve_up.setData(
                    self.up_x[:self.up_count], self.up_y[:self.up_count])
            if self.full_count > 0:
                self.curve_full.setData(
                    self.full_x[:self.full_count], self.full_y[:self.full_count])
            self.points_changed = False

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
