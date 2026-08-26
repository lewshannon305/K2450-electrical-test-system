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
    QCheckBox,
    QSizePolicy,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont

import pyqtgraph as pg

from core.app_base import BaseAppWidget
from core.paths import default_data_directory
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
    verify_current_configuration,
    write_result_metadata,
)
from core.utils import NoScrollComboBox, _0, _1, _2, _3, _4, _5, _6, _7, _8, configure_pyqtgraph


class SwitchMeasurement:
    def __init__(self, params, update_queue, stop_event, force_stop_event):
        self.params = params
        self.update_queue = update_queue
        self.stop_event = stop_event
        self.force_stop_event = force_stop_event
        self.bias_k = None
        self.gate_k = None
        self.gate_monitor_stop = threading.Event()
        self.gate_monitor = None
        self.gate_io_lock = threading.Lock()

    def connect(self):
        try:
            validate_distinct_addresses(
                self.params['bias_address'],
                self.params['gate_address'],
                self.params['gate_enabled'],
            )
            rm = pyvisa.ResourceManager()
            self.bias_k = rm.open_resource(self.params['bias_address'])
            self.bias_k.timeout = 10000
            bias_idn = validate_2450_idn(self.bias_k.query('*IDN?'))
            self.update_queue.put(('log', f"偏压表连接成功: {bias_idn}"))

            if self.params['gate_enabled']:
                self.gate_k = rm.open_resource(self.params['gate_address'])
                self.gate_k.timeout = 10000
                gate_idn = validate_2450_idn(self.gate_k.query('*IDN?'))
                self.update_queue.put(('log', f"栅压表连接成功: {gate_idn}"))
        except Exception as exc:
            self.update_queue.put(('log', f"仪器连接失败: {repr(exc)}"))
            raise

    def setup(self):
        try:
            validate_nplc(self.params['b_nplc'], '偏压NPLC')
            validate_terminal(self.params['bias_terminal'], '偏压端子')
            for key, label in (
                ('sw1_v', '偏压状态1'),
                ('sw2_v', '偏压状态2'),
                ('test_v', '偏压测试电压'),
            ):
                validate_source_voltage(self.params[key], label)
            validate_positive_step(self.params['b_ramp_step'], '偏压爬坡步长')
            validate_current_range_limit(
                self.params['b_range'], self.params['b_ilimit'], '偏压'
            )
            if self.params['gate_enabled']:
                validate_nplc(self.params['g_nplc'], '栅压NPLC')
                validate_terminal(self.params['gate_terminal'], '栅压端子')
                validate_source_voltage(self.params['g_target'], '栅压目标')
                validate_positive_step(self.params['g_ramp_step'], '栅压爬坡步长')
                validate_current_range_limit('AUTO', self.params['g_ilimit'], '栅极')
            validate_program_step_plan('bias_switch', self.params)
            k_b = self.bias_k
            k_b.write('*RST')
            clear_scpi_status(k_b)
            k_b.write(':ABORt')
            k_b.write(':SOUR:FUNC VOLT')
            k_b.write(':SENS:FUNC "CURR"')
            k_b.write('SENS:CURR:RSEN OFF')
            k_b.write(f":ROUT:TERM {self.params['bias_terminal']}")
            k_b.write(':SENS:CURR:AVER OFF')
            k_b.write(':SOUR:VOLT:READ:BACK ON')
            k_b.write(f":SENS:CURR:NPLC {self.params['b_nplc']}")

            r_val = self.params['b_range']
            if str(r_val).upper() == 'AUTO':
                k_b.write(':SENS:CURR:RANG:AUTO ON')
            else:
                k_b.write(':SENS:CURR:RANG:AUTO OFF')
                k_b.write(f":SENS:CURR:RANG {r_val}")
            configure_current_autozero(k_b, 'block_once')

            k_b.write(f":SOUR:VOLT:ILIM {self.params['b_ilimit']}")
            k_b.write(':SOUR:VOLT 0')

            if self.params['gate_enabled']:
                k_g = self.gate_k
                k_g.write('*RST')
                clear_scpi_status(k_g)
                k_g.write(':ABORt')
                k_g.write(':SOUR:FUNC VOLT')
                k_g.write(':SENS:FUNC "CURR"')
                k_g.write('SENS:CURR:RSEN OFF')
                k_g.write(f":ROUT:TERM {self.params['gate_terminal']}")
                k_g.write(':SENS:CURR:AVER OFF')
                k_g.write(':SOUR:VOLT:READ:BACK ON')
                k_g.write(f":SENS:CURR:NPLC {self.params['g_nplc']}")
                k_g.write(':SENS:CURR:RANG:AUTO ON')
                configure_current_autozero(k_g, 'block_once')
                k_g.write(f":SOUR:VOLT:ILIM {self.params['g_ilimit']}")
                k_g.write(':SOUR:VOLT 0')

            if not self._sleep(0.05, "仪器初始化中"):
                return
            verify_current_configuration(
                k_b,
                nplc=self.params['b_nplc'],
                current_range=self.params['b_range'],
                current_limit=self.params['b_ilimit'],
                terminal=self.params['bias_terminal'],
                autozero_mode='block_once',
                label='偏压开关偏压表',
            )
            if self.params['gate_enabled']:
                verify_current_configuration(
                    k_g,
                    nplc=self.params['g_nplc'],
                    current_range='AUTO',
                    current_limit=self.params['g_ilimit'],
                    terminal=self.params['gate_terminal'],
                    autozero_mode='block_once',
                    label='偏压开关栅压表',
                )
        except Exception as exc:
            self.update_queue.put(('log', f"仪器初始化错误: {exc}"))
            raise

    def _sleep(self, duration_s, stage_msg='等待'):
        if duration_s <= 0:
            return True
        steps = int(duration_s / 0.1)
        for i in range(steps):
            if self.stop_event.is_set() or self.force_stop_event.is_set():
                return False
            time.sleep(0.1)
            if i % 10 == 0:
                rem = duration_s - (i * 0.1)
                self.update_queue.put(('stage', f"{stage_msg} ({rem:.1f}s)"))
        rem = duration_s - steps * 0.1
        if rem > 0:
            if self.stop_event.is_set() or self.force_stop_event.is_set():
                return False
            time.sleep(rem)
        return not (self.stop_event.is_set() or self.force_stop_event.is_set())

    def _ramp_voltage(self, inst, target_v, step_abs, step_delay_s, is_gate=False, is_zeroing=False):
        current_v = required_float_query(inst, ':SOUR:VOLT?', '源电压回读')
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
                if not self._sleep(step_delay_s, "电压调整中"):
                    return False

            reading = required_float_query(inst, ':READ?', '爬坡电流读数')

            if is_gate:
                self.update_queue.put(('ramp_g', v, reading))
            else:
                self.update_queue.put(('ramp_b', v, reading))
            if v == target_v:
                break
        return True

    def safe_zeroing(self):
        started = time.perf_counter()
        reports = []
        try:
            if self.bias_k:
                self.update_queue.put(('stage', '偏压归零中...'))
                reports.append(fast_shutdown_zero_2450(
                    self.bias_k, self.params['b_ramp_step'],
                    label='偏压表', force_event=self.force_stop_event,
                ))
            if self.params['gate_enabled'] and self.gate_k:
                self.update_queue.put(('stage', '栅压归零中...'))
                reports.append(fast_shutdown_zero_2450(
                    self.gate_k, self.params['g_ramp_step'],
                    label='栅压表', force_event=self.force_stop_event,
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
                self.update_queue.put(('log',
                    f'安全归零失败，已执行紧急关断：{details or "状态确认失败"}'))
        except Exception as exc:
            self.update_queue.put(
                ('log', f'安全归零失败，已执行紧急关断：{exc}'))

    def gate_monitor_thread(self):
        if not self.params['gate_enabled']:
            return
        while (
            not self.gate_monitor_stop.is_set()
            and not self.stop_event.is_set()
            and not self.force_stop_event.is_set()
        ):
            if self.gate_k is None:
                break
            try:
                with self.gate_io_lock:
                    current = required_float_query(
                        self.gate_k, ':READ?', '栅电流监视读数'
                    )
                self.update_queue.put(('gate_leakage', current))
            except Exception as exc:
                self.update_queue.put(('log', f'栅电流监视失败，停止测量: {exc}'))
                self.stop_event.set()
                break
            self.gate_monitor_stop.wait(1.0)

    def _stop_gate_monitor(self):
        self.gate_monitor_stop.set()
        if self.gate_monitor and self.gate_monitor.is_alive():
            self.gate_monitor.join(timeout=3.0)
        if self.gate_monitor and self.gate_monitor.is_alive():
            self.update_queue.put((
                'log',
                '【严重警告】栅电流监视线程未能退出，'
                '不得关闭栅压表VISA资源，请从仪器面板确认输出。',
            ))
            return False
        return True

    def run(self):
        times, v_outs, i_outs = [], [], []
        vg = self.params['g_target'] if self.params['gate_enabled'] else 0.0
        try:
            self.connect()
            self.setup()

            if self.params['gate_enabled']:
                self.gate_k.write(':OUTP ON')
                if not self._ramp_voltage(self.gate_k, self.params['g_target'], self.params['g_ramp_step'],
                                          self.params['g_step_delay'], is_gate=True):
                    return
                if not self._sleep(self.params['g_settle'], '栅压稳定等待中'):
                    return
                self.gate_monitor_stop.clear()
                self.gate_k.timeout = min(int(self.gate_k.timeout), 1200)
                self.gate_monitor = threading.Thread(
                    target=self.gate_monitor_thread,
                    name='bias-switch-gate-monitor',
                    daemon=False,
                )
                self.gate_monitor.start()

            self.bias_k.write(':OUTP ON')
            if not self._ramp_voltage(self.bias_k, self.params['test_v'], self.params['b_ramp_step'], 0.01, is_gate=False):
                return

            warmup_read_times = []
            for _ in range(10):
                if self.stop_event.is_set() or self.force_stop_event.is_set():
                    return
                warmup_started = time.perf_counter()
                required_float_query(
                    self.bias_k, ':READ?', '偏压开关预热读数'
                )
                warmup_read_times.append(
                    time.perf_counter() - warmup_started
                )

            self.update_queue.put(('stage', '方波测量中'))

            phase_voltages = [
                self.params['test_v'],
                self.params['sw1_v'],
                self.params['test_v'],
                self.params['sw2_v'],
            ]

            p_duration = self.params['phase_duration']
            p_settle = self.params['pre_switch_settle']
            cycles = int(self.params['cycles'])
            plot_interval = self.params['plot_interval']
            batch_t, batch_v, batch_i = [], [], []

            t_start = time.perf_counter()
            est_read_time = max(
                sum(warmup_read_times) / len(warmup_read_times),
                1e-4,
            )

            for cycle in range(1, cycles + 1):
                for target_v in phase_voltages:
                    if self.stop_event.is_set() or self.force_stop_event.is_set():
                        break

                    self.bias_k.write(f':SOUR:VOLT {target_v}')
                    if p_settle > 0:
                        self.update_queue.put(('stage', '切换前等待中'))
                        if not self._sleep(p_settle, "切换前等待中"):
                            break
                    absolute_deadline = time.perf_counter() + p_duration

                    while True:
                        now = time.perf_counter()
                        remaining = absolute_deadline - now

                        # Do not skip a whole voltage phase because the previous
                        # phase's first (range-settling) read was unusually slow.
                        # A read may finish just after the nominal boundary, but
                        # every requested level must contribute real data.
                        if remaining <= 0:
                            break

                        if self.stop_event.is_set() or self.force_stop_event.is_set():
                            break

                        t0 = time.perf_counter()
                        reading = required_float_query(
                            self.bias_k, ':READ?', '偏压开关正式电流读数'
                        )
                        t1 = time.perf_counter()

                        actual_read_time = t1 - t0
                        est_read_time = 0.7 * est_read_time + 0.3 * actual_read_time

                        rel_time = (t0 + t1) / 2 - t_start
                        times.append(rel_time)
                        v_outs.append(target_v)
                        i_outs.append(reading)
                        batch_t.append(rel_time)
                        batch_v.append(target_v)
                        batch_i.append(reading)

                        if len(batch_t) >= plot_interval:
                            self.update_queue.put(
                                ('data_batch', vg, batch_t, batch_v, batch_i, cycle, cycles))
                            batch_t, batch_v, batch_i = [], [], []

                    while time.perf_counter() < absolute_deadline:
                        if self.stop_event.is_set() or self.force_stop_event.is_set():
                            break
                        if not self._sleep(0.0005, ""):
                            break

                if self.stop_event.is_set() or self.force_stop_event.is_set():
                    break

            if batch_t:
                self.update_queue.put(
                    ('data_batch', vg, batch_t, batch_v, batch_i, cycles, cycles))

            prefix = self.params['prefix']
            testv = self.params['test_v']
            sw1v = self.params['sw1_v']
            sw2v = self.params['sw2_v']
            fname = (
                f"{prefix}_Vg={vg:g}V_test{testv:g}V_V1={sw1v:g}V_V2={sw2v:g}V_"
                f"{p_duration}s_{cycles}cycle.txt"
            )
            result_status = (
                'partial'
                if self.stop_event.is_set() or self.force_stop_event.is_set()
                else 'complete'
            )
            self.update_queue.put(
                ('block_done', vg, times, v_outs, i_outs, fname,
                 result_status,
                 None if result_status == 'complete' else '用户停止或强制终止'))

        except Exception as exc:
            self.update_queue.put(('log', f"测量中断或出错: {exc}"))
            if times:
                prefix = self.params['prefix']
                fname = (
                    f"{prefix}_Vg={vg:g}V_test{self.params['test_v']:g}V_"
                    f"V1={self.params['sw1_v']:g}V_V2={self.params['sw2_v']:g}V_"
                    f"{self.params['phase_duration']}s_{int(self.params['cycles'])}cycle.txt"
                )
                self.update_queue.put((
                    'block_done', vg, times, v_outs, i_outs,
                    fname, 'partial', exc,
                ))
        finally:
            monitor_stopped = self._stop_gate_monitor()
            try:
                self.safe_zeroing()
            except Exception as exc:
                self.update_queue.put(('log', f'安全归零失败: {exc}'))
            bias_confirmed, bias_failures = reliable_output_off(
                self.bias_k, '偏压开关偏压表'
            )
            gate_confirmed = True
            gate_failures = []
            if self.gate_k and monitor_stopped:
                gate_confirmed, gate_failures = reliable_output_off(
                    self.gate_k, '偏压开关栅压表'
                )
            elif self.gate_k:
                gate_confirmed = False
                gate_failures = ['栅电流监视线程仍占用VISA资源']
            if not bias_confirmed or not gate_confirmed:
                self.update_queue.put((
                    'log',
                    '【严重警告】无法确认源表输出已关闭，请立即从仪器面板确认。 '
                    + ' | '.join(bias_failures + gate_failures),
                ))
            if self.bias_k:
                try:
                    self.bias_k.close()
                except Exception:
                    pass
            if self.gate_k and monitor_stopped:
                try:
                    self.gate_k.close()
                except Exception:
                    pass

            self.update_queue.put(('finished', None))


class BiasSwitchWidget(BaseAppWidget):
    def __init__(self, run_guard=None, parent=None):
        configure_pyqtgraph(use_opengl=False)
        super().__init__(run_guard=run_guard, parent=parent)

        self.module_id = 'bias_switch'
        self.module_name = '偏压开关测试'

        self.ui_font = QFont('Arial', 12)
        self.ui_font.setWeight(QFont.Weight.Normal)
        self.bold_font = QFont('Arial', 12)
        self.bold_font.setWeight(QFont.Weight.Bold)
        self.setFont(self.ui_font)

        self.capacity = 2000000
        self.data_count = 0
        self.time_data = np.zeros(self.capacity)
        self.v_data = np.zeros(self.capacity)
        self.i_data = np.zeros(self.capacity)
        self.points_changed = False

        self.current_folder = ''

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=3)

        self.graph_widget = pg.GraphicsLayoutWidget()
        left_layout.addWidget(self.graph_widget)
        label_style = {'color': '#000', 'font-size': '12pt'}

        self.plot_v = self.graph_widget.addPlot(title='Bias Voltage vs Time')
        self.plot_v.setLabel('left', text='Bias Voltage', units='V', **label_style)
        self.plot_v.setLabel('bottom', text='Time', units='s', **label_style)
        self.plot_v.getAxis('left').setTickFont(self.ui_font)
        self.plot_v.getAxis('bottom').setTickFont(self.ui_font)
        self.plot_v.showGrid(x=True, y=True, alpha=0.3)
        self.plot_v.setClipToView(True)
        self.plot_v.setDownsampling(auto=True, mode='peak')
        self.curve_v = self.plot_v.plot(pen=pg.mkPen('r', width=1.5))

        self.graph_widget.nextRow()

        self.plot_i = self.graph_widget.addPlot(title='Bias Current vs Time')
        self.plot_i.setLabel('left', text='Bias Current', units='A', **label_style)
        self.plot_i.setLabel('bottom', text='Time', units='s', **label_style)
        self.plot_i.getAxis('left').setTickFont(self.ui_font)
        self.plot_i.getAxis('bottom').setTickFont(self.ui_font)
        self.plot_i.showGrid(x=True, y=True, alpha=0.3)
        self.plot_i.setClipToView(True)
        self.plot_i.setDownsampling(auto=True, mode='peak')
        self.curve_i = self.plot_i.plot(pen=pg.mkPen('b', width=1.5))
        self.plot_i.setXLink(self.plot_v)

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
        self.c_b_addr = NoScrollComboBox()
        self.c_b_addr.setFont(self.ui_font)
        self.c_b_addr.setStyleSheet('font-weight: normal;')
        self.c_b_addr.setEditable(True)
        self.c_b_addr.addItem('GPIB0::1::INSTR')
        self.c_b_addr.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.inputs['bias_address'] = self.c_b_addr
        addr_grid.addWidget(self.c_b_addr, 0, 1)

        lbl_gaddr = QLabel('Gate 表地址:')
        lbl_gaddr.setFont(self.ui_font)
        lbl_gaddr.setStyleSheet('font-weight: normal;')
        addr_grid.addWidget(lbl_gaddr, 0, 2)
        self.c_g_addr = NoScrollComboBox()
        self.c_g_addr.setFont(self.ui_font)
        self.c_g_addr.setStyleSheet('font-weight: normal;')
        self.c_g_addr.setEditable(True)
        self.c_g_addr.addItem('GPIB0::2::INSTR')
        self.c_g_addr.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.inputs['gate_address'] = self.c_g_addr
        addr_grid.addWidget(self.c_g_addr, 0, 3)

        lbl_bterm = QLabel('Bias 表端口:')
        lbl_bterm.setFont(self.ui_font)
        lbl_bterm.setStyleSheet('font-weight: normal;')
        addr_grid.addWidget(lbl_bterm, 1, 0)
        self.c_b_term = NoScrollComboBox()
        self.c_b_term.setFont(self.ui_font)
        self.c_b_term.setStyleSheet('font-weight: normal;')
        self.c_b_term.addItems(['REAR', 'FRONT'])
        self.c_b_term.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.inputs['bias_terminal'] = self.c_b_term
        addr_grid.addWidget(self.c_b_term, 1, 1)

        lbl_gterm = QLabel('Gate 表端口:')
        lbl_gterm.setFont(self.ui_font)
        lbl_gterm.setStyleSheet('font-weight: normal;')
        addr_grid.addWidget(lbl_gterm, 1, 2)
        self.c_g_term = NoScrollComboBox()
        self.c_g_term.setFont(self.ui_font)
        self.c_g_term.setStyleSheet('font-weight: normal;')
        self.c_g_term.addItems(['REAR', 'FRONT'])
        self.c_g_term.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.inputs['gate_terminal'] = self.c_g_term
        addr_grid.addWidget(self.c_g_term, 1, 3)

        btn_scan = QPushButton('扫描设备')
        btn_scan.setFont(self.bold_font)
        btn_scan.setFixedSize(100, 30)
        btn_scan.clicked.connect(self.scan_instruments)
        addr_grid.addWidget(btn_scan, 2, 0, 1, 4,
                            alignment=Qt.AlignmentFlag.AlignCenter)
        box_vbox.addWidget(addr_box)

        self.cb_gate = QCheckBox('启用栅表 (勾选展示栅压参数)')
        self.cb_gate.setFont(self.ui_font)
        self.cb_gate.setStyleSheet('font-weight: normal;')
        self.cb_gate.stateChanged.connect(self.toggle_gate)
        box_vbox.addWidget(self.cb_gate)

        self.gate_box = QGroupBox('栅压参数')
        self.gate_box.setFont(self.bold_font)
        gate_grid = QGridLayout(self.gate_box)
        gate_items = [
            ('目标栅压 (V):', 'g_target', '0.5', '栅压 NPLC:', 'g_nplc', '1.0'),
            ('爬坡/归零步长 (V) (正):', 'g_ramp_step', '0.1',
             '栅压单步延时 (s):', 'g_step_delay', '0.5'),
            ('栅压限流 (A):', 'g_ilimit', '1e-9', '栅压到位等待 (s):', 'g_settle', '20.0'),
        ]
        for i, (l1, k1, d1, l2, k2, d2) in enumerate(gate_items):
            lbl1 = QLabel(l1)
            lbl1.setFont(self.ui_font)
            lbl1.setStyleSheet('font-weight: normal;')
            gate_grid.addWidget(lbl1, i, 0)
            e1 = QLineEdit(d1)
            e1.setFont(self.ui_font)
            e1.setStyleSheet('font-weight: normal;')
            self.inputs[k1] = e1
            gate_grid.addWidget(e1, i, 1)
            lbl2 = QLabel(l2)
            lbl2.setFont(self.ui_font)
            lbl2.setStyleSheet('font-weight: normal;')
            gate_grid.addWidget(lbl2, i, 2)
            e2 = QLineEdit(d2)
            e2.setFont(self.ui_font)
            e2.setStyleSheet('font-weight: normal;')
            self.inputs[k2] = e2
            gate_grid.addWidget(e2, i, 3)
        box_vbox.addWidget(self.gate_box)
        self.gate_box.setVisible(False)

        bias_box = QGroupBox('偏压参数')
        bias_box.setFont(self.bold_font)
        bias_grid = QGridLayout(bias_box)
        bias_items = [
            ('测试偏压 (V):', 'test_v', '0.05', '偏压电流量程 (A / AUTO):', 'b_range', '1e-5'),
            ('开关偏压 1 (V):', 'sw1_v', '0.1', '偏压 NPLC:', 'b_nplc', '0.1'),
            ('开关偏压 2 (V):', 'sw2_v', '-0.1', '循环次数 (Cycles):', 'cycles', '1'),
            ('爬坡/归零步长 (V) (正):', 'b_ramp_step', '0.001',
             '每段时间 (s):', 'phase_duration', '5.0'),
            ('偏压限流 (A):', 'b_ilimit', '1.05e-5',
             '切换前等待时间 (s):', 'pre_switch_settle', '0.0'),
        ]
        for i, (l1, k1, d1, l2, k2, d2) in enumerate(bias_items):
            lbl1 = QLabel(l1)
            lbl1.setFont(self.ui_font)
            lbl1.setStyleSheet('font-weight: normal;')
            bias_grid.addWidget(lbl1, i, 0)
            e1 = QLineEdit(d1)
            e1.setFont(self.ui_font)
            e1.setStyleSheet('font-weight: normal;')
            self.inputs[k1] = e1
            bias_grid.addWidget(e1, i, 1)
            lbl2 = QLabel(l2)
            lbl2.setFont(self.ui_font)
            lbl2.setStyleSheet('font-weight: normal;')
            bias_grid.addWidget(lbl2, i, 2)
            e2 = QLineEdit(d2)
            e2.setFont(self.ui_font)
            e2.setStyleSheet('font-weight: normal;')
            self.inputs[k2] = e2
            bias_grid.addWidget(e2, i, 3)
        box_vbox.addWidget(bias_box)

        path_box = QGroupBox('作图与文件保存路径')
        path_box.setFont(self.bold_font)
        path_grid = QGridLayout(path_box)

        self.cb_plot = QCheckBox('启用实时绘图 (关闭则测量结束后一次性爆发绘制)')
        self.cb_plot.setFont(self.ui_font)
        self.cb_plot.setStyleSheet('font-weight: normal;')
        self.cb_plot.setChecked(True)
        path_grid.addWidget(self.cb_plot, 0, 0, 1, 2)

        lbl_int = QLabel('绘图更新间隔 (点数):')
        lbl_int.setFont(self.ui_font)
        lbl_int.setStyleSheet('font-weight: normal;')
        path_grid.addWidget(lbl_int, 1, 0)
        self.inputs['plot_interval'] = QLineEdit('10')
        self.inputs['plot_interval'].setFont(self.ui_font)
        self.inputs['plot_interval'].setStyleSheet('font-weight: normal;')
        path_grid.addWidget(self.inputs['plot_interval'], 1, 1)

        lbl_prefix = QLabel(
            '文件名前缀 (后缀自动追加栅压、偏压、时间、循环次数):'
        )
        lbl_prefix.setFont(self.ui_font)
        lbl_prefix.setStyleSheet('font-weight: normal;')
        path_grid.addWidget(lbl_prefix, 2, 0, 1, 2)
        self.inputs['prefix'] = QLineEdit('Switch1')
        self.inputs['prefix'].setFont(self.ui_font)
        self.inputs['prefix'].setStyleSheet('font-weight: normal;')
        path_grid.addWidget(self.inputs['prefix'], 3, 0, 1, 2)

        lbl_folder = QLabel('保存文件夹:')
        lbl_folder.setFont(self.ui_font)
        lbl_folder.setStyleSheet('font-weight: normal;')
        path_grid.addWidget(lbl_folder, 4, 0, 1, 2)

        fhbox = QHBoxLayout()
        fhbox.setContentsMargins(0, 0, 0, 0)
        self.ent_folder = QLineEdit(default_data_directory("Bias_Switch"))
        self.ent_folder.setFont(self.ui_font)
        self.ent_folder.setStyleSheet('font-weight: normal;')
        fhbox.addWidget(self.ent_folder)
        btn_br = QPushButton('浏览')
        btn_br.setFont(self.ui_font)
        btn_br.setStyleSheet('font-weight: normal;')
        btn_br.clicked.connect(self.browse_folder)
        fhbox.addWidget(btn_br)
        path_grid.addLayout(fhbox, 5, 0, 1, 2)

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

        try:
            preset = {
                'gate_enabled': self.cb_gate.isChecked(),
                'plot_interval': int(self.inputs['plot_interval'].text().strip()),
                'bias_address': self.c_b_addr.currentText().strip(),
                'gate_address': self.c_g_addr.currentText().strip(),
                'bias_terminal': self.c_b_term.currentText().strip(),
                'gate_terminal': self.c_g_term.currentText().strip(),
                'prefix': self.inputs['prefix'].text().strip(),
            }

            if preset['gate_enabled']:
                preset['g_target'] = float(
                    self.inputs['g_target'].text().strip())
                preset['g_nplc'] = float(self.inputs['g_nplc'].text().strip())
                preset['g_ramp_step'] = float(
                    self.inputs['g_ramp_step'].text().strip())
                if preset['g_ramp_step'] <= 0:
                    raise ValueError(
                        f'栅压爬坡步长必须为正值，当前值: {self.inputs["g_ramp_step"].text().strip()}')
                preset['g_step_delay'] = float(
                    self.inputs['g_step_delay'].text().strip())
                preset['g_ilimit'] = float(
                    self.inputs['g_ilimit'].text().strip())
                preset['g_settle'] = float(
                    self.inputs['g_settle'].text().strip())

            preset['test_v'] = float(self.inputs['test_v'].text().strip())
            preset['sw1_v'] = float(self.inputs['sw1_v'].text().strip())
            preset['sw2_v'] = float(self.inputs['sw2_v'].text().strip())
            preset['b_ramp_step'] = float(
                self.inputs['b_ramp_step'].text().strip())
            if preset['b_ramp_step'] <= 0:
                raise ValueError(
                    f'偏压爬坡步长必须为正值，当前值: {self.inputs["b_ramp_step"].text().strip()}')
            preset['b_ilimit'] = float(self.inputs['b_ilimit'].text().strip())

            r_val = self.inputs['b_range'].text().strip()
            preset['b_range'] = r_val if r_val.upper(
            ) == 'AUTO' else float(r_val)
            preset['b_nplc'] = float(self.inputs['b_nplc'].text().strip())
            preset['cycles'] = int(self.inputs['cycles'].text().strip())

            preset['phase_duration'] = float(
                self.inputs['phase_duration'].text().strip())
            if preset['phase_duration'] < 0.05:
                raise ValueError('每段时间不能小于 0.05s！')

            preset['pre_switch_settle'] = float(
                self.inputs['pre_switch_settle'].text().strip())
            if preset['plot_interval'] <= 0:
                raise ValueError('界面刷新间隔必须为正整数')
            if preset['cycles'] <= 0:
                raise ValueError('循环次数必须大于 0')
            if preset['b_nplc'] <= 0:
                raise ValueError('偏压 NPLC 必须大于 0')
            if preset['b_ilimit'] <= 0:
                raise ValueError('偏压限流必须大于 0')
            if not isinstance(preset['b_range'], str) and preset['b_range'] <= 0:
                raise ValueError('电流量程必须大于 0 或为 AUTO')
            if preset['pre_switch_settle'] < 0:
                raise ValueError('切换前等待时间不能为负值')
            if preset['gate_enabled']:
                if preset['g_nplc'] <= 0:
                    raise ValueError('栅压 NPLC 必须大于 0')
                if preset['g_step_delay'] < 0:
                    raise ValueError('栅压单步延时不能为负值')
                if preset['g_ilimit'] <= 0:
                    raise ValueError('栅压限流必须大于 0')
                if preset['g_settle'] < 0:
                    raise ValueError('栅压到位等待不能为负值')
            validate_program_step_plan('bias_switch', preset)

            folder = self.ent_folder.text().strip()
            self.current_folder = folder

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
        self.time_data.fill(0)
        self.v_data.fill(0)
        self.i_data.fill(0)
        self.curve_v.setData([], [])
        self.curve_i.setData([], [])

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
        SwitchMeasurement(
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
                if '严重警告' in msg[_1]:
                    self.raise_persistent_safety_alarm(msg[_1])
                else:
                    self.log_info(msg[_1])

            elif msg_type == 'stage':
                self.status_labels['stage'].setText(msg[_1])

            elif msg_type == 'ramp_b':
                self.status_labels['bias_v'].setText(f"{msg[_1]:.6f}")
                self.status_labels['bias_i'].setText(f"{msg[_2]:.6e}")

            elif msg_type == 'ramp_g':
                self.status_labels['gate_v'].setText(f"{msg[_1]:.6f}")

            elif msg_type == 'gate_leakage':
                self.status_labels['gate_i'].setText(f"{msg[_1]:.6e}")

            elif msg_type == 'data_batch':
                target_vg, batch_t, batch_v, batch_i, cycle, total_cycles = (
                    msg[_1], msg[_2], msg[_3], msg[_4], msg[_5], msg[_6]
                )

                self.status_labels['stage'].setText(
                    f"方波测量中 ({cycle}/{total_cycles})")

                if batch_t:
                    self.status_labels['gate_v'].setText(f"{target_vg:.6f}")
                    self.status_labels['bias_v'].setText(f"{batch_v[-1]:.6f}")
                    self.status_labels['bias_i'].setText(f"{batch_i[-1]:.6e}")
                    self.status_labels['time'].setText(f"{batch_t[-1]:.3f}")

                    new_count = len(batch_t)
                    self.status_labels['count'].setText(
                        str(self.data_count + new_count))

                    needed = self.data_count + new_count
                    if needed > self.capacity:
                        new_capacity = max(needed, self.capacity * 2)
                        grow = new_capacity - self.capacity
                        self.time_data = np.pad(self.time_data, (0, grow))
                        self.v_data = np.pad(self.v_data, (0, grow))
                        self.i_data = np.pad(self.i_data, (0, grow))
                        self.capacity = new_capacity
                        self.log_info(f'绘图缓存已自动扩容至 {new_capacity} 点')
                    self.time_data[self.data_count:needed] = batch_t
                    self.v_data[self.data_count:needed] = batch_v
                    self.i_data[self.data_count:needed] = batch_i
                    self.data_count = needed
                    self.points_changed = True

            elif msg_type == 'block_done':
                vg, times, v_outs, i_outs, fname = msg[_1], msg[_2], msg[_3], msg[_4], msg[_5]
                status = msg[_6] if len(msg) > 6 else 'complete'
                error = msg[_7] if len(msg) > 7 else None
                self.note_result_status(status, error)
                rate = len(times) / \
                    times[-1] if len(times) > 1 and times[-1] > 0 else 0
                self.status_labels['rate'].setText(f"{rate:.2f}")
                self.submit_save(
                    self.save_data,
                    vg, times, v_outs, i_outs, fname,
                    self.current_folder,
                    status=status, error=error,
                    gate_enabled=self.cb_gate.isChecked(),
                    stopped_at_local=time.strftime('%Y-%m-%d %H:%M:%S'),
                )

                if not self.cb_plot.isChecked() and self.data_count > 0:
                    self.curve_v.setData(
                        self.time_data[:self.data_count], self.v_data[:self.data_count])
                    self.curve_i.setData(
                        self.time_data[:self.data_count], self.i_data[:self.data_count])

            c += 1

    def update_plot(self):
        if self.points_changed and self.data_count > 0:
            if self.cb_plot.isChecked():
                self.curve_v.setData(
                    self.time_data[:self.data_count], self.v_data[:self.data_count])
                self.curve_i.setData(
                    self.time_data[:self.data_count], self.i_data[:self.data_count])
            self.points_changed = False

    def save_data(
        self, vg, times, v_outs, i_outs, filename, output_folder,
        status='complete', error=None, gate_enabled=False,
        stopped_at_local=None,
    ):
        if status != 'complete':
            stem, suffix = os.path.splitext(filename)
            filename = f'{stem}_partial{suffix}'
        filepath = allocate_unique_path(output_folder, filename)
        try:
            with atomic_text_writer(filepath) as file_obj:
                if gate_enabled:
                    file_obj.write(
                        'Time(s)\tGateVoltage(V)\tBiasVoltage(V)\tBiasCurrent(A)\n')
                    for t, v, i_val in zip(times, v_outs, i_outs):
                        file_obj.write(
                            f"{t:.6f}\t{vg:.6f}\t{v:.6f}\t{i_val:.6e}\n")
                else:
                    file_obj.write('Time(s)\tBiasVoltage(V)\tBiasCurrent(A)\n')
                    for t, v, i_val in zip(times, v_outs, i_outs):
                        file_obj.write(f"{t:.6f}\t{v:.6f}\t{i_val:.6e}\n")
            write_result_metadata(
                filepath, status=status, point_count=len(times), error=error,
                stopped_at_local=stopped_at_local,
            )
            self.post_log(f"全循环方波序列数据已落盘保存至: {filepath}")
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
