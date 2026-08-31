import os
import time
import numpy as np
import pyvisa

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QLineEdit, QPushButton, QGroupBox, QTextEdit, QScrollArea, QMessageBox, QSizePolicy, QDialog, QComboBox,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont

import pyqtgraph as pg

from core.app_base import BaseAppWidget
from core.paths import DataRootSettings
from core.hardware_base import (
    allocate_unique_path,
    atomic_text_writer,
    clear_scpi_status,
    check_gate_current_limit,
    configure_gate_meter,
    configure_current_autozero,
    fast_shutdown_zero_2450,
    reliable_output_off,
    shutdown_report_confirmed,
    required_float_query,
    validate_2450_idn,
    validate_current_range_limit,
    validate_distinct_addresses,
    validate_nplc,
    validate_positive_step,
    validate_program_step_plan,
    validate_step_divides_interval,
    validate_source_voltage,
    validate_terminal,
    validate_voltage_range,
    validate_gate_voltage_within_range,
    verify_current_configuration,
    write_result_metadata,
    GATE_CURRENT_RANGE_A,
    GateCurrentLimitError,
)
from core.instrument_config import InstrumentSettings
from core.ui_builder import (
    bind_range_to_limit, combo_config_value, configure_output_path,
    configure_parameter_grid, create_current_range_combo, create_status_group,
    style_parameter_control, style_parameter_label,
)
from core.time_acquisition import (
    ACQUISITION_TRIGGERED,
    GATE_MONITOR_NPLC,
    InternalSegmentCollector,
    RealtimeSampler,
    create_sampling_settings,
    selected_acquisition_mode,
    timing_metadata,
)
from core.utils import _0, _1, _2, _3, _4, _5, _6, configure_pyqtgraph


class WaveformEditorDialog(QDialog):
    def __init__(self, parent, waveform, ui_font, bold_font, ramp_step=None):
        super().__init__(parent)
        self.setWindowTitle("自定义栅压波形编辑器")
        self.resize(1000, 600)
        self.setWindowFlags(self.windowFlags(
        ) | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint)

        self.ui_font = ui_font
        self.bold_font = bold_font
        self.ramp_step = ramp_step

        self.waveform = [row[:] for row in waveform]
        self.changed = False
        self.row_widgets = []

        self.init_ui()
        self.populate_table()
        self.update_preview()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        left_group = QGroupBox("栅压波形定义")
        left_group.setFont(self.bold_font)
        left_layout = QVBoxLayout(left_group)

        header_widget = QWidget()
        h_layout = QHBoxLayout(header_widget)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(6)

        lbl_idx = QLabel("序号")
        lbl_idx.setFont(self.bold_font)
        lbl_idx.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_dur = QLabel("时长 (s)")
        lbl_dur.setFont(self.bold_font)
        lbl_dur.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_vol = QLabel("栅压 (V)")
        lbl_vol.setFont(self.bold_font)
        lbl_vol.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_op = QLabel("操作")
        lbl_op.setFont(self.bold_font)
        lbl_op.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_op.setFixedWidth(66)

        h_layout.addWidget(lbl_idx, 1)
        h_layout.addWidget(lbl_dur, 3)
        h_layout.addWidget(lbl_vol, 3)
        h_layout.addWidget(lbl_op, 0)
        left_layout.addWidget(header_widget)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll.setWidget(self.scroll_content)
        left_layout.addWidget(scroll)
        main_layout.addWidget(left_group, stretch=2)

        right_group = QGroupBox("栅压波形预览")
        right_group.setFont(self.bold_font)
        right_layout = QVBoxLayout(right_group)

        self.graph_widget = pg.GraphicsLayoutWidget()
        right_layout.addWidget(self.graph_widget)

        self.plot_preview = self.graph_widget.addPlot()
        self.plot_preview.setLabel(
            'left', text='Gate Voltage', units='V', **{'color': '#000', 'font-size': '11pt'})
        self.plot_preview.setLabel(
            'bottom', text='Time', units='s', **{'color': '#000', 'font-size': '11pt'})
        self.plot_preview.showGrid(x=True, y=True, alpha=0.4)
        self.curve_preview = self.plot_preview.plot(
            pen=pg.mkPen('r', width=2.0))

        dialog_btn_layout = QHBoxLayout()
        btn_ok = QPushButton("确认")
        btn_ok.setFont(self.bold_font)
        btn_ok.setFixedSize(100, 30)
        btn_ok.setStyleSheet("color: #AA0000;")
        btn_ok.clicked.connect(self.on_ok)

        btn_cancel = QPushButton("取消")
        btn_cancel.setFont(self.bold_font)
        btn_cancel.setFixedSize(100, 30)
        btn_cancel.clicked.connect(self.reject)

        dialog_btn_layout.addStretch()
        dialog_btn_layout.addWidget(btn_cancel)
        dialog_btn_layout.addWidget(btn_ok)
        right_layout.addLayout(dialog_btn_layout)

        main_layout.addWidget(right_group, stretch=3)

    def create_row_widget(self, index, duration, voltage):
        row_widget = QWidget()
        layout = QHBoxLayout(row_widget)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(6)

        lbl_idx = QLabel(str(index))
        lbl_idx.setFont(self.ui_font)
        lbl_idx.setAlignment(Qt.AlignmentFlag.AlignCenter)

        ent_d = QLineEdit(str(duration))
        ent_d.setFont(self.ui_font)
        ent_d.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ent_d.setSizePolicy(QSizePolicy.Policy.Expanding,
                            QSizePolicy.Policy.Fixed)
        ent_d.textChanged.connect(self.update_preview)

        ent_v = QLineEdit(str(voltage))
        ent_v.setFont(self.ui_font)
        ent_v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ent_v.setSizePolicy(QSizePolicy.Policy.Expanding,
                            QSizePolicy.Policy.Fixed)
        ent_v.textChanged.connect(self.update_preview)

        btn_add = QPushButton("+")
        btn_add.setFont(self.bold_font)
        btn_add.setFixedSize(30, 30)

        btn_del = QPushButton("-")
        btn_del.setFont(self.bold_font)
        btn_del.setFixedSize(30, 30)

        layout.addWidget(lbl_idx, 1)
        layout.addWidget(ent_d, 3)
        layout.addWidget(ent_v, 3)
        layout.addWidget(btn_add, 0)
        layout.addWidget(btn_del, 0)

        return row_widget, lbl_idx, ent_d, ent_v, btn_add, btn_del

    def add_row_at(self, index, duration, voltage):
        rw, lbl, ed, ev, btn_add, btn_del = self.create_row_widget(
            0, duration, voltage)
        btn_add.clicked.connect(lambda _, w=rw: self.insert_row_after(w))
        btn_del.clicked.connect(lambda _, w=rw: self.delete_row(w))
        self.scroll_layout.insertWidget(index, rw)
        self.row_widgets.insert(index, (rw, lbl, ed, ev))
        self.renumber_rows()
        self.update_preview()

    def insert_row_after(self, target_rw):
        for i, (rw, lbl, ed, ev) in enumerate(self.row_widgets):
            if rw == target_rw:
                self.add_row_at(i + 1, 1.0, 0.0)
                break

    def delete_row(self, target_rw):
        if len(self.row_widgets) <= 1:
            QMessageBox.warning(self, "提示", "请至少保留一段波形！")
            return
        for i, (rw, lbl, ed, ev) in enumerate(self.row_widgets):
            if rw == target_rw:
                self.scroll_layout.removeWidget(rw)
                rw.deleteLater()
                self.row_widgets.pop(i)
                break
        self.renumber_rows()
        self.update_preview()

    def renumber_rows(self):
        for i, (rw, lbl, ed, ev) in enumerate(self.row_widgets):
            lbl.setText(str(i + 1))

    def populate_table(self):
        for i, (voltage, duration) in enumerate(self.waveform):
            self.add_row_at(i, duration, voltage)

    def extract_data(self):
        wf = []
        for rw, lbl, ent_d, ent_v in self.row_widgets:
            try:
                d = float(ent_d.text().strip())
                v = float(ent_v.text().strip())
                if d <= 0:
                    continue
                wf.append([v, d])
            except ValueError:
                pass
        return wf

    def update_preview(self):
        wf = self.extract_data()
        if not wf:
            self.curve_preview.setData([], [])
            self.plot_preview.setTitle("波形预览 (无有效数据)")
            return

        times = [0.0]
        voltages = [wf[_0][_0]]
        cum_time = 0.0

        for i in range(len(wf)):
            v = wf[i][_0]
            d = wf[i][_1]
            cum_time += d
            times.append(cum_time)
            voltages.append(v)
            if i < len(wf) - 1:
                next_v = wf[i+1][_0]
                times.append(cum_time)
                voltages.append(next_v)

        self.curve_preview.setData(times, voltages)
        self.plot_preview.setTitle(
            f"栅压波形预览 (共 {len(wf)} 段, 单次总时长 {cum_time:.2f} s)", size="12pt")

    def on_ok(self):
        wf = self.extract_data()
        if not wf:
            QMessageBox.warning(self, "警告", "请至少保留一段有效的波形！")
            return
        if self.ramp_step is not None:
            try:
                for index, (voltage, _duration) in enumerate(wf, 1):
                    validate_step_divides_interval(
                        0, voltage, self.ramp_step,
                        f'任意栅压第{index}段',
                    )
            except ValueError as exc:
                QMessageBox.warning(
                    self, '波形与爬坡步长不匹配',
                    f'{exc}\n\n请修改该段电压，或先关闭编辑器修改模块中的栅压爬坡步长。',
                )
                return
        self.waveform = wf
        self.changed = True
        self.accept()


class GateArbMeasurement:
    def __init__(self, params, update_queue, stop_event, force_stop_event):
        self.params = params
        self.update_queue = update_queue
        self.stop_event = stop_event
        self.force_stop_event = force_stop_event
        self.bias_k = None
        self.gate_k = None

    def connect(self):
        try:
            validate_distinct_addresses(
                self.params['bias_address'], self.params['gate_address'], True
            )
            rm = pyvisa.ResourceManager()
            self.bias_k = rm.open_resource(self.params['bias_address'])
            self.bias_k.timeout = 10000
            bias_idn = validate_2450_idn(self.bias_k.query('*IDN?'))
            self.update_queue.put(('log', f"偏压表连接成功: {bias_idn}"))

            self.gate_k = rm.open_resource(self.params['gate_address'])
            self.gate_k.timeout = 10000
            gate_idn = validate_2450_idn(self.gate_k.query('*IDN?'))
            self.update_queue.put(('log', f"栅压表连接成功: {gate_idn}"))
        except Exception as e:
            self.update_queue.put(('log', f"仪器连接失败: {repr(e)}"))
            raise

    def setup(self):
        try:
            validate_nplc(self.params['sample_nplc'], '测量NPLC')
            validate_terminal(self.params['bias_terminal'], '偏压端子')
            validate_terminal(self.params['gate_terminal'], '栅压端子')
            validate_source_voltage(self.params['b_target'], '偏压目标')
            for voltage, _duration in self.params['waveform']:
                validate_source_voltage(voltage, '任意栅压波形电压')
            validate_positive_step(self.params['b_ramp_step'], '偏压爬坡步长')
            validate_positive_step(self.params['g_ramp_step'], '栅压爬坡步长')
            validate_program_step_plan('arbitrary_gate', self.params)
            validate_current_range_limit(
                self.params['b_range'], self.params['b_ilimit'], '偏压'
            )
            check_gate_current_limit(0.0, self.params['g_ilimit'])
            validate_voltage_range(
                self.params['g_voltage_range'], '栅压量程'
            )
            for voltage, _duration in self.params['waveform']:
                validate_gate_voltage_within_range(
                    voltage, self.params['g_voltage_range'], '任意栅压波形电压'
                )
            k_b = self.bias_k
            k_b.write('*RST')
            clear_scpi_status(k_b)
            k_b.write(':ABORt')
            k_b.write(':SOUR:FUNC VOLT')
            k_b.write(':SENS:FUNC "CURR"')
            k_b.write('SENS:CURR:RSEN OFF')
            k_b.write(f':ROUT:TERM {self.params["bias_terminal"]}')
            k_b.write(':SENS:CURR:AVER OFF')
            k_b.write(':SOUR:VOLT:READ:BACK OFF')
            k_b.write(f':SENS:CURR:NPLC {self.params["sample_nplc"]}')

            r_val = self.params['b_range']
            if str(r_val).upper() == 'AUTO':
                k_b.write(':SENS:CURR:RANG:AUTO ON')
            else:
                k_b.write(':SENS:CURR:RANG:AUTO OFF')
                k_b.write(f':SENS:CURR:RANG {r_val}')
            configure_current_autozero(k_b, 'block_once')

            k_b.write(f':SOUR:VOLT:ILIM {self.params["b_ilimit"]}')
            k_b.write(':SOUR:VOLT 0')

            k_g = self.gate_k
            configure_gate_meter(
                k_g,
                voltage_range=self.params['g_voltage_range'],
                current_limit=self.params['g_ilimit'],
                nplc=GATE_MONITOR_NPLC,
                terminal=self.params['gate_terminal'],
                autozero_mode='block_once',
                label='任意栅压表',
            )

            if not self._sleep(0.05, "仪器初始化中"):
                return
            verify_current_configuration(
                k_b,
                nplc=self.params['sample_nplc'],
                current_range=self.params['b_range'],
                current_limit=self.params['b_ilimit'],
                terminal=self.params['bias_terminal'],
                autozero_mode='block_once',
                label='任意栅压偏压表',
            )
        except Exception as e:
            self.update_queue.put(('log', f"仪器初始化错误: {e}"))
            raise

    def _sleep(self, duration_s, status_msg):
        self.update_queue.put(('stage', status_msg))
        if duration_s <= 0:
            return True
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

        for i in range(1, steps+1):
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
                if self.params['acquisition_mode'] != ACQUISITION_TRIGGERED:
                    check_gate_current_limit(reading, self.params['g_ilimit'])
            else:
                self.update_queue.put(('ramp_b', v, reading))
            if v == target_v:
                break
        return True

    def safe_zeroing(self):
        started = time.perf_counter()
        reports = []
        try:
            if self.gate_k:
                self.update_queue.put(('stage', '栅压归零中...'))
                reports.append(fast_shutdown_zero_2450(
                    self.gate_k, self.params['g_ramp_step'],
                    label='栅压表', force_event=self.force_stop_event,
                ))
            if self.bias_k:
                self.update_queue.put(('stage', '偏压归零中...'))
                reports.append(fast_shutdown_zero_2450(
                    self.bias_k, self.params['b_ramp_step'],
                    label='偏压表', force_event=self.force_stop_event,
                ))
            if not reports:
                return
            elapsed = time.perf_counter() - started
            if all(shutdown_report_confirmed(report) for report in reports):
                emergency = any(
                    report['status'] == 'emergency_off' for report in reports
                )
                message = (
                    '强制终止：已紧急归零并关闭输出'
                    if emergency else f'归零完成，用时 {elapsed:.1f} s'
                )
                self.update_queue.put(('log', message))
                self.update_queue.put(('log', '输出已关闭'))
                self.update_queue.put(('ramp_g', 0.0, 0.0))
                self.update_queue.put(('ramp_b', 0.0, 0.0))
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

    def run(self):
        times, v_outs, isd_outs = [], [], []
        collector = None
        metadata = {
            'gate_voltage_range_V': self.params.get('g_voltage_range'),
            'gate_current_range_A': GATE_CURRENT_RANGE_A,
            'gate_current_limit_A': self.params.get('g_ilimit'),
            'gate_software_monitoring': bool(
                self.params.get('acquisition_mode') != ACQUISITION_TRIGGERED
            ),
            'gate_current_limit_tripped': False,
        }
        vb = self.params['b_target']
        try:
            self.connect()
            self.setup()

            self.bias_k.write(':OUTP ON')
            self.gate_k.write(':OUTP ON')

            if not self._ramp_voltage(self.bias_k, self.params['b_target'], self.params['b_ramp_step'], self.params['b_step_delay'], is_gate=False):
                return
            if not self._sleep(self.params['b_settle'], '偏压稳定等待中'):
                return

            waveform = self.params['waveform']
            if not waveform:
                return

            first_g_v = waveform[_0][_0]
            if not self._ramp_voltage(self.gate_k, first_g_v, self.params['g_ramp_step'], 0.01, is_gate=True):
                return

            self.update_queue.put(('stage', '波形输出中'))

            p_settle = self.params['switch_settle']
            cycles = int(self.params['cycles'])
            plot_interval = self.params['plot_interval']
            batch_t, batch_v, batch_isd = [], [], []
            acquisition_mode = self.params['acquisition_mode']
            if acquisition_mode == ACQUISITION_TRIGGERED:
                self.update_queue.put(('gate_leakage_unavailable', None))
                collector = InternalSegmentCollector(
                    self.bias_k, self.update_queue, self.stop_event,
                    self.force_stop_event, self.params['sample_nplc'],
                    prefix='arb_gate',
                )
            else:
                sampler = RealtimeSampler(
                    self.bias_k, self.gate_k,
                    self.params['gate_monitor_interval'],
                    lambda current: self.update_queue.put(('gate_leakage', current)),
                    gate_current_limit=self.params['g_ilimit'],
                )

            for cycle in range(1, cycles + 1):
                for phase_idx, (target_g, p_duration) in enumerate(waveform):
                    if self.stop_event.is_set() or self.force_stop_event.is_set():
                        break

                    self.gate_k.write(f':SOUR:VOLT {target_g}')
                    if p_settle > 0:
                        if not self._sleep(p_settle, "稳定等待中"):
                            break
                    if acquisition_mode == ACQUISITION_TRIGGERED:
                        collector.acquire_segment(target_g, p_duration)
                    else:
                        absolute_deadline = time.perf_counter() + p_duration
                        while time.perf_counter() < absolute_deadline:
                            if self.stop_event.is_set() or self.force_stop_event.is_set():
                                break
                            rel_time, isd = sampler.sample(
                                '任意栅压正式偏压电流读数'
                            )
                            times.append(rel_time)
                            v_outs.append(target_g)
                            isd_outs.append(isd)
                            batch_t.append(rel_time)
                            batch_v.append(target_g)
                            batch_isd.append(isd)
                            if len(batch_t) >= plot_interval:
                                self.update_queue.put(
                                    ('data_batch', vb, batch_t, batch_v,
                                     batch_isd, cycle, cycles))
                                batch_t, batch_v, batch_isd = [], [], []

                if self.stop_event.is_set() or self.force_stop_event.is_set():
                    break

            if collector is not None:
                self.update_queue.put(('stage', '读取2450内部高速缓冲'))
                t, v, i = collector.read_all()
                times, v_outs, isd_outs = t.tolist(), v.tolist(), i.tolist()
                metadata['segments'] = collector.transition_metadata()
                metadata['synchronization'] = 'software_approximate'
                for start in range(0, len(times), max(1000, plot_interval)):
                    end = min(len(times), start + max(1000, plot_interval))
                    self.update_queue.put(('data_batch', vb, times[start:end], v_outs[start:end], isd_outs[start:end], cycles, cycles))
            elif batch_t:
                self.update_queue.put(
                    ('data_batch', vb, batch_t, batch_v,
                     batch_isd, cycles, cycles))

            metadata.update(timing_metadata(times))
            metadata.update({
                'acquisition_mode': acquisition_mode,
                'nplc': float(self.params['sample_nplc']),
            })

            fname = self.params['filename']
            result_status = (
                'partial'
                if self.stop_event.is_set() or self.force_stop_event.is_set()
                else 'complete'
            )
            self.update_queue.put(
                ('block_done', vb, times, v_outs, isd_outs,
                 fname, result_status,
                 None if result_status == 'complete' else '用户停止或强制终止',
                 metadata))

        except Exception as e:
            if isinstance(e, GateCurrentLimitError):
                metadata['gate_current_limit_tripped'] = True
                self.update_queue.put((
                    'log', f'栅电流保护触发，测试已停止: {e}'
                ))
            self.update_queue.put(('log', f"测量中断或出错: {e}"))
            if collector is not None and not times:
                try:
                    t, v, i = collector.read_all()
                    times, v_outs, isd_outs = t.tolist(), v.tolist(), i.tolist()
                    metadata['segments'] = collector.transition_metadata()
                    metadata['synchronization'] = 'software_approximate'
                    metadata.update(timing_metadata(times))
                    metadata.update({
                        'acquisition_mode': ACQUISITION_TRIGGERED,
                        'nplc': float(self.params['sample_nplc']),
                    })
                except Exception as fetch_exc:
                    self.update_queue.put(('log', f'部分高速数据读取失败: {fetch_exc}'))
            if times:
                self.update_queue.put((
                    'block_done', vb, times, v_outs, isd_outs,
                    self.params['filename'], 'partial', e, metadata,
                ))
        finally:
            if collector is not None:
                try:
                    collector.cleanup()
                except Exception as exc:
                    self.update_queue.put(('log', f'高速缓冲清理失败: {exc}'))
            try:
                self.safe_zeroing()
            except Exception as exc:
                self.update_queue.put(('log', f'安全归零失败: {exc}'))
            bias_confirmed, bias_failures = reliable_output_off(
                self.bias_k, '任意栅压偏压表'
            )
            gate_confirmed, gate_failures = reliable_output_off(
                self.gate_k, '任意栅压表'
            )
            if not bias_confirmed or not gate_confirmed:
                self.update_queue.put((
                    'log',
                    '【严重警告】无法确认源表输出已关闭，请立即从仪器面板确认。 '
                    + ' | '.join(bias_failures + gate_failures),
                ))
            if self.bias_k:
                try:
                    self.bias_k.close()
                except:
                    pass
            if self.gate_k:
                try:
                    self.gate_k.close()
                except:
                    pass

            self.update_queue.put(('finished', None))


class ArbitraryGateWidget(BaseAppWidget):
    def __init__(self, run_guard=None, instrument_settings=None, data_settings=None, parent=None):
        configure_pyqtgraph(use_opengl=False)
        super().__init__(run_guard=run_guard, parent=parent)

        self.module_id = "arbitrary_gate"
        self.module_name = "任意栅压波形测试"
        self.instrument_settings = instrument_settings or InstrumentSettings(
            bias_address='GPIB0::1::INSTR',
            gate_address='GPIB0::2::INSTR',
        )
        self.data_settings = data_settings or DataRootSettings(parent=self)

        self.ui_font = QFont("Arial", 12)
        self.ui_font.setWeight(QFont.Weight.Normal)
        self.bold_font = QFont("Arial", 12)
        self.bold_font.setWeight(QFont.Weight.Bold)
        self.setFont(self.ui_font)

        self.capacity = 2000000
        self.data_count = 0
        self.time_data = np.zeros(self.capacity)
        self.v_data = np.zeros(self.capacity)
        self.isd_data = np.zeros(self.capacity)
        self.points_changed = False

        self.current_folder = ""
        self.waveform = [[0.05, 2.0], [0.1, 2.0], [0.05, 2.0], [-0.1, 2.0]]

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=3)

        self.graph_widget = pg.GraphicsLayoutWidget()
        left_layout.addWidget(self.graph_widget)
        label_style = {'color': '#000', 'font-size': '12pt'}

        self.plot_v = self.graph_widget.addPlot(
            title="Gate Voltage vs Time")
        self.plot_v.setLabel('left', text='Gate Voltage',
                             units='V', **label_style)
        self.plot_v.setLabel('bottom', text='Time', units='s', **label_style)
        self.plot_v.getAxis('left').setTickFont(self.ui_font)
        self.plot_v.getAxis('bottom').setTickFont(self.ui_font)
        self.plot_v.showGrid(x=True, y=True, alpha=0.3)
        self.plot_v.setDownsampling(auto=True, mode='peak')
        self.plot_v.setClipToView(True)
        self.curve_v = self.plot_v.plot(pen=pg.mkPen('r', width=1.5))

        self.graph_widget.nextRow()

        self.plot_isd = self.graph_widget.addPlot(
            title="Bias Current vs Time")
        self.plot_isd.setLabel('left', text='Bias Current',
                             units='A', **label_style)
        self.plot_isd.setLabel('bottom', text='Time', units='s', **label_style)
        self.plot_isd.getAxis('left').setTickFont(self.ui_font)
        self.plot_isd.getAxis('bottom').setTickFont(self.ui_font)
        self.plot_isd.showGrid(x=True, y=True, alpha=0.3)
        self.plot_isd.setDownsampling(auto=True, mode='peak')
        self.plot_isd.setClipToView(True)
        self.curve_isd = self.plot_isd.plot(pen=pg.mkPen('b', width=1.5))
        self.plot_isd.setXLink(self.plot_v)

        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, stretch=2)

        status_items = [
            ("偏压 Vsd (V):", "bias_v", 0, 0), ("用时 (s):", "time", 0, 2),
            ("栅压 Vg (V):", "gate_v", 1, 0), ("已采点数:", "count", 1, 2),
            ("偏置电流 Isd (A):", "bias_i", 2, 0), ("采样率 (Hz):", "rate", 2, 2),
            ("栅电流 Ig (A):", "gate_i", 3, 0), ("系统状态:", "stage", 3, 2)
        ]
        status_group, self.status_labels = create_status_group(
            status_items, self.ui_font, self.bold_font
        )

        right_layout.addWidget(status_group)

        param_group = QGroupBox("测量参数")
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

        sampling_box = create_sampling_settings(
            self, self.inputs, self.ui_font, self.bold_font,
            gate_available=True,
        )
        box_vbox.addWidget(sampling_box)

        meas_box = QGroupBox("栅压波形配置")
        meas_box.setFont(self.bold_font)
        meas_grid = QGridLayout(meas_box)
        configure_parameter_grid(meas_grid)

        btn_edit_wave = QPushButton("打开自定义栅压波形编辑器")
        btn_edit_wave.setFont(self.bold_font)
        btn_edit_wave.setStyleSheet("color: #aa00aa;")
        self.btn_edit_wave = btn_edit_wave
        btn_edit_wave.clicked.connect(self.open_waveform_editor)
        meas_grid.addWidget(btn_edit_wave, 0, 0, 1, 2)

        self.lbl_wave_info = QLabel(f"当前栅压波形段数: {len(self.waveform)}")
        self.lbl_wave_info.setFont(self.ui_font)
        self.lbl_wave_info.setStyleSheet(
            "color: #005500; font-weight: normal;")
        meas_grid.addWidget(self.lbl_wave_info, 0, 2, 1, 2,
                            alignment=Qt.AlignmentFlag.AlignRight)

        def add_p(r, c, txt, k, v):
            lb = QLabel(txt)
            style_parameter_label(lb, self.ui_font)
            le = QLineEdit(v)
            style_parameter_control(le, self.ui_font)
            meas_grid.addWidget(lb, r, c)
            meas_grid.addWidget(le, r, c+1)
            self.inputs[k] = le

        add_p(2, 0, "循环次数:", "cycles", "1")
        add_p(3, 0, "跃变缓冲时延 (s):", "switch_settle", "0.0")
        add_p(4, 0, "爬坡/归零步长 (V):", "g_ramp_step", "0.05")
        add_p(2, 2, "电压量程 (V):", "g_voltage_range", "20")
        add_p(3, 2, "电流限制 (A):", "g_ilimit", "1e-9")

        box_vbox.addWidget(meas_box)

        self.bias_box = QGroupBox("偏压参数")
        self.bias_box.setFont(self.bold_font)
        bias_grid = QGridLayout(self.bias_box)
        configure_parameter_grid(bias_grid)

        def add_bp(r, c, txt, k, v):
            lb = QLabel(txt)
            style_parameter_label(lb, self.ui_font)
            le = (
                create_current_range_combo(v, True, self.ui_font)
                if k == 'b_range' else QLineEdit(v)
            )
            style_parameter_control(le, self.ui_font)
            bias_grid.addWidget(lb, r, c)
            bias_grid.addWidget(le, r, c+1)
            self.inputs[k] = le

        add_bp(0, 0, "目标电压 (V):", "b_target", "0.1")
        add_bp(1, 0, "爬坡/归零步长 (V):", "b_ramp_step", "0.001")
        add_bp(2, 0, "偏压单步延时 (s):", "b_step_delay", "0.01")
        add_bp(0, 2, "偏压到位等待 (s):", "b_settle", "2.0")
        add_bp(1, 2, "电流量程 (A):", "b_range", "1e-6")
        add_bp(2, 2, "电流限制 (A):", "b_ilimit", "1.05e-6")
        bind_range_to_limit(self.inputs['b_range'], self.inputs['b_ilimit'])

        box_vbox.addWidget(self.bias_box)

        path_box = QGroupBox("文件保存路径")
        path_box.setFont(self.bold_font)
        path_grid = QGridLayout(path_box)
        self.inputs['filename'] = QLineEdit("arb_gate_test.txt")
        self.inputs['filename'].setFont(self.ui_font)
        self.ent_folder = QLineEdit('Arbitrary_Gate')
        self.ent_folder.setFont(self.ui_font)
        configure_output_path(
            self, path_grid, self.ent_folder, self.inputs['filename'],
            self.data_settings, 'Arbitrary_Gate',
            hint='',
        )
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

    def open_waveform_editor(self):
        try:
            ramp_step = float(self.inputs['g_ramp_step'].text().strip())
        except (TypeError, ValueError):
            ramp_step = None
        editor = WaveformEditorDialog(
            self, self.waveform, self.ui_font, self.bold_font, ramp_step)
        if editor.exec() == QDialog.DialogCode.Accepted and editor.changed:
            self.waveform = editor.waveform
            self.lbl_wave_info.setText(f"当前栅压波形段数: {len(self.waveform)}")

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

        if not self.waveform:
            self.log_info("波形为空，测量取消。")
            self.mark_measurement_finished(self.module_id)
            return

        self.clear_log()

        p = {}
        try:
            for k, widget in self.inputs.items():
                if isinstance(widget, QComboBox):
                    p[k] = combo_config_value(widget)
                elif isinstance(widget, QLineEdit):
                    p[k] = widget.text().strip()

            b_ramp_step = float(p['b_ramp_step'])
            if b_ramp_step <= 0:
                raise ValueError(f'偏压爬坡步长必须为正值，当前值: {p["b_ramp_step"]}')

            g_ramp_step = float(p['g_ramp_step'])
            if g_ramp_step <= 0:
                raise ValueError(f'栅压爬坡步长必须为正值，当前值: {p["g_ramp_step"]}')

            instrument = self.instrument_settings.snapshot(require_gate=True)
            preset = {
                'acquisition_mode': selected_acquisition_mode(self),
                'sample_nplc': float(p['sample_nplc']),
                'gate_monitor_interval': float(p['gate_monitor_interval']),
                'waveform': self.waveform,
                'cycles': int(p['cycles']),
                'switch_settle': float(p['switch_settle']),
                'plot_interval': int(p['plot_interval']),

                'bias_address': instrument['bias_address'],
                'gate_address': instrument['gate_address'],
                'bias_terminal': instrument['bias_terminal'],
                'gate_terminal': instrument['gate_terminal'],

                'b_target': float(p['b_target']),
                'b_ramp_step': b_ramp_step,
                'b_step_delay': float(p['b_step_delay']),
                'b_settle': float(p['b_settle']),
                'b_range': p['b_range'],
                'b_ilimit': float(p['b_ilimit']),

                'g_ramp_step': g_ramp_step,
                'g_voltage_range': float(p['g_voltage_range']),
                'g_ilimit': float(p['g_ilimit']),
            }
            if preset['cycles'] <= 0:
                raise ValueError('循环次数必须大于 0')
            if preset['switch_settle'] < 0:
                raise ValueError('跃变缓冲时延不能为负值')
            if preset['plot_interval'] <= 0:
                raise ValueError('界面刷新间隔必须为正整数')
            validate_nplc(preset['sample_nplc'], '测量 NPLC')
            if preset['gate_monitor_interval'] <= 0:
                raise ValueError('栅电流监测间隔必须大于 0')
            if preset['b_step_delay'] < 0:
                raise ValueError('偏压单步时延不能为负值')
            if preset['b_settle'] < 0:
                raise ValueError('偏压到位等待不能为负值')
            if preset['b_ilimit'] <= 0:
                raise ValueError('偏压限流必须大于 0')
            b_range = str(preset['b_range']).strip()
            if b_range.upper() != 'AUTO' and float(b_range) <= 0:
                raise ValueError('偏压量程必须大于 0 或为 AUTO')
            validate_voltage_range(preset['g_voltage_range'], '栅压量程')
            check_gate_current_limit(0.0, preset['g_ilimit'])
            for _, duration in preset['waveform']:
                if duration <= 0:
                    raise ValueError('波形每段时长必须大于 0')
            validate_program_step_plan('arbitrary_gate', preset)

            folder = self.resolved_output_folder()
            self.current_folder = folder
            os.makedirs(folder, exist_ok=True)
            with open(os.path.join(folder, '.test'), 'w') as f:
                f.write('1')
            os.remove(os.path.join(folder, '.test'))

            preset['filename'] = p['filename']
        except Exception as e:
            self.log_info(f"参数错误: {e}")
            self.show_parameter_error(e)
            self.mark_measurement_finished(self.module_id)
            return

        self.data_count = 0
        self.time_data.fill(0)
        self.v_data.fill(0)
        self.isd_data.fill(0)
        self.curve_v.setData([], [])
        self.curve_isd.setData([], [])

        while not self.update_queue.empty():
            self.update_queue.get()

        self.stop_event.clear()
        self.force_stop_event.clear()
        self.measure_running = True

        self.btn_start.setEnabled(False)
        self.btn_edit_wave.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_force.setEnabled(True)

        self.start_worker(
            target=self._worker,
            args=(preset,),
            name=f'{self.module_id}-worker',
        )

    def _worker(self, preset):
        GateArbMeasurement(
            preset,
            update_queue=self.update_queue,
            stop_event=self.stop_event,
            force_stop_event=self.force_stop_event
        ).run()

    def stop_measurement(self):
        self.stop_event.set()
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText("安全归零中...")
        self.log_info("已触发安全停止...")

    def force_stop_measurement(self):
        if not self.measure_running:
            self._reset_btns()
            return
        self.force_stop_event.set()
        self.stop_event.set()
        self.btn_force.setEnabled(False)
        self.btn_force.setText("强行切断中...")
        self.btn_stop.setEnabled(False)
        self.log_info("强制终止！")
        QTimer.singleShot(500, self._reset_btns)

    def _reset_btns(self):
        self.btn_start.setEnabled(True)
        self.btn_edit_wave.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText("停止")
        self.btn_force.setEnabled(True)
        self.btn_force.setText("强制终止")

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
                self.status_labels['gate_i'].setText(f"{msg[_2]:.6e}")

            elif msg_type == 'gate_leakage':
                self.status_labels['gate_i'].setText(f"{msg[_1]:.6e}")

            elif msg_type == 'gate_leakage_unavailable':
                self.status_labels['gate_i'].setText('未监测（高速模式）')

            elif msg_type == 'data_batch':
                target_vb, batch_t, batch_v, batch_isd, cycle, total_cycles = msg[
                    _1], msg[_2], msg[_3], msg[_4], msg[_5], msg[_6]
                if batch_t:
                    self.status_labels['bias_v'].setText(f"{target_vb:.6f}")
                    self.status_labels['gate_v'].setText(f"{batch_v[-1]:.6f}")
                    self.status_labels['bias_i'].setText(f"{batch_isd[-1]:.6e}")
                    self.status_labels['time'].setText(f"{batch_t[-1]:.3f}")
                    self.status_labels['stage'].setText(
                        f"执行波形 [循环 {cycle}/{total_cycles}]")

                    new_count = len(batch_t)
                    self.status_labels['count'].setText(
                        str(self.data_count + new_count))

                    needed = self.data_count + new_count
                    if needed > self.capacity:
                        new_capacity = max(needed, self.capacity * 2)
                        grow = new_capacity - self.capacity
                        self.time_data = np.pad(self.time_data, (0, grow))
                        self.v_data = np.pad(self.v_data, (0, grow))
                        self.isd_data = np.pad(self.isd_data, (0, grow))
                        self.capacity = new_capacity
                        self.log_info(f'绘图缓存已自动扩容至 {new_capacity} 点')
                    self.time_data[self.data_count:needed] = batch_t
                    self.v_data[self.data_count:needed] = batch_v
                    self.isd_data[self.data_count:needed] = batch_isd
                    self.data_count = needed
                    self.points_changed = True

            elif msg_type == 'block_done':
                vb, times, v_outs, isd_outs, fname = (
                    msg[_1], msg[_2], msg[_3], msg[_4], msg[_5]
                )
                status = msg[6] if len(msg) > 6 else 'complete'
                error = msg[7] if len(msg) > 7 else None
                metadata = msg[8] if len(msg) > 8 else {}
                self.note_result_status(status, error)
                rate = len(times) / \
                    times[-1] if len(times) > 1 and times[-1] > 0 else 0
                self.status_labels['rate'].setText(f"{rate:.2f}")
                self.submit_save(
                    self.save_data,
                    vb, times, v_outs, isd_outs, fname,
                    self.current_folder,
                    status=status, error=error,
                    metadata=metadata,
                    stopped_at_local=time.strftime('%Y-%m-%d %H:%M:%S'),
                )

            c += 1

    def update_plot(self):
        if self.points_changed and self.data_count > 0:
            self.curve_v.setData(
                self.time_data[:self.data_count], self.v_data[:self.data_count])
            self.curve_isd.setData(
                self.time_data[:self.data_count], self.isd_data[:self.data_count])
            self.points_changed = False

    def save_data(
        self, vb, times, v_outs, isd_outs, filename, output_folder,
        status='complete', error=None, metadata=None, stopped_at_local=None,
    ):
        if status != 'complete':
            stem, suffix = os.path.splitext(filename)
            filename = f'{stem}_partial{suffix}'
        fp = allocate_unique_path(output_folder, filename)
        try:
            lengths = {len(times), len(v_outs), len(isd_outs)}
            if len(lengths) != 1:
                raise ValueError('时间、栅压与 Isd 数据长度不一致')
            with atomic_text_writer(fp) as f:
                f.write(
                    "Time(s)\tGateVoltage(V)\tBiasVoltage(V)\t"
                    "BiasCurrent(A)\n"
                )
                for t, v, isd in zip(times, v_outs, isd_outs):
                    f.write(f"{t:.6f}\t{v:.6f}\t{vb:.6f}\t{isd:.6e}\n")
            write_result_metadata(
                fp, status=status, point_count=len(times), error=error,
                extra=metadata, stopped_at_local=stopped_at_local,
            )
            self.post_log(f"保存成功: {fp}")
            return {'paths': [str(fp)], 'status': status, 'error': error}
        except Exception as e:
            self.post_log(f"保存失败: {e}")
            return {'paths': [], 'status': 'error', 'error': e}

    def closeEvent(self, event):
        if self.measure_running:
            reply = QMessageBox.question(
                self, "警告", "测量正在运行，确认退出？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.force_stop_event.set()
                self.stop_event.set()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
