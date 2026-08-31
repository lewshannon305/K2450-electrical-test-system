import os
import time
import numpy as np
import pyvisa

from PyQt6.QtWidgets import (
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
    QMessageBox,
    QComboBox,
    QSizePolicy,
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
    gate_safety_metadata,
    configure_current_autozero,
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
    validate_terminal,
    validate_voltage_range,
    validate_gate_voltage_within_range,
    verify_current_configuration,
    write_result_metadata,
)
from core.instrument_config import InstrumentSettings
from core.ui_builder import (
    bind_range_to_limit, combo_config_value, configure_output_path,
    configure_parameter_grid, create_current_range_combo,
    create_voltage_range_combo, create_status_group,
    style_parameter_control, style_parameter_label,
)
from core.utils import configure_pyqtgraph, G0


_1, _2, _3, _4 = 1, 2, 3, 4


def generate_four_segment_vg(start, first, second, end, step):
    segments = []
    if first > start:
        seg1 = np.arange(start, first + step / 2, step)
    else:
        seg1 = np.arange(start, first - step / 2, -step)
    segments.append(seg1)

    if second > first:
        seg2 = np.arange(first + step, second + step / 2, step)
    else:
        seg2 = np.arange(first - step, second - step / 2, -step)
    segments.append(seg2)

    if first > second:
        seg3 = np.arange(second + step, first + step / 2, step)
    else:
        seg3 = np.arange(second - step, first - step / 2, -step)
    segments.append(seg3)

    if end > first:
        seg4 = np.arange(first + step, end + step / 2, step)
    else:
        seg4 = np.arange(first - step, end - step / 2, -step)
    segments.append(seg4)

    voltages = []
    segment_ids = []
    for idx, seg in enumerate(segments, start=1):
        voltages.extend(seg)
        segment_ids.extend([idx] * len(seg))
    return np.array(voltages), np.array(segment_ids)


class IsdVgMeasurement:
    def __init__(self, preset, update_queue, stop_event, force_stop_event):
        self.preset = preset
        self.update_queue = update_queue
        self.stop_event = stop_event
        self.force_stop_event = force_stop_event

    def _interruptible_sleep(self, duration):
        if duration <= 0:
            return None
        steps = int(duration / 0.05)
        for _ in range(steps):
            if self.force_stop_event.is_set():
                return 'force'
            if self.stop_event.is_set():
                return 'stop'
            time.sleep(0.05)
        rem = duration - steps * 0.05
        if rem > 0:
            if self.force_stop_event.is_set():
                return 'force'
            if self.stop_event.is_set():
                return 'stop'
            time.sleep(rem)
        return None

    def _zeroing_sleep(self, duration):
        """Wait the complete settle time during cleanup unless force-stopped."""
        deadline = time.monotonic() + max(0.0, float(duration))
        while True:
            if self.force_stop_event.is_set():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(0.05, remaining))

    def run(self):
        rm = pyvisa.ResourceManager()
        bias = gate = None
        current_vg = 0
        current_bias = 0
        measurement_bias = None
        current_seg = _1

        user_stopped = False
        threshold_exceeded = False
        force_stopped = False
        finish_reason = 'normal'

        bias_target = self.preset['Bias_target']
        bias_step = self.preset['Bias_step']
        settle_time = self.preset['SETTLE_TIME']

        all_vg, all_isd, all_ig, all_seg = [], [], [], []
        scan_completed = False

        try:
            validate_distinct_addresses(
                self.preset['BIAS_ADDR'], self.preset['GATE_ADDR'], True
            )
            validate_nplc(self.preset['Bias_NPLC'], '偏压NPLC')
            validate_terminal(self.preset['BIAS_TERM'], '偏压端子')
            validate_terminal(self.preset['GATE_TERM'], '栅压端子')
            validate_source_voltage(self.preset['Bias_target'], '偏压目标')
            validate_source_voltage(self.preset['Vg_1st'], '第一栅压目标')
            validate_source_voltage(self.preset['Vg_2nd'], '第二栅压目标')
            validate_positive_step(self.preset['Bias_step'], '偏压步长')
            validate_positive_step(self.preset['Vg_step'], '栅压步长')
            validate_program_step_plan('isd_vg', self.preset)
            validate_voltage_range(
                self.preset['Gate_VOLT_RANGE'], '栅压量程'
            )
            vg_values, _ = generate_four_segment_vg(
                0.0,
                self.preset['Vg_1st'],
                self.preset['Vg_2nd'],
                0.0,
                self.preset['Vg_step'],
            )
            for voltage in vg_values:
                validate_gate_voltage_within_range(
                    voltage,
                    self.preset['Gate_VOLT_RANGE'],
                    'Isd-Vg栅压点',
                )
            validate_current_range_limit(
                self.preset['Bias_RANGE'],
                self.preset['Bias_I_LIMIT'],
                '偏压',
            )
            check_gate_current_limit(0.0, self.preset['Gate_I_LIMIT'])
            try:
                bias = rm.open_resource(self.preset['BIAS_ADDR'])
                bias.timeout = 5000
                bias_idn = validate_2450_idn(bias.query('*IDN?'))
                self.update_queue.put(('log', f"Bias 表连接成功: {bias_idn}"))
            except Exception as exc:
                self.update_queue.put(('alarm', f"连接 Bias 表失败: {exc}"))
                raise Exception('InitError')

            try:
                gate = rm.open_resource(self.preset['GATE_ADDR'])
                gate.timeout = 5000
                gate_idn = validate_2450_idn(gate.query('*IDN?'))
                self.update_queue.put(('log', f"Gate 表连接成功: {gate_idn}"))
            except Exception as exc:
                self.update_queue.put(('alarm', f"连接 Gate 表失败: {exc}"))
                raise Exception('InitError')

            try:
                bias.write('*RST')
                clear_scpi_status(bias)
                bias.write(':SENS:FUNC "CURR"')
                bias_range = self.preset['Bias_RANGE']
                if str(bias_range).upper() == 'AUTO':
                    bias.write(':SENS:CURR:RANG:AUTO ON')
                else:
                    bias.write(':SENS:CURR:RANG:AUTO OFF')
                    bias.write(f':SENS:CURR:RANG {bias_range}')
                bias.write(f":SENS:CURR:NPLC {self.preset['Bias_NPLC']}")
                configure_current_autozero(bias, 'continuous')
                bias.write(f":ROUT:TERM {self.preset['BIAS_TERM']}")
                bias.write(':SOUR:FUNC VOLT')
                bias.write(':SOUR:VOLT 0')
                bias.write(f":SOUR:VOLT:ILIM {self.preset['Bias_I_LIMIT']}")
                bias.write(':SOUR:VOLT:RANG:AUTO ON')

                configure_gate_meter(
                    gate,
                    voltage_range=self.preset['Gate_VOLT_RANGE'],
                    current_limit=self.preset['Gate_I_LIMIT'],
                    nplc=1,
                    terminal=self.preset['GATE_TERM'],
                    label='Isd-Vg栅压表',
                )
                verify_current_configuration(
                    bias,
                    nplc=self.preset['Bias_NPLC'],
                    current_range=self.preset['Bias_RANGE'],
                    current_limit=self.preset['Bias_I_LIMIT'],
                    terminal=self.preset['BIAS_TERM'],
                    autozero_mode='continuous',
                    label='Isd-Vg偏压表',
                )
                bias.write(':OUTP ON')
                gate.write(':OUTP ON')
                self.update_queue.put(('log', 'Bias/Gate 表初始化完成。'))
            except Exception as exc:
                self.update_queue.put(('alarm', f"初始化错误: {exc}"))
                raise Exception('InitError')

            if bias_target != 0:
                self.update_queue.put(('log', f"开始偏压爬坡至 {bias_target} V..."))
                num_steps_bias = int(round(abs(bias_target) / bias_step)) + 1
                bias_ramp = np.linspace(0, bias_target, num_steps_bias)
                for v_b in bias_ramp:
                    if self.force_stop_event.is_set():
                        force_stopped = True
                        raise KeyboardInterrupt('Force')
                    if self.stop_event.is_set():
                        user_stopped = True
                        break

                    bias.write(f':SOUR:VOLT {v_b}')
                    interrupt = self._interruptible_sleep(settle_time)
                    if interrupt == 'force':
                        force_stopped = True
                        raise KeyboardInterrupt('Force')
                    if interrupt == 'stop':
                        user_stopped = True
                        raise KeyboardInterrupt('UserStop')
                    current_bias = v_b
                    isd = required_float_query(
                        bias, ':MEAS:CURR?', '偏压爬坡Isd读数'
                    )
                    ig = required_float_query(
                        gate, ':MEAS:CURR?', '偏压爬坡Ig读数'
                    )
                    self.update_queue.put(
                        ('status_data', (current_vg, current_bias, isd, ig, '偏压爬坡中')))
                    try:
                        check_gate_current_limit(
                            ig, self.preset['Gate_I_LIMIT']
                        )
                    except Exception:
                        threshold_exceeded = True
                        self.update_queue.put((
                            'alarm', f'栅电流达到保护限值: {ig:.2e} A'
                        ))
                        break

                if user_stopped:
                    raise KeyboardInterrupt('UserStop')
                if threshold_exceeded:
                    raise RuntimeError('栅电流保护触发')
                self.update_queue.put(('log', '偏压爬坡完成。'))

            # Preserve the voltage used for the formal scan.  current_bias is
            # subsequently changed by final zeroing and must not be persisted.
            measurement_bias = current_bias

            if self.preset['Bias_Delay'] > 0 and not self.stop_event.is_set():
                self.update_queue.put(
                    ('log', f"等待额外延时 {self.preset['Bias_Delay']} 秒..."))
                delay_remaining = self.preset['Bias_Delay']
                while delay_remaining > 0 and not self.stop_event.is_set() and not self.force_stop_event.is_set():
                    wait = min(1.0, delay_remaining)
                    time.sleep(wait)
                    delay_remaining -= wait
                    self.update_queue.put(
                        ('stage', f'偏压稳定等待中 ({delay_remaining:.1f}s)'))
                if self.force_stop_event.is_set():
                    force_stopped = True
                    raise KeyboardInterrupt('Force')

            vg_list, seg_ids = generate_four_segment_vg(
                start=0.0,
                first=self.preset['Vg_1st'],
                second=self.preset['Vg_2nd'],
                end=0.0,
                step=self.preset['Vg_step'],
            )

            total_points = len(vg_list)
            self.update_queue.put(('log', f"栅压扫描开始，共 {total_points} 个点。"))

            for idx, (vg, seg) in enumerate(zip(vg_list, seg_ids)):
                if self.force_stop_event.is_set():
                    force_stopped = True
                    raise KeyboardInterrupt('Force')
                if self.stop_event.is_set():
                    user_stopped = True
                    break

                try:
                    gate.write(f':SOUR:VOLT {vg}')
                    interrupt = self._interruptible_sleep(settle_time)
                    if interrupt == 'force':
                        force_stopped = True
                        raise KeyboardInterrupt('Force')
                    if interrupt == 'stop':
                        user_stopped = True
                        raise KeyboardInterrupt('UserStop')
                    current_vg = vg
                    current_seg = seg
                    isd = required_float_query(
                        bias, ':MEAS:CURR?', 'Isd-Vg正式Isd读数'
                    )
                    ig = required_float_query(
                        gate, ':MEAS:CURR?', 'Isd-Vg正式Ig读数'
                    )
                except Exception as exc:
                    self.update_queue.put(('alarm', f"读数出错: {exc}"))
                    raise Exception('ReadError')

                all_vg.append(vg)
                all_isd.append(isd)
                all_ig.append(ig)
                all_seg.append(seg)

                try:
                    check_gate_current_limit(ig, self.preset['Gate_I_LIMIT'])
                except Exception:
                    alarm_msg = f"栅电流达到保护限值: {ig:.2e} A"
                    self.update_queue.put(('alarm', alarm_msg))
                    threshold_exceeded = True
                    break

                self.update_queue.put(
                    ('data', (vg, isd, ig, seg, measurement_bias)))

                if idx == len(vg_list) - 1:
                    scan_completed = True

        except KeyboardInterrupt as exc:
            if str(exc) == 'Force':
                finish_reason = 'force'
            else:
                finish_reason = 'user'
        except Exception as exc:
            if str(exc) != 'InitError':
                finish_reason = 'error'

        finally:
            zeroing_error = None
            if not force_stopped:
                if threshold_exceeded:
                    finish_reason = 'threshold'
                    self.update_queue.put(('stage', '保护触发，归零中'))
                elif user_stopped:
                    finish_reason = 'user'
                    self.update_queue.put(('stage', '用户停止，归零中'))
                else:
                    self.update_queue.put(('stage', '准备归零'))

                if abs(current_vg) > 0.001:
                    self.update_queue.put(('stage', '栅压归零中...'))
                    zero_ramp = [current_vg]
                    zero_ramp.extend(generate_exact_ramp_levels(
                        current_vg, 0.0, self.preset['Vg_step']
                    ))
                    try:
                        for vg_z in zero_ramp:
                            if self.force_stop_event.is_set():
                                force_stopped = True
                                break
                            gate.write(f':SOUR:VOLT {vg_z}')
                            if not self._zeroing_sleep(settle_time):
                                force_stopped = True
                                break
                            current_vg = vg_z
                            isd = required_float_query(
                                bias, ':MEAS:CURR?', '栅压归零Isd读数'
                            )
                            ig = required_float_query(
                                gate, ':MEAS:CURR?', '栅压归零Ig读数'
                            )
                            self.update_queue.put(
                                ('zero_data', (
                                    vg_z, isd, ig, current_seg, current_bias
                                ))
                            )
                    except Exception as exc:
                        zeroing_error = exc

                if (
                    not force_stopped
                    and zeroing_error is None
                    and abs(current_bias) > 0.001
                ):
                    self.update_queue.put(('stage', '偏压归零中...'))
                    bias_ramp = [current_bias]
                    bias_ramp.extend(generate_exact_ramp_levels(
                        current_bias, 0.0, self.preset['Bias_step']
                    ))
                    try:
                        for vb_z in bias_ramp:
                            if self.force_stop_event.is_set():
                                force_stopped = True
                                break
                            bias.write(f':SOUR:VOLT {vb_z}')
                            if not self._zeroing_sleep(settle_time):
                                force_stopped = True
                                break
                            current_bias = vb_z
                            isd = required_float_query(
                                bias, ':MEAS:CURR?', '偏压归零Isd读数'
                            )
                            ig = required_float_query(
                                gate, ':MEAS:CURR?', '偏压归零Ig读数'
                            )
                            self.update_queue.put(
                                ('status_data', (
                                    current_vg, current_bias, isd, ig,
                                    '偏压归零中'
                                ))
                            )
                    except Exception as exc:
                        zeroing_error = exc

                if zeroing_error is not None:
                    finish_reason = 'error'
                    self.update_queue.put((
                        'alarm',
                        '安全归零失败，已执行紧急关断：'
                        f'{zeroing_error}',
                    ))
            bias_confirmed, bias_failures = (
                reliable_output_off(bias, 'Isd-Vg偏压表')
                if bias is not None else (True, [])
            )
            gate_confirmed, gate_failures = (
                reliable_output_off(gate, 'Isd-Vg栅压表')
                if gate is not None else (True, [])
            )
            if not bias_confirmed or not gate_confirmed:
                self.update_queue.put((
                    'alarm',
                    '严重警告：无法确认源表输出已关闭，请立即从仪器面板确认。 '
                    + ' | '.join(bias_failures + gate_failures),
                ))
            else:
                self.update_queue.put(('safety_cleared', None))
            if gate:
                try:
                    gate.close()
                except Exception:
                    pass
            if bias:
                try:
                    bias.close()
                except Exception:
                    pass

            self.update_queue.put(
                ('finished', (all_vg, all_isd, all_ig, all_seg,
                 measurement_bias if measurement_bias is not None else current_bias,
                 scan_completed, finish_reason))
            )


class IsdVgSetVsdWidget(BaseAppWidget):
    def __init__(self, run_guard=None, instrument_settings=None, data_settings=None, parent=None):
        configure_pyqtgraph(use_opengl=False)
        super().__init__(run_guard=run_guard, parent=parent)

        self.module_id = 'isd_vg_setvsd'
        self.module_name = '栅压特性扫描'
        self.instrument_settings = instrument_settings or InstrumentSettings(
            bias_address='GPIB0::1::INSTR',
            gate_address='GPIB0::2::INSTR',
        )
        self.data_settings = data_settings or DataRootSettings(parent=self)

        self.ui_font = QFont('Arial', 12)
        self.ui_font.setWeight(QFont.Weight.Normal)
        self.bold_font = QFont('Arial', 12)
        self.bold_font.setWeight(QFont.Weight.Bold)
        self.setFont(self.ui_font)

        self.capacity = 50000
        self.data_count = {_1: 0, _2: 0, _3: 0, _4: 0}
        self.zero_count = {_1: 0, _2: 0, _3: 0, _4: 0}

        self.vg_data = {k: np.zeros(self.capacity) for k in [_1, _2, _3, _4]}
        self.isd_data = {k: np.zeros(self.capacity) for k in [_1, _2, _3, _4]}
        self.ig_data = {k: np.zeros(self.capacity) for k in [_1, _2, _3, _4]}

        self.zero_vg_data = {k: np.zeros(self.capacity)
                             for k in [_1, _2, _3, _4]}
        self.zero_isd_data = {k: np.zeros(self.capacity)
                              for k in [_1, _2, _3, _4]}

        self.points_changed = False

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=3)

        self.graph_widget = pg.GraphicsLayoutWidget()
        left_layout.addWidget(self.graph_widget)

        label_style = {'color': '#000', 'font-size': '12pt'}

        self.plots = {}
        self.curves_isd = {}
        self.curves_zero = {}
        colors = {_1: 'b', _2: 'g', _3: 'r', _4: 'c'}

        for i in [_1, _2, _3, _4]:
            p = self.graph_widget.addPlot(title=f"Isd vs Vg - Segment {i}")
            p.setLabel('left', text='Isd', units='A', **label_style)
            p.setLabel('bottom', text='Vg', units='V', **label_style)
            p.getAxis('left').setTickFont(self.ui_font)
            p.getAxis('bottom').setTickFont(self.ui_font)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setClipToView(True)
            p.setDownsampling(auto=True, mode='subsample')
            p.enableAutoRange()

            self.curves_isd[i] = p.plot(pen=pg.mkPen(colors[i], width=1.5))
            self.curves_zero[i] = p.plot(pen=pg.mkPen(
                'm', style=Qt.PenStyle.DashLine, width=1.5))
            self.plots[i] = p

            if i == _2:
                self.graph_widget.nextRow()

        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, stretch=2)

        status_items = [
            ('偏压 Vsd (V):', 'vsd', 0, 0), ('电导 (G₀):', 'cond', 0, 2),
            ('栅压 Vg (V):', 'vg', 1, 0), ('电阻 (Ω):', 'res', 1, 2),
            ('偏置电流 Isd (A):', 'isd', 2, 0), ('进度:', 'progress', 2, 2),
            ('栅电流 Ig (A):', 'ig', 3, 0), ('系统状态:', 'stage', 3, 2),
        ]
        status_group, self.status_labels = create_status_group(
            status_items, self.ui_font, self.bold_font
        )

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

        bias_box = QGroupBox('偏压参数')
        bias_box.setFont(self.bold_font)
        bias_grid = QGridLayout(bias_box)
        configure_parameter_grid(bias_grid)
        bias_items = [
            ('目标电压 (V):', 'Bias_target', '0.002'),
            ('扫描步长 (V):', 'Bias_step', '0.001'),
            ('测量 NPLC:', 'Bias_NPLC', '10.0'),
            ('电流量程 (A):', 'Bias_RANGE', '1e-7'),
            ('电流限制 (A):', 'Bias_I_LIMIT', '1.05e-7'),
            ('偏压到位等待 (s):', 'Bias_Delay', '3.0'),
        ]
        for i, (label, key, default) in enumerate(bias_items):
            row = i % 3
            col = (i // 3) * 2
            lbl = QLabel(label)
            style_parameter_label(lbl, self.ui_font)
            bias_grid.addWidget(lbl, row, col)
            ent = (
                create_current_range_combo(default, True, self.ui_font)
                if key == 'Bias_RANGE' else QLineEdit(default)
            )
            style_parameter_control(ent, self.ui_font)
            self.inputs[key] = ent
            bias_grid.addWidget(ent, row, col + 1)
        bind_range_to_limit(
            self.inputs['Bias_RANGE'], self.inputs['Bias_I_LIMIT']
        )
        box_vbox.addWidget(bias_box)

        gate_box = QGroupBox('栅压参数')
        gate_box.setFont(self.bold_font)
        gate_grid = QGridLayout(gate_box)
        configure_parameter_grid(gate_grid)
        gate_items = [
            ('第一目标电压 (V):', 'Vg_1st', '5.0'),
            ('第二目标电压 (V):', 'Vg_2nd', '-5.0'),
            ('扫描步长 (V):', 'Vg_step', '0.05'),
            ('电压量程 (V):', 'Gate_VOLT_RANGE', '20.0'),
            ('电流限制 (A):', 'Gate_I_LIMIT', '1e-9'),
            ('栅压稳定时间 (s):', 'SETTLE_TIME', '0.1'),
        ]
        for i, (label, key, default) in enumerate(gate_items):
            row = i % 3
            col = (i // 3) * 2
            lbl = QLabel(label)
            style_parameter_label(lbl, self.ui_font)
            gate_grid.addWidget(lbl, row, col)
            ent = (
                create_voltage_range_combo(default, self.ui_font)
                if key == 'Gate_VOLT_RANGE' else QLineEdit(default)
            )
            style_parameter_control(ent, self.ui_font)
            self.inputs[key] = ent
            gate_grid.addWidget(ent, row, col + 1)
        box_vbox.addWidget(gate_box)

        path_box = QGroupBox('文件保存路径')
        path_box.setFont(self.bold_font)
        path_layout_box = QGridLayout(path_box)

        self.file_prefix = QLineEdit('Isd_Vg')
        self.file_prefix.setFont(self.ui_font)
        self.inputs['FILE_PREFIX'] = self.file_prefix
        self.folder_input = QLineEdit('Isd_Vg')
        self.folder_input.setFont(self.ui_font)
        self.inputs['FILE_FOLDER'] = self.folder_input
        configure_output_path(
            self, path_layout_box, self.folder_input, self.file_prefix,
            self.data_settings, 'Isd_Vg', filename_is_prefix=True,
            hint='后缀自动追加：_seg1 至 _seg4',
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
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(5, 5, 5, 5)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(self.ui_font)
        self.log_text.setStyleSheet(
            'background-color: #FFF0F0; color: #333333;')
        self.log_text.setFont(self.ui_font)  # standardized

        self.log_text.setFixedHeight(60)
        log_layout.addWidget(self.log_text)

        btn_clear_log = QPushButton('清除信息')
        btn_clear_log.setFont(self.bold_font)
        btn_clear_log.setFixedWidth(100)
        btn_clear_log.setFixedHeight(30)
        btn_clear_log.clicked.connect(self.clear_log)
        log_layout.addWidget(
            btn_clear_log, alignment=Qt.AlignmentFlag.AlignCenter)

        right_layout.addWidget(log_group, stretch=0)

        btn_area = QWidget()
        btn_layout = QHBoxLayout(btn_area)
        btn_layout.setContentsMargins(0, 10, 0, 10)

        self.start_btn = QPushButton('开始')
        self.start_btn.setFixedSize(100, 30)
        self.start_btn.setFont(self.bold_font)
        self.start_btn.clicked.connect(self.start_measurement)

        self.stop_btn = QPushButton('停止')
        self.stop_btn.setFixedSize(100, 30)
        self.stop_btn.setFont(self.bold_font)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_measurement)

        self.force_stop_btn = QPushButton('强制终止')
        self.force_stop_btn.setFixedSize(100, 30)
        self.force_stop_btn.setFont(self.bold_font)
        self.force_stop_btn.setStyleSheet('color: #AA0000;')
        self.force_stop_btn.clicked.connect(self.force_stop_measurement)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.force_stop_btn)

        right_layout.addWidget(btn_area)

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
            instrument = self.instrument_settings.snapshot(require_gate=True)
            preset['BIAS_ADDR'] = instrument['bias_address']
            preset['GATE_ADDR'] = instrument['gate_address']
            preset['BIAS_TERM'] = instrument['bias_terminal']
            preset['GATE_TERM'] = instrument['gate_terminal']

            for key, entry in self.inputs.items():
                if isinstance(entry, QComboBox):
                    preset[key] = combo_config_value(entry)
                    continue
                val = entry.text().strip()
                if key in ['Vg_step', 'Bias_step']:
                    step_val = float(val)
                    if step_val <= 0:
                        raise ValueError(f'{key} 必须为正值，当前值: {val}')
                    preset[key] = step_val
                elif key in ['FILE_PREFIX', 'FILE_FOLDER']:
                    preset[key] = val
                else:
                    preset[key] = float(val)
            if preset['Bias_I_LIMIT'] <= 0:
                raise ValueError('偏压限流必须大于 0')
            if preset['Bias_NPLC'] <= 0:
                raise ValueError('偏压 NPLC 必须大于 0')
            if (
                str(preset['Bias_RANGE']).upper() != 'AUTO'
                and float(preset['Bias_RANGE']) <= 0
            ):
                raise ValueError('电流量程必须大于 0 或为 AUTO')
            if preset['Bias_Delay'] < 0:
                raise ValueError('偏压后延时不能为负值')
            if preset['Gate_VOLT_RANGE'] <= 0:
                raise ValueError('栅压量程必须大于 0')
            check_gate_current_limit(0.0, preset['Gate_I_LIMIT'])
            if preset['SETTLE_TIME'] < 0:
                raise ValueError('稳定时间不能为负值')
            validate_program_step_plan('isd_vg', preset)
            self._gate_safety_metadata = gate_safety_metadata(
                preset['Gate_VOLT_RANGE'], preset['Gate_I_LIMIT'], True
            )
        except ValueError as exc:
            self.log_info(f"参数格式错误：{exc}")
            self.show_parameter_error(exc)
            self.mark_measurement_finished(self.module_id)
            return

        folder = self.resolved_output_folder()
        try:
            os.makedirs(folder, exist_ok=True)
            test_file = os.path.join(folder, '.write_test')
            with open(test_file, 'w') as file_obj:
                file_obj.write('test')
            os.remove(test_file)
        except Exception as exc:
            self.log_info(f"文件夹不可写: {exc}")
            self.show_final_status('error', exc)
            self.mark_measurement_finished(self.module_id)
            return

        for i in [_1, _2, _3, _4]:
            self.data_count[i] = 0
            self.zero_count[i] = 0
            self.vg_data[i].fill(0)
            self.isd_data[i].fill(0)
            self.ig_data[i].fill(0)
            self.zero_vg_data[i].fill(0)
            self.zero_isd_data[i].fill(0)
            self.curves_isd[i].setData([], [])
            self.curves_zero[i].setData([], [])

        self.points_changed = True
        self.status_labels['progress'].setText('0/4')
        self.status_labels['stage'].setText('仪器初始化中')

        while not self.update_queue.empty():
            self.update_queue.get()

        self.stop_event.clear()
        self.force_stop_event.clear()
        self.measure_running = True

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.force_stop_btn.setEnabled(True)

        self.start_worker(
            target=self._worker_thread,
            args=(preset,),
            name=f'{self.module_id}-worker',
        )

    def _worker_thread(self, preset):
        meas = IsdVgMeasurement(
            preset,
            update_queue=self.update_queue,
            stop_event=self.stop_event,
            force_stop_event=self.force_stop_event,
        )
        meas.run()

    def stop_measurement(self):
        self.stop_event.set()
        self.note_result_status('partial', '用户停止')
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText('归零中...')
        self.log_info('已触发停止，正在步进安全归零...')

    def force_stop_measurement(self):
        if not self.measure_running:
            self._reset_btns()
            return

        self.force_stop_event.set()
        self.stop_event.set()
        self.force_stop_btn.setEnabled(False)
        self.force_stop_btn.setText('强制终止中...')
        self.stop_btn.setEnabled(False)
        self.status_labels['stage'].setText('强制断电')
        QTimer.singleShot(500, self._reset_btns)

    def _reset_btns(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText('停止')
        self.force_stop_btn.setEnabled(True)
        self.force_stop_btn.setText('强制终止')

    def poll_queue(self):
        count = 0
        while count < 500 and not self.update_queue.empty():
            msg_type, payload = self.update_queue.get_nowait()

            if msg_type == 'alarm':
                self.note_status_from_message(payload)
                if '严重警告' in payload:
                    self.raise_persistent_safety_alarm(payload)
                else:
                    self.log_info(payload)

            elif msg_type == 'safety_cleared':
                self.clear_persistent_safety_alarm()

            elif msg_type == 'log':
                pass

            elif msg_type == 'status':
                self.status_labels['stage'].setText(payload)

            elif msg_type == 'stage':
                self.status_labels['stage'].setText(payload)

            elif msg_type == 'status_data':
                vg, vsd, isd, ig, st_text = payload
                self.status_labels['vsd'].setText(f"{vsd:.6f}")
                self.status_labels['vg'].setText(f"{vg:.6f}")
                self.status_labels['isd'].setText(f"{isd:.2e}")
                self.status_labels['ig'].setText(f"{ig:.2e}")
                self.status_labels['stage'].setText(st_text)
                if vsd != 0:
                    self.status_labels['cond'].setText(f"{(isd / vsd) / G0:.6e}")
                    self.status_labels['res'].setText(f"{vsd / isd:.2e}" if isd != 0 else 'inf')
                else:
                    self.status_labels['cond'].setText('0')
                    self.status_labels['res'].setText('inf')

            elif msg_type == 'data':
                vg, isd, ig, seg, vsd_fixed = payload
                self.status_labels['vsd'].setText(f"{vsd_fixed:.6f}")
                self.status_labels['vg'].setText(f"{vg:.6f}")
                self.status_labels['isd'].setText(f"{isd:.2e}")
                self.status_labels['ig'].setText(f"{ig:.2e}")
                self.status_labels['progress'].setText(f'{int(seg)}/4')
                self.status_labels['stage'].setText('栅压扫描中')
                if vsd_fixed != 0:
                    self.status_labels['cond'].setText(f"{(isd / vsd_fixed) / G0:.6e}")
                    self.status_labels['res'].setText(f"{vsd_fixed / isd:.2e}" if isd != 0 else 'inf')
                else:
                    self.status_labels['cond'].setText('0')
                    self.status_labels['res'].setText('inf')

                idx = self.data_count[seg]
                if idx >= self.capacity:
                    self._expand_plot_capacity()
                self.vg_data[seg][idx] = vg
                self.isd_data[seg][idx] = isd
                self.ig_data[seg][idx] = ig
                self.data_count[seg] += 1
                self.points_changed = True

            elif msg_type == 'zero_data':
                vg, isd, ig, current_seg, vsd_fixed = payload
                self.status_labels['vsd'].setText(f"{vsd_fixed:.6f}")
                self.status_labels['vg'].setText(f"{vg:.6f}")
                self.status_labels['isd'].setText(f"{isd:.2e}")
                self.status_labels['ig'].setText(f"{ig:.2e}")
                self.status_labels['stage'].setText('栅压归零中')
                if vsd_fixed != 0:
                    self.status_labels['cond'].setText(f"{(isd / vsd_fixed) / G0:.6e}")
                    self.status_labels['res'].setText(f"{vsd_fixed / isd:.2e}" if isd != 0 else 'inf')
                else:
                    self.status_labels['cond'].setText('0')
                    self.status_labels['res'].setText('inf')

                idx = self.zero_count[current_seg]
                if idx >= self.capacity:
                    self._expand_plot_capacity()
                self.zero_vg_data[current_seg][idx] = vg
                self.zero_isd_data[current_seg][idx] = isd
                self.zero_count[current_seg] += 1
                self.points_changed = True

            elif msg_type == 'finished':
                all_vg, all_isd, all_ig, all_seg, vsd_fixed, scan_completed, reason = payload
                if scan_completed:
                    self.status_labels['progress'].setText('4/4')
                self._reset_btns()
                result_status = 'complete' if scan_completed else (
                    'error' if reason == 'error' else 'partial'
                )
                self.show_final_status(
                    result_status, None if scan_completed else reason
                )
                terminal_logs = {
                    'user': '用户停止：安全归零流程已结束。',
                    'force': '强制终止：紧急关断流程已结束。',
                    'threshold': '保护触发：安全归零流程已结束。',
                    'error': '测量异常结束，已执行输出关闭流程。',
                }
                self.log_info(
                    '测量完成。' if scan_completed
                    else terminal_logs.get(reason, '测量流程已结束。')
                )

                if all_vg:
                    self.submit_save(
                        self.save_data,
                        all_vg,
                        all_isd,
                        all_ig,
                        all_seg,
                        vsd_fixed,
                        self.resolved_output_folder(),
                        self.file_prefix.text().strip() or 'IsdVg',
                        result_status,
                        None if scan_completed else reason,
                        stopped_at_local=time.strftime('%Y-%m-%d %H:%M:%S'),
                    )

                self.measure_running = False
                self.mark_measurement_finished(self.module_id)
                break

            count += 1

    def _expand_plot_capacity(self):
        old_capacity = self.capacity
        self.capacity *= 2
        extra = self.capacity - old_capacity
        for store in (
            self.vg_data,
            self.isd_data,
            self.ig_data,
            self.zero_vg_data,
            self.zero_isd_data,
        ):
            for segment in store:
                store[segment] = np.pad(store[segment], (0, extra))
        self.log_info(f'绘图缓存已自动扩容至 {self.capacity} 点')

    def update_plot(self):
        if self.points_changed:
            for i in [_1, _2, _3, _4]:
                c = self.data_count[i]
                if c > 0:
                    self.curves_isd[i].setData(
                        self.vg_data[i][:c], self.isd_data[i][:c])
                zc = self.zero_count[i]
                if zc > 0:
                    self.curves_zero[i].setData(
                        self.zero_vg_data[i][:zc], self.zero_isd_data[i][:zc])
                if c > 0 or zc > 0:
                    self.plots[i].autoRange()
            self.points_changed = False

    def save_data(
        self,
        all_vg,
        all_isd,
        all_ig,
        all_seg,
        vsd_fixed,
        filepath,
        prefix,
        status='complete',
        error=None,
        stopped_at_local=None,
    ):
        saved_paths = []
        save_errors = []
        for seg in range(1, 5):
            mask = np.array(all_seg) == seg
            vg_seg = np.array(all_vg)[mask]
            isd_seg = np.array(all_isd)[mask]
            ig_seg = np.array(all_ig)[mask]
            if len(vg_seg) == 0:
                continue

            marker = '' if status == 'complete' else '_partial'
            filename = f"{prefix}_seg{seg}{marker}.txt"
            full_path = allocate_unique_path(filepath, filename)
            try:
                with atomic_text_writer(full_path) as file_obj:
                    file_obj.write('# Vsd (V)\tVg (V)\tIsd (A)\tIg (A)\n')
                    for i in range(len(vg_seg)):
                        file_obj.write(
                            f"{vsd_fixed:.6f}\t{vg_seg[i]:.6f}\t{isd_seg[i]:.6e}\t{ig_seg[i]:.6e}\n"
                        )
                write_result_metadata(
                    full_path,
                    status=status,
                    point_count=len(vg_seg),
                    error=error,
                    stopped_at_local=stopped_at_local,
                    extra={
                        **getattr(self, '_gate_safety_metadata', {}),
                        'bias_voltage_V': float(vsd_fixed),
                        'gate_current_limit_tripped': status != 'complete'
                        and str(error) == 'threshold',
                    },
                )
                self.post_log(f"段{seg}数据已保存至: {full_path}")
                saved_paths.append(str(full_path))
            except Exception as exc:
                self.post_log(f"保存段{seg}数据失败: {exc}")
                save_errors.append(str(exc))
        if save_errors:
            return {
                'paths': saved_paths,
                'status': 'error',
                'error': ' | '.join(save_errors),
            }
        return {'paths': saved_paths, 'status': status, 'error': error}

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
