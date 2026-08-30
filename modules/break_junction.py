import os
import queue
import threading
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
    configure_current_autozero,
    fast_shutdown_zero_2450,
    reliable_output_off,
    required_float_query,
    validate_2450_idn,
    validate_current_range_limit,
    validate_nplc,
    validate_positive_step,
    validate_program_step_plan,
    validate_source_voltage,
    validate_terminal,
    verify_current_configuration,
    write_result_metadata,
)
from core.instrument_config import InstrumentSettings
from core.ui_builder import (
    configure_output_path, configure_parameter_grid, create_status_group,
    style_parameter_control, style_parameter_label,
)
from core.utils import G0, NoScrollComboBox, _0, _1, _2, configure_pyqtgraph


class BreakMeasurement:
    def __init__(self, preset, control, update_queue, alarm_queue, param_queue,
                 stop_event, force_stop_event, user_stop_event):
        self.preset = preset
        self.control = control
        self.update_queue = update_queue
        self.alarm_queue = alarm_queue
        self.param_queue = param_queue
        self.stop_event = stop_event
        self.force_stop_event = force_stop_event
        self.user_stop_event = user_stop_event
        self.keithley = None
        self.normal_completion_reason = None

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

    def _measurement_was_aborted(self):
        return (
            self.force_stop_event.is_set()
            or self.user_stop_event.is_set()
            or (
                self.stop_event.is_set()
                and self.normal_completion_reason is None
            )
        )

    def connect(self):
        try:
            rm = pyvisa.ResourceManager()
            self.keithley = rm.open_resource(self.preset['RESOURCE_NAME'])
            self.keithley.timeout = 5000
            idn = validate_2450_idn(self.keithley.query('*IDN?'))
            self.alarm_queue.put(f"仪器已连接: {idn}")
        except Exception as exc:
            self.alarm_queue.put(f"连接失败: {repr(exc)}")
            raise

    def setup(self):
        k = self.keithley
        validate_nplc(self.preset['NPLC'])
        validate_terminal(self.preset['TERMINAL'])
        validate_source_voltage(self.preset['V0'], '断裂结起始电压')
        validate_source_voltage(self.preset['V_MAX'], '断裂结最大电压')
        validate_positive_step(self.preset['ZERO_STEP_V'], '断裂结归零步长')
        validate_program_step_plan('break_junction', self.preset)
        validate_current_range_limit(
            'AUTO', self.preset['CURRENT_LIMIT'], '断裂结'
        )
        k.write('*RST')
        clear_scpi_status(k)
        k.write(':ABORt')
        k.write(':SOUR:FUNC VOLT')
        k.write(':SENS:FUNC "CURR"')
        k.write(f":ROUT:TERM {self.preset['TERMINAL']}")
        k.write(f":SENS:CURR:NPLC {self.preset['NPLC']}")
        k.write(':SENS:CURR:RANG:AUTO ON')
        configure_current_autozero(k, 'block_once')
        k.write(f":SOUR:VOLT:ILIM {self.preset['CURRENT_LIMIT']}")
        k.write(':SOUR:VOLT 0')
        self._interruptible_sleep(0.05)
        verify_current_configuration(
            k,
            nplc=self.preset['NPLC'],
            current_range='AUTO',
            current_limit=self.preset['CURRENT_LIMIT'],
            terminal=self.preset['TERMINAL'],
            autozero_mode='block_once',
            label='断裂结源表',
        )
        self.update_queue.put(('stage', '仪器初始化中'))

    def ramp_to_start(self):
        self.update_queue.put(('stage', '偏压爬坡中'))
        v_start = self.preset['V0']
        step_abs = self.preset['ZERO_STEP_V']
        if v_start != 0 and step_abs > 0:
            num_steps = int(round(abs(v_start) / step_abs)) + 1
            ramp_voltages = np.linspace(0, v_start, num_steps)
            for v in ramp_voltages:
                if self.stop_event.is_set() or self.force_stop_event.is_set():
                    return False
                self.keithley.write(f':SOUR:VOLT {v}')
                curr = required_float_query(
                    self.keithley, ':MEAS:CURR?', '断裂结爬坡电流读数'
                )
                self.update_queue.put(('ramp', v, curr))
        return True

    def safe_ramp_to_zero(self):
        if self.keithley is None:
            return
        if self.force_stop_event.is_set():
            self.update_queue.put(('stage', '强制断电'))
            reliable_output_off(self.keithley, '断裂结源表')
            return
        self.update_queue.put(('stage', '安全归零中...'))
        report = fast_shutdown_zero_2450(
            self.keithley,
            self.preset['ZERO_STEP_V'],
            label='断裂结源表',
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

    def measure_loop(self):
        self.update_queue.put(('stage', '数据采集'))
        cycle = 0
        v0 = self.preset['V0']
        v_max = self.preset['V_MAX']
        g_aim = self.preset['G_AIM_G0']

        first = True
        step_v = self.control['step_mV'] / 1000.0

        while not self.stop_event.is_set() and not self.force_stop_event.is_set():
            while not self.param_queue.empty():
                new_params = self.param_queue.get()
                self.control.update(new_params)
                step_v = self.control['step_mV'] / 1000.0

            cycle += 1
            v = v0
            i_max = 0

            while v <= v_max and not self.stop_event.is_set() and not self.force_stop_event.is_set():
                self.keithley.write(f':SOUR:VOLT {v}')
                try:
                    i_val = required_float_query(
                        self.keithley, ':MEAS:CURR?', '断裂结正式电流读数'
                    )
                except Exception as exc:
                    self.alarm_queue.put(f"读数失败: {exc}")
                    raise

                self.update_queue.put((v, i_val, i_max, cycle))

                g_g0 = (i_val / v) / G0 if v != 0 else 0

                if not first and g_g0 < g_aim:
                    self.alarm_queue.put(
                        f"电导 {g_g0:.2e} G₀ 低于目标值 {g_aim:.2e} G₀")
                    self.normal_completion_reason = 'target_reached'
                    self.stop_event.set()
                    break

                if first:
                    if g_g0 < g_aim:
                        self.alarm_queue.put(f"初始电导 {g_g0:.2e} G₀ 已低于目标值，断裂成功")
                        self.normal_completion_reason = 'target_reached'
                        self.stop_event.set()
                        break
                    first = False

                if i_val > i_max:
                    i_max = i_val

                if v >= self.control['min_feedback_voltage'] and i_max > 0:
                    change_ratio = (i_val - i_max) / i_max * 100.0
                    if change_ratio <= -self.control['delta_percent']:
                        break

                v += step_v

                if step_v > self.control['max_step_mV'] / 1000.0:
                    self.alarm_queue.put(
                        f"步长 {step_v * 1000:.3f} mV 超过设定最大步长，强制终止。"
                    )
                    self.stop_event.set()
                    break

            if not self.stop_event.is_set() and not self.force_stop_event.is_set():
                if v > v_max:
                    self.alarm_queue.put('到达最大电压，扫描结束。')
                    self.normal_completion_reason = 'voltage_limit_reached'
                    self.stop_event.set()
                    break

                step_v += self.control['speed_mV'] / 1000.0
                self.update_queue.put(('step_update', step_v * 1000.0))

    def run(self):
        result_status = 'complete'
        result_error = None
        try:
            self.connect()
            self.setup()
            self.keithley.write(':OUTP ON')
            self._interruptible_sleep(0.01)
            if not self.ramp_to_start():
                result_status = 'partial'
                result_error = '用户停止或强制终止'
                return
            self.measure_loop()
            if self._measurement_was_aborted():
                result_status = 'partial'
                result_error = '用户停止或强制终止'
        except Exception as exc:
            result_status = 'partial'
            result_error = exc
            self.alarm_queue.put(f"测量异常: {exc}")
        finally:
            try:
                self.safe_ramp_to_zero()
            except Exception as exc:
                if not self.force_stop_event.is_set():
                    self.alarm_queue.put(f"安全归零异常: {exc}")
            confirmed, failures = reliable_output_off(
                self.keithley, '断裂结源表'
            )
            if not confirmed:
                self.alarm_queue.put(
                    '严重警告：无法确认断裂结源表输出已关闭，请立即从仪器面板确认。 '
                    + ' | '.join(failures)
                )

            if self.keithley:
                try:
                    self.keithley.close()
                except Exception:
                    pass

            self.update_queue.put(('result_status', result_status, result_error))
            self.update_queue.put(None)


class BreakJunctionWidget(BaseAppWidget):
    def __init__(self, run_guard=None, instrument_settings=None, data_settings=None, parent=None):
        configure_pyqtgraph(use_opengl=True)
        super().__init__(run_guard=run_guard, parent=parent)

        self.module_id = 'break_junction'
        self.module_name = '断裂结'
        self.instrument_settings = instrument_settings or InstrumentSettings(
            bias_address='GPIB0::1::INSTR'
        )
        self.data_settings = data_settings or DataRootSettings(parent=self)

        self.ui_font = QFont('Arial', 12)
        self.ui_font.setWeight(QFont.Weight.Normal)
        self.bold_font = QFont('Arial', 12)
        self.bold_font.setWeight(QFont.Weight.Bold)
        self.setFont(self.ui_font)

        self.param_queue = queue.Queue()
        self.user_stop_event = threading.Event()

        self.capacity = 100000
        self.data_count = 0
        self.result_status = 'complete'
        self.result_error = None
        self.v_data = np.zeros(self.capacity)
        self.i_data = np.zeros(self.capacity)
        self.g_data = np.zeros(self.capacity)
        self.points_changed = False

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, stretch=3)

        self.graph_widget = pg.GraphicsLayoutWidget()
        left_layout.addWidget(self.graph_widget)

        self.plot_iv = self.graph_widget.addPlot(title='I-V Curve')
        self.plot_iv.setTitle('I-V Curve', size='12pt')
        label_style = {'color': '#000', 'font-size': '12pt'}
        self.plot_iv.setLabel('left', text='Current', units='A', **label_style)
        self.plot_iv.setLabel('bottom', text='Voltage',
                              units='V', **label_style)
        self.plot_iv.getAxis('left').setTickFont(self.ui_font)
        self.plot_iv.getAxis('bottom').setTickFont(self.ui_font)
        self.plot_iv.showGrid(x=True, y=True, alpha=0.3)
        self.plot_iv.setClipToView(True)
        self.plot_iv.setDownsampling(auto=False)
        self.curve_iv = self.plot_iv.plot(pen=pg.mkPen('b', width=1.5))

        self.graph_widget.nextRow()

        self.plot_g = self.graph_widget.addPlot(title='Conductance vs Voltage')
        self.plot_g.setTitle('Conductance vs Voltage', size='12pt')
        self.plot_g.setLabel('left', text='Conductance (G₀)', **label_style)
        self.plot_g.setLabel('bottom', text='Voltage',
                             units='V', **label_style)
        self.plot_g.getAxis('left').setTickFont(self.ui_font)
        self.plot_g.getAxis('bottom').setTickFont(self.ui_font)
        self.plot_g.showGrid(x=True, y=True, alpha=0.3)
        self.plot_g.setClipToView(True)
        self.plot_g.setDownsampling(auto=False)
        self.curve_g = self.plot_g.plot(pen=pg.mkPen('r', width=1.5))

        self.target_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(
            'g', style=Qt.PenStyle.DashLine, width=2))

        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, stretch=2)

        status_items = [
            ('电压 (V):', 'V', 0, 0),
            ('电流 (A):', 'I', 1, 0),
            ('电导 (G₀):', 'G', 2, 0),
            ('电阻 (Ω):', 'R', 3, 0),
            ('当前轮次        :', 'cycle', 0, 2),
            ('本轮最大电流 (A):', 'Imax_cycle', 1, 2),
            ('电流变化比例 (%):', 'change_ratio', 2, 2),
            ('系统状态:', 'stage', 3, 2),
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
        scroll_content_layout = QVBoxLayout(scroll_content)
        scroll_content_layout.setContentsMargins(0, 0, 0, 0)

        self.preset_inputs = {}
        self.control_inputs = {}
        self.inputs = {}

        preset_group = QGroupBox('预设参数 (启动后不可更改)')
        preset_group.setFont(self.bold_font)
        preset_grid = QGridLayout(preset_group)
        configure_parameter_grid(preset_grid)

        preset_items = [
            ('起始电压 (V):', 'V0', '0.01'),
            ('最大电压 (V):', 'V_MAX', '10.0'),
            ('目标电导 (G₀):', 'G_AIM_G0', '0.05'),
            ('电流限制 (A):', 'CURRENT_LIMIT', '0.1'),
            ('测量 NPLC:', 'NPLC', '0.01'),
            ('归零步长 (V):', 'ZERO_STEP_V', '0.005'),
        ]

        r, col_idx = 0, 0
        for label, key, default in preset_items:
            lbl = QLabel(label)
            style_parameter_label(lbl, self.ui_font)
            ent = QLineEdit(default)
            style_parameter_control(ent, self.ui_font)
            preset_grid.addWidget(lbl, r, col_idx * 2)
            preset_grid.addWidget(ent, r, col_idx * 2 + 1)
            self.preset_inputs[key] = ent
            col_idx += 1
            if col_idx > 1:
                col_idx = 0
                r += 1

        scroll_content_layout.addWidget(preset_group)

        control_group = QGroupBox('控制参数 (支持实时更改)')
        control_group.setFont(self.bold_font)
        control_grid = QGridLayout(control_group)
        control_grid.setColumnMinimumWidth(0, 140)
        control_grid.setColumnStretch(1, 1)
        control_grid.setHorizontalSpacing(10)
        control_grid.setVerticalSpacing(6)

        control_items = [
            ('步长 (mV):', 'step_mV', '0.100'),
            ('步长增量 (mV/轮):', 'speed_mV', '0.020'),
            ('最大步长 (mV):', 'max_step_mV', '40.000'),
            ('最小反馈电压 (V):', 'min_feedback_voltage', '0.050'),
            ('电流变化阈值 (%):', 'delta_percent', '1.0'),
        ]
        r = 0
        for label, key, default in control_items:
            lbl = QLabel(label)
            style_parameter_label(lbl, self.ui_font)
            ent = QLineEdit(default)
            style_parameter_control(ent, self.ui_font)
            if key == 'step_mV':
                lbl.setStyleSheet('color: #AA0000;')
                ent.setStyleSheet(
                    'background-color: #FFF0F0; color: #8B0000;'
                )
            control_grid.addWidget(lbl, r, 0)
            control_grid.addWidget(ent, r, 1)
            self.control_inputs[key] = ent
            r += 1

        btn_apply = QPushButton('应用修改')
        btn_apply.setFont(self.bold_font)
        btn_apply.setFixedSize(100, 30)
        btn_apply.clicked.connect(self.apply_params)
        control_grid.addWidget(btn_apply, r, 0, 1, 2,
                               alignment=Qt.AlignmentFlag.AlignCenter)
        scroll_content_layout.addWidget(control_group)

        path_group = QGroupBox('文件保存路径')
        path_group.setFont(self.bold_font)
        path_grid = QGridLayout(path_group)

        ent_filename = QLineEdit('Break.txt')
        ent_filename.setFont(self.ui_font)
        self.preset_inputs['FILENAME'] = ent_filename
        self.folder_input = QLineEdit('Break')
        self.folder_input.setFont(self.ui_font)
        configure_output_path(
            self, path_grid, self.folder_input, ent_filename,
            self.data_settings, 'Break',
            hint='',
        )
        scroll_content_layout.addWidget(path_group)

        self.inputs.update(self.preset_inputs)
        self.inputs.update(self.control_inputs)
        self.inputs['folder'] = self.folder_input

        scroll_content_layout.addStretch()
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
        self.log_text.clear()

        try:
            control_params = {k: float(v.text())
                              for k, v in self.control_inputs.items()}
            if control_params['step_mV'] <= 0:
                raise ValueError('初始步长必须大于 0')
            if control_params['speed_mV'] < 0:
                raise ValueError('步长增量不能为负值')
            if control_params['max_step_mV'] <= 0:
                raise ValueError('最大步长必须大于 0')
            if control_params['min_feedback_voltage'] < 0:
                raise ValueError('反馈起始电压不能为负值')
            if control_params['delta_percent'] <= 0:
                raise ValueError('电流变化比例阈值必须大于 0')
        except ValueError as exc:
            self.log_info(f'控制参数格式错误：{exc}')
            self.show_parameter_error(exc)
            self.mark_measurement_finished(self.module_id)
            return

        preset_params = {}
        try:
            instrument = self.instrument_settings.snapshot(require_gate=False)
            preset_params.update({
                'RESOURCE_NAME': instrument['bias_address'],
                'TERMINAL': instrument['bias_terminal'],
            })
            for key, entry in self.preset_inputs.items():
                val = entry.currentText().strip() if isinstance(
                    entry, NoScrollComboBox) else entry.text().strip()
                if key in ['RESOURCE_NAME', 'FILENAME', 'TERMINAL']:
                    preset_params[key] = val
                else:
                    preset_params[key] = float(val)
            if preset_params['V_MAX'] <= preset_params['V0']:
                raise ValueError('最大电压必须大于起始电压')
            if preset_params['G_AIM_G0'] <= 0:
                raise ValueError('目标电导必须大于 0')
            if preset_params['CURRENT_LIMIT'] <= 0:
                raise ValueError('电流限制必须大于 0')
            if preset_params['NPLC'] <= 0:
                raise ValueError('NPLC 必须大于 0')
            if preset_params['ZERO_STEP_V'] <= 0:
                raise ValueError('归零步长必须大于 0')
            validate_program_step_plan(
                'break_junction', {**preset_params, **control_params}
            )
        except ValueError as exc:
            self.log_info(f"预设参数格式错误：{exc}")
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

        self.data_count = 0
        self.result_status = 'complete'
        self.result_error = None
        self.v_data.fill(0)
        self.i_data.fill(0)
        self.g_data.fill(0)
        self.points_changed = True

        self.curve_iv.setData([], [])
        self.curve_g.setData([], [])

        aim_g0 = preset_params['G_AIM_G0']
        if self.target_line not in self.plot_g.items:
            self.plot_g.addItem(self.target_line)
        self.target_line.setValue(aim_g0)

        while not self.update_queue.empty():
            self.update_queue.get()
        while not self.alarm_queue.empty():
            self.alarm_queue.get()
        while not self.param_queue.empty():
            self.param_queue.get()

        self.stop_event.clear()
        self.force_stop_event.clear()
        self.user_stop_event.clear()
        self.measure_running = True

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.force_stop_btn.setEnabled(True)

        self.start_worker(
            target=self._worker_thread,
            args=(preset_params, control_params),
            name=f'{self.module_id}-worker',
        )

    def _worker_thread(self, preset, control):
        meas = BreakMeasurement(
            preset,
            control,
            update_queue=self.update_queue,
            alarm_queue=self.alarm_queue,
            param_queue=self.param_queue,
            stop_event=self.stop_event,
            force_stop_event=self.force_stop_event,
            user_stop_event=self.user_stop_event,
        )
        meas.run()

    def stop_measurement(self):
        self.user_stop_event.set()
        self.stop_event.set()
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
        self.log_info('执行强制终止，切断输出...')
        QTimer.singleShot(500, self._reset_btns)

    def _reset_btns(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText('停止')
        self.force_stop_btn.setEnabled(True)
        self.force_stop_btn.setText('强制终止')

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
                self._reset_btns()
                if self.data_count > 0:
                    self.submit_save(
                        self.save_data,
                        self.resolved_output_folder(),
                        self.preset_inputs['FILENAME'].text().strip()
                        or 'Break.txt',
                        self.v_data[:self.data_count].copy(),
                        self.i_data[:self.data_count].copy(),
                        self.g_data[:self.data_count].copy(),
                        self.result_status,
                        self.result_error,
                        stopped_at_local=time.strftime('%Y-%m-%d %H:%M:%S'),
                    )
                self.log_info('流程结束。')
                self.show_final_status()
                self.measure_running = False
                self.mark_measurement_finished(self.module_id)
                break

            if not isinstance(msg, tuple):
                continue

            if msg[_0] == 'result_status':
                self.result_status = msg[_1]
                self.result_error = msg[_2]
                self.note_result_status(self.result_status, self.result_error)
                continue

            length = len(msg)

            if length == 2 and msg[_0] == 'step_update':
                new_step = msg[_1]
                self.control_inputs['step_mV'].setText(f"{new_step:.3f}")
                continue

            if length == 2 and msg[_0] == 'stage':
                self.status_labels['stage'].setText(msg[_1])
                continue

            if length == 3:
                msg_type, v, i_val = msg
                cycle = int(self.status_labels['cycle'].text(
                )) if self.status_labels['cycle'].text().isdigit() else 0
                imax = 0.0
                if msg_type == 'ramp':
                    self.status_labels['stage'].setText('偏压爬坡中')
                elif msg_type == 'zeroing':
                    self.status_labels['stage'].setText('偏压归零中')
            elif length == 4:
                v, i_val, imax, cycle = msg
                msg_type = 'measure'
            else:
                continue

            self.status_labels['V'].setText(f"{v:.6f}")
            self.status_labels['I'].setText(f"{i_val:.6e}")
            self.status_labels['cycle'].setText(str(cycle))
            self.status_labels['Imax_cycle'].setText(f"{imax:.6e}")

            if v != 0:
                self.status_labels['G'].setText(f"{(i_val / v) / G0:.6e}")
                self.status_labels['R'].setText(
                    f"{v / i_val:.2e}" if i_val != 0 else 'inf')
            else:
                self.status_labels['G'].setText('0')
                self.status_labels['R'].setText('inf')

            if imax > 0:
                change_ratio = (i_val - imax) / imax * 100.0
                self.status_labels['change_ratio'].setText(
                    f"{change_ratio:.2f}")

            if msg_type not in ('ramp', 'zeroing'):
                if self.data_count >= self.capacity:
                    new_cap = self.capacity * 2
                    new_v = np.zeros(new_cap)
                    new_i = np.zeros(new_cap)
                    new_g = np.zeros(new_cap)
                    new_v[:self.capacity] = self.v_data
                    new_i[:self.capacity] = self.i_data
                    new_g[:self.capacity] = self.g_data
                    self.v_data = new_v
                    self.i_data = new_i
                    self.g_data = new_g
                    self.capacity = new_cap

                self.v_data[self.data_count] = v
                self.i_data[self.data_count] = i_val
                self.g_data[self.data_count] = (
                    i_val / v) / G0 if v != 0 else 0
                self.data_count += 1
                self.points_changed = True

            count += 1

    def update_plot(self):
        if self.points_changed and self.data_count > 0:
            self.curve_iv.setData(
                self.v_data[:self.data_count], self.i_data[:self.data_count])
            self.curve_g.setData(
                self.v_data[:self.data_count], self.g_data[:self.data_count])
            self.points_changed = False

    def save_data(
        self,
        folder,
        filename,
        v_data,
        i_data,
        g_data,
        status='complete',
        error=None,
        stopped_at_local=None,
    ):
        if status != 'complete':
            stem, suffix = os.path.splitext(filename)
            filename = f'{stem}_partial{suffix}'
        try:
            os.makedirs(folder, exist_ok=True)
            full_path = allocate_unique_path(folder, filename)
            with atomic_text_writer(full_path) as file_obj:
                file_obj.write(
                    '# Voltage (V)\tCurrent (A)\tConductance (G0)\n')
                for v, i_val, g_val in zip(v_data, i_data, g_data):
                    file_obj.write(f"{v:.6f}\t{i_val:.6e}\t{g_val:.6e}\n")
            write_result_metadata(
                full_path,
                status=status,
                point_count=len(v_data),
                error=error,
                stopped_at_local=stopped_at_local,
            )
            self.post_log(f"数据成功保存至: {full_path}")
            return {
                'paths': [str(full_path)], 'status': status, 'error': error,
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

    def apply_params(self):
        new_params = {}
        for key, entry in self.control_inputs.items():
            try:
                new_params[key] = float(entry.text())
            except ValueError:
                pass
        if new_params:
            self.param_queue.put(new_params)
            self.log_info('控制参数已投递，将在下一步进生效。')
