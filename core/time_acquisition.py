"""Shared acquisition primitives for the three time-domain modules."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from core.hardware_base import assert_no_scpi_errors, required_float_query


ACQUISITION_TRIGGERED = "triggered"
ACQUISITION_REALTIME = "realtime"
GATE_MONITOR_NPLC = 1.0
MAX_BUFFER_POINTS = 1_000_000


def estimate_2450_rate(nplc: float) -> float:
    """Conservative 50 Hz estimate with source readback disabled."""
    return 1.0 / (float(nplc) / 50.0 + 0.00028)


def timing_metadata(times) -> dict:
    values = np.asarray(times, dtype=float)
    result = {
        "points": int(values.size),
        "sample_rate_hz": 0.0,
        "mean_interval_s": None,
        "median_interval_s": None,
        "std_interval_s": None,
        "min_interval_s": None,
        "max_interval_s": None,
    }
    if values.size < 2:
        return result
    intervals = np.diff(values)
    intervals = intervals[np.isfinite(intervals) & (intervals >= 0)]
    if not intervals.size:
        return result
    span = float(values[-1] - values[0])
    result.update({
        "sample_rate_hz": float((values.size - 1) / span) if span > 0 else 0.0,
        "mean_interval_s": float(np.mean(intervals)),
        "median_interval_s": float(np.median(intervals)),
        "std_interval_s": float(np.std(intervals)),
        "min_interval_s": float(np.min(intervals)),
        "max_interval_s": float(np.max(intervals)),
    })
    return result


@dataclass
class SegmentRecord:
    buffer_name: str
    value: float
    requested_duration_s: float
    wall_start_s: float
    wall_end_s: float
    point_count: int
    aborted: bool = False


class InternalSegmentCollector:
    """Acquire multiple voltage phases in separate 2450 buffers.

    No reading data are transferred while a phase is active. Buffers are read
    only after every phase has completed (or after an abort), so USB/GPIB
    traffic cannot disturb the active internal trigger loop.
    """

    def __init__(self, instrument, update_queue, stop_event, force_stop_event,
                 nplc, chunk_points=10_000, prefix="time"):
        self.instrument = instrument
        self.update_queue = update_queue
        self.stop_event = stop_event
        self.force_stop_event = force_stop_event
        self.nplc = float(nplc)
        self.chunk_points = max(1000, int(chunk_points))
        self.prefix = prefix
        self.records: list[SegmentRecord] = []
        self._created: list[str] = []
        self._origin = None

    def _name(self):
        return f"{self.prefix}_{time.monotonic_ns() & 0xFFFFFFFFFFFF:x}"

    def acquire_segment(self, value: float, duration_s: float) -> SegmentRecord:
        duration_s = float(duration_s)
        expected = int(math.ceil(duration_s * estimate_2450_rate(self.nplc) * 1.20)) + 100
        if expected > MAX_BUFFER_POINTS:
            raise ValueError(
                f"单个波形段预计需要 {expected} 点，超过2450单缓冲上限 "
                f"{MAX_BUFFER_POINTS} 点；请缩短该段时长。"
            )
        name = self._name()
        self.instrument.write(f':TRAC:MAKE "{name}", {max(100, expected)}')
        self._created.append(name)
        self.instrument.write(f':TRIG:LOAD "DurationLoop", {duration_s}, 0, "{name}"')
        assert_no_scpi_errors(self.instrument, "高速波形段缓冲与触发配置")

        wall_start = time.perf_counter()
        if self._origin is None:
            self._origin = wall_start
        self.instrument.write(":INIT")
        aborted = False
        quiet_until = wall_start + max(0.0, duration_s * 0.85)
        while time.perf_counter() < quiet_until:
            if self.stop_event.is_set() or self.force_stop_event.is_set():
                self.instrument.write(":ABOR")
                aborted = True
                break
            time.sleep(min(0.05, max(0.0, quiet_until - time.perf_counter())))
        while not aborted:
            if self.stop_event.is_set() or self.force_stop_event.is_set():
                self.instrument.write(":ABOR")
                aborted = True
                break
            state = self.instrument.query(":TRIG:STAT?").strip().upper()
            if state.startswith(("IDLE", "ABORTED")):
                break
            if state.startswith("FAILED"):
                raise RuntimeError(f"2450内部触发模型失败: {state}")
            time.sleep(0.05)
        wall_end = time.perf_counter()
        actual = int(float(self.instrument.query(f':TRAC:ACT? "{name}"')))
        record = SegmentRecord(
            buffer_name=name,
            value=float(value),
            requested_duration_s=duration_s,
            wall_start_s=wall_start,
            wall_end_s=wall_end,
            point_count=actual,
            aborted=aborted,
        )
        self.records.append(record)
        return record

    def read_all(self):
        all_times, all_values, all_currents = [], [], []
        previous_end = None
        for record in self.records:
            seg_times, seg_currents = [], []
            for start in range(1, record.point_count + 1, self.chunk_points):
                if self.force_stop_event.is_set() and all_times:
                    break
                end = min(record.point_count, start + self.chunk_points - 1)
                response = self.instrument.query(
                    f':TRAC:DATA? {start}, {end}, "{record.buffer_name}", READ, REL'
                )
                values = np.fromstring(response.replace(";", ","), sep=",")
                expected = (end - start + 1) * 2
                if values.size != expected:
                    raise RuntimeError(
                        f"高速缓冲数据长度异常：期望 {expected}，实际 {values.size}"
                    )
                seg_currents.append(values[0::2])
                seg_times.append(values[1::2])
            if not seg_times:
                continue
            t = np.concatenate(seg_times)
            i = np.concatenate(seg_currents)
            # REL is local to a named buffer. Anchor each phase to the host-side
            # INIT timestamp; this preserves real inter-phase gaps.
            host_anchor = record.wall_start_s - self._origin
            if previous_end is not None:
                host_anchor = max(host_anchor, previous_end)
            t = t - t[0] + host_anchor
            previous_end = float(t[-1])
            all_times.append(t)
            all_currents.append(i)
            all_values.append(np.full(t.size, record.value, dtype=float))
        if not all_times:
            empty = np.array([], dtype=float)
            return empty, empty.copy(), empty.copy()
        return (
            np.concatenate(all_times),
            np.concatenate(all_values),
            np.concatenate(all_currents),
        )

    def transition_metadata(self):
        rows = []
        previous = None
        for item in self.records:
            rows.append({
                "value_V": item.value,
                "requested_duration_s": item.requested_duration_s,
                "acquisition_elapsed_s": item.wall_end_s - item.wall_start_s,
                "transition_from_previous_s": (
                    None if previous is None else
                    max(0.0, item.wall_start_s - previous.wall_end_s)
                ),
                "points": item.point_count,
                "aborted": item.aborted,
                "start_offset_s": (
                    item.wall_start_s - self._origin if self._origin is not None else 0.0
                ),
            })
            previous = item
        return rows

    def cleanup(self):
        errors = []
        for name in list(self._created):
            try:
                self.instrument.write(f':TRAC:DEL "{name}"')
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        self._created.clear()
        if errors:
            raise RuntimeError("; ".join(errors))


class RealtimeSampler:
    """Serial primary-current sampling with optional low-rate gate monitoring."""

    def __init__(
        self, primary, gate, monitor_interval_s, on_gate_current=None,
        gate_current_limit=None,
    ):
        self.primary = primary
        self.gate = gate
        self.monitor_interval_s = float(monitor_interval_s)
        self.on_gate_current = on_gate_current
        self.gate_current_limit = gate_current_limit
        self.origin = time.perf_counter()
        self.next_monitor = self.origin

    def sample(self, primary_label="正式偏压电流读数"):
        t0 = time.perf_counter()
        current = required_float_query(self.primary, ":READ?", primary_label)
        t1 = time.perf_counter()
        now = t1
        if self.gate is not None and now >= self.next_monitor:
            gate_current = required_float_query(
                self.gate, ":READ?", "栅电流监视读数"
            )
            if self.on_gate_current is not None:
                self.on_gate_current(gate_current)
            if self.gate_current_limit is not None:
                from core.hardware_base import check_gate_current_limit
                check_gate_current_limit(gate_current, self.gate_current_limit)
            self.next_monitor = now + self.monitor_interval_s
        return (t0 + t1) / 2.0 - self.origin, current


def create_sampling_settings(owner, inputs, ui_font, bold_font,
                             gate_available):
    """Build the identical sampling group used by all time-domain pages."""
    from PyQt6.QtWidgets import (
        QButtonGroup, QGridLayout, QGroupBox, QLabel, QLineEdit,
        QRadioButton, QSizePolicy,
    )
    from core.ui_builder import (
        configure_parameter_grid, style_parameter_control,
        style_parameter_label,
    )

    box = QGroupBox("采样设置")
    box.setFont(bold_font)
    grid = QGridLayout(box)
    configure_parameter_grid(grid)

    def label(text):
        widget = QLabel(text)
        style_parameter_label(widget, ui_font)
        return widget

    grid.addWidget(label("采样模式:"), 0, 0)
    owner.rb_sample_triggered = QRadioButton("高速触发")
    owner.rb_sample_realtime = QRadioButton("实时采样")
    for button in (owner.rb_sample_triggered, owner.rb_sample_realtime):
        button.setFont(ui_font)
        button.setStyleSheet("font-weight: normal;")
    owner.rb_sample_triggered.setChecked(True)
    owner.sample_mode_group = QButtonGroup(owner)
    owner.sample_mode_group.addButton(owner.rb_sample_triggered)
    owner.sample_mode_group.addButton(owner.rb_sample_realtime)
    grid.addWidget(owner.rb_sample_triggered, 0, 1)
    grid.addWidget(owner.rb_sample_realtime, 0, 2, 1, 2)

    grid.addWidget(label("测量 NPLC:"), 1, 0)
    inputs["sample_nplc"] = QLineEdit("0.05")
    style_parameter_control(inputs["sample_nplc"], ui_font)
    grid.addWidget(inputs["sample_nplc"], 1, 1)

    owner.lbl_sample_help = label("")
    owner.lbl_sample_help.setMinimumWidth(0)
    owner.lbl_sample_help.setMaximumWidth(16777215)
    owner.lbl_sample_help.setSizePolicy(
        QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
    )
    owner.lbl_sample_help.setStyleSheet("font-weight: normal; color: #555555;")
    grid.addWidget(owner.lbl_sample_help, 1, 2, 1, 2)

    owner.lbl_plot_interval = label("界面刷新间隔 (点):")
    inputs["plot_interval"] = QLineEdit("50")
    style_parameter_control(inputs["plot_interval"], ui_font)
    grid.addWidget(owner.lbl_plot_interval, 2, 0)
    grid.addWidget(inputs["plot_interval"], 2, 1)

    owner.lbl_gate_monitor_interval = label("栅电流监测间隔 (s):")
    inputs["gate_monitor_interval"] = QLineEdit("1.0")
    style_parameter_control(inputs["gate_monitor_interval"], ui_font)
    grid.addWidget(owner.lbl_gate_monitor_interval, 2, 2)
    grid.addWidget(inputs["gate_monitor_interval"], 2, 3)

    owner.lbl_gate_monitor_note = label("Ig 仅慢速监测，不保存、不绘图")
    owner.lbl_gate_monitor_note.setMinimumWidth(0)
    owner.lbl_gate_monitor_note.setMaximumWidth(16777215)
    owner.lbl_gate_monitor_note.setStyleSheet(
        "font-weight: normal; color: #777777;"
    )
    grid.addWidget(owner.lbl_gate_monitor_note, 3, 0, 1, 4)

    owner._sampling_gate_available = bool(gate_available)
    def refresh():
        realtime = owner.rb_sample_realtime.isChecked()
        inputs["plot_interval"].setEnabled(realtime)
        monitor_enabled = realtime and owner._sampling_gate_available
        owner.lbl_gate_monitor_interval.setEnabled(monitor_enabled)
        inputs["gate_monitor_interval"].setEnabled(monitor_enabled)
        if realtime:
            owner.lbl_sample_help.setText("电脑逐点读取并实时更新曲线")
            owner.lbl_gate_monitor_note.setText(
                "Ig 仅慢速监测，不保存、不绘图"
                if owner._sampling_gate_available else
                "未启用栅压表，不监测 Ig"
            )
        else:
            owner.lbl_sample_help.setText("仪器内部缓冲采集，采集完成后绘图")
            owner.lbl_gate_monitor_note.setText("高速模式不监测 Ig")

    owner.refresh_sampling_controls = refresh
    owner.set_sampling_gate_available = lambda available: (
        setattr(owner, "_sampling_gate_available", bool(available)), refresh()
    )[-1]
    owner.rb_sample_triggered.toggled.connect(refresh)
    owner.rb_sample_realtime.toggled.connect(refresh)
    refresh()
    return box


def selected_acquisition_mode(owner):
    return (
        ACQUISITION_REALTIME
        if owner.rb_sample_realtime.isChecked()
        else ACQUISITION_TRIGGERED
    )
