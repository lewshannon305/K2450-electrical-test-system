from PyQt6.QtWidgets import QWidget, QMessageBox
from PyQt6.QtCore import QTimer, pyqtSignal
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time
import uuid

from core.hardware_base import result_metadata_path


class BaseAppWidget(QWidget):
    save_jobs_drained = pyqtSignal()
    result_ready = pyqtSignal(object)

    def __init__(self, run_guard=None, parent=None):
        super().__init__(parent)
        self.run_guard = run_guard
        self.update_queue = queue.Queue()
        self.alarm_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.force_stop_event = threading.Event()
        self.measure_running = False
        self.worker_thread = None
        self._save_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix='measurement-save',
        )
        self._save_futures = set()
        self._save_lock = threading.Lock()
        self._deferred_finish_module_id = None
        self._run_id = None
        self._run_started_at = None
        self._result_records = []
        self._result_emitted = False
        self._display_result_status = 'complete'
        self._display_result_error = None
        self._safety_alarm_active = False
        self._safety_alarm_text = ''
        self.save_jobs_drained.connect(self._complete_deferred_finish)

        self.timer_poll = QTimer()
        self.timer_poll.timeout.connect(self.poll_queue)
        self.timer_poll.start(20)

        self.timer_plot = QTimer()
        self.timer_plot.timeout.connect(self.update_plot)
        self.timer_plot.start(100)

    def request_start(self, module_id, module_name):
        if self.run_guard is None:
            self._begin_result_run()
            return True
        if self.run_guard.request_start(module_id, module_name):
            self._begin_result_run()
            return True

        self._show_silent_message("提示", "当前已有测量正在运行，请先结束再启动其他模块。")
        return False

    def _begin_result_run(self):
        with self._save_lock:
            self._run_id = uuid.uuid4().hex[:12]
            self._run_started_at = time.time()
            self._result_records = []
            self._result_emitted = False
            self._display_result_status = 'complete'
            self._display_result_error = None

    def reset_status_display(self, stage_text='仪器初始化中...'):
        """Clear values from a previous run without hiding a safety alarm."""
        labels = getattr(self, 'status_labels', {})
        if not isinstance(labels, dict):
            return
        for key, label in labels.items():
            if key != 'stage':
                label.setText('-')
        stage = labels.get('stage')
        if stage is None:
            return
        if self._safety_alarm_active:
            self._render_persistent_safety_alarm()
        else:
            stage.setText(str(stage_text))

    def note_result_status(self, status='complete', error=None):
        """Keep the most severe result seen before the worker-finished message."""
        normalized = str(status or 'complete').lower()
        if normalized not in ('complete', 'partial', 'error'):
            normalized = 'error'
        severity = {'complete': 0, 'partial': 1, 'error': 2}
        if severity[normalized] >= severity[self._display_result_status]:
            self._display_result_status = normalized
            if error is not None:
                self._display_result_error = str(error)

    def show_final_status(self, status=None, error=None):
        """Render an honest terminal state while preserving critical alarms."""
        if status is not None:
            self.note_result_status(status, error)
        if self._safety_alarm_active:
            self._render_persistent_safety_alarm()
            return
        labels = getattr(self, 'status_labels', {})
        stage = labels.get('stage') if isinstance(labels, dict) else None
        if stage is None:
            return

        status = self._display_result_status
        detail = (self._display_result_error or '').lower()
        force_event = getattr(self, 'force_stop_event', None)
        force_requested = bool(
            force_event is not None
            and hasattr(force_event, 'is_set')
            and force_event.is_set()
        )
        if status == 'complete':
            text = '测试完成'
        elif force_requested or (
            any(token in detail for token in ('force', '强制'))
            and not any(token in detail for token in ('user', '用户停止'))
        ):
            text = '强制终止'
        elif any(token in detail for token in ('threshold', '保护', '漏电')):
            text = '保护触发，测试已停止'
        elif any(token in detail for token in ('user', '用户停止')):
            text = '用户停止'
        elif status == 'error':
            text = '错误中断'
        else:
            text = '测试未完整完成'
        stage.setText(text)

    def note_status_from_message(self, message):
        """Promote terminal failure messages into the displayed run result."""
        text = str(message)
        if any(token in text for token in ('失败', '错误', '异常', '中断')):
            self.note_result_status('error', text)
            if not self.measure_running:
                self.show_final_status()

    def record_saved_result(self, paths, status='complete', error=None):
        if paths is None:
            normalized = []
        elif isinstance(paths, (str, bytes)):
            normalized = [str(paths)]
        else:
            normalized = [str(path) for path in paths]
        with self._save_lock:
            self._result_records.append({
                'paths': normalized,
                'status': str(status),
                'error': None if error is None else str(error),
            })

    def mark_measurement_finished(self, module_id):
        self.measure_running = False
        if self._safety_alarm_active:
            self._render_persistent_safety_alarm()
        with self._save_lock:
            if self._save_futures:
                self._deferred_finish_module_id = module_id
                return
        if self.run_guard is not None:
            self.run_guard.finish(module_id)
        self._emit_result_ready(module_id)

    def _complete_deferred_finish(self):
        with self._save_lock:
            if self._save_futures:
                return
            module_id = self._deferred_finish_module_id
            self._deferred_finish_module_id = None
        if module_id is not None and self.run_guard is not None:
            self.run_guard.finish(module_id)
        if module_id is not None:
            self._emit_result_ready(module_id)

    def _emit_result_ready(self, module_id):
        with self._save_lock:
            if self._result_emitted:
                return
            self._result_emitted = True
            records = list(self._result_records)
            run_id = self._run_id
            started_at = self._run_started_at
        statuses = [record.get('status', 'complete') for record in records]
        if 'error' in statuses:
            status = 'error'
        elif 'partial' in statuses:
            status = 'partial'
        else:
            status = 'complete'
        data_files = []
        errors = []
        for record in records:
            data_files.extend(record.get('paths', []))
            if record.get('error'):
                errors.append(record['error'])
        metadata_files = []
        for path in data_files:
            current_metadata = result_metadata_path(path)
            if current_metadata.exists():
                metadata_files.append(str(current_metadata))
        self.result_ready.emit({
            'run_id': run_id,
            'module_id': module_id,
            'module_name': getattr(self, 'module_name', module_id),
            'status': status,
            'started_at': started_at,
            'finished_at': time.time(),
            'data_files': data_files,
            'metadata_files': metadata_files,
            'errors': errors,
        })

    def submit_save(self, function, *args, **kwargs):
        future = self._save_executor.submit(function, *args, **kwargs)
        with self._save_lock:
            self._save_futures.add(future)

        def completed(done_future):
            try:
                result = done_future.result()
                if isinstance(result, dict):
                    self.record_saved_result(
                        result.get('paths', []),
                        status=result.get('status', 'complete'),
                        error=result.get('error'),
                    )
                elif result is not None:
                    self.record_saved_result(result)
            except Exception as exc:
                self.record_saved_result([], status='error', error=exc)
                self.post_log(f'保存任务失败: {exc}')
            with self._save_lock:
                self._save_futures.discard(done_future)
                drained = not self._save_futures
            if drained:
                self.save_jobs_drained.emit()

        future.add_done_callback(completed)
        return future

    def post_log(self, text):
        alarm_queue = getattr(self, 'alarm_queue', None)
        if alarm_queue is not None:
            alarm_queue.put(str(text))
        else:
            self.update_queue.put(('log', str(text)))

    def raise_persistent_safety_alarm(self, text):
        self._safety_alarm_active = True
        self._safety_alarm_text = str(text)
        self._render_persistent_safety_alarm()
        self.log_info(self._safety_alarm_text)

    def clear_persistent_safety_alarm(self):
        self._safety_alarm_active = False
        self._safety_alarm_text = ''
        labels = getattr(self, 'status_labels', {})
        stage = labels.get('stage') if isinstance(labels, dict) else None
        if stage is not None:
            stage.setWordWrap(False)
            stage.setMinimumHeight(0)
            stage.setStyleSheet('')

    def _render_persistent_safety_alarm(self):
        labels = getattr(self, 'status_labels', {})
        stage = labels.get('stage') if isinstance(labels, dict) else None
        if stage is not None:
            stage.setText('严重：输出状态未确认，请从仪器面板确认')
            stage.setWordWrap(True)
            stage.setMinimumHeight(42)
            stage.setStyleSheet(
                'background-color:#b00020;color:white;font-weight:bold;'
                'padding:4px;'
            )

    def has_pending_saves(self):
        with self._save_lock:
            return bool(self._save_futures)

    def start_worker(self, target, args=(), kwargs=None, name=None):
        self.worker_thread = threading.Thread(
            target=target,
            args=args,
            kwargs=kwargs or {},
            name=name,
            daemon=False,
        )
        self.worker_thread.start()
        return self.worker_thread

    def is_worker_alive(self):
        return self.worker_thread is not None and self.worker_thread.is_alive()

    def is_measurement_active(self):
        return self.measure_running or self.is_worker_alive() or self.has_pending_saves()

    def request_shutdown_for_close(self, force=True):
        if force:
            self.force_stop_event.set()
        self.stop_event.set()

    @staticmethod
    def require_positive(value, label):
        if value <= 0:
            raise ValueError(f"{label}必须大于 0，当前值: {value}")
        return value

    @staticmethod
    def require_non_negative(value, label):
        if value < 0:
            raise ValueError(f"{label}不能为负值，当前值: {value}")
        return value

    @staticmethod
    def require_positive_int(value, label):
        if int(value) != value or value <= 0:
            raise ValueError(f"{label}必须为正整数，当前值: {value}")
        return int(value)

    def poll_queue(self):
        pass

    def update_plot(self):
        pass

    def _show_silent_message(self, title, text):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.NoIcon)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def show_parameter_error(self, text):
        self.note_result_status('error', text)
        self.show_final_status()
        self._show_silent_message("参数错误，无法开始测量", str(text))
