"""Two-phase, main-thread-safe shutdown coordination for the recorder."""

from __future__ import annotations

import os
import signal
import threading
import time
from typing import Callable

from aloha.record_trigger import RecordingPhase
from aloha.safety_state import (
    DEFAULT_SAFETY_STATE_PATH,
    publish_safety_state,
)


def hold_unsafe_until_safe(
    *,
    controller: "SafeStopController",
    report,
    retry_recovery,
    publish_state=publish_safety_state,
    logger: Callable[[str], None] = print,
    log_interval_seconds: float = 10.0,
    retry_poll_seconds: float = 1.0,
    clock: Callable[[], float] = time.monotonic,
):
    """Block runtime teardown until an explicit retry verifies every robot."""

    controller.enter_unsafe_hold()
    current_report = report
    last_log = float("-inf")
    while not current_report.safe_to_stop:
        now = clock()
        if now - last_log >= log_interval_seconds:
            unresolved = [
                name
                for name, result in current_report.results.items()
                if result.status.value != "slept_verified"
            ]
            logger(
                "[UNSAFE_HOLD] 禁止关闭；未确认 sleep: "
                f"{', '.join(sorted(unresolved))}. "
                "反馈恢复后执行显式重试。"
            )
            last_log = now

        if not controller.wait_for_safety_retry(
            timeout=retry_poll_seconds
        ):
            continue

        logger("[UNSAFE_HOLD] 收到显式重试，重新检查并逐臂回收。")
        publish_state("RECOVERY_IN_PROGRESS", report=current_report)
        try:
            current_report = retry_recovery()
        except Exception as exc:
            logger(f"[UNSAFE_HOLD] 重试异常，继续保持运行: {exc}")
            publish_state("UNSAFE_HOLD", report=current_report)
            continue
        if not current_report.safe_to_stop:
            publish_state("UNSAFE_HOLD", report=current_report)

    controller.leave_unsafe_hold()
    return current_report


def should_defer_s_wakeup(phase: RecordingPhase) -> bool:
    """Keep SIGINT out of robot return phases that own a motion thread."""
    return phase in {
        RecordingPhase.RETURNING_TO_RETRY,
        RecordingPhase.RETURNING_TO_SAVE,
    }


def send_sigint_to_process() -> None:
    """Wake the main thread with a real signal, including during blocking waits."""
    os.kill(os.getpid(), signal.SIGINT)


class SafeStopController:
    """Request graceful no-save shutdown and escalate blocked main threads."""

    def __init__(
        self,
        stop_no_save: threading.Event,
        stop_and_save: threading.Event,
        skip_sleep: threading.Event,
        *,
        lock: threading.RLock | None = None,
        interrupt_main: Callable[[], None] = send_sigint_to_process,
        logger: Callable[[str], None] = print,
        retry_input_available: bool = True,
    ) -> None:
        self._stop_no_save = stop_no_save
        self._stop_and_save = stop_and_save
        self._skip_sleep = skip_sleep
        self._interrupt_main = interrupt_main
        self._logger = logger
        self._lock = lock or threading.RLock()
        self._s_interrupt_sent = False
        self._allow_pose_deviation = False
        self._pose_deviation_policy_frozen = False
        self._cleanup_started = False
        self._unsafe_hold = False
        self._safety_retry = threading.Event()
        self._retry_input_available = bool(retry_input_available)

    @property
    def allow_pose_deviation(self) -> bool:
        with self._lock:
            return self._allow_pose_deviation

    @property
    def retry_input_available(self) -> bool:
        with self._lock:
            return self._retry_input_available

    def set_retry_input_available(self, available: bool) -> None:
        with self._lock:
            self._retry_input_available = bool(available)

    def retry_guidance(self) -> str:
        with self._lock:
            if self._retry_input_available:
                return "恢复反馈后按 s 显式重试。"
            return (
                "当前无交互键盘；保持 UNSAFE_HOLD。请由运维在"
                "交互终端接管并执行独立恢复。"
            )

    def handle_sigint(self) -> None:
        self._handle_termination("Ctrl+C")

    def handle_sigterm(self) -> None:
        self._handle_termination("SIGTERM")

    def _handle_termination(self, source: str) -> None:
        with self._lock:
            if self._unsafe_hold:
                self._logger(
                    f"\n[{source}] 当前为 UNSAFE_HOLD；仍有机械臂未确认 "
                    f"sleep，拒绝关闭。{self.retry_guidance()}"
                )
                return
            if self._cleanup_started:
                self._logger(
                    f"\n[{source}] 当前为 RECOVERY_IN_PROGRESS；"
                    "忽略中断，必须先完成逐臂 sleep 验证。"
                )
                return
            if self._stop_no_save.is_set():
                self._logger(
                    f"\n[{source}] 第二次中断：强制退出阻塞并进入 sleep 清理..."
                )
                raise KeyboardInterrupt
            if not self._pose_deviation_policy_frozen:
                self._allow_pose_deviation = False
                self._pose_deviation_policy_frozen = True
            self._skip_sleep.clear()
            self._stop_and_save.clear()
            self._stop_no_save.set()
            self._logger(
                f"\n[{source}] 请求安全停止：不保存数据，随后回到 sleep 位..."
            )

    def request_from_s(self, *, wake_main: bool = True) -> None:
        """Request authoritative no-save shutdown.

        ``wake_main=False`` is used while the robot is already returning. That
        motion loop will observe the stop event itself, avoiding a SIGINT that
        could unwind while its background move thread is active.
        """
        with self._lock:
            if self._unsafe_hold:
                self._safety_retry.set()
                self._logger(
                    "\n[s] UNSAFE_HOLD 显式重试已请求；"
                    "将重试当前独立进程或启动全新 safe-sleep。"
                )
                return
            self._request_no_save_locked("s", wake_main=wake_main)

    def request_no_save(
        self,
        *,
        source: str,
        wake_main: bool = True,
    ) -> None:
        with self._lock:
            if self._unsafe_hold:
                self._logger(
                    f"\n[{source}] 当前为 UNSAFE_HOLD；"
                    "保持安全恢复流程，不接受新的停止请求。"
                )
                return
            self._request_no_save_locked(source, wake_main=wake_main)

    def _request_no_save_locked(self, source: str, *, wake_main: bool) -> None:
        if self._cleanup_started:
            self._logger(
                f"\n[{source}] 已进入机器人清理阶段，忽略重复请求。"
            )
            return
        if self._s_interrupt_sent:
            self._logger(f"\n[{source}] 安全停止已在进行，忽略重复请求。")
            return
        self._s_interrupt_sent = True
        if not self._pose_deviation_policy_frozen:
            self._allow_pose_deviation = source == "s"
            self._pose_deviation_policy_frozen = True
        self._skip_sleep.clear()
        self._stop_and_save.clear()
        self._stop_no_save.set()
        if source == "s":
            self._logger(
                "\n[s] 已收到：停止采集并丢弃当前未完成 episode；"
                "随后启动独立 safe-sleep。"
            )
        else:
            self._logger(
                f"\n[{source}] 请求安全停止：不保存数据，"
                "随后回到 sleep 位并退出..."
            )
        if wake_main:
            self._interrupt_main()

    def request_save(self, *, skip_sleep: bool, source: str) -> bool:
        """Request a save unless no-save shutdown or cleanup already owns the run."""
        with self._lock:
            if self._stop_no_save.is_set() or self._cleanup_started:
                self._logger(
                    f"\n[{source}] 安全停止或清理已开始，忽略保存请求。"
                )
                return False
            self._stop_and_save.set()
            if skip_sleep:
                self._skip_sleep.set()
            else:
                self._skip_sleep.clear()
            return True

    def force_no_save(self, reason: str) -> None:
        """Make cleanup failure authoritative without interrupting cleanup."""
        with self._lock:
            if not self._pose_deviation_policy_frozen:
                self._allow_pose_deviation = False
                self._pose_deviation_policy_frozen = True
            self._skip_sleep.clear()
            self._stop_and_save.clear()
            self._stop_no_save.set()
            self._logger(f"\n[cleanup] 强制不保存：{reason}")

    def begin_cleanup(self) -> None:
        """Prevent terminal `s` from interrupting robot sleep cleanup."""
        with self._lock:
            self._cleanup_started = True
            if self._stop_no_save.is_set():
                self._skip_sleep.clear()

    def enter_unsafe_hold(self) -> None:
        with self._lock:
            self._cleanup_started = True
            self._unsafe_hold = True
            self._safety_retry.clear()

    def leave_unsafe_hold(self) -> None:
        with self._lock:
            self._unsafe_hold = False
            self._safety_retry.clear()

    def wait_for_safety_retry(self, timeout: float) -> bool:
        requested = self._safety_retry.wait(timeout)
        if requested:
            self._safety_retry.clear()
        return requested
