import threading

from aloha.command_coordinator import RecorderCommandCoordinator
from aloha.record_trigger import (
    RecordingEvents,
    RecordingPhase,
    RecordingTriggerController,
    TriggerResult,
)
from aloha.safe_stop import SafeStopController


def make_coordinator(*, logger=lambda _message: None):
    lock = threading.RLock()
    events = RecordingEvents.create()
    interrupts = []
    trigger = RecordingTriggerController(events, start_trigger="b", lock=lock)
    safe_stop = SafeStopController(
        events.stop_no_save,
        events.stop_and_save,
        events.skip_sleep,
        lock=lock,
        interrupt_main=lambda: interrupts.append("interrupt"),
        logger=logger,
    )
    coordinator = RecorderCommandCoordinator(trigger, safe_stop, lock=lock)
    assert trigger.complete_preparation() is True
    return events, trigger, safe_stop, coordinator, interrupts


def test_named_no_save_uses_phase_aware_main_thread_wakeup():
    events, trigger, _, coordinator, interrupts = make_coordinator()
    assert coordinator.handle_b() is TriggerResult.STARTED

    coordinator.request_no_save(source="foot-pedal-failure")

    assert events.stop_no_save.is_set()
    assert interrupts == ["interrupt"]


def test_named_no_save_is_deferred_during_return_motion():
    events, trigger, _, coordinator, interrupts = make_coordinator()
    assert coordinator.handle_b() is TriggerResult.STARTED
    assert trigger.mark_sample_recorded() is True
    assert coordinator.handle_b() is TriggerResult.STOPPED
    assert trigger.phase is RecordingPhase.RETURNING_TO_SAVE

    coordinator.request_no_save(source="foot-pedal-failure")

    assert events.stop_no_save.is_set()
    assert interrupts == []


def test_d_transition_authoritatively_rejects_m_and_r():
    events, trigger, _, coordinator, _ = make_coordinator()
    assert coordinator.handle_b() is TriggerResult.STARTED
    assert coordinator.handle_d() is TriggerResult.DISCARD_STARTED

    assert coordinator.request_save(skip_sleep=False, source="m") is False
    assert coordinator.request_save(skip_sleep=True, source="r") is False
    assert trigger.phase is RecordingPhase.RETURNING_TO_RETRY
    assert events.discard_and_retry.is_set()
    assert not events.stop_and_save.is_set()
    assert not events.skip_sleep.is_set()


def test_m_winning_before_d_makes_later_d_ignored():
    events, _, _, coordinator, _ = make_coordinator()
    assert coordinator.handle_b() is TriggerResult.STARTED
    assert coordinator.request_save(skip_sleep=False, source="m") is True

    assert coordinator.handle_d() is TriggerResult.IGNORED
    assert events.stop_and_save.is_set()
    assert not events.discard_and_retry.is_set()


def test_s_phase_snapshot_and_deferred_request_share_one_lock():
    events, trigger, _, coordinator, interrupts = make_coordinator()
    assert coordinator.handle_b() is TriggerResult.STARTED
    assert trigger.mark_sample_recorded() is True
    assert coordinator.handle_b() is TriggerResult.STOPPED
    assert trigger.phase is RecordingPhase.RETURNING_TO_SAVE
    assert events.return_to_start.is_set()
    assert not events.stop_no_save.is_set()
    assert not events.stop_and_save.is_set()
    assert not events.skip_sleep.is_set()

    coordinator.request_no_save_from_s()

    assert trigger.phase is RecordingPhase.RETURNING_TO_SAVE
    assert events.stop_no_save.is_set()
    assert not events.stop_and_save.is_set()
    assert not events.skip_sleep.is_set()
    assert interrupts == []


def test_s_while_recording_still_wakes_main():
    events, _, _, coordinator, interrupts = make_coordinator()
    assert coordinator.handle_b() is TriggerResult.STARTED

    coordinator.request_no_save_from_s()

    assert events.stop_no_save.is_set()
    assert interrupts == ["interrupt"]


def test_s_routes_authoritative_discard_then_standalone_sleep_message():
    logs = []
    events, _, _, coordinator, interrupts = make_coordinator(
        logger=logs.append
    )
    coordinator.request_no_save_from_s()

    assert events.stop_no_save.is_set()
    assert interrupts == ["interrupt"]
    assert logs == [
        "\n[s] 已收到：停止采集并丢弃当前未完成 episode；"
        "随后启动独立 safe-sleep。"
    ]


def test_d_holds_shared_lock_until_transition_so_concurrent_m_cannot_pass():
    events, trigger, _, coordinator, _ = make_coordinator()
    assert coordinator.handle_b() is TriggerResult.STARTED
    d_inside_lock = threading.Event()
    release_d = threading.Event()
    m_finished = threading.Event()
    results = {}
    original_handle_d = trigger.handle_d

    def delayed_handle_d():
        d_inside_lock.set()
        release_d.wait(timeout=1.0)
        return original_handle_d()

    trigger.handle_d = delayed_handle_d
    d_thread = threading.Thread(
        target=lambda: results.setdefault("d", coordinator.handle_d())
    )
    m_thread = threading.Thread(
        target=lambda: (
            results.setdefault(
                "m",
                coordinator.request_save(skip_sleep=False, source="m"),
            ),
            m_finished.set(),
        )
    )

    d_thread.start()
    assert d_inside_lock.wait(timeout=1.0)
    m_thread.start()
    assert not m_finished.wait(timeout=0.05)
    release_d.set()
    d_thread.join(timeout=1.0)
    m_thread.join(timeout=1.0)

    assert results == {
        "d": TriggerResult.DISCARD_STARTED,
        "m": False,
    }
    assert events.discard_and_retry.is_set()
    assert not events.stop_and_save.is_set()


def test_s_holds_shared_lock_between_phase_snapshot_and_no_save_mutation():
    events, trigger, safe_stop, coordinator, interrupts = make_coordinator()
    assert coordinator.handle_b() is TriggerResult.STARTED
    s_inside_lock = threading.Event()
    release_s = threading.Event()
    b_finished = threading.Event()
    results = {}
    original_request = safe_stop.request_from_s

    def delayed_request_from_s(*, wake_main):
        assert wake_main is True
        s_inside_lock.set()
        release_s.wait(timeout=1.0)
        return original_request(wake_main=wake_main)

    safe_stop.request_from_s = delayed_request_from_s
    s_thread = threading.Thread(target=coordinator.request_no_save_from_s)
    b_thread = threading.Thread(
        target=lambda: (
            results.setdefault("b", coordinator.handle_b()),
            b_finished.set(),
        )
    )

    s_thread.start()
    assert s_inside_lock.wait(timeout=1.0)
    b_thread.start()
    assert not b_finished.wait(timeout=0.05)
    release_s.set()
    s_thread.join(timeout=1.0)
    b_thread.join(timeout=1.0)

    assert events.stop_no_save.is_set()
    assert interrupts == ["interrupt"]
    assert results["b"] is TriggerResult.IGNORED
    assert trigger.phase is RecordingPhase.RECORDING
