import threading
from concurrent.futures import ThreadPoolExecutor

from aloha.record_trigger import (
    RecordingEvents,
    RecordingPhase,
    RecordingTriggerController,
    TriggerResult,
)


def make_controller(start_trigger="b", *, ready=True):
    events = RecordingEvents.create()
    controller = RecordingTriggerController(events, start_trigger=start_trigger)
    if ready:
        assert controller.complete_preparation() is True
    return events, controller


def test_new_controller_rejects_b_until_preparation_completes():
    events, controller = make_controller(ready=False)

    assert controller.phase is RecordingPhase.PREPARING
    assert controller.handle_b() is TriggerResult.NOT_READY
    assert not events.recording_started.is_set()
    assert not events.return_to_start.is_set()

    assert controller.complete_preparation() is True
    assert controller.phase is RecordingPhase.WAITING_FOR_B
    assert controller.handle_b() is TriggerResult.STARTED


def test_preparation_can_auto_start_gripper_recording():
    events, controller = make_controller(start_trigger="gripper", ready=False)

    assert controller.complete_preparation(auto_start=True) is True

    assert controller.phase is RecordingPhase.RECORDING
    assert events.recording_started.is_set()
    assert not events.return_to_start.is_set()


def test_b_before_first_sample_does_not_stop_or_create_return_event():
    events, controller = make_controller()
    assert controller.handle_b() is TriggerResult.STARTED

    assert controller.handle_b() is TriggerResult.NO_SAMPLES

    assert controller.phase is RecordingPhase.RECORDING
    assert events.recording_started.is_set()
    assert not events.return_to_start.is_set()


def test_b_after_first_sample_keeps_normal_stop_transition():
    events, controller = make_controller()
    assert controller.handle_b() is TriggerResult.STARTED

    assert controller.mark_sample_recorded() is True
    assert controller.handle_b() is TriggerResult.STOPPED

    assert controller.phase is RecordingPhase.RETURNING_TO_SAVE
    assert events.return_to_start.is_set()


def test_existing_five_event_constructor_gets_clear_discard_event():
    events = RecordingEvents(
        recording_started=threading.Event(),
        return_to_start=threading.Event(),
        stop_and_save=threading.Event(),
        stop_no_save=threading.Event(),
        skip_sleep=threading.Event(),
    )

    assert isinstance(events.discard_and_retry, threading.Event)
    assert not events.discard_and_retry.is_set()


def test_two_b_presses_start_then_stop():
    events, controller = make_controller()

    assert controller.phase is RecordingPhase.WAITING_FOR_B
    assert controller.handle_b() is TriggerResult.STARTED
    assert controller.phase is RecordingPhase.RECORDING
    assert events.recording_started.is_set()

    assert controller.mark_sample_recorded() is True
    assert controller.handle_b() is TriggerResult.STOPPED
    assert controller.phase is RecordingPhase.RETURNING_TO_SAVE
    assert events.return_to_start.is_set()
    assert not events.stop_and_save.is_set()
    assert not events.stop_no_save.is_set()
    assert not events.skip_sleep.is_set()


def test_b_before_recording_is_rejected_in_gripper_mode():
    events, controller = make_controller(start_trigger="gripper")

    assert controller.handle_b() is TriggerResult.WRONG_START_MODE
    assert not events.recording_started.is_set()
    assert not events.stop_and_save.is_set()


def test_b_stops_after_gripper_started_recording():
    events, controller = make_controller(start_trigger="gripper")
    events.recording_started.set()

    assert controller.mark_sample_recorded() is True
    assert controller.handle_b() is TriggerResult.STOPPED
    assert controller.phase is RecordingPhase.RETURNING_TO_SAVE
    assert events.return_to_start.is_set()
    assert not events.stop_and_save.is_set()
    assert not events.stop_no_save.is_set()
    assert not events.skip_sleep.is_set()


def test_commands_after_stop_are_ignored():
    _, controller = make_controller()
    controller.handle_b()
    controller.mark_sample_recorded()
    controller.handle_b()

    assert controller.handle_b() is TriggerResult.IGNORED


def test_external_stop_events_make_b_ignored():
    events, controller = make_controller()
    events.stop_no_save.set()

    assert controller.handle_b() is TriggerResult.IGNORED
    assert not events.recording_started.is_set()


def test_d_while_recording_starts_discard_and_retry():
    events, controller = make_controller()
    assert controller.handle_b() is TriggerResult.STARTED

    assert controller.handle_d() is TriggerResult.DISCARD_STARTED

    assert controller.phase is RecordingPhase.RETURNING_TO_RETRY
    assert events.discard_and_retry.is_set()
    assert events.return_to_start.is_set()
    assert not events.stop_and_save.is_set()


def test_d_before_recording_reports_not_recording():
    events, controller = make_controller()

    assert controller.handle_d() is TriggerResult.NOT_RECORDING
    assert controller.phase is RecordingPhase.WAITING_FOR_B
    assert not events.discard_and_retry.is_set()


def test_repeated_d_and_b_are_ignored_while_returning_to_retry():
    _, controller = make_controller()
    controller.handle_b()
    assert controller.handle_d() is TriggerResult.DISCARD_STARTED

    assert controller.handle_d() is TriggerResult.IGNORED
    assert controller.handle_b() is TriggerResult.IGNORED


def test_complete_retry_resets_attempt_state_and_allows_another_b():
    events, controller = make_controller()
    controller.handle_b()
    controller.handle_d()

    controller.complete_retry()

    assert controller.phase is RecordingPhase.WAITING_FOR_B
    assert not events.recording_started.is_set()
    assert not events.discard_and_retry.is_set()
    assert not events.return_to_start.is_set()
    assert controller.handle_b() is TriggerResult.STARTED
    assert controller.phase is RecordingPhase.RECORDING


def test_complete_retry_can_auto_start_gripper_attempt():
    events, controller = make_controller(start_trigger="gripper")
    events.recording_started.set()
    assert controller.handle_d() is TriggerResult.DISCARD_STARTED

    controller.complete_retry(auto_start=True)

    assert controller.phase is RecordingPhase.RECORDING
    assert events.recording_started.is_set()
    assert not events.discard_and_retry.is_set()
    assert not events.return_to_start.is_set()


def test_complete_retry_preserves_retry_state_after_process_stop():
    for stop_event_name in ("stop_no_save", "stop_and_save"):
        events, controller = make_controller()
        controller.handle_b()
        controller.handle_d()
        getattr(events, stop_event_name).set()

        assert controller.complete_retry() is False

        assert controller.phase is RecordingPhase.RETURNING_TO_RETRY
        assert events.recording_started.is_set()
        assert events.discard_and_retry.is_set()
        assert events.return_to_start.is_set()


def test_complete_save_resets_state_and_allows_another_b():
    events, controller = make_controller()
    assert controller.handle_b() is TriggerResult.STARTED
    assert controller.mark_sample_recorded() is True
    assert controller.handle_b() is TriggerResult.STOPPED

    assert controller.complete_save() is True

    assert controller.phase is RecordingPhase.PREPARING
    assert not events.recording_started.is_set()
    assert not events.return_to_start.is_set()
    assert controller.complete_preparation() is True
    assert controller.handle_b() is TriggerResult.STARTED
    assert controller.phase is RecordingPhase.RECORDING


def test_complete_save_handoff_allows_rearm_before_publication():
    events, controller = make_controller()
    assert controller.handle_b() is TriggerResult.STARTED
    assert controller.mark_sample_recorded() is True
    assert controller.handle_b() is TriggerResult.STOPPED

    assert controller.complete_save_handoff() is True

    assert controller.phase is RecordingPhase.PREPARING
    assert not events.recording_started.is_set()
    assert not events.return_to_start.is_set()


def test_sample_latch_resets_after_save_handoff_and_next_preparation():
    _, controller = make_controller()
    assert controller.handle_b() is TriggerResult.STARTED
    assert controller.mark_sample_recorded() is True
    assert controller.handle_b() is TriggerResult.STOPPED
    assert controller.complete_save_handoff() is True
    assert controller.complete_preparation() is True
    assert controller.handle_b() is TriggerResult.STARTED

    assert controller.handle_b() is TriggerResult.NO_SAMPLES


def test_complete_save_preserves_return_state_when_process_stop_wins():
    for stop_event_name in ("stop_no_save", "stop_and_save"):
        events, controller = make_controller()
        controller.handle_b()
        controller.mark_sample_recorded()
        controller.handle_b()
        getattr(events, stop_event_name).set()

        assert controller.complete_save() is False

        assert controller.phase is RecordingPhase.RETURNING_TO_SAVE
        assert events.recording_started.is_set()
        assert events.return_to_start.is_set()
        assert getattr(events, stop_event_name).is_set()


def test_external_process_stop_events_reject_d():
    for stop_event_name in ("stop_no_save", "stop_and_save"):
        events, controller = make_controller()
        controller.handle_b()
        getattr(events, stop_event_name).set()

        assert controller.handle_d() is TriggerResult.IGNORED
        assert not events.discard_and_retry.is_set()


def test_d_is_ignored_after_second_b_commits_save():
    events, controller = make_controller()
    assert controller.handle_b() is TriggerResult.STARTED
    assert controller.mark_sample_recorded() is True
    assert controller.handle_b() is TriggerResult.STOPPED

    assert controller.handle_d() is TriggerResult.IGNORED
    assert controller.phase is RecordingPhase.RETURNING_TO_SAVE
    assert not events.discard_and_retry.is_set()


def test_concurrent_b_and_d_choose_exactly_one_return_transition():
    events, controller = make_controller()
    assert controller.handle_b() is TriggerResult.STARTED
    assert controller.mark_sample_recorded() is True
    barrier = threading.Barrier(8)

    def press_b():
        barrier.wait()
        return controller.handle_b()

    def press_d():
        barrier.wait()
        return controller.handle_d()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(press_b) for _ in range(4)]
        futures += [executor.submit(press_d) for _ in range(4)]
        results = [future.result() for future in futures]

    committed = [
        result
        for result in results
        if result in {TriggerResult.STOPPED, TriggerResult.DISCARD_STARTED}
    ]
    assert len(committed) == 1
    assert results.count(TriggerResult.IGNORED) == 7
    assert events.return_to_start.is_set()
    expected_phase = (
        RecordingPhase.RETURNING_TO_SAVE
        if committed[0] is TriggerResult.STOPPED
        else RecordingPhase.RETURNING_TO_RETRY
    )
    assert controller.phase is expected_phase
    assert events.discard_and_retry.is_set() is (
        expected_phase is RecordingPhase.RETURNING_TO_RETRY
    )
    assert not events.stop_and_save.is_set()
    assert not events.stop_no_save.is_set()
    assert not events.skip_sleep.is_set()


def test_concurrent_second_press_stops_once():
    events, controller = make_controller()
    assert controller.handle_b() is TriggerResult.STARTED
    assert controller.mark_sample_recorded() is True
    barrier = threading.Barrier(8)

    def press():
        barrier.wait()
        return controller.handle_b()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: press(), range(8)))

    assert results.count(TriggerResult.STOPPED) == 1
    assert results.count(TriggerResult.IGNORED) == 7
    assert events.return_to_start.is_set()
    assert controller.phase is RecordingPhase.RETURNING_TO_SAVE
    assert not events.stop_and_save.is_set()
    assert not events.stop_no_save.is_set()
    assert not events.skip_sleep.is_set()
