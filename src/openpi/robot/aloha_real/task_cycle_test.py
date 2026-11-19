"""Offline home/queue integration tests: no ROS, network, model or actuators."""
from unittest.mock import Mock, patch

import numpy as np
import pytest

from openpi.robot.aloha_real.runtime import Runtime
from openpi.robot.aloha_real.task_cycle import HomeReturnGate


def pose(x=0):
    q = np.zeros(14)
    q[0] = x
    return q


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setenv('ALOHA_TASK_MODE', 'exp_insert')
    with patch('openpi.robot.aloha_real.runtime.PolicyRecording'):
        r = Runtime(Mock(), Mock(), [], max_hz=50)
    r._environment.get_joint_feedback.return_value = {'qpos': pose(), 'qvel': pose()}
    r._task_cycle.set_home(pose())
    r._publish_runtime_state = Mock()
    r._recording.snapshot.return_value = {'status': 'idle'}
    r._is_waiting_for_task = True
    return r


def task(number, command_id=None):
    return {'task_num': f'bottle_{number}', 'task_name': 'untrusted input', 'command_id': command_id}


def test_initial_home_never_completes_and_fast_crossing_does_not_complete():
    g = HomeReturnGate()
    g.set_home(pose())
    for t in range(10):
        assert g.update(pose(), pose(), t) == 'running'
    assert g.update(pose(.3), pose(), 10) == 'running'
    assert g.update(pose(), pose(.5), 11) == 'running'
    assert g.update(pose(), pose(), 12) == 'settling'
    assert g.update(pose(.06), pose(), 12.2) == 'settling'
    assert g.update(pose(), pose(), 13) == 'settling'
    assert g.update(pose(), pose(), 13.31) == 'complete'


@pytest.mark.parametrize('value', [None, [0]*13, [float('nan')]*14, [float('inf')]*14])
def test_bad_feedback_fails_closed(value):
    g = HomeReturnGate()
    g.set_home(pose())
    with pytest.raises(ValueError):
        g.update(value, pose(), 0)


def test_hold_preserves_grippers_and_uses_home_arm_reference():
    g = HomeReturnGate()
    g.set_home(pose())
    q = pose(.03)
    q[[6, 13]] = [.7, .9]
    action = g.hold_action(q)['actions']
    np.testing.assert_array_equal(action[g.ARM_INDICES], np.zeros(12))
    np.testing.assert_array_equal(action[[6, 13]], [.7, .9])


def test_queue_replace_keeps_current_prompt_and_does_not_reset_or_finish_recording(runtime):
    r = runtime
    r._handle_task(task(1, 'first'))
    r._policy.reset.reset_mock()
    r._recording.snapshot.return_value = {'status': 'recording'}
    r._handle_task(task(2, 'second'))
    r._handle_task(task(3, 'third'))
    assert r._current_task['task_name'] == r._BOTTLE_PROMPTS['bottle_1']
    assert r._pending_task['task_num'] == 'bottle_3'
    r._policy.reset.assert_not_called()
    r._recording.finish.assert_not_called()
    r._handle_task(task(2, 'second'))  # duplicate must not replace the newer choice
    assert r._pending_task['task_num'] == 'bottle_3'


@pytest.mark.parametrize('queued', [False, True])
def test_completion_waits_or_starts_next_only_after_reset(runtime, queued):
    r = runtime
    r._handle_task(task(1, 'first'))
    if queued:
        r._handle_task(task(2, 'second'))
    r._task_cycle.update(pose(.3), pose(), 0)
    calls = []
    r._policy.reset.side_effect = lambda: calls.append('reset')
    sub = Mock()
    sub.on_episode_start.side_effect = lambda: calls.append('start')
    r._subscribers = [sub]
    with patch('openpi.robot.aloha_real.runtime.time.monotonic', side_effect=[1, 1.31]):
        assert r._advance_home_cycle({'qpos': pose(), 'qvel': pose()})
        assert r._current_task['task_num'] == 'bottle_1'
        assert r._advance_home_cycle({'qpos': pose(), 'qvel': pose()})
    r._policy.infer.assert_not_called()
    assert r._pending_task is None
    sub.on_episode_end.assert_called_once()
    if queued:
        assert r._current_task['task_num'] == 'bottle_2'
        assert calls.index('reset') < calls.index('start')
        assert not r._is_waiting_for_task
    else:
        assert r._current_task is None and r._is_waiting_for_task
        assert calls == ['reset']


def test_stop_drops_next_task_and_repeated_command_cannot_restart(runtime):
    r = runtime
    r._handle_task(task(1, 'first'))
    r._handle_task(task(2, 'second'))
    r._handle_task({'task_num': '4'})
    assert r._pending_task is None and r._current_task is None
    assert r._is_waiting_for_task
    r._handle_task(task(2, 'second'))
    assert r._current_task is None


def test_pause_arriving_during_reset_blocks_next_task(runtime):
    r = runtime
    r._handle_task(task(1))
    r._handle_task(task(2))
    r._task_cycle.update(pose(.3), pose(), 0)
    r._task_cycle.update(pose(), pose(), 1)
    r._policy.reset.side_effect = r._recording_hold.set
    with patch('openpi.robot.aloha_real.runtime.time.monotonic', return_value=2):
        r._advance_home_cycle({'qpos': pose(), 'qvel': pose()})
    assert r._current_task is None and r._pending_task is None


def test_keyboard_three_is_bottle_and_h_is_manual(runtime):
    r = runtime
    assert r._build_task_from_key('3')['task_num'] == 'bottle_3'
    assert r._build_task_from_key('h', prompt_for_manual_dataset=False)['task_num'] == '3'
    assert r._build_task_from_key('2', allowed_task_nums=r._model_task_nums)['task_num'] == 'bottle_2'


def test_start_away_from_home_is_rejected(runtime):
    r = runtime
    r._environment.get_joint_feedback.return_value = {'qpos': pose(.3)}
    r._handle_task(task(1))
    assert r._current_task is None
    r._policy.reset.assert_not_called()


def test_legacy_mode_still_switches_immediately(monkeypatch):
    monkeypatch.setenv('ALOHA_TASK_MODE', 'legacy')
    with patch('openpi.robot.aloha_real.runtime.PolicyRecording'):
        r = Runtime(Mock(), Mock(), [])
    r._handle_task({'task_num': '1'})
    r._handle_task({'task_num': '2'})
    assert r._task_cycle is None and r._current_task['task_num'] == '2'
    assert r._build_task_from_key('3', prompt_for_manual_dataset=False)['task_num'] == '3'


def test_stop_received_during_reset_prevents_queued_start(runtime):
    r = runtime
    r._handle_task(task(1))
    r._handle_task(task(2))
    r._task_cycle.update(pose(.3), pose(), 0)
    r._task_cycle.update(pose(), pose(), 1)
    r._policy.reset.side_effect = lambda: setattr(r, '_latest_task', {'task_num': '4'})
    with patch('openpi.robot.aloha_real.runtime.time.monotonic', return_value=2):
        r._advance_home_cycle({'qpos': pose(), 'qvel': pose()})
    assert r._current_task is None and r._pending_task is None


def test_home_hold_is_recorded_before_episode_finishes(runtime):
    r = runtime
    r._handle_task(task(1))
    r._recording.snapshot.return_value = {'status': 'recording'}
    r._task_cycle.update(pose(.3), pose(), 0)
    r._task_cycle.update(pose(), pose(), 1)
    with patch('openpi.robot.aloha_real.runtime.time.monotonic', return_value=2):
        r._advance_home_cycle({'qpos': pose(), 'qvel': pose(), 'origin_observation': {'frame': 'home'}})
    assert r._recording.append.call_args.args[0] == {'frame': 'home'}
    r._recording.finish.assert_called_once()


def test_step_uses_hold_without_consuming_policy_chunk(runtime):
    r = runtime
    r._handle_task(task(1))
    r._task_cycle.update(pose(.3), pose(), 0)
    r._environment.get_observation.return_value = {'qpos': pose(), 'qvel': pose()}
    with patch('openpi.robot.aloha_real.runtime.time.monotonic', return_value=1):
        r._step()
    r._policy.infer.assert_not_called()
    assert r._task_cycle.settling
    r._environment.apply_action.assert_called_once()


def test_prompts_match_approved_dataset():
    import json
    from pathlib import Path
    annotations = Path(__file__).parents[4] / 'reports/exp_insert_qwen_formal_20261119/global_instructions.json'
    approved = json.loads(annotations.read_text())['instructions']
    assert Runtime._BOTTLE_PROMPTS == {f'bottle_{i}': text for i, text in approved.items()}
