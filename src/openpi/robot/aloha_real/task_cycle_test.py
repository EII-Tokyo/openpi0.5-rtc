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
    with patch('openpi.robot.aloha_real.runtime.PolicyRecording'):
        r = Runtime(Mock(), Mock(), [], max_hz=50, policy_metadata={"runtime": {"task_control": {"mode": "exp_insert", "phase_field": "task_phase"}}})
    r._environment.refresh_observation.return_value = {'qpos': pose(), 'qvel': pose()}
    r._task_cycle.set_home(pose())
    r._publish_runtime_state = Mock()
    r._recording.snapshot.return_value = {'status': 'idle'}
    r._is_waiting_for_task = True
    return r


def task(number, command_id=None):
    return {'task_num': f'bottle_{number}', 'task_name': 'untrusted input', 'command_id': command_id}


def start(runtime, command):
    runtime._handle_task(command)
    runtime._activate_pending_task()


def returning(runtime):
    runtime._task_cycle.observe_phase('return_home')


def test_initial_home_never_completes_and_fast_crossing_does_not_complete():
    g = HomeReturnGate()
    g.set_home(pose())
    g.observe_phase("return_home")
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
    start(r, task(1, 'first'))
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
    start(r, task(1, 'first'))
    if queued:
        r._handle_task(task(2, 'second'))
    returning(r)
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
    r._activate_pending_task()
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
    start(r, task(1, 'first'))
    r._handle_task(task(2, 'second'))
    r._handle_task({'task_num': '4'})
    assert r._pending_task is None and r._current_task is None
    assert r._is_waiting_for_task
    r._handle_task(task(2, 'second'))
    assert r._current_task is None


def test_pause_arriving_during_reset_blocks_next_task(runtime):
    r = runtime
    start(r, task(1))
    r._handle_task(task(2))
    returning(r)
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
    r._environment.refresh_observation.return_value = {'qpos': pose(.3)}
    start(r, task(1))
    assert r._current_task is None
    r._policy.infer.assert_not_called()


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
    start(r, task(1))
    r._handle_task(task(2))
    returning(r)
    r._task_cycle.update(pose(.3), pose(), 0)
    r._task_cycle.update(pose(), pose(), 1)
    r._policy.reset.side_effect = lambda: setattr(r, '_latest_task', {'task_num': '4'})
    with patch('openpi.robot.aloha_real.runtime.time.monotonic', return_value=2):
        r._advance_home_cycle({'qpos': pose(), 'qvel': pose()})
    assert r._current_task is None and r._pending_task is None


def test_home_hold_is_recorded_before_episode_finishes(runtime):
    r = runtime
    start(r, task(1))
    r._recording.snapshot.return_value = {'status': 'recording'}
    returning(r)
    r._task_cycle.update(pose(.3), pose(), 0)
    r._task_cycle.update(pose(), pose(), 1)
    with patch('openpi.robot.aloha_real.runtime.time.monotonic', return_value=2):
        r._advance_home_cycle({'qpos': pose(), 'qvel': pose(), 'origin_observation': {'frame': 'home'}})
    assert r._recording.append.call_args.args[0] == {'frame': 'home'}
    r._recording.finish.assert_called_once()


def test_step_uses_hold_without_consuming_policy_chunk(runtime):
    r = runtime
    start(r, task(1))
    returning(r)
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


def test_home_without_final_phase_cannot_finish_or_take_over_actions():
    g = HomeReturnGate()
    g.set_home(pose())
    g.observe_phase('pick')
    g.update(pose(.3), pose(), 0)
    assert g.update(pose(), pose(), 1) == 'running'
    assert g.update(pose(), pose(), 10) == 'running'
    assert not g.settling
    g.observe_phase('return_home')
    assert g.update(pose(), pose(), 11) == 'settling'
    assert g.update(pose(), pose(), 11.31) == 'complete'


def test_latest_selection_during_cache_reset_wins(runtime):
    r = runtime
    start(r, task(1))
    r._handle_task(task(2))
    returning(r)
    r._task_cycle.update(pose(.3), pose(), 0)
    r._task_cycle.update(pose(), pose(), 1)
    r._policy.reset.side_effect = lambda: setattr(r, '_latest_task', task(3, 'latest-c'))
    with patch('openpi.robot.aloha_real.runtime.time.monotonic', return_value=2):
        r._advance_home_cycle({'qpos': pose(), 'qvel': pose()})
    assert r._current_task is None  # Boundary is not a task-start decision.
    r._activate_pending_task()
    assert r._current_task['task_num'] == 'bottle_3'
    assert r._pending_task is None and r._latest_task is None


def test_missing_phase_stops_before_motor_command(runtime):
    r = runtime
    start(r, task(1))
    r._environment.get_observation.return_value = {'qpos': pose(), 'qvel': pose(), 'images': {}}
    r._video_memory_preview = Mock()
    r._policy.infer.return_value = {'actions': pose()}
    r._step()
    r._environment.apply_action.assert_not_called()
    assert r._recording_hold.is_set() and r._recording_paused


def test_policy_metadata_controls_mode_and_low_level_checkpoint_is_rejected(monkeypatch):
    from openpi.robot.aloha_real.task_contract import task_contract
    monkeypatch.setenv('ALOHA_TASK_MODE', 'exp_insert')
    assert task_contract({}) == 'legacy'  # Process environment cannot mislabel a loaded model.
    assert task_contract({'runtime': {'task_control': {'mode': 'exp_insert', 'phase_field': 'task_phase'}}}) == 'exp_insert'
    with pytest.raises(ValueError, match='low-level only'):
        task_contract({'runtime': {'task_control': {'mode': 'exp_insert', 'phase_field': None}}})


def test_first_policy_input_after_wait_uses_refreshed_scene(runtime):
    # Execute the actual environment methods without importing ROS/hardware drivers.
    import ast
    import copy
    from pathlib import Path
    from types import SimpleNamespace, MethodType
    from collections import namedtuple
    source = ast.parse(Path(__file__).with_name('real_env.py').read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'AlohaRealEnvironment')
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {'refresh_observation', 'get_observation'}]
    namespace = {'copy': copy}
    exec(compile(ast.Module(body=methods, type_ignores=[]), '<actual environment methods>', 'exec'), namespace)
    old = {'qpos': pose(), 'qvel': pose(), 'effort': pose(), 'images': {'cam_high': 'old scene'}}
    fresh = {**old, 'images': {'cam_high': 'new scene'}}
    hardware = Mock()
    hardware.get_observation.return_value = fresh
    env = SimpleNamespace(_ts=namedtuple('TimeStep', 'observation')(old), _env=hardware,
                          _joint_history=None, _capture_joint_position=Mock(), apply_action=Mock(), image_recorder=Mock())
    for name, method in namespace.items():
        if name in {'refresh_observation', 'get_observation'}:
            setattr(env, name, MethodType(method, env))
    r = runtime
    r._environment = env
    r._video_memory_preview = Mock()
    r._policy.infer.return_value = {'actions': pose(), 'task_phase': 'pick'}
    start(r, task(1))
    hardware.step.assert_not_called()
    env.apply_action.assert_not_called()
    r._step()
    assert r._policy.infer.call_args.args[0]['images']['cam_high'] == 'new scene'


def test_current_training_config_does_not_claim_high_level_support():
    import ast
    from pathlib import Path
    from openpi.robot.aloha_real.task_contract import task_contract
    root = Path(__file__).parents[4]
    tree = ast.parse((root / 'src/openpi/training/config.py').read_text())
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        fields = {kw.arg: kw.value for kw in call.keywords}
        name = fields.get('name')
        if isinstance(name, ast.Constant) and name.value == 'eii_hxz_exp_insert_3repo_cam4_fullft_h200x2_448_no_mem_5k_20261119':
            metadata = ast.literal_eval(fields['policy_metadata'])
            with pytest.raises(ValueError, match='low-level only'):
                task_contract(metadata)
            break
    else:
        pytest.fail('Training config not found')
