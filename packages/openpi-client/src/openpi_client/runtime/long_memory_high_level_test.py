from concurrent.futures import Future
import base64
import io
import json
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest

from openpi.serving.high_level_archive import reconstruct_request
from openpi.shared.completed_step_memory import CURRENT_TASKS, COMPLETED_FACTS, SUBTASKS
from openpi.shared.long_memory import MemoryState, decode_memory, memory_prompt
from openpi_client.runtime.long_memory_high_level import (
    CAMERAS, SYSTEM_PROMPT, MEMORY_SCHEMA, GptMemoryBackend, LongMemoryPolicy,
    MemorySupervisor, MemoryUpdate, parse_update,
)


def payload(index=2, memory=COMPLETED_FACTS['pickup'], description='green bottle'):
    return {'bottle_description': description, 'memory': memory, 'current_task_index': index}


STAMP = {'camera_timestamps': {c: 2. for c in CAMERAS}, 'camera_timestamp_domain': 'unix_seconds'}


def images(value=3):
    return {c: np.full((9, 11, 3), value + i, np.uint8) for i, c in enumerate(CAMERAS)}


def result(value):
    return json.dumps({'id': 'fake-response', 'model': 'gpt-6.2-sol', 'status': 'completed',
                       'usage': {'input_tokens': 100, 'output_tokens': 20},
                       'output': [{'content': [{'type': 'output_text', 'text': json.dumps(value)}]}]}).encode()


def fake_dispatch(monkeypatch, value, before=None):
    sent = []
    def open_(request, **kwargs):
        sent.append(request)
        if before:
            before(request)
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = result(value)
        return response
    monkeypatch.setattr('urllib.request.urlopen', open_)
    return sent


def test_exact_approved_prompt_and_three_field_wire_schema():
    root = Path(__file__).resolve().parents[5]
    assert SYSTEM_PROMPT == (root / 'docs/sol-eight-task-prompt-en.txt').read_text().strip()
    assert MEMORY_SCHEMA['properties']['current_task_index']['enum'] == list(range(1, 9))
    assert set(MEMORY_SCHEMA['properties']) == {'bottle_description', 'memory', 'current_task_index'}
    assert 'dropout' not in SYSTEM_PROMPT.lower() and 'dataloader' not in SYSTEM_PROMPT.lower()


def test_request_is_archived_before_dispatch_and_uses_only_latest_four_original_views(tmp_path, monkeypatch):
    def before(request):
        assert list(tmp_path.rglob('request.json'))
        assert len(list((tmp_path/'images').rglob('*.png'))) == 4
    sent = fake_dispatch(monkeypatch, payload(), before)
    backend = GptMemoryBackend(api_key='mock-secret-never-persist', archive_root=tmp_path)
    old = MemoryState('old target', '', CURRENT_TASKS[0])
    update = backend(**STAMP, task='twist and dispose', memory=old, frames=[(1., images(40)), (2., images(3))], epoch=4)
    body = json.loads(sent[0].data)
    assert body['model'] == 'gpt-6.2-sol' and body['input'][0]['content'] == SYSTEM_PROMPT
    user = body['input'][1]['content']
    assert json.loads(user[0]['text'])['previous_state']['current_task'] == CURRENT_TASKS[0]
    assert len([x for x in user if x['type'] == 'input_image']) == 4
    for i, item in enumerate(x for x in user if x['type'] == 'input_image'):
        a = np.asarray(Image.open(io.BytesIO(base64.b64decode(item['image_url'].split(',')[1]))))
        np.testing.assert_array_equal(a, images(3)[CAMERAS[i]])
    assert reconstruct_request(tmp_path, Path(update.archive_directory)) == body
    rec = json.loads((Path(update.archive_directory)/'result.json').read_text())
    assert rec['usage']['input_tokens'] == 100 and rec['prediction'] == payload()
    assert rec['camera_timestamps'] == {c: 2. for c in CAMERAS}
    assert rec['known_cost_usd'] is None  # No invented price or paid test.
    assert 'mock-secret-never-persist' not in ''.join(p.read_text() for p in tmp_path.rglob('*.json'))
    backend.record_outcome(update, applied=True, reason='applied')
    assert json.loads((Path(update.archive_directory)/'application.json').read_text())['applied']


def test_archive_failure_blocks_paid_dispatch(tmp_path, monkeypatch):
    calls = fake_dispatch(monkeypatch, payload())
    backend = GptMemoryBackend(api_key='mock', archive_root=tmp_path)
    backend.archive.prepare = Mock(side_effect=OSError('disk full'))
    with pytest.raises(OSError):backend(**STAMP, task='goal', memory=MemoryState(), frames=[(0, images())])
    assert not calls


def test_missing_fourth_camera_never_dispatches(tmp_path, monkeypatch):
    calls = fake_dispatch(monkeypatch, payload())
    backend = GptMemoryBackend(api_key='mock', archive_root=tmp_path)
    photo = images();photo.pop('cam_low')
    with pytest.raises(ValueError):backend(**STAMP, task='goal', memory=MemoryState(), frames=[(0, photo)])
    assert not calls


def test_repeated_direct_requests_have_distinct_archive_records(tmp_path, monkeypatch):
    fake_dispatch(monkeypatch, payload())
    backend = GptMemoryBackend(api_key='mock', archive_root=tmp_path)
    a = backend(**STAMP, task='goal', memory=MemoryState(), frames=[(0, images())])
    b = backend(**STAMP, task='goal', memory=MemoryState(), frames=[(1, images())])
    assert a.archive_directory != b.archive_directory


def test_invalid_wire_response_is_saved_but_never_accepted(tmp_path, monkeypatch):
    fake_dispatch(monkeypatch, {**payload(), 'cycle_complete': False})
    backend = GptMemoryBackend(api_key='mock', archive_root=tmp_path)
    with pytest.raises(ValueError):backend(**STAMP, task='goal', memory=MemoryState(), frames=[(0, images())])
    record = next(tmp_path.rglob('result.json'))
    assert json.loads(record.read_text())['status'] == 'error'
    assert (record.parent/'response.json').exists()
    assert json.loads((record.parent/'application.json').read_text())['applied'] is False


def set_reply(supervisor, value, *, generation=None, age=.1):
    supervisor.future_started = time.monotonic()
    supervisor.future = Future()
    supervisor.future.set_result((supervisor.generation if generation is None else generation,
                                 1., parse_update(value), age))


def test_gpt_can_reset_old_bottle_at_new_pickup_and_correct_wrong_memory():
    s = MemorySupervisor(lambda **kwargs: None)
    try:
        s.memory = MemoryState('old bottle', COMPLETED_FACTS['pickup'] + ' ' + COMPLETED_FACTS['uncap'], CURRENT_TASKS[7])
        set_reply(s, payload(1, '', 'new bottle'));s._poll()
        assert s.memory == MemoryState('new bottle', '', CURRENT_TASKS[0])
        s.memory = MemoryState('new bottle', COMPLETED_FACTS['pickup'] + ' ' + COMPLETED_FACTS['uncap'], CURRENT_TASKS[1])
        set_reply(s, payload(2));s._poll()
        assert s.memory.memory == COMPLETED_FACTS['pickup']  # Complete authoritative replacement, no local append.
    finally:s.close()


def test_box_reset_preserves_provided_memory_and_final_return_keeps_task7():
    s = MemorySupervisor(lambda **kwargs: None)
    try:
        set_reply(s, payload(8));s._poll();assert s.memory.memory == COMPLETED_FACTS['pickup']
        set_reply(s, payload(7, '', ''));s._poll()
        assert s.memory == MemoryState('', '', CURRENT_TASKS[6])
    finally:s.close()


def test_late_reply_or_explicit_episode_reset_cannot_resurrect_old_target():
    backend = Mock();s = MemorySupervisor(backend)
    try:
        old_generation = s.generation;s.reset()
        set_reply(s, payload(), generation=old_generation);s._poll();assert s.memory == MemoryState()
        set_reply(s, payload(), age=9.);s._poll();assert s.memory == MemoryState()
        backend.record_outcome.assert_called_with(parse_update(payload()), applied=False, reason='late_or_reset')
        s.future = Future();pending = s.future;s.reset()
        # Simulate a request already in flight, which cannot be cancelled.
        pending2 = Future();pending2.set_running_or_notify_cancel();s.future = pending2;s.reset()
        pending2.set_result((old_generation, 1., parse_update(payload()), .1))
        assert s.memory == MemoryState()
        backend.record_outcome.assert_called_with(parse_update(payload()), applied=False, reason='episode_reset')
    finally:s.close()


def test_real_backend_to_low_level_preserves_separate_fields_and_training_text(tmp_path, monkeypatch):
    fake_dispatch(monkeypatch, payload())
    backend = GptMemoryBackend(api_key='mock', archive_root=tmp_path)
    s = MemorySupervisor(backend);low = Mock();low.infer.return_value = {'actions': np.zeros((2, 14))}
    p = LongMemoryPolicy(low, s, first_result_timeout=2)
    obs = {'prompt': 'twist and dispose', 'images': images(), 'state': np.zeros(14), **STAMP}
    try:
        p.infer(obs, chunking_mode='training_time', action_prefix=np.zeros((2, 14)))
        args, kwargs = low.infer.call_args;state = decode_memory(args[0]['long_memory'])
        assert state.memory == COMPLETED_FACTS['pickup'] and state.current_task == CURRENT_TASKS[1]
        assert args[0]['prompt'] == obs['prompt'] and kwargs['chunking_mode'] == 'training_time'
        assert memory_prompt(state) == ('Bottle description: green bottle\nMemory: I picked up the bottle with my left gripper.\nCurrent task: Unscrew the bottle cap.')
        assert 'current_task_index' not in json.loads(args[0]['long_memory'])
        assert json.loads(next(tmp_path.rglob('application.json')).read_text())['applied']
    finally:s.close()


@pytest.mark.parametrize('text', [COMPLETED_FACTS['adjust'], COMPLETED_FACTS['bottle_bin'], COMPLETED_FACTS['cap_bin']])
def test_merged_task_rejects_adjustment_only_or_half_compound_records(text):
    with pytest.raises(ValueError):parse_update(payload(memory=text))


def test_whole_compound_record_is_accepted():
    assert parse_update(payload(1, SUBTASKS[2]['memory_en'])).state.memory == SUBTASKS[2]['memory_en']


def test_duplicate_completion_is_rejected():
    with pytest.raises(ValueError):parse_update(payload(memory=COMPLETED_FACTS['pickup'] + ' ' + COMPLETED_FACTS['pickup']))


def test_failed_old_request_does_not_poison_new_episode_but_archive_failure_blocks():
    backend = Mock();s = MemorySupervisor(backend)
    try:
        f = Future();f.set_running_or_notify_cancel();s.future=f;s.reset()
        f.set_exception(RuntimeError('network failed'))
        assert s.archive_error is None and s.memory == MemoryState()
        f = Future();f.set_running_or_notify_cancel();s.future=f;s.reset()
        backend.record_outcome.side_effect=OSError('archive full')
        f.set_result((1, 1., parse_update(payload()), .1))
        with pytest.raises(RuntimeError, match='archive failure'):s.observe(images(), now_s=0, task='goal')
    finally:s.close()


def test_missing_or_misaligned_camera_timestamps_block_dispatch(tmp_path, monkeypatch):
    calls = fake_dispatch(monkeypatch, payload())
    backend = GptMemoryBackend(api_key='mock', archive_root=tmp_path)
    with pytest.raises(ValueError, match='timestamps'):backend(task='goal',memory=MemoryState(),frames=[(0,images())])
    stamps={**STAMP['camera_timestamps'],'cam_low':3.}
    with pytest.raises(ValueError, match='alignment'):backend(task='goal',memory=MemoryState(),frames=[(0,images())],camera_timestamps=stamps,camera_timestamp_domain='unix_seconds')
    assert not calls
