import asyncio
import json
from unittest.mock import Mock

import pytest

from . import redis_commands
from .voice_session import VoiceAssistantEngine


@pytest.mark.parametrize('text,expected', [('1', 'bottle_1'), ('2号', 'bottle_2'), ('三号瓶子。', 'bottle_3'),
                                         ('bottle 3', 'bottle_3'), ('停止3号', None), ('1号或2号', None),
                                         ('人工接管', None), ('不要3号', None)])
def test_explicit_selection(monkeypatch, text, expected):
    assert redis_commands.bottle_task_from_text(text) == expected


def test_legacy_does_not_interpret_bottle_numbers(monkeypatch):
    redis = Mock()
    redis.get.return_value = None
    assert not redis_commands.runtime_supports_bottles(redis)


def test_voice_number_dispatch_without_language_model(monkeypatch):
    monkeypatch.setitem(redis_commands.TASK_MAPPING, 'bottle_3', 'Bottle 3')
    redis = Mock()
    redis.get.side_effect = lambda key: json.dumps({'mode': 'exp_insert', 'task_ids': ['bottle_1','bottle_2','bottle_3','3','4','5','6']}) if key == 'aloha_task_capabilities' else None
    engine = VoiceAssistantEngine.__new__(VoiceAssistantEngine)
    engine._redis = redis
    engine._openai = None
    response = asyncio.run(engine.process_text('3号', language='zh'))
    message = json.loads(redis.publish.call_args.args[1])
    assert response.task_number == 'bottle_3'
    assert message['task'] == 'bottle_3' and message['command_id']
    assert '暂存' in response.reply_text


def test_original_routes_remain_and_unsupported_model_returns_conflict(monkeypatch):
    from fastapi.testclient import TestClient
    from . import main
    redis = Mock()
    capabilities = {'mode': 'legacy', 'task_ids': ['1', '2', '3', '4', '5', '6']}
    redis.get.side_effect = lambda key: json.dumps(capabilities) if key == 'aloha_task_capabilities' else None
    monkeypatch.setattr(main, 'redis_client', redis)
    client = TestClient(main.app)
    for number in ('1', '2'):
        assert client.post('/api/tasks/' + number).status_code == 200
    assert client.post('/api/tasks/bottle_1').status_code == 409
    capabilities.update(mode='exp_insert', task_ids=['bottle_1', 'bottle_2', 'bottle_3', '3', '4', '5', '6'])
    for number in ('1', '2'):
        response = client.post('/api/tasks/' + number)
        assert response.status_code == 409  # Existing route, incompatible loaded model; never 404 or reinterpretation.
    assert client.post('/api/tasks/bottle_3').status_code == 200
    assert json.loads(redis.publish.call_args.args[1])['task'] == 'bottle_3'
