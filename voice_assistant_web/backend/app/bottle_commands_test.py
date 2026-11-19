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
    monkeypatch.setattr(redis_commands, 'EXP_INSERT', True)
    assert redis_commands.bottle_task_from_text(text) == expected


def test_legacy_does_not_interpret_bottle_numbers(monkeypatch):
    monkeypatch.setattr(redis_commands, 'EXP_INSERT', False)
    assert redis_commands.bottle_task_from_text('3') is None


def test_voice_number_dispatch_without_language_model(monkeypatch):
    monkeypatch.setattr(redis_commands, 'EXP_INSERT', True)
    monkeypatch.setitem(redis_commands.TASK_MAPPING, 'bottle_3', 'Bottle 3')
    redis = Mock()
    redis.get.return_value = None
    engine = VoiceAssistantEngine.__new__(VoiceAssistantEngine)
    engine._redis = redis
    engine._openai = None
    response = asyncio.run(engine.process_text('3号', language='zh'))
    message = json.loads(redis.publish.call_args.args[1])
    assert response.task_number == 'bottle_3'
    assert message['task'] == 'bottle_3' and message['command_id']
    assert '暂存' in response.reply_text
