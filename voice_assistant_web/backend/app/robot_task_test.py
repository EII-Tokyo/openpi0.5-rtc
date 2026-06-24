import json
import time

import pytest
from fastapi import HTTPException

from voice_assistant_web.backend.app import main
from voice_assistant_web.backend.app.schemas import RobotTaskRequest


class _FakeRedis:
    def __init__(self):
        self.messages = []

    def publish(self, channel, payload):
        self.messages.append((channel, payload))


class _FakeRobotStateBridge:
    def __init__(self, snapshot):
        self._snapshot = snapshot

    def snapshot(self):
        return dict(self._snapshot)


def test_robot_task_rejects_when_runtime_heartbeat_is_stale(monkeypatch):
    fake_redis = _FakeRedis()
    monkeypatch.setattr(main, "redis_client", fake_redis)
    monkeypatch.setattr(
        main,
        "robot_state_bridge",
        _FakeRobotStateBridge({"runtime_timestamp": time.time() - 10.0, "mode": "sleep"}),
    )

    with pytest.raises(HTTPException) as exc:
        main.robot_task(RobotTaskRequest(task_num="4", source="test"))

    assert exc.value.status_code == 409
    assert "runtime is not listening" in str(exc.value.detail)
    assert fake_redis.messages == []


def test_robot_task_allows_shutdown_and_live_runtime_commands(monkeypatch):
    fake_redis = _FakeRedis()
    monkeypatch.setattr(main, "redis_client", fake_redis)
    monkeypatch.setattr(
        main,
        "robot_state_bridge",
        _FakeRobotStateBridge({"runtime_timestamp": time.time(), "mode": "waiting"}),
    )

    response = main.robot_task(RobotTaskRequest(task_num="9", source="test"))

    assert response == {"status": "ok", "task_num": "9", "task_name": "shutdown"}
    payload = json.loads(fake_redis.messages[-1][1])
    assert payload["type"] == "robot_task"
    assert payload["task_num"] == "9"
