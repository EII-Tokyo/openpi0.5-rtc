import json
import threading
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from voice_assistant_web.backend.app import main, redis_commands
from openpi.robot.aloha_real.runtime import Runtime


class Redis:
    def __init__(self):
        self.values = {}
        self.messages = []

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value

    def publish(self, channel, message):
        self.messages.append(json.loads(message))

    def pipeline(self):
        return self

    def execute(self):
        return []


@pytest.fixture
def client(monkeypatch):
    redis = Redis()
    monkeypatch.setattr(main, "redis_client", redis)
    # Deliberately do not enter the lifespan: no ROS nodes, workers or robot start.
    return TestClient(main.app), redis


def test_settings_and_runtime_end_to_end(client):
    http, redis = client
    assert http.get("/api/inference-settings").json() == {
        "operator": None,
        "rating": None,
        "operators": ["Jia Long", "Xinzheng He", "Zhaonan Ma"],
    }
    assert http.put("/api/inference-settings", json={"operator": "Jia Long", "rating": 5}).status_code == 200
    assert redis.messages[-1]["type"] == "inference_settings"
    assert "task" not in redis.messages[-1]  # Saving must not issue home/start.
    assert http.post("/api/tasks/1").status_code == 200
    message = redis.messages[-1]
    assert message["operator"] == "Jia Long" and message["rating"] == 5
    runtime = Runtime.__new__(Runtime)
    runtime._redis_client = redis
    runtime._redis_running = True
    runtime._task_lock = threading.Lock()

    def receive(**kwargs):
        runtime._redis_running = False
        return {"type": "message", "data": json.dumps(message)}

    redis.pubsub = lambda: SimpleNamespace(subscribe=lambda *a: None, get_message=receive, close=lambda: None)
    runtime._redis_listener()
    assert runtime._latest_task["operator"] == "Jia Long"
    runtime._model_task_nums = {"1", "2"}
    runtime._subscribers = []
    runtime._policy = Mock()
    runtime._publish_runtime_state = Mock()
    runtime._handle_task(runtime._latest_task)
    runtime._environment = Mock()
    runtime._environment.get_observation.return_value = {}
    runtime._policy.infer.return_value = {}
    runtime._recent_puppet_actions = deque()
    runtime._step()
    observation = runtime._policy.infer.call_args.args[0]
    assert observation["operator"] == "Jia Long" and observation["rating"] == 5
    assert "subtask" not in observation
    # Keyboard/old clients also pick up settings on the next model task.
    runtime._handle_task({"task_num": "2", "task_name": "rinse bottle"})
    assert runtime._current_task["rating"] == 5
    assert http.put("/api/inference-settings", json={}).status_code == 200
    runtime._handle_task({"task_num": "1", "task_name": "rinse bottle"})
    assert "operator" not in runtime._current_task and "rating" not in runtime._current_task


@pytest.mark.parametrize(
    "payload", [{"operator": "other"}, {"rating": 0}, {"rating": 6}, {"rating": True}, {"rating": 2.5}, {"rating": "5"}]
)
def test_invalid_settings(client, payload):
    http, redis = client
    assert http.put("/api/inference-settings", json=payload).status_code == 422
    assert not redis.values and not redis.messages


def test_stop_commands_do_not_read_conditioning(client):
    http, redis = client
    redis.get = Mock(side_effect=AssertionError("Stop must not depend on saved settings"))
    for task in ["3", "4", "5", "6"]:
        assert http.post("/api/tasks/" + task).status_code == 200


def test_voice_publish_uses_same_settings():
    redis = Redis()
    redis.set(redis_commands.INFERENCE_SETTINGS_KEY, json.dumps({"operator": "Zhaonan Ma", "rating": 3}))
    assert redis_commands.publish_task(redis, "2")["rating"] == 3


def test_live_settings_update_without_restarting_episode(client):
    http, redis = client
    runtime = Runtime.__new__(Runtime)
    runtime._task_lock = threading.Lock()
    runtime._pending_inference_settings = None
    runtime._redis_client = redis
    runtime._redis_running = True
    runtime._current_task = {"task_num": "1", "task_name": "rinse bottle", "operator": "Jia Long", "rating": 1}
    runtime._policy = Mock()
    runtime._policy.infer.return_value = {}
    runtime._environment = Mock()
    runtime._environment.get_observation.return_value = {'origin_observation': {}}
    runtime._subscribers = [Mock()]
    runtime._recent_puppet_actions = deque()
    runtime._publish_runtime_state = Mock()
    for payload in [{"operator": "Xinzheng He", "rating": 5}, {}]:
        assert http.put("/api/inference-settings", json=payload).status_code == 200
        message = redis.messages[-1]
        assert "task" not in message

        def receive(**kwargs):
            runtime._redis_running = False
            return {"type": "message", "data": json.dumps(message)}

        runtime._redis_running = True
        redis.pubsub = lambda: SimpleNamespace(subscribe=lambda *a: None, get_message=receive, close=lambda: None)
        runtime._redis_listener()
        runtime._step()
        observation = runtime._policy.infer.call_args.args[0]
        assert observation.get("operator") == payload.get("operator")
        assert observation.get("rating") == payload.get("rating")
        assert runtime._current_task["task_num"] == "1"
    assert runtime._policy.reset.call_count == 2
    runtime._environment.stop.assert_not_called()
    runtime._subscribers[0].on_episode_start.assert_not_called()
    runtime._subscribers[0].on_episode_end.assert_not_called()
