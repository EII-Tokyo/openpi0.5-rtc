from types import SimpleNamespace

import pytest

from aloha import real_env
from aloha.camera_runtime import CameraRuntime


class FakeNode:
    def __init__(self, events):
        self.events = events

    def destroy_node(self):
        self.events.append("destroy_node")


class FakeExecutor:
    def __init__(
        self,
        events,
        *,
        shutdown_error=None,
        shutdown_result=True,
    ):
        self.events = events
        self.node = None
        self.shutdown_error = shutdown_error
        self.shutdown_result = shutdown_result

    def add_node(self, node):
        self.node = node
        self.events.append("add_node")
        return True

    def remove_node(self, node):
        assert node is self.node
        self.events.append("remove_node")

    def spin(self):
        self.events.append("spin")

    def shutdown(self, *, timeout_sec):
        assert timeout_sec in (None, 1.0)
        self.events.append("executor_shutdown")
        if self.shutdown_error is not None:
            raise self.shutdown_error
        return self.shutdown_result


class FakeThread:
    def __init__(self, events, *, target, name, daemon, alive=False):
        self.events = events
        self.target = target
        self.name = name
        self.daemon = daemon
        self.alive = alive

    def start(self):
        self.events.append("thread_start")

    def join(self, *, timeout):
        assert timeout == 1.0
        self.events.append("thread_join")

    def is_alive(self):
        return self.alive


def dependencies(
    *,
    image_error=None,
    thread_alive=False,
    shutdown_error=None,
    shutdown_result=True,
):
    events = []
    node = FakeNode(events)
    executor = FakeExecutor(
        events,
        shutdown_error=shutdown_error,
        shutdown_result=shutdown_result,
    )
    image_recorder = SimpleNamespace(name="isolated-images")
    thread_holder = {}

    def make_node():
        events.append("create_node")
        return node

    def make_executor():
        events.append("create_executor")
        return executor

    def make_images(*, config, node):
        events.append("create_image_recorder")
        assert config == {"cameras": {}}
        assert node is node_instance
        if image_error is not None:
            raise image_error
        return image_recorder

    def make_thread(**kwargs):
        thread = FakeThread(
            events,
            alive=thread_alive,
            **kwargs,
        )
        thread_holder["thread"] = thread
        return thread

    node_instance = node
    factories = {
        "node_factory": make_node,
        "executor_factory": make_executor,
        "image_recorder_factory": make_images,
        "thread_factory": make_thread,
    }
    return (
        events,
        factories,
        node,
        executor,
        image_recorder,
        thread_holder,
    )


def test_camera_runtime_owns_one_node_executor_and_thread():
    events, factories, node, executor, images, threads = dependencies()

    runtime = CameraRuntime.create(
        config={"cameras": {}},
        context=object(),
        logger=lambda message: events.append(("log", message)),
        **factories,
    )

    assert runtime.node is node
    assert runtime.executor is executor
    assert runtime.image_recorder is images
    assert runtime.thread is threads["thread"]
    assert runtime.thread.name == "aloha-camera-executor"
    assert runtime.thread.daemon
    assert events[:5] == [
        "create_node",
        "create_executor",
        "add_node",
        "create_image_recorder",
        "thread_start",
    ]

    runtime.close()
    runtime.close()

    assert events.count("executor_shutdown") == 1
    assert events.count("thread_join") == 1
    assert events.count("remove_node") == 1
    assert events.count("destroy_node") == 1


def test_camera_runtime_rolls_back_partial_creation():
    expected = RuntimeError("camera init failed")
    events, factories, *_rest = dependencies(image_error=expected)

    with pytest.raises(RuntimeError, match="camera init failed"):
        CameraRuntime.create(
            config={"cameras": {}},
            context=object(),
            **factories,
        )

    assert events == [
        "create_node",
        "create_executor",
        "add_node",
        "create_image_recorder",
        "executor_shutdown",
        "remove_node",
        "destroy_node",
    ]


def test_logger_failure_does_not_rollback_started_camera_runtime():
    events, factories, *_rest = dependencies()

    def fail_log(_message):
        raise BrokenPipeError("stdout closed")

    runtime = CameraRuntime.create(
        config={"cameras": {}},
        context=object(),
        logger=fail_log,
        **factories,
    )

    assert "thread_start" in events
    assert "executor_shutdown" not in events
    assert "remove_node" not in events
    assert "destroy_node" not in events

    runtime.close()
    assert "destroy_node" in events


def test_constructor_rollback_destroys_node_when_executor_shutdown_fails():
    events, factories, *_rest = dependencies(
        image_error=RuntimeError("camera init failed"),
        shutdown_error=RuntimeError("executor shutdown failed"),
    )

    with pytest.raises(RuntimeError, match="camera init failed"):
        CameraRuntime.create(
            config={"cameras": {}},
            context=object(),
            logger=lambda _message: None,
            **factories,
        )

    assert "remove_node" in events
    assert "destroy_node" in events


def test_camera_runtime_does_not_destroy_node_while_spin_thread_is_alive():
    events, factories, *_rest = dependencies(thread_alive=True)
    runtime = CameraRuntime.create(
        config={"cameras": {}},
        context=object(),
        **factories,
    )

    with pytest.raises(RuntimeError, match="did not stop"):
        runtime.close()

    assert "executor_shutdown" in events
    assert "thread_join" in events
    assert "remove_node" not in events
    assert "destroy_node" not in events
    assert not runtime._closed


def test_camera_runtime_close_can_retry_after_spin_thread_stops():
    events, factories, *_rest = dependencies(thread_alive=True)
    runtime = CameraRuntime.create(
        config={"cameras": {}},
        context=object(),
        **factories,
    )
    with pytest.raises(RuntimeError, match="did not stop"):
        runtime.close()

    runtime.thread.alive = False
    runtime.close()

    assert events.count("remove_node") == 1
    assert events.count("destroy_node") == 1
    assert runtime._closed


def test_camera_runtime_does_not_destroy_node_when_shutdown_is_incomplete():
    events, factories, *_rest = dependencies(shutdown_result=False)
    runtime = CameraRuntime.create(
        config={"cameras": {}},
        context=object(),
        **factories,
    )

    with pytest.raises(RuntimeError, match="shutdown did not complete"):
        runtime.close()

    assert "remove_node" not in events
    assert "destroy_node" not in events
    assert not runtime._closed


def test_real_env_uses_injected_image_recorder(monkeypatch):
    supplied = object()
    monkeypatch.setattr(
        real_env,
        "ImageRecorder",
        lambda **_kwargs: pytest.fail(
            "must not create a shared-node image recorder"
        ),
    )
    monkeypatch.setattr(
        real_env,
        "InterbotixManipulatorXS",
        lambda **_kwargs: SimpleNamespace(),
    )

    env = real_env.RealEnv(
        node=object(),
        setup_robots=False,
        config={
            "leader_arms": [
                {"name": "leader_left", "model": "wx250s"},
            ],
        },
        image_recorder=supplied,
    )

    assert env.image_recorder is supplied
