import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from aloha import robot_utils
from aloha.robot_utils import ImageRecorder
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    ReliabilityPolicy,
)


def camera_config(*names):
    return {
        "cameras": {
            "common_parameters": {
                "color_image_topic_name": "{}/camera/color/image_raw",
            },
            "camera_instances": [
                {"name": name}
                for name in names
            ],
        },
    }


class FakeNode:
    def __init__(self):
        self.calls = []

    def create_subscription(
        self,
        message_type,
        topic,
        callback,
        qos,
        *,
        callback_group=None,
    ):
        subscription = SimpleNamespace(
            message_type=message_type,
            topic=topic,
            callback=callback,
            qos=qos,
            callback_group=callback_group,
        )
        self.calls.append(subscription)
        return subscription


def test_each_camera_uses_a_distinct_retained_mutually_exclusive_group(
    monkeypatch,
):
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

    monkeypatch.setattr(robot_utils.time, "sleep", lambda _seconds: None)
    node = FakeNode()

    recorder = ImageRecorder(
        config=camera_config("camera_high", "camera_low"),
        node=node,
    )

    assert [call.topic for call in node.calls] == [
        "camera_high/camera/color/image_raw",
        "camera_low/camera/color/image_raw",
    ]
    qos_profiles = [call.qos for call in node.calls]
    assert all(profile.depth == 1 for profile in qos_profiles)
    assert all(
        profile.history == HistoryPolicy.KEEP_LAST
        for profile in qos_profiles
    )
    assert all(
        profile.reliability == ReliabilityPolicy.BEST_EFFORT
        for profile in qos_profiles
    )
    assert all(
        profile.durability == DurabilityPolicy.VOLATILE
        for profile in qos_profiles
    )
    groups = [call.callback_group for call in node.calls]
    assert all(
        isinstance(group, MutuallyExclusiveCallbackGroup)
        for group in groups
    )
    assert groups[0] is not groups[1]
    assert recorder.camera_callback_groups == {
        "camera_high": groups[0],
        "camera_low": groups[1],
    }
    assert recorder.camera_subscriptions == {
        "camera_high": node.calls[0],
        "camera_low": node.calls[1],
    }


def test_callback_replaces_latest_frame_and_preserves_previous_snapshot(
    monkeypatch,
):
    monkeypatch.setattr(robot_utils.time, "sleep", lambda _seconds: None)
    node = FakeNode()
    recorder = ImageRecorder(
        config=camera_config("camera_high"),
        node=node,
    )
    frames = [
        np.full((2, 3, 3), 1, dtype=np.uint8),
        np.full((2, 3, 3), 2, dtype=np.uint8),
    ]
    recorder.bridge = SimpleNamespace(
        imgmsg_to_cv2=(
            lambda message, desired_encoding: frames[message.index]
        ),
    )
    callback = node.calls[0].callback

    callback(
        SimpleNamespace(
            index=0,
            header=SimpleNamespace(
                stamp=SimpleNamespace(sec=10, nanosec=100),
            ),
        )
    )
    first = recorder.get_images()
    callback(
        SimpleNamespace(
            index=1,
            header=SimpleNamespace(
                stamp=SimpleNamespace(sec=11, nanosec=200),
            ),
        )
    )
    second = recorder.get_images()

    assert np.all(first["camera_high"] == 1)
    assert np.all(second["camera_high"] == 2)
    assert recorder.camera_high_secs == 11
    assert recorder.camera_high_nsecs == 200


def test_callback_and_reader_share_snapshot_lock(monkeypatch):
    class CountingLock:
        def __init__(self):
            self.entries = 0

        def __enter__(self):
            self.entries += 1

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(robot_utils.time, "sleep", lambda _seconds: None)
    node = FakeNode()
    recorder = ImageRecorder(
        config=camera_config("camera_high"),
        node=node,
    )
    recorder.bridge = SimpleNamespace(
        imgmsg_to_cv2=lambda *_args, **_kwargs: np.zeros(
            (2, 3, 3),
            dtype=np.uint8,
        ),
    )
    lock = CountingLock()
    recorder._snapshot_lock = lock

    node.calls[0].callback(
        SimpleNamespace(
            header=SimpleNamespace(
                stamp=SimpleNamespace(sec=10, nanosec=100),
            ),
        )
    )
    recorder.get_images()

    assert lock.entries == 2


def test_blocked_camera_callback_does_not_block_default_group_robot_callback(
    monkeypatch,
):
    rclpy = pytest.importorskip("rclpy")
    from rclpy.context import Context
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from std_msgs.msg import Empty

    monkeypatch.setattr(robot_utils.time, "sleep", lambda _seconds: None)
    context = Context()
    rclpy.init(context=context)
    node = Node("image_callback_group_regression", context=context)
    executor = MultiThreadedExecutor(num_threads=2, context=context)
    image_started = threading.Event()
    release_image = threading.Event()
    robot_seen = threading.Event()
    spin_thread = None
    try:
        recorder = ImageRecorder(
            config=camera_config("camera_probe"),
            node=node,
        )

        def blocking_image_callback(_camera_name, _message):
            image_started.set()
            assert release_image.wait(timeout=2.0)

        recorder.image_cb = blocking_image_callback
        node.create_subscription(
            Empty,
            "robot_probe",
            lambda _message: robot_seen.set(),
            10,
        )
        image_publisher = node.create_publisher(
            Image,
            "camera_probe/camera/color/image_raw",
            10,
        )
        robot_publisher = node.create_publisher(
            Empty,
            "robot_probe",
            10,
        )
        executor.add_node(node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        deadline = time.monotonic() + 1.0
        while not image_started.is_set() and time.monotonic() < deadline:
            image_publisher.publish(Image())
            time.sleep(0.01)
        assert image_started.is_set()

        robot_publisher.publish(Empty())
        assert robot_seen.wait(timeout=0.5)
        assert not release_image.is_set()
    finally:
        release_image.set()
        executor.shutdown(timeout_sec=1.0)
        if spin_thread is not None:
            spin_thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown(context=context)
