import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "check_collect_ready.py"
)
SPEC = importlib.util.spec_from_file_location("check_collect_ready", SCRIPT)
ready = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = ready
SPEC.loader.exec_module(ready)


def stationary_config():
    return {
        "robot": {
            "leader_arms": [
                {"name": "leader_left"},
                {"name": "leader_right"},
            ],
            "follower_arms": [
                {"name": "follower_left"},
                {"name": "follower_right"},
            ],
            "cameras": {
                "common_parameters": {
                    "color_image_topic_name": (
                        "{}/camera/color/image_raw"
                    ),
                },
                "camera_instances": [
                    {"name": "camera_high"},
                    {"name": "camera_low"},
                    {"name": "camera_wrist_left"},
                    {"name": "camera_wrist_right"},
                ],
            },
        },
    }


def test_expected_graph_is_derived_from_robot_config():
    graph = ready.build_expected_graph(stationary_config())

    assert graph.nodes == frozenset(
        {
            "/leader_left/xs_sdk",
            "/leader_right/xs_sdk",
            "/follower_left/xs_sdk",
            "/follower_right/xs_sdk",
            "/camera_high/camera",
            "/camera_low/camera",
            "/camera_wrist_left/camera",
            "/camera_wrist_right/camera",
        }
    )
    assert graph.joint_topics == frozenset(
        {
            "/leader_left/joint_states",
            "/leader_right/joint_states",
            "/follower_left/joint_states",
            "/follower_right/joint_states",
        }
    )
    assert graph.camera_topics == frozenset(
        {
            "/camera_high/camera/color/image_raw",
            "/camera_low/camera/color/image_raw",
            "/camera_wrist_left/camera/color/image_raw",
            "/camera_wrist_right/camera/color/image_raw",
        }
    )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda config: config["robot"]["leader_arms"].append(
                {"name": "leader_left"}
            ),
            "unique",
        ),
        (
            lambda config: config["robot"]["cameras"].update(
                {"camera_instances": []}
            ),
            "camera",
        ),
    ],
)
def test_invalid_robot_config_is_rejected(mutate, match):
    config = stationary_config()
    mutate(config)

    with pytest.raises(ValueError, match=match):
        ready.build_expected_graph(config)


def test_graph_classification_distinguishes_empty_partial_and_complete():
    expected = ready.build_expected_graph(stationary_config())

    empty = ready.classify_graph(expected, set(), set())
    assert empty.state == "empty"

    partial = ready.classify_graph(
        expected,
        {"/leader_left/xs_sdk"},
        {"/leader_left/joint_states"},
    )
    assert partial.state == "partial"
    assert "/leader_right/xs_sdk" in partial.missing_nodes

    complete = ready.classify_graph(
        expected,
        set(expected.nodes),
        set(expected.topics),
    )
    assert complete.state == "complete"
    assert complete.missing_nodes == ()
    assert complete.missing_topics == ()


def test_message_validation_rejects_empty_joint_and_image_payloads():
    assert ready.valid_joint_message(
        SimpleNamespace(name=["waist"], position=[0.1])
    )
    assert not ready.valid_joint_message(
        SimpleNamespace(name=[], position=[])
    )
    assert not ready.valid_joint_message(
        SimpleNamespace(name=["waist"], position=[])
    )
    assert ready.valid_image_message(
        SimpleNamespace(width=640, height=480, data=b"pixels")
    )
    assert not ready.valid_image_message(
        SimpleNamespace(width=0, height=480, data=b"pixels")
    )
    assert not ready.valid_image_message(
        SimpleNamespace(width=640, height=480, data=b"")
    )


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


def test_graph_must_be_complete_on_two_consecutive_polls():
    expected = ready.build_expected_graph(stationary_config())
    clock = FakeClock()
    observations = iter(
        [
            (set(), set()),
            (set(expected.nodes), set(expected.topics)),
            ({"/leader_left/xs_sdk"}, set()),
            (set(expected.nodes), set(expected.topics)),
            (set(expected.nodes), set(expected.topics)),
        ]
    )
    reports = []

    remaining = ready.wait_for_stable_graph(
        expected,
        observe=lambda: next(observations),
        timeout=10.0,
        interval=1.0,
        stable_polls=2,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        report=reports.append,
    )

    assert remaining == pytest.approx(6.0)
    assert any("[WAIT]" in report for report in reports)
    assert reports[-1].startswith("[READY]")


def test_graph_wait_timeout_names_missing_resources():
    expected = ready.build_expected_graph(stationary_config())
    clock = FakeClock()

    with pytest.raises(TimeoutError, match="camera_high"):
        ready.wait_for_stable_graph(
            expected,
            observe=lambda: (
                set(expected.nodes) - {"/camera_high/camera"},
                set(expected.topics),
            ),
            timeout=2.0,
            interval=1.0,
            stable_polls=2,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            report=lambda _message: None,
        )


def test_message_wait_requires_every_topic():
    clock = FakeClock()
    required = {"/arm/joint_states", "/camera/color/image_raw"}
    callbacks = {}

    def subscribe(topic, callback):
        callbacks[topic] = callback
        return topic

    spins = 0

    def spin_once(_seconds):
        nonlocal spins
        spins += 1
        if spins == 1:
            callbacks["/arm/joint_states"]()
        elif spins == 2:
            callbacks["/camera/color/image_raw"]()
        clock.sleep(0.1)

    ready.wait_for_messages(
        required,
        subscribe=subscribe,
        spin_once=spin_once,
        timeout=1.0,
        monotonic=clock.monotonic,
        report=lambda _message: None,
    )

    assert spins == 2


def test_message_wait_timeout_names_silent_topic():
    clock = FakeClock()

    with pytest.raises(TimeoutError, match="camera"):
        ready.wait_for_messages(
            {"/camera"},
            subscribe=lambda _topic, _callback: object(),
            spin_once=lambda seconds: clock.sleep(seconds),
            timeout=0.2,
            monotonic=clock.monotonic,
            report=lambda _message: None,
        )
