#!/usr/bin/env python3
"""Read-only ROS readiness gate for one-command ALOHA collection."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Iterable


@dataclass(frozen=True)
class ExpectedGraph:
    nodes: frozenset[str]
    joint_topics: frozenset[str]
    camera_topics: frozenset[str]

    @property
    def topics(self) -> frozenset[str]:
        return self.joint_topics | self.camera_topics


@dataclass(frozen=True)
class GraphStatus:
    state: str
    missing_nodes: tuple[str, ...]
    missing_topics: tuple[str, ...]


def _required_names(items, *, kind: str) -> tuple[str, ...]:
    names = tuple(item.get("name", "") for item in items)
    if not names or any(
        not isinstance(name, str) or not name.strip()
        for name in names
    ):
        raise ValueError(f"{kind} names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{kind} names must be unique")
    return names


def build_expected_graph(config: dict) -> ExpectedGraph:
    robot = config.get("robot", {})
    leader_names = _required_names(
        robot.get("leader_arms", []),
        kind="leader arm",
    )
    follower_names = _required_names(
        robot.get("follower_arms", []),
        kind="follower arm",
    )
    arm_names = leader_names + follower_names
    if len(set(arm_names)) != len(arm_names):
        raise ValueError("all arm names must be unique")

    cameras = robot.get("cameras", {})
    camera_names = _required_names(
        cameras.get("camera_instances", []),
        kind="camera",
    )
    template = cameras.get("common_parameters", {}).get(
        "color_image_topic_name",
        "{}/camera/color/image_raw",
    )
    if not isinstance(template, str) or template.count("{}") != 1:
        raise ValueError(
            "camera color_image_topic_name must contain one {}"
        )

    arm_nodes = {f"/{name}/xs_sdk" for name in arm_names}
    camera_nodes = {f"/{name}/camera" for name in camera_names}
    joint_topics = {
        f"/{name}/joint_states" for name in arm_names
    }
    camera_topics = {
        "/" + template.format(name).lstrip("/")
        for name in camera_names
    }
    return ExpectedGraph(
        nodes=frozenset(arm_nodes | camera_nodes),
        joint_topics=frozenset(joint_topics),
        camera_topics=frozenset(camera_topics),
    )


def classify_graph(
    expected: ExpectedGraph,
    actual_nodes: Iterable[str],
    actual_topics: Iterable[str],
) -> GraphStatus:
    node_set = set(actual_nodes)
    topic_set = set(actual_topics)
    missing_nodes = tuple(sorted(expected.nodes - node_set))
    missing_topics = tuple(sorted(expected.topics - topic_set))
    present_count = (
        len(expected.nodes & node_set)
        + len(expected.topics & topic_set)
    )
    if not missing_nodes and not missing_topics:
        state = "complete"
    elif present_count == 0:
        state = "empty"
    else:
        state = "partial"
    return GraphStatus(
        state=state,
        missing_nodes=missing_nodes,
        missing_topics=missing_topics,
    )


def valid_joint_message(message) -> bool:
    names = tuple(getattr(message, "name", ()))
    positions = tuple(getattr(message, "position", ()))
    return bool(names) and len(names) == len(positions)


def valid_image_message(message) -> bool:
    return (
        int(getattr(message, "width", 0)) > 0
        and int(getattr(message, "height", 0)) > 0
        and bool(getattr(message, "data", b""))
    )


def _missing_text(status: GraphStatus) -> str:
    parts = []
    if status.missing_nodes:
        parts.append("nodes=" + ",".join(status.missing_nodes))
    if status.missing_topics:
        parts.append("topics=" + ",".join(status.missing_topics))
    return " ".join(parts)


def wait_for_stable_graph(
    expected: ExpectedGraph,
    *,
    observe: Callable[[], tuple[set[str], set[str]]],
    timeout: float,
    interval: float = 2.0,
    stable_polls: int = 2,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    report: Callable[[str], None] = print,
) -> float:
    if timeout <= 0 or interval <= 0 or stable_polls < 1:
        raise ValueError("wait timing values must be positive")
    deadline = monotonic() + timeout
    consecutive = 0
    last_status = classify_graph(expected, set(), set())
    while monotonic() < deadline:
        nodes, topics = observe()
        last_status = classify_graph(expected, nodes, topics)
        if last_status.state == "complete":
            consecutive += 1
            if consecutive >= stable_polls:
                report("[READY] ROS graph is stable")
                return max(0.0, deadline - monotonic())
        else:
            consecutive = 0
            report("[WAIT] " + _missing_text(last_status))
        sleep(min(interval, max(0.0, deadline - monotonic())))
    raise TimeoutError(
        "ROS graph readiness timed out: "
        + _missing_text(last_status)
    )


def wait_for_messages(
    required_topics: Iterable[str],
    *,
    subscribe: Callable[[str, Callable[[], None]], object],
    spin_once: Callable[[float], None],
    timeout: float,
    monotonic: Callable[[], float] = time.monotonic,
    report: Callable[[str], None] = print,
) -> None:
    required = set(required_topics)
    if not required or timeout <= 0:
        raise ValueError("message topics and timeout must be positive")
    seen: set[str] = set()
    subscriptions = []

    for topic in sorted(required):
        subscriptions.append(
            subscribe(topic, lambda topic=topic: seen.add(topic))
        )

    deadline = monotonic() + timeout
    last_missing = required
    while monotonic() < deadline:
        last_missing = required - seen
        if not last_missing:
            report("[READY] all required ROS topics are live")
            return
        report("[WAIT] live topics=" + ",".join(sorted(last_missing)))
        spin_once(min(0.1, max(0.0, deadline - monotonic())))
    raise TimeoutError(
        "ROS message readiness timed out: "
        + ",".join(sorted(last_missing))
    )


def _full_node_name(name: str, namespace: str) -> str:
    prefix = "" if namespace == "/" else namespace.rstrip("/")
    return f"{prefix}/{name}"


def _load_config(path: Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError(f"robot config must be a mapping: {path}")
    return loaded


def _run_ros(args) -> int:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, JointState

    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        repo_root / "config" / "robot" / f"{args.robot}.yaml"
    )
    expected = build_expected_graph(_load_config(config_path))

    rclpy.init()
    node = Node("aloha_collect_readiness")
    try:
        def observe():
            nodes = {
                _full_node_name(name, namespace)
                for name, namespace
                in node.get_node_names_and_namespaces()
            }
            topics = {
                name for name, _types in node.get_topic_names_and_types()
            }
            return nodes, topics

        if args.classify_graph:
            deadline = time.monotonic() + args.timeout
            best = classify_graph(expected, set(), set())
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
                nodes, topics = observe()
                current = classify_graph(expected, nodes, topics)
                if current.state == "complete":
                    best = current
                    break
                if current.state == "partial":
                    best = current
            print(best.state)
            return 0

        remaining = wait_for_stable_graph(
            expected,
            observe=observe,
            timeout=args.timeout,
        )

        def subscribe(topic, callback):
            message_type = (
                JointState
                if topic in expected.joint_topics
                else Image
            )
            validator = (
                valid_joint_message
                if topic in expected.joint_topics
                else valid_image_message
            )

            def accept_valid(message):
                if validator(message):
                    callback()

            return node.create_subscription(
                message_type,
                topic,
                accept_valid,
                qos_profile_sensor_data,
            )

        wait_for_messages(
            expected.topics,
            subscribe=subscribe,
            spin_once=lambda seconds: rclpy.spin_once(
                node,
                timeout_sec=seconds,
            ),
            timeout=remaining,
        )
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Read-only ALOHA collection readiness gate.",
    )
    parser.add_argument("--robot", default="aloha_stationary")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--classify-graph",
        action="store_true",
        help="Print empty, partial, or complete without waiting.",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main(argv=None) -> int:
    try:
        return _run_ros(parse_args(argv))
    except (FileNotFoundError, TimeoutError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
