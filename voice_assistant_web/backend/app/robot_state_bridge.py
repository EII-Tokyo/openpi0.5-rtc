from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

import numpy as np

from .config import settings
from .redis_commands import create_redis_client


class RobotStateBridge:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "timestamp": None,
            "mode": "waiting",
            "current_task": None,
            "qpos": [],
            "latest_action": [],
            "hierarchical": {},
        }
        self._running = False
        self._poll_thread: threading.Thread | None = None
        self._redis_thread: threading.Thread | None = None
        self._left_qpos: list[float] | None = None
        self._right_qpos: list[float] | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_ros_state, daemon=True)
        self._redis_thread = threading.Thread(target=self._listen_runtime_state, daemon=True)
        self._poll_thread.start()
        self._redis_thread.start()

    def stop(self) -> None:
        self._running = False
        for thread in (self._poll_thread, self._redis_thread):
            if thread and thread.is_alive():
                thread.join(timeout=1.0)

    def _poll_ros_state(self) -> None:
        try:
            import rospy
            from sensor_msgs.msg import JointState

            if not rospy.core.is_initialized():
                rospy.init_node("voice_assistant_web_backend", anonymous=True, disable_signals=True)

            def left_callback(message: JointState) -> None:
                self._left_qpos = [float(v) for v in message.position]

            def right_callback(message: JointState) -> None:
                self._right_qpos = [float(v) for v in message.position]

            left_subscriber = rospy.Subscriber("/puppet_left/joint_states", JointState, left_callback)
            right_subscriber = rospy.Subscriber("/puppet_right/joint_states", JointState, right_callback)

            while self._running and not rospy.is_shutdown():
                if self._left_qpos is not None and self._right_qpos is not None:
                    qpos = self._combine_qpos(self._left_qpos, self._right_qpos)
                    with self._lock:
                        self._state["qpos"] = qpos
                        if not self._state["latest_action"]:
                            self._state["latest_action"] = qpos.copy()
                        self._state["timestamp"] = time.time()
                time.sleep(0.05)

            left_subscriber.unregister()
            right_subscriber.unregister()
        except Exception:
            logging.exception("Robot state ROS polling failed")

    def _listen_runtime_state(self) -> None:
        while self._running:
            try:
                redis_client = create_redis_client()
                pubsub = redis_client.pubsub()
                pubsub.subscribe(settings.runtime_state_channel)
                while self._running:
                    message = pubsub.get_message(timeout=1.0)
                    if not message or message["type"] != "message":
                        continue
                    payload = json.loads(message["data"])
                    with self._lock:
                        self._state.update(
                            {
                                "timestamp": payload.get("timestamp", time.time()),
                                "mode": payload.get("mode", self._state.get("mode", "waiting")),
                                "current_task": payload.get("current_task"),
                                "latest_action": payload.get("latest_action", self._state.get("latest_action", [])),
                                "qpos": payload.get("qpos", self._state.get("qpos", [])),
                                "hierarchical": payload.get("hierarchical", self._state.get("hierarchical", {})),
                            }
                        )
            except Exception:
                logging.exception("Runtime state redis listener failed")
                if self._running:
                    time.sleep(1.0)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "timestamp": self._state.get("timestamp"),
                "mode": self._state.get("mode", "waiting"),
                "current_task": self._state.get("current_task"),
                "qpos": list(self._state.get("qpos", [])),
                "latest_action": list(self._state.get("latest_action", [])),
                "hierarchical": dict(self._state.get("hierarchical", {})),
            }

    def _combine_qpos(self, left_qpos: list[float], right_qpos: list[float]) -> list[float]:
        left = np.asarray(left_qpos, dtype=float)
        right = np.asarray(right_qpos, dtype=float)
        if left.size < 7 or right.size < 7:
            return []
        return list(left[:6]) + [float(left[6])] + list(right[:6]) + [float(right[6])]
