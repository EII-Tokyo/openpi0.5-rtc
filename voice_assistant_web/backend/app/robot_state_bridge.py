from __future__ import annotations

import json
import logging
import os
import sys
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
            "effort": [],
            "joint_effort": {},
            "joint_temperature": {},
            "latest_action": [],
            "rlt_actor_enabled": False,
            "rlt_chunk_q_min": None,
            "rlt_vla_chunk_q_min": None,
            "rlt_actor_chunk_q_min": None,
        }
        self._running = False
        self._poll_thread: threading.Thread | None = None
        self._redis_thread: threading.Thread | None = None
        self._left_qpos: list[float] | None = None
        self._right_qpos: list[float] | None = None
        self._left_effort: list[float] | None = None
        self._right_effort: list[float] | None = None
        self._joint_effort: dict[str, dict[str, list[float] | list[str]]] = {}
        self._joint_temperature: dict[str, dict[str, list[float] | list[str]]] = {}

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
            register_service_type = self._load_register_service_type()

            if not rospy.core.is_initialized():
                rospy.init_node("voice_assistant_web_backend", anonymous=True)

            def left_callback(message: JointState) -> None:
                self._left_qpos = [float(v) for v in message.position]
                self._left_effort = [float(v) for v in message.effort]
                self._record_joint_effort("puppet_left", message)

            def right_callback(message: JointState) -> None:
                self._right_qpos = [float(v) for v in message.position]
                self._right_effort = [float(v) for v in message.effort]
                self._record_joint_effort("puppet_right", message)

            def master_left_callback(message: JointState) -> None:
                self._record_joint_effort("master_left", message)

            def master_right_callback(message: JointState) -> None:
                self._record_joint_effort("master_right", message)

            left_subscriber = rospy.Subscriber("/puppet_left/joint_states", JointState, left_callback)
            right_subscriber = rospy.Subscriber("/puppet_right/joint_states", JointState, right_callback)
            master_left_subscriber = rospy.Subscriber("/master_left/joint_states", JointState, master_left_callback)
            master_right_subscriber = rospy.Subscriber("/master_right/joint_states", JointState, master_right_callback)
            temperature_proxies = self._create_temperature_proxies(rospy, register_service_type)
            last_temperature_poll = 0.0

            while self._running and not rospy.is_shutdown():
                now = time.time()
                if temperature_proxies and now - last_temperature_poll >= 2.0:
                    self._poll_temperatures(temperature_proxies)
                    last_temperature_poll = now
                if self._left_qpos is not None and self._right_qpos is not None:
                    qpos = self._combine_qpos(self._left_qpos, self._right_qpos)
                    with self._lock:
                        self._state["qpos"] = qpos
                        if self._left_effort is not None and self._right_effort is not None:
                            self._state["effort"] = self._combine_joint_values(self._left_effort, self._right_effort)
                        self._state["joint_effort"] = dict(self._joint_effort)
                        self._state["joint_temperature"] = dict(self._joint_temperature)
                        if not self._state["latest_action"]:
                            self._state["latest_action"] = qpos.copy()
                        self._state["timestamp"] = time.time()
                time.sleep(0.05)

            left_subscriber.unregister()
            right_subscriber.unregister()
            master_left_subscriber.unregister()
            master_right_subscriber.unregister()
        except Exception:
            logging.exception("Robot state ROS polling failed")

    def _listen_runtime_state(self) -> None:
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
                            "effort": payload.get("effort", self._state.get("effort", [])),
                            "joint_effort": payload.get("joint_effort", self._state.get("joint_effort", {})),
                            "joint_temperature": payload.get(
                                "joint_temperature", self._state.get("joint_temperature", {})
                            ),
                            "rlt_actor_enabled": payload.get(
                                "rlt_actor_enabled", self._state.get("rlt_actor_enabled", False)
                            ),
                            "rlt_chunk_q_min": payload.get(
                                "rlt_chunk_q_min", self._state.get("rlt_chunk_q_min")
                            ),
                            "rlt_vla_chunk_q_min": payload.get(
                                "rlt_vla_chunk_q_min", self._state.get("rlt_vla_chunk_q_min")
                            ),
                            "rlt_actor_chunk_q_min": payload.get(
                                "rlt_actor_chunk_q_min", self._state.get("rlt_actor_chunk_q_min")
                            ),
                        }
                    )
        except Exception:
            logging.exception("Runtime state redis listener failed")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "timestamp": self._state.get("timestamp"),
                "mode": self._state.get("mode", "waiting"),
                "current_task": self._state.get("current_task"),
                "qpos": list(self._state.get("qpos", [])),
                "effort": list(self._state.get("effort", [])),
                "joint_effort": dict(self._state.get("joint_effort", {})),
                "joint_temperature": dict(self._state.get("joint_temperature", {})),
                "latest_action": list(self._state.get("latest_action", [])),
                "rlt_actor_enabled": bool(self._state.get("rlt_actor_enabled", False)),
                "rlt_chunk_q_min": self._state.get("rlt_chunk_q_min"),
                "rlt_vla_chunk_q_min": self._state.get("rlt_vla_chunk_q_min"),
                "rlt_actor_chunk_q_min": self._state.get("rlt_actor_chunk_q_min"),
            }

    def _combine_qpos(self, left_qpos: list[float], right_qpos: list[float]) -> list[float]:
        left = np.asarray(left_qpos, dtype=float)
        right = np.asarray(right_qpos, dtype=float)
        if left.size < 7 or right.size < 7:
            return []
        return list(left[:6]) + [float(left[6])] + list(right[:6]) + [float(right[6])]

    def _combine_joint_values(self, left_values: list[float], right_values: list[float]) -> list[float]:
        left = np.asarray(left_values, dtype=float)
        right = np.asarray(right_values, dtype=float)
        if left.size < 7 or right.size < 7:
            return []
        return list(left[:7]) + list(right[:7])

    def _record_joint_effort(self, robot_name: str, message: Any) -> None:
        self._joint_effort[robot_name] = {
            "names": [str(name) for name in message.name[:7]],
            "values": [float(value) for value in message.effort[:7]],
        }

    def _load_register_service_type(self) -> Any | None:
        interbotix_python = "/root/interbotix_ws/devel/lib/python3/dist-packages"
        if os.path.isdir(interbotix_python) and interbotix_python not in sys.path:
            sys.path.insert(0, interbotix_python)
        try:
            from interbotix_xs_msgs.srv import RegisterValues
        except Exception:
            logging.exception("Unable to load interbotix_xs_msgs/RegisterValues; motor temperatures disabled")
            return None
        return RegisterValues

    def _create_temperature_proxies(self, rospy: Any, register_service_type: Any | None) -> dict[str, Any]:
        if register_service_type is None:
            return {}
        proxies = {}
        for robot_name in ("master_left", "master_right", "puppet_left", "puppet_right"):
            service_name = f"/{robot_name}/get_motor_registers"
            try:
                rospy.wait_for_service(service_name, timeout=1.0)
                proxies[robot_name] = rospy.ServiceProxy(service_name, register_service_type)
            except Exception:
                logging.exception("Unable to connect to %s; motor temperature disabled for %s", service_name, robot_name)
        return proxies

    def _poll_temperatures(self, proxies: dict[str, Any]) -> None:
        for robot_name, proxy in proxies.items():
            try:
                arm_response = proxy("group", "arm", "Present_Temperature", 0)
                gripper_response = proxy("single", "gripper", "Present_Temperature", 0)
                values = [float(value) for value in list(arm_response.values[:6]) + list(gripper_response.values[:1])]
                self._joint_temperature[robot_name] = {
                    "names": ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper"],
                    "values": values,
                }
            except Exception:
                logging.exception("Unable to poll motor temperatures for %s", robot_name)
