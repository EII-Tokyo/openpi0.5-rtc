from dataclasses import dataclass

from tools.aloha1_mapping.home_sleep_ros1_adapter import build_ros_adapter
from tools.aloha1_mapping.home_sleep_ros1_adapter import live_adapter_gate
from tools.aloha1_mapping.home_sleep_ros1_adapter import run_read_only_preflight
from tools.aloha1_mapping.home_sleep_ros1_adapter import serialize_joint_group_command


class FakeJointGroupCommand:
    def __init__(self) -> None:
        self.name = ""
        self.cmd: list[float] = []


@dataclass
class FakeRosFactory:
    publisher_count: int = 0

    def read_only_snapshot(self) -> dict[str, object]:
        return {
            "joint_states_topic": "/puppet_left/joint_states",
            "joint_states_type": "sensor_msgs/JointState",
            "joint_names": [
                "waist",
                "shoulder",
                "elbow",
                "forearm_roll",
                "wrist_angle",
                "wrist_rotate",
            ],
            "command_topic": "/puppet_left/commands/joint_group",
            "command_type": "interbotix_xs_msgs/JointGroupCommand",
            "camera_topic": "/cam_high",
            "camera_type": "aloha.msg/RGBGrayscaleImage",
            "operating_mode": "position",
            "group_name": "arm",
            "stop_path_verified": False,
        }

    def create_publisher(self) -> object:
        self.publisher_count += 1
        return object()


def _passing_live_gates() -> dict[str, bool]:
    return {
        "real_access_authorized": True,
        "real_motion_authorized": True,
        "operator_workspace_clear": True,
        "stop_path_verified": True,
        "joint_order_verified": True,
        "camera_ready": True,
        "manifest_hash_match": True,
        "digital_gate_pass": True,
    }


def test_ros_adapter_serializes_exact_six_joint_group_command() -> None:
    message = serialize_joint_group_command(
        FakeJointGroupCommand, "arm", [0, 1, 2, 3, 4, 5]
    )

    assert message.name == "arm"
    assert message.cmd == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_ros_adapter_rejects_nonfinite_or_wrong_count_commands() -> None:
    for values in ([0, 1], [0, 1, 2, 3, 4, float("nan")]):
        try:
            serialize_joint_group_command(FakeJointGroupCommand, "arm", values)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid command must be rejected")


def test_ros_import_is_deferred_until_live_gate_passes() -> None:
    import_calls: list[str] = []

    def reject_import(name: str) -> object:
        import_calls.append(name)
        raise AssertionError("ROS import must not happen while live gate is blocked")

    report = build_ros_adapter(
        _passing_live_gates() | {"real_motion_authorized": False},
        module_importer=reject_import,
    )

    assert report["status"] == "NOT_RUN_AUTHORIZATION_REQUIRED"
    assert import_calls == []
    assert report["publisher_constructed"] is False


def test_read_only_preflight_never_constructs_publisher() -> None:
    factory = FakeRosFactory()

    report = run_read_only_preflight(factory)

    assert factory.publisher_count == 0
    assert report["status"] == "PARTIAL"
    assert report["publisher_constructed"] is False
    assert report["failed_gates"] == ["stop_path_verified"]


def test_unverified_stop_path_blocks_live_status() -> None:
    report = live_adapter_gate(
        _passing_live_gates() | {"stop_path_verified": False}
    )

    assert report["status"] == "BLOCKED"
    assert "stop_path_verified" in report["failed_gates"]


def test_all_live_gates_are_required() -> None:
    report = live_adapter_gate({})

    assert report["status"] == "BLOCKED"
    assert set(report["failed_gates"]) == set(_passing_live_gates())
