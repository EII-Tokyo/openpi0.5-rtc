from tools.aloha1_mapping.home_sleep_103_preflight import REMOTE_READ_ONLY_SCRIPT
from tools.aloha1_mapping.home_sleep_103_preflight import classify_remote_snapshot
from tools.aloha1_mapping.home_sleep_103_preflight import parse_remote_snapshot


def _snapshot_text() -> str:
    return """PROJECT_ROOT=/home/eii/openpi0.5-rtc-reward-learning
GIT_HEAD=ea818494bf9ee7756c955864ba3b0d62be6ce649
GIT_BRANCH=paper_actor_sample
GIT_DIRTY_COUNT=45
ROBOT_CONTAINER_COUNT=0
ROS_MASTER_PORT_LISTENERS=0
ROSBRIDGE_PORT_LISTENERS=0
IMAGE_ID=sha256:fa02
IMAGE_DIGEST=lyl472324464/robot@sha256:fa02
COMPOSE_SHA256=04dd806b4a79796c80e39fa4d290ee523933271729695faf50850499fdd30cfd
ROBOT_UTILS_SHA256=479dd5845d639f32775460b1225d9f4b2d8611a2588649574de37237d83489d7
CONSTANTS_SHA256=19a72e01cba604ecbf9775067a3df5926ed489b8369fdbec4bde4597cd0389db
STATIC_JOINT_ORDER_PRESENT=1
STATIC_JOINT_STATE_TOPIC_PRESENT=1
STATIC_JOINT_COMMAND_TOPIC_PRESENT=1
STATIC_CAM_HIGH_PRESENT=1
ALOHA_LAUNCH_COMMAND_PRESENT=1
EXTERNAL_ALOHA_MOUNT_DECLARED=1
"""


def test_parse_remote_snapshot_preserves_values() -> None:
    snapshot = parse_remote_snapshot(_snapshot_text())

    assert snapshot["PROJECT_ROOT"] == "/home/eii/openpi0.5-rtc-reward-learning"
    assert snapshot["GIT_DIRTY_COUNT"] == "45"
    assert snapshot["IMAGE_DIGEST"].endswith("sha256:fa02")


def test_stopped_robot_stack_is_partial_not_pass() -> None:
    report = classify_remote_snapshot(parse_remote_snapshot(_snapshot_text()))

    assert report["status"] == "PARTIAL_RUNTIME_STACK_STOPPED"
    assert report["static_project_evidence"] == "PASS"
    assert report["runtime_ros_evidence"] == "NOT_RUN_ROBOT_STACK_STOPPED"
    assert report["real_motion_authorized"] is False
    assert "authorization_to_start_robot_driver" in report["remaining_gates"]


def test_read_only_script_contains_no_mutating_ros_or_docker_actions() -> None:
    forbidden = (
        "docker compose up",
        "docker start",
        "docker run",
        "rostopic pub",
        "rosservice call",
        "roslaunch",
        "torque_enable",
        "commands/joint_group",
    )

    assert all(token not in REMOTE_READ_ONLY_SCRIPT for token in forbidden)
    assert "cd /home/eii/openpi0.5-rtc-reward-learning" in REMOTE_READ_ONLY_SCRIPT
