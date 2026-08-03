from pathlib import Path

from tools.aloha1_mapping.home_sleep_external_aloha_audit import REMOTE_EXTERNAL_READ_ONLY_SCRIPT
from tools.aloha1_mapping.home_sleep_external_aloha_audit import audit_external_aloha_source
from tools.aloha1_mapping.home_sleep_external_aloha_audit import build_external_audit_report
from tools.aloha1_mapping.home_sleep_external_aloha_audit import parse_snapshot
from tools.aloha1_mapping.home_sleep_external_aloha_audit import snapshot_file
from tools.aloha1_mapping.home_sleep_external_aloha_audit import validate_left_only_launch
from tools.audit_aloha1_home_sleep_external_aloha import build_markdown

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "configs/aloha1_home_sleep_puppet_left_only_candidate.launch"

FOUR_ARM_LAUNCH = """<launch>
  <arg name="robot_name_master_left" value="master_left"/>
  <arg name="robot_name_master_right" value="master_right"/>
  <arg name="robot_name_puppet_left" value="puppet_left"/>
  <arg name="robot_name_puppet_right" value="puppet_right"/>
  <include file="$(find interbotix_xsarm_control)/launch/xsarm_control.launch">
    <arg name="robot_model" value="wx250s"/>
    <arg name="robot_name" value="$(arg robot_name_master_left)"/>
  </include>
  <include file="$(find interbotix_xsarm_control)/launch/xsarm_control.launch">
    <arg name="robot_model" value="wx250s"/>
    <arg name="robot_name" value="$(arg robot_name_master_right)"/>
  </include>
  <include file="$(find interbotix_xsarm_control)/launch/xsarm_control.launch">
    <arg name="robot_model" value="vx300s"/>
    <arg name="robot_name" value="$(arg robot_name_puppet_left)"/>
  </include>
  <include file="$(find interbotix_xsarm_control)/launch/xsarm_control.launch">
    <arg name="robot_model" value="vx300s"/>
    <arg name="robot_name" value="$(arg robot_name_puppet_right)"/>
  </include>
  <node name="realsense_publisher" pkg="aloha" type="realsense_publisher.py"/>
</launch>
"""

PUPPET_LEFT_MODES = """port: /dev/ttyDXL_puppet_left
groups:
  arm:
    operating_mode: position
    torque_enable: true
singles:
  gripper:
    operating_mode: linear_position
    torque_enable: true
"""

FOUR_CAMERA_SOURCE = """camera_names = [
    'cam_left_wrist', 'cam_high', 'cam_low', 'cam_right_wrist'
]
missing_cams = [name for name in camera_names if name not in device_ids]
if missing_cams:
    raise Exception(f'Cameras missing:{missing_cams}')
for dev in devices:
    dev.hardware_reset()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, FPS)
color_frame = color_frame[..., ::-1]
cv_bridge.cv2_to_imgmsg(color_frame, encoding="bgr8")
"""

SLEEP_SOURCE = """puppet_bot_left = InterbotixManipulatorXS(robot_name='puppet_left')
puppet_bot_right = InterbotixManipulatorXS(robot_name='puppet_right')
all_bots = [puppet_bot_left, puppet_bot_right]
move_arms(all_bots, [puppet_sleep_position] * 2, move_time=2)
"""


def test_external_launch_is_rejected_for_left_only_replay() -> None:
    report = audit_external_aloha_source(
        ros_nodes_launch=FOUR_ARM_LAUNCH,
        puppet_left_modes=PUPPET_LEFT_MODES,
        realsense_source=FOUR_CAMERA_SOURCE,
        sleep_source=SLEEP_SOURCE,
    )

    assert report["status"] == "REJECTED_FOR_LEFT_ONLY_SUPERVISED_REPLAY"
    assert report["driver_scope"]["robot_names"] == [
        "master_left",
        "master_right",
        "puppet_left",
        "puppet_right",
    ]
    assert report["puppet_left_mode"]["arm_operating_mode"] == "position"
    assert report["puppet_left_mode"]["arm_torque_enable"] is True
    assert report["puppet_left_mode"]["gripper_torque_enable"] is True
    assert report["camera_scope"]["requires_four_cameras"] is True
    assert report["camera_scope"]["hardware_reset_present"] is True
    assert report["camera_scope"]["color_semantics"] == (
        "MISMATCH_RGB_BYTES_LABELED_BGR8"
    )
    assert report["sleep_scope"]["commands_both_puppets"] is True


def test_candidate_launch_contains_only_puppet_left_driver() -> None:
    report = validate_left_only_launch(CANDIDATE.read_text(encoding="utf-8"))

    assert report["status"] == "PASS_STATIC_LEFT_ONLY_SCOPE"
    assert report["robot_names"] == ["puppet_left"]
    assert report["robot_model"] == "vx300s"
    assert report["mode_config"].endswith("puppet_modes_left.yaml")
    assert report["load_configs"] == "false"
    assert report["use_sim"] == "false"
    assert report["camera_nodes"] == []
    assert report["real_execution"] == "NOT_RUN_AUTHORIZATION_REQUIRED"


def test_external_read_only_script_cannot_start_or_command_hardware() -> None:
    forbidden = (
        "roslaunch",
        "rostopic pub",
        "rosservice call",
        "docker run",
        "docker start",
        "docker compose up",
        "torque_enable",
        "/dev/ttyDXL",
    )

    assert all(token not in REMOTE_EXTERNAL_READ_ONLY_SCRIPT for token in forbidden)
    assert "cd /home/eii/openpi0.5-rtc/third_party/aloha" in (
        REMOTE_EXTERNAL_READ_ONLY_SCRIPT
    )


def test_snapshot_round_trip_builds_fail_closed_report() -> None:
    import base64

    files = {
        "launch/ros_nodes.launch": FOUR_ARM_LAUNCH,
        "config/puppet_modes_left.yaml": PUPPET_LEFT_MODES,
        "aloha_scripts/realsense_publisher.py": FOUR_CAMERA_SOURCE,
        "aloha_scripts/sleep.py": SLEEP_SOURCE,
        "LICENSE": "MIT License\n",
    }
    lines = [
        "EXTERNAL_ROOT=/home/eii/openpi0.5-rtc/third_party/aloha",
        "GIT_TOPLEVEL=/home/eii/openpi0.5-rtc",
        "GIT_HEAD=f2e6",
        "GIT_BRANCH=codex/minimal-aloha-real",
        "GIT_DIRTY_COUNT=11",
        "GIT_ORIGIN=https://github.com/EII-Tokyo/openpi0.5-rtc.git",
    ]
    for path, text in files.items():
        key = path.translate(str.maketrans({"/": "_", ".": "_"})).upper()
        lines.append(f"FILE_{key}_SHA256={'a' * 64}")
        encoded = base64.b64encode(text.encode()).decode()
        lines.append(f"FILE_{key}_B64={encoded}")
    snapshot = parse_snapshot("\n".join(lines))

    assert snapshot_file(snapshot, "launch/ros_nodes.launch") == FOUR_ARM_LAUNCH
    report = build_external_audit_report(
        snapshot,
        candidate_text=CANDIDATE.read_text(encoding="utf-8"),
    )

    assert report["status"] == "READY_FOR_MINIMAL_START_AUTHORIZATION"
    assert report["existing_deployment"]["status"] == (
        "REJECTED_FOR_LEFT_ONLY_SUPERVISED_REPLAY"
    )
    assert report["left_only_candidate"]["status"] == (
        "PASS_STATIC_LEFT_ONLY_SCOPE"
    )
    assert report["authorization"]["driver_started"] is False
    assert report["authorization"]["commands_published"] == 0
    assert report["remaining_gates"] == [
        "explicit_authorization_to_start_puppet_left_driver",
        "runtime_joint_order_and_position_mode_readback",
        "operator_tested_stop_hold_path",
        "cam_high_single_camera_runtime_path",
        "operator_workspace_clear",
        "explicit_real_motion_authorization",
    ]

    markdown = build_markdown(report)
    assert "READY_FOR_MINIMAL_START_AUTHORIZATION" in markdown
    assert "REJECTED_FOR_LEFT_ONLY_SUPERVISED_REPLAY" in markdown
    assert "NOT_RUN_AUTHORIZATION_REQUIRED" in markdown
