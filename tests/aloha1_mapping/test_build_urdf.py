from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from tools.aloha1_mapping.build_urdf import build_all_from_config
from tools.aloha1_mapping.build_urdf import prepare_generated_urdf


def test_prepare_generated_urdf_sanitizes_names_and_resolves_resources(
    tmp_path: Path,
) -> None:
    package = tmp_path / "description"
    mesh = package / "meshes" / "finger.stl"
    texture = package / "meshes" / "black.png"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"mesh")
    texture.write_bytes(b"texture")
    source = tmp_path / "raw.urdf"
    source.write_text(
        """\
<robot name="follower/left">
  <material name="black/material">
    <texture filename="package://description/meshes/black.png"/>
  </material>
  <link name="follower/left/base">
    <visual>
      <geometry>
        <mesh filename="package://description/meshes/finger.stl"/>
      </geometry>
    </visual>
  </link>
  <link name="follower/left/finger"/>
  <joint name="finger/joint" type="prismatic">
    <parent link="follower/left/base"/>
    <child link="follower/left/finger"/>
    <mimic joint="finger/joint" multiplier="1" offset="0"/>
  </joint>
</robot>
"""
    )
    output = tmp_path / "prepared.urdf"

    result = prepare_generated_urdf(
        source_path=source,
        output_path=output,
        package_map={"description": package},
    )

    root = ET.parse(output).getroot()
    assert root.get("name") == "follower_left"
    assert [item.get("name") for item in root.findall("link")] == [
        "follower_left_base",
        "follower_left_finger",
    ]
    joint = root.find("joint")
    assert joint is not None
    assert joint.get("name") == "finger_joint"
    assert joint.find("parent").get("link") == "follower_left_base"
    assert joint.find("child").get("link") == "follower_left_finger"
    assert joint.find("mimic").get("joint") == "finger_joint"
    filenames = [
        element.get("filename")
        for element in root.findall(".//*[@filename]")
    ]
    assert filenames == [texture.resolve().as_uri(), mesh.resolve().as_uri()]
    assert result["name_replacements"] == {
        "finger/joint": "finger_joint",
        "follower/left": "follower_left",
        "follower/left/base": "follower_left_base",
        "follower/left/finger": "follower_left_finger",
        "black/material": "black_material",
    }
    assert result["resource_count"] == 2


def test_prepare_generated_urdf_refuses_unresolved_or_colliding_names(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw.urdf"
    source.write_text(
        """\
<robot name="fixture">
  <link name="a/b">
    <visual><geometry><mesh filename="package://missing/a.stl"/></geometry></visual>
  </link>
  <link name="a_b"/>
</robot>
"""
    )

    try:
        prepare_generated_urdf(
            source_path=source,
            output_path=tmp_path / "prepared.urdf",
            package_map={},
        )
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("expected prepare_generated_urdf to fail")

    assert "name collision" in message
    assert not (tmp_path / "prepared.urdf").exists()


def test_prepare_generated_urdf_manifest_is_json_serializable(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw.urdf"
    source.write_text("<robot name='fixture'><link name='base'/></robot>\n")

    result = prepare_generated_urdf(
        source_path=source,
        output_path=tmp_path / "prepared.urdf",
        package_map={},
    )

    assert json.loads(json.dumps(result))["output_sha256"]


def test_prepare_generated_urdf_can_assign_instance_robot_name(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw.urdf"
    source.write_text("<robot name='aloha_vx300s'><link name='base'/></robot>\n")
    output = tmp_path / "prepared.urdf"

    result = prepare_generated_urdf(
        source_path=source,
        output_path=output,
        package_map={},
        target_robot_name="follower_left",
    )

    assert ET.parse(output).getroot().get("name") == "follower_left"
    assert result["name_replacements"]["aloha_vx300s"] == "follower_left"


def test_build_all_from_config_verifies_commit_and_emits_audited_outputs(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=project, check=True)
    subprocess.run(
        ["git", "config", "user.email", "audit@example.invalid"],
        cwd=project,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Audit Test"],
        cwd=project,
        check=True,
    )
    package = project / "description"
    package.mkdir()
    xacro = package / "robot.urdf.xacro"
    xacro.write_text("<robot name='source'/>\n")
    subprocess.run(["git", "add", "."], cwd=project, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=project, check=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        text=True,
    ).strip()

    fake_xacro = tmp_path / "fake_xacro"
    fake_xacro.write_text(
        """#!/usr/bin/env python3
import pathlib
import sys

if "--deps" in sys.argv:
    print(pathlib.Path(sys.argv[-1]).resolve())
    raise SystemExit(0)
output = pathlib.Path(sys.argv[sys.argv.index("-o") + 1])
robot_name = next(
    value.split(":=", 1)[1]
    for value in sys.argv
    if value.startswith("robot_name:=")
)
output.write_text(
    f'''<robot name="source">
  <link name="{robot_name}/base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
</robot>\\n'''
)
"""
    )
    fake_xacro.chmod(fake_xacro.stat().st_mode | 0o111)
    config = project / "config.yaml"
    config.write_text(
        f"""\
schema_version: 1
source:
  repository_root: {project}
  commit: {commit}
  package_name: description
  package_path: {package}
xacro:
  executable: {fake_xacro}
  python_path: ""
  ament_prefix_base: ""
common_args:
  base_link_frame: base_link
  use_gripper: "true"
  show_gripper_bar: "true"
  show_gripper_fingers: "true"
  use_world_frame: "false"
  external_urdf_loc: ""
  hardware_type: actual
outputs:
  directory: generated/urdf
  report_directory: reports/aloha1_mapping
robots:
  - name: follower_left
    xacro: description/robot.urdf.xacro
"""
    )

    report = build_all_from_config(config, project_root=project)

    assert report["status"] == "PASS"
    generated = project / "generated/urdf/follower_left.urdf"
    assert generated.is_file()
    assert ET.parse(generated).getroot().get("name") == "follower_left"
    assert json.loads(
        (project / "reports/aloha1_mapping/urdf_audit.json").read_text()
    )["status"] == "PASS"
    assert (project / "reports/aloha1_mapping/joint_inventory.csv").is_file()
    assert (project / "reports/aloha1_mapping/mesh_inventory.csv").is_file()
    assert "AMENT_PREFIX_PATH" not in os.environ
