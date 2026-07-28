from __future__ import annotations

from pathlib import Path

from tools.aloha1_mapping.urdf_audit import audit_urdf


def _write_valid_urdf(tmp_path: Path) -> Path:
    mesh = tmp_path / "finger.stl"
    mesh.write_bytes(b"solid finger\nendsolid finger\n")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        f"""\
<robot name="fixture">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
  <link name="arm">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
  <link name="left_finger">
    <visual><geometry><mesh filename="{mesh.name}"/></geometry></visual>
    <collision><geometry><mesh filename="{mesh.name}"/></geometry></collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.1"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <link name="right_finger">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.1"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <joint name="waist" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <parent link="base"/>
    <child link="arm"/>
    <limit lower="-1" upper="1" effort="10" velocity="2"/>
  </joint>
  <joint name="left_finger_joint" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <parent link="arm"/>
    <child link="left_finger"/>
    <limit lower="0.01" upper="0.05" effort="5" velocity="1"/>
  </joint>
  <joint name="right_finger_joint" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 -1 0"/>
    <parent link="arm"/>
    <child link="right_finger"/>
    <limit lower="0.01" upper="0.05" effort="5" velocity="1"/>
    <mimic joint="left_finger_joint" multiplier="1" offset="0"/>
  </joint>
</robot>
"""
    )
    return urdf


def test_audit_preserves_source_order_and_inventories_mimic_and_mesh(
    tmp_path: Path,
) -> None:
    urdf = _write_valid_urdf(tmp_path)

    report = audit_urdf(urdf, package_map={})

    assert report["status"] == "PASS"
    assert report["link_order"] == [
        "base",
        "arm",
        "left_finger",
        "right_finger",
    ]
    assert report["joint_order"] == [
        "waist",
        "left_finger_joint",
        "right_finger_joint",
    ]
    assert report["root_links"] == ["base"]
    assert report["mimic"] == [
        {
            "joint": "right_finger_joint",
            "parent": "left_finger_joint",
            "multiplier": 1.0,
            "offset": 0.0,
        }
    ]
    assert report["meshes"][0]["exists"] is True
    assert report["meshes"][0]["sha256"]
    assert report["missing_dynamics"] == []


def test_audit_resolves_package_uri_from_explicit_package_map(
    tmp_path: Path,
) -> None:
    package = tmp_path / "description"
    mesh = package / "meshes" / "base.stl"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"mesh")
    urdf = _write_valid_urdf(tmp_path)
    text = urdf.read_text().replace(
        'filename="finger.stl"',
        'filename="package://robot_description/meshes/base.stl"',
    )
    urdf.write_text(text)

    report = audit_urdf(
        urdf,
        package_map={"robot_description": package},
    )

    assert report["status"] == "PASS"
    assert report["meshes"][0]["resolved_path"] == str(mesh.resolve())


def test_audit_fails_duplicate_names_invalid_limits_and_non_tree_parent(
    tmp_path: Path,
) -> None:
    urdf = _write_valid_urdf(tmp_path)
    text = urdf.read_text()
    text = text.replace(
        "</robot>",
        """\
  <link name="arm"/>
  <joint name="bad_joint" type="revolute">
    <parent link="base"/>
    <child link="left_finger"/>
    <limit lower="1" upper="-1" effort="0" velocity="0"/>
  </joint>
</robot>
""",
    )
    urdf.write_text(text)

    report = audit_urdf(urdf, package_map={})

    assert report["status"] == "FAIL"
    codes = {issue["code"] for issue in report["issues"]}
    assert "DUPLICATE_LINK_NAME" in codes
    assert "MULTIPLE_PARENT_JOINTS" in codes
    assert "MISSING_JOINT_ORIGIN" in codes
    assert "MISSING_JOINT_AXIS" in codes
    assert "INVALID_POSITION_LIMIT" in codes
    assert "INVALID_EFFORT_LIMIT" in codes
    assert "INVALID_VELOCITY_LIMIT" in codes


def test_audit_lists_missing_inertia_and_unresolved_mesh_without_defaults(
    tmp_path: Path,
) -> None:
    urdf = _write_valid_urdf(tmp_path)
    text = urdf.read_text()
    text = text.replace(
        '<link name="arm">\n    <inertial>',
        '<link name="arm">\n    <visual><geometry><mesh '
        'filename="package://missing/arm.stl"/></geometry></visual>\n'
        "    <inertial>",
    )
    text = text.replace(
        """\
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
  <link name="left_finger">""",
        """\
  </link>
  <link name="left_finger">""",
        1,
    )
    urdf.write_text(text)

    report = audit_urdf(urdf, package_map={})

    assert report["status"] == "FAIL"
    assert {"link": "arm", "missing": ["inertial"]} in report[
        "missing_dynamics"
    ]
    codes = {issue["code"] for issue in report["issues"]}
    assert "UNRESOLVED_PACKAGE_URI" in codes
    assert "MISSING_INERTIAL" in codes


def test_audit_records_implicit_identity_inertial_origin_without_failure(
    tmp_path: Path,
) -> None:
    urdf = _write_valid_urdf(tmp_path)
    text = urdf.read_text().replace(
        '<origin xyz="0 0 0" rpy="0 0 0"/>\n      <mass value="1"/>',
        '<mass value="1"/>',
        1,
    )
    urdf.write_text(text)

    report = audit_urdf(urdf, package_map={})

    assert report["status"] == "PASS"
    assert report["missing_dynamics"] == []
    assert report["dynamics"][0]["origin_explicit"] is False
    assert report["dynamics"][0]["center_of_mass_xyz"] == [0.0, 0.0, 0.0]
