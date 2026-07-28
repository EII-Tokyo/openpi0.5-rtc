from __future__ import annotations

from pathlib import Path

import pytest

from tools.import_aloha1_to_usd import _urdf_links_without_visuals
from tools.import_aloha1_to_usd import build_import_plan


def test_import_plan_pins_conservative_isaac_5_1_settings(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "generated" / "urdf"
    generated.mkdir(parents=True)
    for name in ("follower_left", "follower_right", "leader_left", "leader_right"):
        (generated / f"{name}.urdf").write_text(
            f"<robot name='{name}'><link name='base'/></robot>\n"
        )

    plan = build_import_plan(project_root=tmp_path, enable_leaders=False)

    assert [item["name"] for item in plan["robots"]] == [
        "follower_left",
        "follower_right",
    ]
    assert plan["output_strategy"] == "direct_to_stable_destination"
    assert plan["post_import_dependency_check"] is True
    assert plan["settings"] == {
        "merge_fixed_joints": False,
        "replace_cylinders_with_capsules": False,
        "convex_decomp": False,
        "import_inertia_tensor": True,
        "fix_base": True,
        "self_collision": False,
        "density": 0.0,
        "distance_scale": 1.0,
        "default_drive_type": "JOINT_DRIVE_POSITION",
        "default_drive_strength": 1000.0,
        "default_position_drive_damping": 100.0,
        "make_default_prim": True,
        "parse_mimic": True,
        "create_physics_scene": False,
        "collision_from_visuals": False,
        "override_joint_dynamics": False,
        "mesh_merge_requested": False,
        "requires_complete_urdf_dynamics": True,
    }
    assert plan["robots"][0]["output_dir"] == str(
        (
            tmp_path
            / "assets/Trossen/ALOHA1/1.0/follower_vx300s/follower_left"
        ).resolve()
    )
    assert plan["robots"][0]["imported_usd"].endswith(
        "/source/follower_left_imported.usd"
    )
    assert "/." not in plan["robots"][0]["imported_usd"]


def test_import_plan_enables_leaders_only_with_explicit_switch(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "generated" / "urdf"
    generated.mkdir(parents=True)
    for name in ("follower_left", "follower_right", "leader_left", "leader_right"):
        (generated / f"{name}.urdf").write_text(
            f"<robot name='{name}'><link name='base'/></robot>\n"
        )

    plan = build_import_plan(project_root=tmp_path, enable_leaders=True)

    assert [item["name"] for item in plan["robots"]] == [
        "follower_left",
        "follower_right",
        "leader_left",
        "leader_right",
    ]
    assert plan["robots"][2]["output_dir"].endswith(
        "/leader_wx250s/leader_left"
    )


def test_import_plan_refuses_missing_generated_urdf(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="follower_left.urdf"):
        build_import_plan(project_root=tmp_path, enable_leaders=False)


def test_empty_visual_links_are_derived_from_urdf_not_hard_coded(
    tmp_path: Path,
) -> None:
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """
<robot name="r">
  <link name="has_visual"><visual><geometry><box size="1 1 1"/></geometry></visual></link>
  <link name="inertial_only"><inertial><mass value="1"/></inertial></link>
</robot>
""".strip()
    )

    assert _urdf_links_without_visuals(urdf) == ["inertial_only"]
