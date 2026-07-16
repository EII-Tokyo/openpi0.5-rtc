from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG


VX300S_MESH_NAME_MAP = {
    "vx300s_1_base.stl": "base.stl",
    "vx300s_2_shoulder.stl": "shoulder.stl",
    "vx300s_3_upper_arm.stl": "upper_arm.stl",
    "vx300s_4_upper_forearm.stl": "upper_forearm.stl",
    "vx300s_5_lower_forearm.stl": "lower_forearm.stl",
    "vx300s_6_wrist.stl": "wrist.stl",
    "vx300s_7_gripper.stl": "gripper.stl",
    "vx300s_8_gripper_prop.stl": "gripper_prop.stl",
    "vx300s_9_gripper_bar.stl": "gripper_bar.stl",
    "vx300s_10_gripper_finger.stl": "gripper_finger.stl",
}

ARM_ONLY_LINK_SUFFIXES = {
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "upper_forearm_link",
    "lower_forearm_link",
    "wrist_link",
    "gripper_link",
}
ARM_ONLY_JOINTS = {"waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"}


def _make_resolved_urdf(source: Path, dest: Path, robot_name: str, package_dir: Path) -> dict[str, str]:
    text = source.read_text()
    text = re.sub(r'<robot\s+name="[^"]+"', f'<robot name="{robot_name}"', text, count=1)
    text = text.replace("package://interbotix_xsarm_descriptions/meshes/vx300s_meshes", str(package_dir / "meshes/vx300s_meshes"))
    mesh_rewrites = {}
    for old_name, new_name in VX300S_MESH_NAME_MAP.items():
        old_path = str(package_dir / "meshes/vx300s_meshes" / old_name)
        new_path = str(package_dir / "meshes/vx300s_meshes" / new_name)
        text = text.replace(old_path, new_path)
        mesh_rewrites[old_path] = new_path
    text = text.replace("package://interbotix_xsarm_descriptions", str(package_dir))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    return mesh_rewrites


def _write_arm_only_urdf(source: Path, dest: Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()

    def keep_link(name: str | None) -> bool:
        if name is None:
            return False
        suffix = name.rsplit("/", 1)[-1]
        return suffix in ARM_ONLY_LINK_SUFFIXES

    def keep_joint(name: str | None) -> bool:
        return name in ARM_ONLY_JOINTS

    for child in list(root):
        tag = child.tag
        name = child.attrib.get("name")
        keep = True
        if tag == "link":
            keep = keep_link(name)
        elif tag == "joint":
            keep = keep_joint(name)
        elif tag == "transmission":
            joint = child.find("joint")
            keep = keep_joint(joint.attrib.get("name") if joint is not None else None)
        elif tag == "gazebo":
            reference = child.attrib.get("reference")
            keep = keep_link(reference) or keep_joint(reference)
        if not keep:
            root.remove(child)

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass
    tree.write(dest, encoding="unicode", xml_declaration=True)


def _jsonable_vec(value) -> list[float] | None:
    if value is None:
        return None
    try:
        return [float(x) for x in value]
    except TypeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Import audited Original ALOHA vx300s URDFs into an Isaac USD stage.")
    parser.add_argument("--left-urdf", default="reports/aloha_model_audit/raw/robot_descriptions/puppet_left_robot_description.urdf")
    parser.add_argument("--right-urdf", default="reports/aloha_model_audit/raw/robot_descriptions/puppet_right_robot_description.urdf")
    parser.add_argument(
        "--package-dir",
        default="external/ros2-essentials/aloha_ws/src/interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_descriptions",
    )
    parser.add_argument("--output-root", default="assets/isaac/original_stationary_aloha")
    parser.add_argument("--merge-fixed-joints", action="store_true")
    parser.add_argument("--arm-only", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    generated = output_root / "generated"
    reports = output_root / "reports"
    generated.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    left_urdf = Path(args.left_urdf).resolve()
    right_urdf = Path(args.right_urdf).resolve()
    package_dir = Path(args.package_dir).resolve()
    left_resolved = generated / "puppet_left_vx300s_resolved.urdf"
    right_resolved = generated / "puppet_right_vx300s_resolved.urdf"
    left_mesh_rewrites = _make_resolved_urdf(left_urdf, left_resolved, "puppet_left_vx300s", package_dir)
    right_mesh_rewrites = _make_resolved_urdf(right_urdf, right_resolved, "puppet_right_vx300s", package_dir)
    import_left_urdf = left_resolved
    import_right_urdf = right_resolved
    arm_only_urdfs = {}
    if args.arm_only:
        import_left_urdf = generated / "puppet_left_vx300s_arm_only_resolved.urdf"
        import_right_urdf = generated / "puppet_right_vx300s_arm_only_resolved.urdf"
        _write_arm_only_urdf(left_resolved, import_left_urdf)
        _write_arm_only_urdf(right_resolved, import_right_urdf)
        arm_only_urdfs = {"left": str(import_left_urdf), "right": str(import_right_urdf)}

    from isaacsim import SimulationApp

    app = SimulationApp(dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG))
    try:
        import omni.kit.app
        import omni.kit.commands
        import omni.usd
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not status:
            raise RuntimeError("URDFCreateImportConfig failed")
        for attr, value in {
            "merge_fixed_joints": False,
            "import_inertia_tensor": True,
            "fix_base": True,
            "make_default_prim": False,
            "create_physics_scene": False,
            "self_collision": False,
        }.items():
            if hasattr(import_config, attr):
                setattr(import_config, attr, value)
        if hasattr(import_config, "merge_fixed_joints"):
            import_config.merge_fixed_joints = bool(args.merge_fixed_joints)

        def _set_default_prim(stage, side: str) -> str:
            root_candidates = [
                prim
                for prim in stage.GetPseudoRoot().GetChildren()
                if prim.IsValid() and prim.GetName() not in {"World", "physicsScene"}
            ]
            if not root_candidates:
                raise RuntimeError(f"Imported {side} USD has no root prim candidate for defaultPrim")
            if len(root_candidates) > 1:
                names = [str(prim.GetPath()) for prim in root_candidates]
                raise RuntimeError(f"Imported {side} USD has multiple root prim candidates: {names}")
            root = root_candidates[0]
            stage.SetDefaultPrim(root)
            stage.GetRootLayer().Save()
            return str(root.GetPath())

        def import_one(side: str, urdf: Path, usd_path: Path) -> dict[str, object]:
            usd_context = omni.usd.get_context()
            usd_context.new_stage()
            for _ in range(3):
                omni.kit.app.get_app().update()
            status, prim_path = omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=str(urdf),
                import_config=import_config,
                dest_path=str(usd_path.resolve()),
                get_articulation_root=True,
            )
            if not status:
                raise RuntimeError(f"URDFParseAndImportFile failed for {side}: {urdf}")
            for _ in range(5):
                omni.kit.app.get_app().update()
            stage = Usd.Stage.Open(str(usd_path.resolve()))
            default_prim_path = _set_default_prim(stage, side)
            report = inspect_stage(stage, str(prim_path))
            report["default_prim"] = default_prim_path
            return report

        def inspect_stage(stage, imported_root: str) -> dict[str, object]:
            joint_rows = []
            articulation_roots = []
            rigid_bodies = 0
            collisions = 0
            meshes = 0
            for prim in Usd.PrimRange(stage.GetPseudoRoot()):
                path = str(prim.GetPath())
                type_name = prim.GetTypeName()
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    articulation_roots.append(path)
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid_bodies += 1
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    collisions += 1
                if type_name == "Mesh":
                    meshes += 1
                if type_name.startswith("Physics") and type_name.endswith("Joint"):
                    joint_rows.append(
                        {
                            "path": path,
                            "type": type_name,
                            "axis": prim.GetAttribute("physics:axis").Get() if prim.HasAttribute("physics:axis") else None,
                            "lower": prim.GetAttribute("physics:lowerLimit").Get()
                            if prim.HasAttribute("physics:lowerLimit")
                            else None,
                            "upper": prim.GetAttribute("physics:upperLimit").Get()
                            if prim.HasAttribute("physics:upperLimit")
                            else None,
                        }
                    )
            return {
                "imported_root": imported_root,
                "stage_meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
                "articulation_roots": articulation_roots,
                "rigid_body_count": rigid_bodies,
                "collision_count": collisions,
                "mesh_count": meshes,
                "joint_count": len(joint_rows),
                "joints": joint_rows,
            }

        side_usds = {
            "left": generated / "vx300s_left.usd",
            "right": generated / "vx300s_right.usd",
        }
        side_reports = {
            "left": import_one("left", import_left_urdf, side_usds["left"]),
            "right": import_one("right", import_right_urdf, side_usds["right"]),
        }

        combined_usd = generated / "original_stationary_aloha.usd"
        combined_stage = Usd.Stage.CreateNew(str(combined_usd.resolve()))
        UsdGeom.SetStageMetersPerUnit(combined_stage, 1.0)
        world = UsdGeom.Xform.Define(combined_stage, Sdf.Path("/World"))
        combined_stage.SetDefaultPrim(world.GetPrim())
        for side, usd_path in side_usds.items():
            prim = UsdGeom.Xform.Define(combined_stage, Sdf.Path(f"/World/{side}_vx300s")).GetPrim()
            default_prim = side_reports[side]["default_prim"]
            prim.GetReferences().AddReference(str(usd_path.resolve()), Sdf.Path(default_prim))
        combined_stage.Save()

        combined_reopened = Usd.Stage.Open(str(combined_usd.resolve()))
        combined_report = inspect_stage(combined_reopened, "/World")

        report = {
            "status": "PASS",
            "source_urdfs": {"left": str(left_urdf), "right": str(right_urdf)},
            "resolved_urdfs": {"left": str(left_resolved), "right": str(right_resolved)},
            "import_urdfs": {"left": str(import_left_urdf), "right": str(import_right_urdf)},
            "arm_only": bool(args.arm_only),
            "arm_only_urdfs": arm_only_urdfs,
            "package_dir": str(package_dir),
            "mesh_rewrites": {"left": left_mesh_rewrites, "right": right_mesh_rewrites},
            "side_usds": {side: str(path) for side, path in side_usds.items()},
            "combined_usd": str(combined_usd),
            "side_reports": side_reports,
            "combined_report": combined_report,
            "base_pose_source": "reports/aloha_model_audit/raw/remote_103_focused_audit.txt static_transform_publisher: puppet_left and puppet_right at translation [0, 0.25, 0], rpy [0, 0, 0]",
            "base_pose_note": "The audited ROS launch gives both puppet_left and puppet_right the same world transform; this is recorded as source evidence, but physical left/right table separation remains a later calibration item.",
            "merge_fixed_joints": bool(args.merge_fixed_joints),
        }
        (reports / "import_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    finally:
        app.close(skip_cleanup=True)


if __name__ == "__main__":
    raise SystemExit(main())
