#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Build the isolated supplier-CAD follower_right robot-local asset.

The robot structure is explicitly referenced from the version-pinned
``follower_right/follower_right.usd`` import.  It is never produced by
renaming or mirroring the accepted follower_left diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RIGHT_ASSET = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/follower_vx300s/"
    "follower_right/follower_right.usd"
)
SOURCE_RIGHT_URDF = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/follower_vx300s/"
    "follower_right/source/follower_right.urdf"
)
SUPPLIER_GEOMETRY = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_finger_task5_convex_hull/geometry/supplier_cad_finger_mesh.usda"
)
APPROVED_LEFT_REVIEW_STAGE = (
    ROOT / "local_eval_assets/aloha_isaac_assets/aloha_viperx.usd"
)
CAD_IDENTITY_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_follower_left_right_cad_identity.json"
)
OUTPUT_ROOT = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "supplier_cad_follower_right/1.0"
)
OUTPUT_REPORT = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_supplier_cad_follower_right_asset.json"
)
EXPECTED_HASHES = {
    "source_right_asset": (
        "86d850cea5b35fb2969d3a78834317b51e2ac0d301f09aaaa9dad191f9bb3d5d"
    ),
    "source_right_urdf": (
        "268a5e5b56ce48af679256380008606f64358d6ead2619b52946b37f47fb624b"
    ),
    "supplier_geometry": (
        "781613d408843737b17d9f9a75e8c1b037ecc45749358d4b34ab48a8a7e98d4f"
    ),
    "approved_left_review_stage": (
        "b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e"
    ),
}
ROOT_PRIM = "/follower_right"
ROBOT_PRODUCT_PRIM = "/follower_right/vx300s_right"
SOURCE_PRODUCT_PRIM = "/follower_right"
DOF_NAMES = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
    "gripper",
    "left_finger",
    "right_finger",
)
FINGERS = {
    "left_finger": {
        "link": "follower_right_left_finger_link",
        "source_prim": "/CadFingerGeometry/left_finger",
        "cad_side": "+X",
        "source_obj_sha256": (
            "c6710d0fe5b2030a32722d9df5c0b553c771c9d61d92b8ddaec36c94c5963488"
        ),
    },
    "right_finger": {
        "link": "follower_right_right_finger_link",
        "source_prim": "/CadFingerGeometry/right_finger",
        "cad_side": "-X",
        "source_obj_sha256": (
            "b0979c5d55fee448dab512dc75b1251bab17d94892decd01de9a6e76c01482d1"
        ),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str) -> str:
    actual = _sha256(path.resolve(strict=True))
    if actual != expected:
        raise RuntimeError(
            f"protected hash mismatch: {path}: {actual} != {expected}"
        )
    return actual


def _relative(target: Path, owner: Path) -> str:
    return Path(
        os.path.relpath(target.resolve(), owner.resolve().parent)
    ).as_posix()


def _layer_for_path(stage: Any, path: Path) -> Any:
    for layer in stage.GetLayerStack(includeSessionLayers=False):
        if layer.realPath and Path(layer.realPath).resolve() == path.resolve():
            return layer
    raise RuntimeError(f"layer not found in stack: {path}")


def _create_layers(output_root: Path) -> dict[str, Path]:
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom

    if output_root.exists():
        raise FileExistsError(
            f"diagnostic output already exists: {output_root}"
        )
    geometry_dir = output_root / "geometry"
    configuration_dir = output_root / "configuration"
    physics_dir = output_root / "physics"
    geometry_dir.mkdir(parents=True)
    configuration_dir.mkdir()
    physics_dir.mkdir()
    paths = {
        "wrapper": output_root / "supplier_cad_follower_right.usda",
        "geometry": (
            geometry_dir / "supplier_cad_follower_right_geometry.usda"
        ),
        "configuration": (
            configuration_dir
            / "supplier_cad_follower_right_configuration.usda"
        ),
        "physics": (
            physics_dir / "supplier_cad_follower_right_physics.usda"
        ),
    }
    for key in ("geometry", "configuration", "physics"):
        layer = Sdf.Layer.CreateNew(str(paths[key]))
        if layer is None:
            raise RuntimeError(f"unable to create {key} layer")
        layer.Save()

    stage = Usd.Stage.CreateNew(str(paths["wrapper"]))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, ROOT_PRIM).GetPrim()
    product = UsdGeom.Xform.Define(stage, ROBOT_PRODUCT_PRIM).GetPrim()
    if not product.GetReferences().AddReference(
        _relative(SOURCE_RIGHT_ASSET, paths["wrapper"]),
        Sdf.Path("/follower_right"),
    ):
        raise RuntimeError("unable to reference explicit follower_right root")
    stage.GetRootLayer().subLayerPaths = [
        _relative(paths["physics"], paths["wrapper"]),
        _relative(paths["configuration"], paths["wrapper"]),
        _relative(paths["geometry"], paths["wrapper"]),
    ]
    stage.SetDefaultPrim(root)
    stage.GetRootLayer().Save()
    return paths


def _author_geometry(paths: dict[str, Path]) -> dict[str, Any]:
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom

    stage = Usd.Stage.Open(str(paths["wrapper"]), Usd.Stage.LoadAll)
    target = _layer_for_path(stage, paths["geometry"])
    stage.SetEditTarget(target)
    records = {}
    for name, spec in FINGERS.items():
        link_path = f"{ROBOT_PRODUCT_PRIM}/{spec['link']}"
        for role in ("visuals", "collisions"):
            role_path = f"{link_path}/{role}"
            role_prim = stage.GetPrimAtPath(role_path)
            if not role_prim.IsValid():
                raise RuntimeError(f"finger role parent missing: {role_path}")
            role_prim.SetInstanceable(False)  # noqa: FBT003
            authored_name = f"diagnostic_supplier_cad_{name}"
            authored_path = f"{role_path}/{authored_name}"
            xform = UsdGeom.Xform.Define(stage, authored_path).GetPrim()
            xform.SetCustomDataByKey(
                "aloha1:diagnosticRole",
                (
                    "SUPPLIER_CAD_V2_VISUAL_DIAGNOSTIC_NOT_FINAL"
                    if role == "visuals"
                    else "SUPPLIER_CAD_V2_CONVEX_HULL_DIAGNOSTIC_NOT_FINAL"
                ),
            )
            xform.SetCustomDataByKey(
                "aloha1:sourceObjSha256",
                spec["source_obj_sha256"],
            )
            if not xform.GetReferences().AddReference(
                _relative(SUPPLIER_GEOMETRY, paths["geometry"]),
                Sdf.Path(spec["source_prim"]),
            ):
                raise RuntimeError(
                    f"unable to reference supplier finger: {authored_path}"
                )
            records[f"{name}_{role}"] = authored_path
    target.Save()
    return records


def _author_configuration(
    paths: dict[str, Path],
    geometry_records: dict[str, Any],
) -> dict[str, Any]:
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdPhysics
    from usd.schema.isaac import robot_schema

    stage = Usd.Stage.Open(str(paths["wrapper"]), Usd.Stage.LoadAll)
    target = _layer_for_path(stage, paths["configuration"])
    stage.SetEditTarget(target)
    root = stage.GetPrimAtPath(ROOT_PRIM)
    product = stage.GetPrimAtPath(ROBOT_PRODUCT_PRIM)
    if not root.IsValid() or not product.IsValid():
        raise RuntimeError("right diagnostic root/product missing")
    root.SetCustomDataByKey(
        "aloha1:scope",
        "ROBOT_LOCAL_DIAGNOSTIC_NOT_WORKCELL_PLACEMENT",
    )
    root.SetCustomDataByKey(
        "aloha1:workcellPlacementAuthored",
        False,  # noqa: FBT003
    )
    product.SetCustomDataByKey(
        "aloha1:robotGeometryMirrored",
        False,  # noqa: FBT003
    )
    robot_schema.ApplyRobotAPI(root)

    generic_deactivated = []
    for spec in FINGERS.values():
        link_path = f"{ROBOT_PRODUCT_PRIM}/{spec['link']}"
        for role in ("visuals", "collisions"):
            role_path = f"{link_path}/{role}"
            stage.OverridePrim(role_path).SetInstanceable(
                False  # noqa: FBT003
            )
            generic_path = f"{role_path}/gripper_finger"
            if not stage.GetPrimAtPath(generic_path).IsValid():
                raise RuntimeError(
                    f"generic finger source missing: {generic_path}"
                )
            stage.OverridePrim(generic_path).SetActive(
                False  # noqa: FBT003
            )
            generic_deactivated.append(generic_path)

    link_paths = []
    joint_paths = []
    for prim in Usd.PrimRange(
        product,
        Usd.TraverseInstanceProxies(),
    ):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            path = str(prim.GetPath())
            robot_schema.ApplyLinkAPI(stage.GetPrimAtPath(path))
            link_paths.append(path)
        if prim.IsA(UsdPhysics.Joint):
            path = str(prim.GetPath())
            robot_schema.ApplyJointAPI(stage.GetPrimAtPath(path))
            joint_paths.append(path)
    if not link_paths or not joint_paths:
        raise RuntimeError("robot link/joint inventory is empty")

    relationships = {}
    for relation, targets in (
        (robot_schema.Relations.ROBOT_LINKS, link_paths),
        (robot_schema.Relations.ROBOT_JOINTS, joint_paths),
    ):
        relationship = root.GetRelationship(relation.name)
        relationship.ClearTargets(True)  # noqa: FBT003
        for path in targets:
            relationship.AddTarget(
                Sdf.Path(path),
                Usd.ListPositionBackOfPrependList,
            )
        relationships[relation.name] = [
            str(path) for path in relationship.GetTargets()
        ]
    target.Save()
    return {
        "generic_deactivated": generic_deactivated,
        "geometry_records": geometry_records,
        "link_paths": link_paths,
        "joint_paths": joint_paths,
        "relationships": relationships,
    }


def _author_physics(
    paths: dict[str, Path],
    geometry_records: dict[str, Any],
) -> dict[str, Any]:
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    stage = Usd.Stage.Open(str(paths["wrapper"]), Usd.Stage.LoadAll)
    target = _layer_for_path(stage, paths["physics"])
    stage.SetEditTarget(target)
    colliders = {}
    for name in FINGERS:
        xform_path = geometry_records[f"{name}_collisions"]
        mesh_path = f"{xform_path}/mesh"
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        if not mesh_prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"supplier collision mesh missing: {mesh_path}")
        UsdPhysics.CollisionAPI.Apply(mesh_prim)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        mesh_collision.CreateApproximationAttr().Set(
            UsdPhysics.Tokens.convexHull
        )
        UsdGeom.Imageable(stage.GetPrimAtPath(xform_path)).MakeInvisible()
        UsdGeom.Imageable(mesh_prim).CreatePurposeAttr().Set(
            UsdGeom.Tokens.guide
        )
        colliders[name] = mesh_path
    target.Save()
    return {
        "approximation": "convexHull",
        "collider_paths": colliders,
    }


def _readback(
    paths: dict[str, Path],
    configuration: dict[str, Any],
    physics: dict[str, Any],
) -> dict[str, Any]:
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics
    from usd.schema.isaac import robot_schema

    stage = Usd.Stage.Open(str(paths["wrapper"]), Usd.Stage.LoadAll)
    root = stage.GetDefaultPrim()
    product = stage.GetPrimAtPath(ROBOT_PRODUCT_PRIM)
    if str(root.GetPath()) != ROOT_PRIM:
        raise RuntimeError("unexpected default prim")
    articulation_roots = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(
            root,
            Usd.TraverseInstanceProxies(),
        )
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if articulation_roots != [f"{ROBOT_PRODUCT_PRIM}/root_joint"]:
        raise RuntimeError(
            f"unexpected articulation roots: {articulation_roots}"
        )

    invalid_body_targets = []
    for path in configuration["joint_paths"]:
        prim = stage.GetPrimAtPath(path)
        joint = UsdPhysics.Joint(prim)
        for relation_name, relationship in (
            ("body0", joint.GetBody0Rel()),
            ("body1", joint.GetBody1Rel()),
        ):
            invalid_body_targets.extend(
                {
                    "joint": path,
                    "relation": relation_name,
                    "target": str(target),
                }
                for target in relationship.GetTargets()
                if not stage.GetPrimAtPath(target).IsValid()
            )

    root_links = root.GetRelationship(
        robot_schema.Relations.ROBOT_LINKS.name
    ).GetTargets()
    root_joints = root.GetRelationship(
        robot_schema.Relations.ROBOT_JOINTS.name
    ).GetTargets()
    invalid_link_targets = [
        str(path)
        for path in root_links
        if not stage.GetPrimAtPath(path).IsValid()
    ]
    invalid_joint_targets = [
        str(path)
        for path in root_joints
        if not stage.GetPrimAtPath(path).IsValid()
    ]
    all_relationship_targets = [*root_links, *root_joints]
    all_under_product = all(
        str(path).startswith(f"{ROBOT_PRODUCT_PRIM}/")
        for path in all_relationship_targets
    )

    dof_paths = [f"{ROBOT_PRODUCT_PRIM}/joints/{name}" for name in DOF_NAMES]
    for path in dof_paths:
        prim = stage.GetPrimAtPath(path)
        if not (
            prim.IsA(UsdPhysics.RevoluteJoint)
            or prim.IsA(UsdPhysics.PrismaticJoint)
        ):
            raise RuntimeError(f"expected DOF joint missing: {path}")

    mesh_readback = {}
    supplier_fingers = {}
    for name, spec in FINGERS.items():
        mesh_path = physics["collider_paths"][name]
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        mesh = UsdGeom.Mesh(mesh_prim)
        approximation = UsdPhysics.MeshCollisionAPI(
            mesh_prim
        ).GetApproximationAttr().Get()
        if approximation != UsdPhysics.Tokens.convexHull:
            raise RuntimeError(
                f"collider approximation readback failed: {mesh_path}"
            )
        mesh_readback[name] = {
            "point_count": len(mesh.GetPointsAttr().Get() or []),
            "face_count": len(mesh.GetFaceVertexCountsAttr().Get() or []),
        }
        supplier_fingers[name] = {
            **spec,
            "visual_prim": configuration["geometry_records"][
                f"{name}_visuals"
            ],
            "collider_prim": mesh_path,
            "reference_xform_matrix": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "reference_xform_determinant": 1.0,
            "unit_scale_m_per_source_unit": 1.0,
        }
    if mesh_readback != {
        "left_finger": {"point_count": 831, "face_count": 1662},
        "right_finger": {"point_count": 831, "face_count": 1662},
    }:
        raise RuntimeError(f"unexpected supplier mesh readback: {mesh_readback}")
    for path in configuration["generic_deactivated"]:
        if stage.GetPrimAtPath(path).IsActive():
            raise RuntimeError(f"generic finger remained active: {path}")

    return {
        "root_prim": str(root.GetPath()),
        "robot_product_prim": str(product.GetPath()),
        "articulation_roots": articulation_roots,
        "dof_paths": dof_paths,
        "relationship_validation": {
            "invalid_joint_body_targets": invalid_body_targets,
            "invalid_robot_link_targets": invalid_link_targets,
            "invalid_robot_joint_targets": invalid_joint_targets,
            "all_targets_under_robot_product": all_under_product,
        },
        "supplier_fingers": supplier_fingers,
        "new_mesh_readback": mesh_readback,
        "generic_856_face_active": False,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Supplier-CAD follower_right robot-local asset",
            "",
            f"- Status: `{report['status']}`",
            f"- Scope: `{report['scope']}`",
            f"- Root prim: `{report['root_prim']}`",
            f"- Robot product: `{report['robot_product_prim']}`",
            f"- Articulation count: `{report['articulation_count']}`",
            f"- DOF order: `{', '.join(report['dof_order'])}`",
            "- Workcell placement: `NOT_AUTHORED`",
            "- Geometry mirroring: `false`",
            "- Task 8: `NOT_RUN`",
            "",
            "The robot structure is an explicit reference to the pinned "
            "`follower_right` import. Supplier embedded-v2 handed finger "
            "geometry is composed in separate diagnostic geometry, "
            "configuration, and physics layers. No left-asset string rename, "
            "robot mirroring, final-collider promotion, or workcell transform "
            "is used.",
            "",
            "## Remaining blocker",
            "",
            "- `HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM`",
            "",
        ]
    )


def build(
    *,
    output_root: Path,
    output_report: Path,
    audit_existing: bool = False,
) -> dict[str, Any]:
    protected = {
        "source_right_asset": (
            SOURCE_RIGHT_ASSET,
            _require_hash(
                SOURCE_RIGHT_ASSET,
                EXPECTED_HASHES["source_right_asset"],
            ),
        ),
        "source_right_urdf": (
            SOURCE_RIGHT_URDF,
            _require_hash(
                SOURCE_RIGHT_URDF,
                EXPECTED_HASHES["source_right_urdf"],
            ),
        ),
        "supplier_geometry": (
            SUPPLIER_GEOMETRY,
            _require_hash(
                SUPPLIER_GEOMETRY,
                EXPECTED_HASHES["supplier_geometry"],
            ),
        ),
        "approved_left_review_stage": (
            APPROVED_LEFT_REVIEW_STAGE,
            _require_hash(
                APPROVED_LEFT_REVIEW_STAGE,
                EXPECTED_HASHES["approved_left_review_stage"],
            ),
        ),
    }
    identity = json.loads(
        CAD_IDENTITY_REPORT.resolve(strict=True).read_text(encoding="utf-8")
    )
    if identity["classification"] != (
        "VERIFIED_SINGLE_REUSABLE_ROBOT_PRODUCT"
    ):
        raise RuntimeError("CAD identity gate is not verified reusable")
    if audit_existing:
        paths = {
            "wrapper": output_root / "supplier_cad_follower_right.usda",
            "geometry": (
                output_root
                / "geometry/supplier_cad_follower_right_geometry.usda"
            ),
            "configuration": (
                output_root
                / "configuration/"
                "supplier_cad_follower_right_configuration.usda"
            ),
            "physics": (
                output_root
                / "physics/supplier_cad_follower_right_physics.usda"
            ),
        }
        if not all(path.is_file() for path in paths.values()):
            raise FileNotFoundError("existing diagnostic layer set is incomplete")
        prior = json.loads(
            output_report.resolve(strict=True).read_text(encoding="utf-8")
        )
        configuration = prior["configuration"]
        physics = prior["physics"]
    else:
        paths = _create_layers(output_root)
        geometry = _author_geometry(paths)
        configuration = _author_configuration(paths, geometry)
        physics = _author_physics(paths, geometry)
    readback = _readback(paths, configuration, physics)
    unchanged = all(_sha256(path) == sha for path, sha in protected.values())
    if not unchanged:
        raise RuntimeError("a protected source changed during the build")
    files = {
        name: {
            "absolute_path": str(path.resolve()),
            "sha256": _sha256(path),
        }
        for name, path in paths.items()
    }
    report = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "ROBOT_LOCAL_DIAGNOSTIC_NOT_WORKCELL_PLACEMENT",
        "root_prim": readback["root_prim"],
        "robot_product_prim": readback["robot_product_prim"],
        "articulation_roots": readback["articulation_roots"],
        "articulation_count": len(readback["articulation_roots"]),
        "dof_order": list(DOF_NAMES),
        "dof_paths": readback["dof_paths"],
        "dof_count": len(DOF_NAMES),
        "construction": {
            "method": (
                "EXPLICIT_REFERENCE_TO_VERSION_PINNED_FOLLOWER_RIGHT_PRODUCT_"
                "PLUS_SUPPLIER_CAD_FINGER_LAYERS"
            ),
            "source_product_prim": SOURCE_PRODUCT_PRIM,
            "robot_geometry_mirrored": False,
            "workcell_placement_authored": False,
            "string_rename_used": False,
        },
        "source_right_asset": {
            "absolute_path": str(SOURCE_RIGHT_ASSET.resolve()),
            "sha256": protected["source_right_asset"][1],
            "source_prim": SOURCE_PRODUCT_PRIM,
            "modified": False,
        },
        "source_right_urdf": {
            "absolute_path": str(SOURCE_RIGHT_URDF.resolve()),
            "sha256": protected["source_right_urdf"][1],
            "modified": False,
        },
        "supplier_geometry_source": {
            "absolute_path": str(SUPPLIER_GEOMETRY.resolve()),
            "sha256": protected["supplier_geometry"][1],
            "modified": False,
        },
        "approved_left_review_stage": {
            "absolute_path": str(APPROVED_LEFT_REVIEW_STAGE.resolve()),
            "sha256": protected["approved_left_review_stage"][1],
            "modified": False,
            "role": "PROTECTED_REFERENCE_EVIDENCE_NOT_RIGHT_STRUCTURE_SOURCE",
        },
        "cad_identity_report": {
            "absolute_path": str(CAD_IDENTITY_REPORT.resolve()),
            "sha256": _sha256(CAD_IDENTITY_REPORT),
            "classification": identity["classification"],
        },
        "files": files,
        "configuration": configuration,
        "physics": physics,
        "relationship_validation": readback["relationship_validation"],
        "supplier_fingers": {
            **readback["supplier_fingers"],
            "mirrored": False,
            "generic_856_face_active": readback[
                "generic_856_face_active"
            ],
            "new_mesh_readback": readback["new_mesh_readback"],
        },
        "protected_inputs_unchanged": unchanged,
        "hard_blockers": [
            "HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM"
        ],
        "final_default_collider_modified": False,
        "license": {
            "status": "UNKNOWN_HARD_BLOCKER",
            "source_cad_redistributed": False,
        },
        "isaac_version_contract": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
        },
        "task8": "NOT_RUN",
    }
    output_report = output_report.resolve()
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_report.with_suffix(".md").write_text(
        _render_markdown(report),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--output-report", type=Path, default=OUTPUT_REPORT)
    parser.add_argument("--audit-existing", action="store_true")
    args = parser.parse_args()
    report = build(
        output_root=args.output_root.resolve(),
        output_report=args.output_report,
        audit_existing=args.audit_existing,
    )
    print(f"status={report['status']}")
    print(f"wrapper={report['files']['wrapper']['absolute_path']}")
    print(f"report={args.output_report.resolve()}")
    return 0


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    exit_code = 1
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        for extension_id in (
            "isaacsim.robot.schema",
            "isaacsim.asset.validation",
        ):
            if not manager.is_extension_enabled(extension_id):
                manager.set_extension_enabled_immediate(
                    extension_id,
                    True,  # noqa: FBT003
                )
            if not manager.is_extension_enabled(extension_id):
                raise RuntimeError(
                    f"required extension disabled: {extension_id}"
                )
        exit_code = main()
    except Exception:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
