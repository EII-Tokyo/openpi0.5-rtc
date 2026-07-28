"""Read-only audit helpers for the user-approved ALOHA Viper review Stage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_REVIEW_STAGE_SHA256 = (
    "b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e"
)

REQUIRED_REVIEW_STAGE_PRIMS = (
    "/workcell",
    "/workcell/vx300s_left",
    "/workcell/vx300s_left/vx300s_left",
    "/workcell/vx300s_left/vx300s_left_gripper_link",
    "/workcell/vx300s_left/vx300s_left_left_finger_link",
    "/workcell/vx300s_left/vx300s_left_right_finger_link",
    "/workcell/joints/vx300s_left_left_finger",
    "/workcell/joints/vx300s_left_right_finger",
)


def evaluate_stage_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    missing_required_prims = [
        path
        for path in REQUIRED_REVIEW_STAGE_PRIMS
        if not snapshot.get("required_prims", {})
        .get(path, {})
        .get("valid", False)
    ]
    source_immutable = (
        snapshot.get("source_sha256_before") == EXPECTED_REVIEW_STAGE_SHA256
        and snapshot.get("source_sha256_after") == EXPECTED_REVIEW_STAGE_SHA256
    )
    root_prim_ok = (
        snapshot.get("default_prim") == "/workcell"
        and snapshot.get("meters_per_unit") == 1.0
        and snapshot.get("up_axis") == "Z"
    )
    used_layers = snapshot.get("used_layers", [])
    layer_stack_ok = len(used_layers) >= 3 and all(
        layer.get("exists")
        and len(layer.get("sha256", "")) == 64
        for layer in used_layers
    )
    finger_branches = snapshot.get("finger_branches", {})
    instance_structure_ok = all(
        finger_branches.get(side, {}).get(key) is True
        for side in ("left", "right")
        for key in (
            "visuals_instanceable",
            "visual_mesh_is_instance_proxy",
            "collisions_instanceable",
            "collision_mesh_is_instance_proxy",
        )
    )
    gates = {
        "source_immutable_gate": source_immutable,
        "root_prim_gate": root_prim_ok,
        "required_key_prims_gate": not missing_required_prims,
        "layer_stack_gate": layer_stack_ok,
        "instance_structure_gate": instance_structure_ok,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    return {
        **snapshot,
        "schema_version": 1,
        "status": status,
        "authorization": {
            "classification": (
                "USER_APPROVED_ISOLATED_DIAGNOSTIC_REVIEW_STAGE"
            ),
            "scope": (
                "Read-only source inspection and independent diagnostic "
                "layer composition only"
            ),
            "source_stage_mutation_allowed": False,
            "default_or_final_collider_mutation_allowed": False,
        },
        "source_immutable_gate": (
            "PASS" if source_immutable else "FAIL"
        ),
        "root_prim_gate": "PASS" if root_prim_ok else "FAIL",
        "required_key_prims_status": (
            "PASS" if not missing_required_prims else "FAIL"
        ),
        "layer_stack_status": "PASS" if layer_stack_ok else "FAIL",
        "instance_structure_status": (
            "PASS" if instance_structure_ok else "FAIL"
        ),
        "missing_required_prims": missing_required_prims,
        "instance_proxy_strategy": (
            "DEINSTANCE_VISUAL_BRANCH_IN_DIAGNOSTIC_LAYER_ONLY"
            if instance_structure_ok
            else "UNRESOLVED_SOURCE_STRUCTURE"
        ),
        "gates": gates,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mesh_record(stage: Any, branch_path: str) -> dict[str, Any]:
    from pxr import Usd
    from pxr import UsdGeom

    branch = stage.GetPrimAtPath(branch_path)
    meshes = [
        prim
        for prim in Usd.PrimRange(
            branch,
            Usd.TraverseInstanceProxies(),
        )
        if prim.IsA(UsdGeom.Mesh)
    ]
    if len(meshes) != 1:
        return {
            "mesh_count": len(meshes),
            "mesh_path": None,
            "mesh_is_instance_proxy": False,
        }
    mesh_prim = meshes[0]
    mesh = UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get() or []
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get() or []
    return {
        "mesh_count": 1,
        "mesh_path": str(mesh_prim.GetPath()),
        "mesh_is_instance_proxy": mesh_prim.IsInstanceProxy(),
        "point_count": len(points),
        "face_count": len(face_vertex_counts),
        "prim_stack": [
            {
                "layer": spec.layer.realPath or spec.layer.identifier,
                "path": str(spec.path),
                "specifier": str(spec.specifier),
            }
            for spec in mesh_prim.GetPrimStack()
        ],
    }


def collect_stage_snapshot(stage_path: Path) -> dict[str, Any]:
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    stage_path = stage_path.resolve()
    source_sha256_before = _sha256(stage_path)
    stage = Usd.Stage.Open(str(stage_path), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Unable to open USD Stage: {stage_path}")

    required_prims = {}
    for path in REQUIRED_REVIEW_STAGE_PRIMS:
        prim = stage.GetPrimAtPath(path)
        required_prims[path] = {
            "valid": prim.IsValid(),
            "type_name": prim.GetTypeName() if prim.IsValid() else None,
            "is_instance": prim.IsInstance() if prim.IsValid() else False,
            "is_instance_proxy": (
                prim.IsInstanceProxy() if prim.IsValid() else False
            ),
            "prim_stack": [
                {
                    "layer": spec.layer.realPath or spec.layer.identifier,
                    "path": str(spec.path),
                    "specifier": str(spec.specifier),
                }
                for spec in prim.GetPrimStack()
            ]
            if prim.IsValid()
            else [],
        }

    used_layers = []
    for layer in stage.GetUsedLayers():
        real_path = Path(layer.realPath).resolve() if layer.realPath else None
        if real_path is None:
            continue
        used_layers.append(
            {
                "absolute_path": str(real_path),
                "exists": real_path.is_file(),
                "sha256": _sha256(real_path) if real_path.is_file() else None,
                "sublayer_paths": list(layer.subLayerPaths),
            }
        )
    used_layers.sort(key=lambda item: item["absolute_path"])

    finger_branches = {}
    for side in ("left", "right"):
        link_path = (
            "/workcell/vx300s_left/"
            f"vx300s_left_{side}_finger_link"
        )
        visual_path = f"{link_path}/visuals"
        collision_path = f"{link_path}/collisions"
        visual_branch = stage.GetPrimAtPath(visual_path)
        collision_branch = stage.GetPrimAtPath(collision_path)
        visual_record = _mesh_record(stage, visual_path)
        collision_record = _mesh_record(stage, collision_path)
        finger_branches[side] = {
            "link_path": link_path,
            "visuals_path": visual_path,
            "visuals_instanceable": visual_branch.IsInstanceable(),
            **visual_record,
            "visual_mesh_is_instance_proxy": visual_record[
                "mesh_is_instance_proxy"
            ],
            "collisions_path": collision_path,
            "collisions_instanceable": collision_branch.IsInstanceable(),
            "collision_mesh_path": collision_record["mesh_path"],
            "collision_mesh_count": collision_record["mesh_count"],
            "collision_mesh_is_instance_proxy": collision_record[
                "mesh_is_instance_proxy"
            ],
            "collision_point_count": collision_record.get("point_count"),
            "collision_face_count": collision_record.get("face_count"),
        }

    joints = {}
    for side in ("left", "right"):
        path = f"/workcell/joints/vx300s_left_{side}_finger"
        prim = stage.GetPrimAtPath(path)
        joint = UsdPhysics.PrismaticJoint(prim)
        joints[side] = {
            "path": path,
            "axis": joint.GetAxisAttr().Get(),
            "lower_limit": joint.GetLowerLimitAttr().Get(),
            "upper_limit": joint.GetUpperLimitAttr().Get(),
            "body0": [str(item) for item in joint.GetBody0Rel().GetTargets()],
            "body1": [str(item) for item in joint.GetBody1Rel().GetTargets()],
        }

    articulation_roots = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    default_prim = stage.GetDefaultPrim()
    snapshot = {
        "absolute_path": str(stage_path),
        "read_only": True,
        "source_sha256_before": source_sha256_before,
        "source_sha256_after": _sha256(stage_path),
        "default_prim": (
            str(default_prim.GetPath()) if default_prim.IsValid() else None
        ),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "root_payload": str(default_prim.GetMetadata("payload")),
        "root_variant_selections": {
            name: default_prim.GetVariantSet(name).GetVariantSelection()
            for name in default_prim.GetVariantSets().GetNames()
        },
        "required_prims": required_prims,
        "used_layers": used_layers,
        "articulation_roots": articulation_roots,
        "target_follower_articulation_root": (
            "/workcell/vx300s_left/vx300s_left"
        ),
        "finger_branches": finger_branches,
        "finger_joints": joints,
    }
    return snapshot


def write_stage_audit(
    report: dict[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Authorized ALOHA Viper Review Stage Audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Stage: `{report['absolute_path']}`",
        f"- Source SHA-256 before: `{report['source_sha256_before']}`",
        f"- Source SHA-256 after: `{report['source_sha256_after']}`",
        f"- Default prim: `{report['default_prim']}`",
        f"- Stage units: `{report['meters_per_unit']} m/unit`",
        f"- Up axis: `{report['up_axis']}`",
        f"- Required key prims: `{report['required_key_prims_status']}`",
        f"- Layer stack: `{report['layer_stack_status']}`",
        f"- Instance structure: `{report['instance_structure_status']}`",
        "",
        "## Authorization boundary",
        "",
        "- Source Stage mutation: `FORBIDDEN`",
        "- Default/final collider mutation: `FORBIDDEN`",
        "- Allowed output: independent diagnostic layer only",
        "",
        "## Required prims",
        "",
    ]
    for path, record in report["required_prims"].items():
        lines.append(
            f"- `{path}`: `{'PASS' if record['valid'] else 'FAIL'}` "
            f"({record['type_name']})"
        )
    lines.extend(["", "## Used file-backed layers", ""])
    for layer in report["used_layers"]:
        lines.append(
            f"- `{layer['absolute_path']}` — `{layer['sha256']}`"
        )
    lines.extend(
        [
            "",
            "## Instance-proxy consequence",
            "",
            (
                "Both finger visual and collision branches are instanceable, "
                "and their Mesh prims are instance proxies. The permitted "
                "strategy is to de-instance only the visual branch in the "
                "independent diagnostic layer; collision branches remain "
                "unchanged until separately audited."
            ),
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
