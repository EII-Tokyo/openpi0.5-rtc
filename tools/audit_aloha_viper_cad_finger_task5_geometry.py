#!/usr/bin/env python3
"""Audit supplier-CAD finger convex hulls against internal gripper colliders."""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np

from tools.aloha1_mapping.cad_finger_task5_structure import FINGER_DOF_NAMES
from tools.aloha1_mapping.cad_finger_task5_structure import LEGAL_POSES_M
from tools.aloha1_mapping.convex_geometry_audit import collider_summary
from tools.aloha1_mapping.convex_geometry_audit import convex_pair_relation

ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_finger_task5_convex_hull/aloha_viperx_supplier_cad_task5.usda"
)
OUTPUT_JSON = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha_viper_cad_finger_task5_geometry_audit.json"
)
OUTPUT_MD = OUTPUT_JSON.with_suffix(".md")
ARTICULATION_PATH = "/workcell/vx300s_left/vx300s_left"
COLLIDER_PREFIXES = (
    "/workcell/vx300s_left/vx300s_left_gripper_link/collisions/",
    "/workcell/vx300s_left/vx300s_left_gripper_prop_link/collisions/",
    "/workcell/vx300s_left/vx300s_left_left_finger_link/collisions/",
    "/workcell/vx300s_left/vx300s_left_right_finger_link/collisions/",
)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_physx_transforms_to_usd(physx_interface: Any) -> None:
    physx_interface.update_transformations(
        True, True, False, False  # noqa: FBT003
    )


def _collider_role(path: str) -> str:
    if "left_finger_link" in path:
        return "left_finger"
    if "right_finger_link" in path:
        return "right_finger"
    if "gripper_bar" in path:
        return "gripper_bar"
    if "gripper_prop_link" in path:
        return "sliding_carriage"
    if "gripper_link" in path:
        return "gripper_shell"
    return "other"


def _world_points(prim: Any) -> np.ndarray:
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get() or []
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    return np.asarray(
        [list(transform.Transform(point)) for point in points],
        dtype=np.float64,
    )


def _collision_inventory(stage: Any) -> list[dict[str, Any]]:
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    root = stage.GetPrimAtPath("/workcell/vx300s_left")
    records = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if (
            not any(path.startswith(prefix) for prefix in COLLIDER_PREFIXES)
            or not prim.IsA(UsdGeom.Mesh)
            or not prim.HasAPI(UsdPhysics.CollisionAPI)
        ):
            continue
        approximation = (
            UsdPhysics.MeshCollisionAPI(prim)
            .GetApproximationAttr()
            .Get()
        )
        if approximation != "convexHull":
            raise RuntimeError(
                f"unexpected approximation for geometry gate: "
                f"{path}={approximation}"
            )
        points = _world_points(prim)
        records.append(
            {
                "path": path,
                "role": _collider_role(path),
                "approximation": approximation,
                "instance_proxy": prim.IsInstanceProxy(),
                "world_points": points,
            }
        )
    return records


def _pair_scope(role_a: str, role_b: str) -> str:
    roles = {role_a, role_b}
    if roles == {"left_finger", "right_finger"}:
        return "FINGER_TO_FINGER_UNEXPECTED_IF_OVERLAP"
    if roles & {"left_finger", "right_finger"}:
        return (
            "FINGER_TO_ATTACHMENT_COMPONENT_"
            "REQUIRES_ASSEMBLY_SEMANTIC_REVIEW"
        )
    return "OUT_OF_SCOPE_NONFINGER_PAIR"


def _state_geometry(stage: Any) -> dict[str, Any]:
    inventory = _collision_inventory(stage)
    colliders = []
    for record in inventory:
        points = record.pop("world_points")
        colliders.append(
            {
                **record,
                **collider_summary(points),
                "_points": points,
            }
        )
    pairs = []
    for first, second in combinations(colliders, 2):
        scope = _pair_scope(first["role"], second["role"])
        if scope == "OUT_OF_SCOPE_NONFINGER_PAIR":
            continue
        relation = convex_pair_relation(
            first["_points"],
            second["_points"],
        )
        pairs.append(
            {
                "collider_a": first["path"],
                "role_a": first["role"],
                "collider_b": second["path"],
                "role_b": second["role"],
                "scope": scope,
                **relation,
            }
        )
    for collider in colliders:
        collider.pop("_points")
    return {"colliders": colliders, "pairs": pairs}


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA ViperX CAD finger Task 5 geometry audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Stage: `{report['stage']['absolute_path']}`",
        f"- Stage SHA-256: `{report['stage']['sha256']}`",
        "- Runtime mutation: none saved; legal poses were session-only.",
        "- Collider approximation audited: `convexHull`.",
        (
            "- Method: world-transformed source points → numerical convex "
            "hull → normalized halfspace LP and intersection volume."
        ),
        (
            "- Boundary: attachment-component overlap is reported "
            "numerically and is not automatically called an error."
        ),
        "",
    ]
    for state in report["states"]:
        lines.extend(
            [
                f"## {state['state']}",
                "",
                "| Pair | Relation | Margin m | Overlap m³ | Scope |",
                "|---|---:|---:|---:|---|",
            ]
        )
        lines.extend(
            (
                f"| `{pair['role_a']} ↔ {pair['role_b']}` | "
                f"`{pair['relation']}` | "
                f"`{pair['signed_chebyshev_margin_m']:.9g}` | "
                f"`{pair['overlap_volume_m3']:.9g}` | "
                f"`{pair['scope']}` |"
            )
            for pair in state["pairs"]
        )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            f"- Finger-to-finger overlap gate: "
            f"`{report['gates']['no_finger_to_finger_overlap']}`.",
            (
                "- Finger-to-shell/bar/carriage relations remain assembly "
                "evidence. A volumetric common region may be a designed "
                "mounting interface and requires CAD assembly semantics."
            ),
            (
                "- This static audit does not prove dynamic collision "
                "resolution, drive tracking, contact, or bottle hold."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    from isaacsim import SimulationApp

    stage_path = STAGE.resolve(strict=True)
    stage_hash_before = _sha256(stage_path)
    app = SimulationApp({"headless": True})
    exit_code = 1
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.stage import open_stage
        from omni.physx import get_physx_interface

        if not open_stage(str(stage_path)):
            raise RuntimeError(f"failed to open {stage_path}")
        stage = get_current_stage()
        world = World(
            stage_units_in_meters=1.0,
            backend="numpy",
            device="cpu",
            physics_dt=1.0 / 60.0,
            rendering_dt=1.0 / 60.0,
        )
        articulation = SingleArticulation(
            prim_path=ARTICULATION_PATH,
            name="supplier_cad_task5_geometry_audit",
            reset_xform_properties=False,
        )
        world.scene.add(articulation)
        world.reset()
        order = list(articulation.dof_names)
        left_index = order.index(FINGER_DOF_NAMES[0])
        right_index = order.index(FINGER_DOF_NAMES[1])
        home_by_name = {
            "vx300s_left_waist": 0.0,
            "vx300s_left_shoulder": -0.96,
            "vx300s_left_elbow": 1.16,
            "vx300s_left_forearm_roll": 0.0,
            "vx300s_left_wrist_angle": -0.3,
            "vx300s_left_wrist_rotate": 0.0,
        }
        base = np.asarray(
            [home_by_name.get(name, 0.0) for name in order],
            dtype=np.float32,
        )
        physx_interface = get_physx_interface()
        states = []
        for state, targets in LEGAL_POSES_M.items():
            world.reset()
            world.pause()
            qpos = base.copy()
            qpos[left_index], qpos[right_index] = targets
            articulation.set_joint_positions(qpos)
            _sync_physx_transforms_to_usd(physx_interface)
            state_record = _state_geometry(stage)
            state_record.update(
                {
                    "state": state,
                    "target_m": list(targets),
                    "readback_m": [
                        float(articulation.get_joint_positions()[left_index]),
                        float(articulation.get_joint_positions()[right_index]),
                    ],
                }
            )
            states.append(state_record)

        finger_pair_relations = [
            pair["relation"]
            for state in states
            for pair in state["pairs"]
            if pair["scope"] == "FINGER_TO_FINGER_UNEXPECTED_IF_OVERLAP"
        ]
        gates = {
            "three_legal_states_audited": len(states) == 3,
            "all_colliders_convex_hull": all(
                collider["approximation"] == "convexHull"
                for state in states
                for collider in state["colliders"]
            ),
            "no_finger_to_finger_overlap": (
                len(finger_pair_relations) == 3
                and "OVERLAP" not in finger_pair_relations
            ),
            "source_stage_immutable": _sha256(stage_path) == stage_hash_before,
        }
        attachment_overlaps = [
            {
                "state": state["state"],
                "role_a": pair["role_a"],
                "role_b": pair["role_b"],
                "overlap_volume_m3": pair["overlap_volume_m3"],
                "signed_chebyshev_margin_m": (
                    pair["signed_chebyshev_margin_m"]
                ),
            }
            for state in states
            for pair in state["pairs"]
            if pair["scope"].endswith("REQUIRES_ASSEMBLY_SEMANTIC_REVIEW")
            and pair["relation"] == "OVERLAP"
        ]
        status = "PASS" if all(gates.values()) else "FAIL"
        if attachment_overlaps and status == "PASS":
            status = "PARTIAL"
        report = {
            "schema_version": 1,
            "status": status,
            "stage": {
                "absolute_path": str(stage_path),
                "sha256": stage_hash_before,
            },
            "runtime": {
                "isaac_sim": "5.1.0.0",
                "kit": "107.3.3",
                "physx": "107.3.26",
                "pose_method": (
                    "fresh World reset; full legal qpos injection; "
                    "PhysX transforms synced to unsaved USD session"
                ),
            },
            "method": {
                "cooked_geometry_model": (
                    "UsdPhysics approximation=convexHull represented by "
                    "scipy.spatial.ConvexHull of world-transformed mesh points"
                ),
                "pair_test": (
                    "normalized halfspace Chebyshev-margin linear program"
                ),
                "overlap_volume": (
                    "HalfspaceIntersection followed by ConvexHull volume"
                ),
                "known_unsafe_api_not_retried": (
                    "PhysXSceneQuery.overlap_shape_any"
                ),
            },
            "states": states,
            "gates": gates,
            "attachment_component_overlaps": attachment_overlaps,
            "acceptance_boundary": {
                "attachment_overlap_is_automatically_unexpected": False,
                "dynamic_collision_resolution": "NOT_RUN",
                "drive_tracking": "FAIL",
                "mimic_or_controller_coupling": "FAIL",
                "bottle_contact_grasp": "NOT_RUN",
                "task8": "NOT_RUN",
            },
        }
        OUTPUT_JSON.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        OUTPUT_MD.write_text(_render_markdown(report), encoding="utf-8")
        print(f"status={status}")
        print(f"report={OUTPUT_JSON.resolve()}")
        print(f"markdown={OUTPUT_MD.resolve()}")
        exit_code = 0 if status in {"PASS", "PARTIAL"} else 1
    except Exception:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
