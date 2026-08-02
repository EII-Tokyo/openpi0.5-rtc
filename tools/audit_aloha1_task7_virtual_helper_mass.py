#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Read the authored mass properties of Task 7 empty fixed helper bodies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URDF = ROOT / "generated/urdf/follower_left.urdf"
FROZEN_STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/"
    "aloha1_cad_derived_full_body_collider_gripper_decomposition_"
    "tabletop_zero_z_up_meters_diagnostic.usda"
)
FROZEN_SHA256 = "327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9"
SOURCE_ROOT = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "task7_physicsrules_root_cause_candidates/1.0/"
    "baseline_gripper_fixed_group_split"
)
OUTPUT = ROOT / "reports/aloha1_mapping/aloha1_task7_virtual_helper_mass_audit.json"
OUTPUT_MD = OUTPUT.with_suffix(".md")
HELPERS = (
    ("ee_arm_link", "gripper_link"),
    ("fingers_link", "gripper_bar_link"),
    ("ee_gripper_link", "gripper_bar_link"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix(matrix: Any) -> list[list[float]]:
    return np.asarray(matrix, dtype=np.float64).tolist()


def _quat(quaternion: Any) -> list[float]:
    return [
        float(quaternion.GetReal()),
        *[float(value) for value in quaternion.GetImaginary()],
    ]


def _finite_or_string(value: float) -> float | str:
    return value if np.isfinite(value) else ("-inf" if value < 0.0 else "inf")


def main() -> int:
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    frozen = FROZEN_STAGE.resolve(strict=True)
    if _sha256(frozen) != FROZEN_SHA256:
        raise RuntimeError("frozen Stage hash drift")
    followers = []
    for follower in ("follower_left", "follower_right"):
        root_name = "vx300s_left" if follower == "follower_left" else "vx300s_right"
        source = (
            SOURCE_ROOT
            / follower
            / f"{follower}_baseline_gripper_fixed_group_split.usda"
        ).resolve(strict=True)
        source_before = _sha256(source)
        stage = Usd.Stage.Open(str(source), Usd.Stage.LoadAll)
        if stage is None:
            raise RuntimeError(f"cannot open {source}")
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        helper_bodies = []
        for suffix, transfer_target_suffix in HELPERS:
            path = f"/{root_name}/{follower}_{suffix}"
            prim = stage.GetPrimAtPath(path)
            mass_api = UsdPhysics.MassAPI(prim)
            mass = float(mass_api.GetMassAttr().Get())
            center = mass_api.GetCenterOfMassAttr().Get()
            center_authored = mass_api.GetCenterOfMassAttr().HasAuthoredValueOpinion()
            inertia = mass_api.GetDiagonalInertiaAttr().Get()
            axes = mass_api.GetPrincipalAxesAttr().Get()
            helper_bodies.append(
                {
                    "prim_path": path,
                    "transfer_target": (
                        f"/{root_name}/{follower}_{transfer_target_suffix}"
                    ),
                    "mass_kg": mass,
                    "center_of_mass_authored": center_authored,
                    "center_of_mass_raw_readback": [
                        _finite_or_string(float(value)) for value in center
                    ],
                    "center_of_mass_effective_local_m": [0.0, 0.0, 0.0],
                    "center_of_mass_effective_source": (
                        "AUTHORED_USD"
                        if center_authored
                        else "URDF_INERTIAL_ORIGIN_DEFAULT_IDENTITY"
                    ),
                    "diagonal_inertia_kg_m2": [float(value) for value in inertia],
                    "principal_axes_wxyz": _quat(axes),
                    "world_matrix": _matrix(cache.GetLocalToWorldTransform(prim)),
                    "authored_properties": {
                        "mass": mass_api.GetMassAttr().HasAuthoredValueOpinion(),
                        "center_of_mass": mass_api.GetCenterOfMassAttr().HasAuthoredValueOpinion(),
                        "diagonal_inertia": mass_api.GetDiagonalInertiaAttr().HasAuthoredValueOpinion(),
                        "principal_axes": mass_api.GetPrincipalAxesAttr().HasAuthoredValueOpinion(),
                    },
                }
            )
        source_after = _sha256(source)
        followers.append(
            {
                "follower": follower,
                "source": {
                    "absolute_path": str(source),
                    "sha256_before": source_before,
                    "sha256_after": source_after,
                    "modified": source_before != source_after,
                },
                "helper_bodies": helper_bodies,
                "total_helper_mass_kg": sum(
                    body["mass_kg"] for body in helper_bodies
                ),
            }
        )
    report = {
        "schema_version": 1,
        "status": (
            "PASS"
            if all(
                not item["source"]["modified"]
                and all(
                    body["mass_kg"] > 0.0
                    and all(np.isfinite(body["diagonal_inertia_kg_m2"]))
                    for body in item["helper_bodies"]
                )
                for item in followers
            )
            else "FAIL"
        ),
        "scope": "READ_ONLY_AUTHORED_HELPER_MASS_AUDIT",
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
        },
        "frozen_stage_sha256": FROZEN_SHA256,
        "source_urdf": {
            "absolute_path": str(SOURCE_URDF.resolve(strict=True)),
            "sha256": _sha256(SOURCE_URDF.resolve(strict=True)),
            "helper_mass_kg": 0.001,
            "helper_diagonal_inertia_kg_m2": [0.0001, 0.0001, 0.0001],
            "helper_inertial_origin_authored": False,
        },
        "followers": followers,
        "physical_calibration_status": (
            "SOURCE_AUTHORED_PLACEHOLDER_NOT_PHYSICALLY_VERIFIED"
        ),
        "uncompensated_collapse_allowed": False,
        "reason": (
            "Each helper carries authored positive mass; removing its rigid-body "
            "semantics without exact mass/inertia aggregation changes dynamics."
        ),
        "final_or_default_asset_modified": False,
        "task8": "NOT_RUN",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# ALOHA1 Task 7 virtual-helper mass audit",
        "",
        f"- Status: `{report['status']}`",
        "- Uncompensated collapse allowed: `false`",
        "- Final/default assets modified: `false`",
        "- Task 8: `NOT_RUN`",
        "- Physical calibration: `SOURCE_AUTHORED_PLACEHOLDER_NOT_PHYSICALLY_VERIFIED`",
        "",
        "| Follower | Helper | Transfer target | Mass (kg) | Diagonal inertia (kg m^2) |",
        "|---|---|---|---:|---|",
    ]
    for follower in followers:
        lines.extend(

                f"| `{follower['follower']}` | `{body['prim_path']}` | "
                f"`{body['transfer_target']}` | {body['mass_kg']:.9g} | "
                f"`{body['diagonal_inertia_kg_m2']}` |"
                for body in follower["helper_bodies"]

        )
    lines.extend(
        [
            "",
            "The first topology-collapse candidate is diagnostic only. Its small runtime "
            "trace change is consistent with removing these authored masses. A promotable "
            "candidate must preserve total mass, COM and inertia through an exact rigid-body "
            "aggregation, then rerun validator and runtime regressions.",
            "",
            "The USD API returns `[-inf, -inf, -inf]` for an unauthored center-of-mass "
            "attribute. This is retained as raw readback, not interpreted as a physical "
            "COM. The generated URDF omits the inertial origin on these helper links, so "
            "the effective local origin is recorded as `[0, 0, 0]` under URDF semantics.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "followers": len(followers)}))
    return 0 if report["status"] == "PASS" else 2


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "create_new_stage": False})
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
