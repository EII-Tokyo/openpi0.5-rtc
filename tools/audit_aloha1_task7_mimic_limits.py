#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Read-only Task 7 audit of negative-gearing finger mimic limits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback
from typing import Any
import xml.etree.ElementTree as ET

from tools.aloha1_mapping.task7_physicsrules_root_cause import mapped_physx_mimic_interval

ROOT = Path(__file__).resolve().parents[1]
FROZEN_STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0"
    / "aloha1_cad_derived_full_body_collider_gripper_decomposition_"
    "tabletop_zero_z_up_meters_diagnostic.usda"
)
FROZEN_SHA256 = (
    "327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9"
)
CANDIDATES = {
    "follower_left": (
        ROOT
        / "assets/Trossen/ALOHA1/1.0/diagnostics/"
        "cad_derived_task7_rule_candidates/1.0/Trossen/vx300s_left/1.0/"
        "vx300s_left.usda",
        "/vx300s_left",
    ),
    "follower_right": (
        ROOT
        / "assets/Trossen/ALOHA1/1.0/diagnostics/"
        "cad_derived_task7_rule_candidates/1.0/Trossen/vx300s_right/1.0/"
        "vx300s_right.usda",
        "/vx300s_right",
    ),
}
OUTPUT_JSON = ROOT / "reports/aloha1_mapping/aloha1_task7_mimic_limit_audit.json"
OUTPUT_MD = OUTPUT_JSON.with_suffix(".md")
RULE_SOURCE = (
    ROOT
    / ".venv_issac/lib/python3.11/site-packages/isaacsim/exts/"
    "isaacsim.asset.validation/isaacsim/asset/validation/joint_rules.py"
)
SCHEMA_SOURCE = (
    ROOT
    / ".venv_issac/lib/python3.11/site-packages/isaacsim/extscache/"
    "omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353/plugins/"
    "PhysxSchema/resources/generatedSchema.usda"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _urdf_mimic(robot: str) -> dict[str, Any]:
    urdf = (ROOT / "generated/urdf" / f"{robot}.urdf").resolve(strict=True)
    root = ET.parse(urdf).getroot()
    joints = {str(joint.get("name")): joint for joint in root.findall("joint")}
    target = joints["right_finger"]
    reference = joints["left_finger"]
    mimic = target.find("mimic")
    target_limit = target.find("limit")
    reference_limit = reference.find("limit")
    if mimic is None or target_limit is None or reference_limit is None:
        raise RuntimeError(f"incomplete URDF mimic definition: {urdf}")
    return {
        "absolute_path": str(urdf),
        "sha256": _sha256(urdf),
        "mimic_joint": str(target.get("name")),
        "reference_joint": str(mimic.get("joint")),
        "gearing": float(mimic.get("multiplier", "1")),
        "offset": float(mimic.get("offset", "0")),
        "mimic_limits": {
            "lower": float(target_limit.get("lower")),
            "upper": float(target_limit.get("upper")),
        },
        "reference_limits": {
            "lower": float(reference_limit.get("lower")),
            "upper": float(reference_limit.get("upper")),
        },
    }


def _official_issue(side: str) -> dict[str, Any]:
    path = (
        ROOT
        / "reports/aloha1_mapping"
        / f"aloha1_cad_derived_task7_candidate_{side}_physics.json"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    issues = [item for item in report["issues"] if item.get("rule") == "MimicAPICheck"]
    if len(issues) != 1:
        raise RuntimeError(f"expected one MimicAPICheck finding in {path}")
    return dict(issues[0])


def _mimic_axis(prim: Any) -> str:
    axes = [
        schema.split(":", maxsplit=1)[1]
        for schema in prim.GetAppliedSchemas()
        if schema.startswith("PhysxMimicJointAPI:")
    ]
    if len(axes) != 1:
        raise RuntimeError(f"expected one mimic API on {prim.GetPath()}: {axes}")
    return axes[0]


def _record(stage: Any, root_path: str, robot: str) -> dict[str, Any]:
    from pxr import PhysxSchema
    from pxr import UsdPhysics

    mimic_path = f"{root_path}/joints/right_finger"
    mimic_prim = stage.GetPrimAtPath(mimic_path)
    if not mimic_prim.IsValid():
        raise RuntimeError(f"missing mimic joint {mimic_path}")
    axis = _mimic_axis(mimic_prim)
    api = PhysxSchema.PhysxMimicJointAPI(mimic_prim, axis)
    targets = api.GetReferenceJointRel().GetTargets()
    if len(targets) != 1:
        raise RuntimeError(f"invalid reference relationship on {mimic_path}")
    reference_path = str(targets[0])
    reference_prim = stage.GetPrimAtPath(reference_path)
    mimic_joint = UsdPhysics.PrismaticJoint(mimic_prim)
    reference_joint = UsdPhysics.PrismaticJoint(reference_prim)
    gearing = float(api.GetGearingAttr().Get())
    mimic_lower = float(mimic_joint.GetLowerLimitAttr().Get())
    mimic_upper = float(mimic_joint.GetUpperLimitAttr().Get())
    reference_lower = float(reference_joint.GetLowerLimitAttr().Get())
    reference_upper = float(reference_joint.GetUpperLimitAttr().Get())
    offset = float(api.GetOffsetAttr().Get())
    mapped = mapped_physx_mimic_interval(
        reference_lower=reference_lower,
        reference_upper=reference_upper,
        gearing=gearing,
        offset=offset,
    )
    if gearing < 0.0:
        validator_predicates = {
            "branch": "negative",
            "reference_lower_times_gearing_gt_mimic_lower": (
                reference_lower * gearing > mimic_lower
            ),
            "mimic_upper_gt_reference_upper_times_gearing": (
                mimic_upper > reference_upper * gearing
            ),
        }
    else:
        validator_predicates = {
            "branch": "nonnegative",
            "reference_lower_times_gearing_lt_mimic_upper": (
                reference_lower * gearing < mimic_upper
            ),
            "mimic_lower_lt_reference_upper_times_gearing": (
                mimic_lower < reference_upper * gearing
            ),
        }
    validator_pass = all(
        value for key, value in validator_predicates.items() if key != "branch"
    )
    authored_order_valid = mimic_lower <= mimic_upper
    sorted_authored = sorted((mimic_lower, mimic_upper))
    mapped_enclosed_after_sort = (
        sorted_authored[0] <= mapped[0] and mapped[1] <= sorted_authored[1]
    )
    effective_multiplier = -gearing
    effective_offset = -offset
    urdf = _urdf_mimic(robot)
    physx_relation_matches_urdf = (
        abs(effective_multiplier - urdf["gearing"]) < 1.0e-8
        and abs(effective_offset - urdf["offset"]) < 1.0e-8
    )
    if (
        authored_order_valid
        and mapped_enclosed_after_sort
        and physx_relation_matches_urdf
        and not validator_pass
    ):
        classification = "VALIDATOR_1_1_0_FORMULA_MISMATCH"
    else:
        classification = "INCONCLUSIVE"
    return {
        "follower": robot,
        "mimic_joint": mimic_path,
        "reference_joint": reference_path,
        "mimic_api_instance": axis,
        "gearing": gearing,
        "offset": offset,
        "physx_equation": (
            "jointPosition + gearing * referenceJointPosition + offset = 0"
        ),
        "effective_reference_multiplier": effective_multiplier,
        "effective_reference_offset": effective_offset,
        "physx_relation_matches_urdf": physx_relation_matches_urdf,
        "natural_frequency": float(
            mimic_prim.GetAttribute(
                f"physxMimicJoint:{axis}:naturalFrequency"
            ).Get()
        ),
        "damping_ratio": float(
            mimic_prim.GetAttribute(
                f"physxMimicJoint:{axis}:dampingRatio"
            ).Get()
        ),
        "authored_mimic_limits": {"lower": mimic_lower, "upper": mimic_upper},
        "authored_reference_limits": {
            "lower": reference_lower,
            "upper": reference_upper,
        },
        "mapped_reference_interval": list(mapped),
        "authored_limit_order_valid": authored_order_valid,
        "sorted_authored_interval": sorted_authored,
        "mapped_interval_enclosed_after_sort": mapped_enclosed_after_sort,
        "validator_predicates": validator_predicates,
        "validator_predicates_pass": validator_pass,
        "classification": classification,
        "urdf_source": urdf,
        "usd_vs_urdf": {
            "reference_limits_match": (
                abs(reference_lower - urdf["reference_limits"]["lower"]) < 1.0e-8
                and abs(reference_upper - urdf["reference_limits"]["upper"]) < 1.0e-8
            ),
            "mimic_limits_match": (
                abs(mimic_lower - urdf["mimic_limits"]["lower"]) < 1.0e-8
                and abs(mimic_upper - urdf["mimic_limits"]["upper"]) < 1.0e-8
            ),
        },
        "official_finding": _official_issue(robot.removeprefix("follower_")),
        "prim_stack": [
            {
                "layer": str(spec.layer.identifier),
                "path": str(spec.path),
                "specifier": str(spec.specifier),
            }
            for spec in mimic_prim.GetPrimStack()
        ],
        "usd_modified": False,
    }


def build_report() -> dict[str, Any]:
    from pxr import Usd

    frozen = FROZEN_STAGE.resolve(strict=True)
    frozen_before = _sha256(frozen)
    if frozen_before != FROZEN_SHA256:
        raise RuntimeError("frozen Stage hash mismatch")
    findings = []
    candidates = {}
    for robot, (candidate_path, root_path) in CANDIDATES.items():
        candidate = candidate_path.resolve(strict=True)
        before = _sha256(candidate)
        stage = Usd.Stage.Open(str(candidate), Usd.Stage.LoadAll)
        if stage is None:
            raise RuntimeError(f"cannot open {candidate}")
        findings.append(_record(stage, root_path, robot))
        after = _sha256(candidate)
        candidates[robot] = {
            "absolute_path": str(candidate),
            "sha256_before": before,
            "sha256_after": after,
            "modified": before != after,
        }
    frozen_after = _sha256(frozen)
    classes = sorted({item["classification"] for item in findings})
    return {
        "schema_version": 1,
        "status": (
            "PASS"
            if classes == ["VALIDATOR_1_1_0_FORMULA_MISMATCH"]
            else "PARTIAL"
        ),
        "finding_count": len(findings),
        "classification": classes[0] if len(classes) == 1 else "INCONCLUSIVE",
        "stage": {
            "absolute_path": str(frozen),
            "sha256_before": frozen_before,
            "sha256_after": frozen_after,
            "modified": frozen_before != frozen_after,
        },
        "candidate_stages": candidates,
        "findings": findings,
        "local_rule_source": {
            "absolute_path": str(RULE_SOURCE.resolve(strict=True)),
            "sha256": _sha256(RULE_SOURCE.resolve(strict=True)),
            "class": "MimicAPICheck",
        },
        "local_schema_source": {
            "absolute_path": str(SCHEMA_SOURCE.resolve(strict=True)),
            "sha256": _sha256(SCHEMA_SOURCE.resolve(strict=True)),
            "class": "PhysxMimicJointAPI",
            "equation": (
                "jointPosition + gearing * referenceJointPosition + offset = 0"
            ),
        },
        "candidate_authoring_allowed": False,
        "next_step": "DO_NOT_CHANGE_VALID_MIMIC_TO_SATISFY_CONFLICTING_RULE",
        "final_or_default_asset_modified": False,
        "task8": "NOT_RUN",
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 Task 7 mimic-limit audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Classification: `{report['classification']}`",
        "- Candidate authoring: `NOT_RUN`",
        "- Task 8: `NOT_RUN`",
        "",
        "| Follower | USD mimic limits | Reference limits | Mapped interval | Ordered |",
        "|---|---|---|---|---:|",
    ]
    lines.extend(
        "| {follower} | [{lo:.9g}, {hi:.9g}] | [{rlo:.9g}, {rhi:.9g}] | "
        "[{mlo:.9g}, {mhi:.9g}] | {ordered} |".format(
            follower=item["follower"],
            lo=item["authored_mimic_limits"]["lower"],
            hi=item["authored_mimic_limits"]["upper"],
            rlo=item["authored_reference_limits"]["lower"],
            rhi=item["authored_reference_limits"]["upper"],
            mlo=item["mapped_reference_interval"][0],
            mhi=item["mapped_reference_interval"][1],
            ordered=item["authored_limit_order_valid"],
        )
        for item in report["findings"]
    )
    lines.extend(
        [
            "",
            "The installed 107.3 schema defines `q_right + gearing*q_left + "
            "offset = 0`. Therefore PhysX gearing `+1` is equivalent to the URDF "
            "multiplier `-1`. The installed validator 1.1.0 limit test treats gearing "
            "as a direct multiplier and rejects the otherwise valid negative mapped "
            "interval. No mimic, drive, limit, or USD property was changed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    report = build_report()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUTPUT_MD.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(OUTPUT_JSON.resolve())}))
    return 0 if report["status"] == "PASS" else 1


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "create_new_stage": False,
            "disable_viewport_updates": True,
        }
    )
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
