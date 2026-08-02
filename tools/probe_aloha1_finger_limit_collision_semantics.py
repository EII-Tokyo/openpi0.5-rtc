#!/usr/bin/env python3
"""Probe source/composed/live ALOHA finger limits in fresh Isaac 5.1 processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ISAAC_PYTHON = ROOT / ".venv_issac/bin/python"
SOURCE_URDF = ROOT / "generated/urdf/follower_left.urdf"
FROZEN_STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_derived_full_body_colliders/1.0/"
    "aloha1_cad_derived_full_body_collider_gripper_decomposition_"
    "tabletop_zero_z_up_meters_diagnostic.usda"
)
FROZEN_STAGE_SHA256 = (
    "327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9"
)
ARTICULATION_PRIM = "/World/follower_left/vx300s_left/root_joint"
ROBOT_ROOT = "/World/follower_left/vx300s_left"
CANDIDATE_ROOT = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "finger_limit_pair_collision_candidate/1.0"
)
CANDIDATE_LAYER = CANDIDATE_ROOT / "configuration/finger_source_limits.usda"
CANDIDATE_STAGE = CANDIDATE_ROOT / "aloha1_finger_source_limit_candidate.usda"
OUTPUT_JSON = (
    ROOT
    / "reports/aloha1_mapping/aloha1_finger_limit_collision_semantics.json"
)
OUTPUT_MD = OUTPUT_JSON.with_suffix(".md")
ARTIFACT_ROOT = (
    ROOT
    / ".codex/artifacts/20260802-aloha1-five-pose-finger-safety/"
    "task4/runtime"
)
FINGER_NAMES = ("left_finger", "right_finger")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_signature(payload: dict[str, Any]) -> str:
    excluded = {"output", "process_id", "runtime_s"}

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): clean(item)
                for key, item in sorted(value.items())
                if str(key) not in excluded
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, float):
            return round(value, 10)
        return value

    encoded = json.dumps(
        clean(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_source_urdf(path: Path = SOURCE_URDF) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    root = ET.parse(resolved).getroot()
    joints = {str(joint.get("name")): joint for joint in root.findall("joint")}
    limits: dict[str, dict[str, float]] = {}
    for name in FINGER_NAMES:
        limit = joints[name].find("limit")
        if limit is None:
            raise RuntimeError(f"missing source limit for {name}")
        limits[name] = {
            "lower": float(limit.get("lower", "nan")),
            "upper": float(limit.get("upper", "nan")),
        }
    mimic = joints["right_finger"].find("mimic")
    if mimic is None:
        raise RuntimeError("missing right_finger mimic definition")
    return {
        "absolute_path": str(resolved),
        "sha256": _sha256(resolved),
        "limits": limits,
        "mimic": {
            "joint": str(mimic.get("joint")),
            "multiplier": float(mimic.get("multiplier", "1")),
            "offset": float(mimic.get("offset", "0")),
        },
    }


def _limits_match(
    first: dict[str, dict[str, float]],
    second: dict[str, dict[str, float]],
    *,
    tolerance: float = 1.0e-7,
) -> bool:
    return all(
        abs(float(first[name][bound]) - float(second[name][bound]))
        <= tolerance
        for name in FINGER_NAMES
        for bound in ("lower", "upper")
    )


def validate_session_layer_probe(
    *,
    record: dict[str, Any],
    source_limits: dict[str, dict[str, float]],
    expected_stage_sha256: str,
    expected_layer_path: str,
) -> dict[str, Any]:
    """Evaluate an isolated session-layer limit readback without promotion."""

    stage = record.get("stage", {})
    application = record.get("session_sublayer_application", {})
    expected_path = str(Path(expected_layer_path).resolve())
    gates = {
        "runtime_pass": record.get("status") == "PASS",
        "source_stage_hash_unchanged": (
            stage.get("sha256_before") == expected_stage_sha256
            and stage.get("sha256_after") == expected_stage_sha256
        ),
        "root_sublayers_unchanged": (
            stage.get("root_sublayers_before")
            == stage.get("root_sublayers_after")
        ),
        "session_layer_inserted_exactly_once": (
            application.get("status") == "PASS"
            and application.get("inserted_paths") == [expected_path]
            and application.get("after") == [expected_path]
        ),
        "root_layer_not_saved": application.get("root_layer_saved") is False,
        "authored_limits_match_source": _limits_match(
            source_limits,
            record.get("composed_usd", {}).get("authored_limits", {}),
        ),
        "runtime_limits_match_source": _limits_match(
            source_limits,
            record.get("runtime_readback", {}).get("dof_limits", {}),
        ),
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "classification": "SESSION_ONLY_SOURCE_DERIVED_LIMIT_READBACK",
        "gates": gates,
        "source_stage_sha256": expected_stage_sha256,
        "session_layer_absolute_path": expected_path,
        "final_or_default_asset_modified": False,
        "task8": "NOT_RUN",
    }


def aggregate_report(
    *,
    source_urdf: dict[str, Any],
    runtime_records: list[dict[str, Any]],
    candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    if len(runtime_records) != 2:
        raise ValueError("exactly two fresh runtime records are required")
    signatures = [str(item["deterministic_signature"]) for item in runtime_records]
    if len(set(signatures)) != 1:
        raise ValueError("fresh runtime records are not deterministic")
    first = runtime_records[0]
    if any(item.get("status") != "PASS" for item in runtime_records):
        limit_status = "INCONCLUSIVE"
    else:
        source_limits = source_urdf["limits"]
        authored_match = _limits_match(
            source_limits,
            first["composed_usd"]["authored_limits"],
        )
        runtime_match = _limits_match(
            source_limits,
            first["runtime_readback"]["dof_limits"],
        )
        mimic = first["composed_usd"]["mimic_api"]
        source_mimic = source_urdf["mimic"]
        mimic_match = (
            abs(
                float(mimic["effective_multiplier"])
                - float(source_mimic["multiplier"])
            )
            <= 1.0e-7
            and abs(
                float(mimic["effective_offset"])
                - float(source_mimic["offset"])
            )
            <= 1.0e-7
        )
        if authored_match and runtime_match and mimic_match:
            limit_status = "VERIFIED_EQUIVALENT"
        elif mimic_match and (not authored_match or not runtime_match):
            limit_status = "VERIFIED_USD_LIMIT_DEFECT"
        else:
            limit_status = "INCONCLUSIVE"
    support_records = [
        item.get("pair_collision_support_probe", {}).get("status")
        for item in runtime_records
    ]
    pair_status = (
        str(support_records[0])
        if len(set(support_records)) == 1
        and support_records[0]
        in {
            "SUPPORTED_LOCAL_5_1",
            "NOT_SUPPORTED_LOCAL_5_1",
        }
        else "INCONCLUSIVE"
    )
    return {
        "schema_version": 1,
        "status": "PASS" if limit_status != "INCONCLUSIVE" else "PARTIAL",
        "source_urdf": source_urdf,
        "composed_usd": first["composed_usd"],
        "runtime_readback": first["runtime_readback"],
        "filtered_pair_inventory": first["filtered_pair_inventory"],
        "limit_semantics_status": limit_status,
        "pair_collision_support_status": pair_status,
        "candidate_created": candidate is not None,
        "candidate": candidate,
        "fresh_process_determinism": {
            "status": "PASS",
            "run_count": 2,
            "signatures": signatures,
        },
        "runtime_records": runtime_records,
        "final_or_default_asset_modified": False,
        "task8": "NOT_RUN",
    }


def _write_candidate(source: dict[str, Any]) -> dict[str, Any]:
    CANDIDATE_LAYER.parent.mkdir(parents=True, exist_ok=True)
    limits = source["limits"]
    layer_text = f"""#usda 1.0
(
    customLayerData = {{
        string classification = "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
        string sourceUrdfSha256 = "{source['sha256']}"
    }}
)

over "World" {{
    over "follower_left" {{
        over "vx300s_left" {{
            over "joints" {{
                over "right_finger" {{
                    float physics:lowerLimit = {limits['right_finger']['lower']}
                    float physics:upperLimit = {limits['right_finger']['upper']}
                }}
            }}
        }}
    }}
}}
"""
    CANDIDATE_LAYER.write_text(layer_text, encoding="utf-8")
    frozen_relative = os.path.relpath(FROZEN_STAGE, CANDIDATE_ROOT)
    layer_relative = os.path.relpath(CANDIDATE_LAYER, CANDIDATE_ROOT)
    stage_text = f"""#usda 1.0
(
    defaultPrim = "World"
    subLayers = [
        @{layer_relative}@,
        @{frozen_relative}@
    ]
    customLayerData = {{
        string classification = "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
        string promotionStatus = "NOT_PROMOTED"
    }}
)
"""
    CANDIDATE_STAGE.write_text(stage_text, encoding="utf-8")
    return {
        "status": "CREATED_NOT_PROMOTED",
        "root_stage": {
            "absolute_path": str(CANDIDATE_STAGE.resolve()),
            "sha256": _sha256(CANDIDATE_STAGE),
        },
        "configuration_layer": {
            "absolute_path": str(CANDIDATE_LAYER.resolve()),
            "sha256": _sha256(CANDIDATE_LAYER),
        },
        "changed_fields": [
            f"{ROBOT_ROOT}/joints/right_finger.physics:lowerLimit",
            f"{ROBOT_ROOT}/joints/right_finger.physics:upperLimit",
        ],
        "source_limits_m": source["limits"],
        "pair_collision_authored": False,
        "final_or_default_modified": False,
    }


def _run_runtime(
    stage_path: Path,
    output: Path,
    *,
    session_layer_path: Path | None = None,
    session_layer_sha256: str | None = None,
) -> int:
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "width": 640,
            "height": 480,
            "/app/useFabricSceneDelegate": False,
        }
    )
    exit_code = 0
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.stage import open_stage
        from pxr import PhysxSchema
        from pxr import UsdPhysics

        from tools.aloha1_mapping.grasp_20cm_runtime import apply_verified_session_sublayers

        stage_sha256_before = _sha256(stage_path)
        if not open_stage(str(stage_path.resolve(strict=True))):
            raise RuntimeError(f"cannot open Stage: {stage_path}")
        stage = get_current_stage()
        root_sublayers_before = list(stage.GetRootLayer().subLayerPaths)
        session_application = {
            "status": "NOT_APPLIED",
            "before": list(stage.GetSessionLayer().subLayerPaths),
            "after": list(stage.GetSessionLayer().subLayerPaths),
            "inserted_paths": [],
            "already_present_paths": [],
            "root_layer_saved": False,
        }
        if session_layer_path is not None:
            resolved_layer = session_layer_path.resolve(strict=True)
            actual_layer_sha256 = _sha256(resolved_layer)
            if session_layer_sha256 is None:
                raise RuntimeError(
                    "session layer requires an expected SHA-256"
                )
            if actual_layer_sha256 != session_layer_sha256:
                raise RuntimeError(
                    "session layer SHA-256 mismatch: "
                    f"{actual_layer_sha256} != {session_layer_sha256}"
                )
            session_application = apply_verified_session_sublayers(
                stage=stage,
                records=[
                    {
                        "absolute_path": str(resolved_layer),
                        "sha256": actual_layer_sha256,
                    }
                ],
            )
        World.clear_instance()
        world = World(
            stage_units_in_meters=1.0,
            backend="numpy",
            device="cpu",
            physics_dt=1.0 / 60.0,
            rendering_dt=1.0 / 60.0,
        )
        articulation = SingleArticulation(
            prim_path=ARTICULATION_PRIM,
            name="aloha1_finger_limit_semantics_probe",
            reset_xform_properties=False,
        )
        world.scene.add(articulation)
        world.reset()
        dof_order = list(articulation.dof_names)
        limits = articulation._articulation_view.get_dof_limits()  # noqa: SLF001
        if getattr(limits, "ndim", 0) == 3:
            limits = limits[0]
        dof_limits = {
            name: {
                "lower": float(limits[dof_order.index(name)][0]),
                "upper": float(limits[dof_order.index(name)][1]),
            }
            for name in FINGER_NAMES
        }
        authored_limits: dict[str, dict[str, float]] = {}
        for name in FINGER_NAMES:
            joint = UsdPhysics.PrismaticJoint(
                stage.GetPrimAtPath(f"{ROBOT_ROOT}/joints/{name}")
            )
            authored_limits[name] = {
                "lower": float(joint.GetLowerLimitAttr().Get()),
                "upper": float(joint.GetUpperLimitAttr().Get()),
            }
        mimic_prim = stage.GetPrimAtPath(
            f"{ROBOT_ROOT}/joints/right_finger"
        )
        mimic_axes = [
            schema.split(":", maxsplit=1)[1]
            for schema in mimic_prim.GetAppliedSchemas()
            if schema.startswith("PhysxMimicJointAPI:")
        ]
        if len(mimic_axes) != 1:
            raise RuntimeError(f"expected one mimic API: {mimic_axes}")
        mimic_api = PhysxSchema.PhysxMimicJointAPI(
            mimic_prim,
            mimic_axes[0],
        )
        gearing = float(mimic_api.GetGearingAttr().Get())
        offset = float(mimic_api.GetOffsetAttr().Get())
        filtered_pairs: list[dict[str, Any]] = []
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.FilteredPairsAPI):
                continue
            targets = [
                str(path)
                for path in UsdPhysics.FilteredPairsAPI(prim)
                .GetFilteredPairsRel()
                .GetTargets()
            ]
            if targets:
                filtered_pairs.append(
                    {"prim_path": str(prim.GetPath()), "targets": targets}
                )
        record: dict[str, Any] = {
            "status": "PASS",
            "stage": {
                "absolute_path": str(stage_path.resolve()),
                "sha256": stage_sha256_before,
                "sha256_before": stage_sha256_before,
                "sha256_after": _sha256(stage_path),
                "default_prim": str(stage.GetDefaultPrim().GetPath()),
                "sublayers": list(stage.GetRootLayer().subLayerPaths),
                "root_sublayers_before": root_sublayers_before,
                "root_sublayers_after": list(
                    stage.GetRootLayer().subLayerPaths
                ),
            },
            "session_sublayer_application": session_application,
            "runtime": {
                "isaac_sim": "5.1.0.0",
                "kit": "107.3.3",
                "physx": "107.3.26",
            },
            "runtime_readback": {
                "dof_order": dof_order,
                "dof_limits": dof_limits,
                "self_collision": bool(
                    articulation.get_enabled_self_collisions()
                ),
            },
            "composed_usd": {
                "authored_limits": authored_limits,
                "mimic_api": {
                    "instance": mimic_axes[0],
                    "gearing": gearing,
                    "offset": offset,
                    "effective_multiplier": -gearing,
                    "effective_offset": -offset,
                    "equation": (
                        "jointPosition + gearing * referenceJointPosition "
                        "+ offset = 0"
                    ),
                },
            },
            "filtered_pair_inventory": filtered_pairs,
            "pair_collision_support_probe": {
                "status": "INCONCLUSIVE",
                "global_self_collision_readback": bool(
                    articulation.get_enabled_self_collisions()
                ),
                "filtered_pairs_semantics": "DISABLE_SELECTED_PAIR",
                "direct_positive_pair_enable_api": False,
                "reason": (
                    "pair-only collision would require articulation-wide self "
                    "collision plus exhaustive filters; not mutated in this "
                    "read-only probe"
                ),
            },
        }
        record["deterministic_signature"] = _canonical_signature(record)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        exit_code = 1
        error = {
            "status": "FAIL",
            "stage": str(stage_path.resolve()),
            "traceback": traceback.format_exc(),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(error, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(error["traceback"], file=sys.stderr, flush=True)
    finally:
        app.close()
    return exit_code


def _fresh_runs(
    *,
    stage_path: Path,
    label: str,
    artifact_root: Path,
) -> list[dict[str, Any]]:
    records = []
    for index in (1, 2):
        output = artifact_root / f"{label}_run{index}.json"
        log = artifact_root / f"{label}_run{index}.log"
        command = [
            str(ISAAC_PYTHON),
            str(Path(__file__).resolve()),
            "--runtime-output",
            str(output),
            "--stage",
            str(stage_path),
        ]
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=900,
            )
        if result.returncode != 0 or not output.is_file():
            raise RuntimeError(
                f"fresh Isaac probe failed ({label} run {index}); log={log}"
            )
        record = json.loads(output.read_text(encoding="utf-8"))
        if record.get("status") != "PASS":
            raise RuntimeError(
                f"fresh Isaac probe reported failure ({label} run {index}); "
                f"output={output}; log={log}"
            )
        record["process"] = {
            "command": command,
            "exit_code": result.returncode,
            "log_absolute_path": str(log.resolve()),
            "output_absolute_path": str(output.resolve()),
        }
        records.append(record)
    return records


def _render_markdown(report: dict[str, Any]) -> str:
    source = report["source_urdf"]["limits"]
    live = report["runtime_readback"]["dof_limits"]
    candidate = report.get("candidate")
    return "\n".join(
        [
            "# ALOHA1 finger limit and pair-collision semantics",
            "",
            f"- Status: `{report['status']}`",
            f"- Limit semantics: `{report['limit_semantics_status']}`",
            f"- Pair-collision support: `{report['pair_collision_support_status']}`",
            f"- Candidate created: `{report['candidate_created']}`",
            f"- Task 8: `{report['task8']}`",
            "",
            "## Source versus live limits",
            "",
            f"- URDF left: `{source['left_finger']}`",
            f"- Live left: `{live['left_finger']}`",
            f"- URDF right: `{source['right_finger']}`",
            f"- Live right: `{live['right_finger']}`",
            "",
            "The source URDF remains the admissible runtime interval. A wider "
            "imported/composed right-finger interval is an asset defect, not a "
            "license to command beyond the source limit.",
            "",
            "## Pair-collision boundary",
            "",
            "Local 5.1 exposes articulation-wide self-collision and a filtered-"
            "pairs API that disables selected pairs. This read-only probe did not "
            "enable all internal self-collisions merely to obtain one positive "
            "finger pair, so no pair-collision candidate is claimed.",
            "",
            "## Isolated candidate",
            "",
            (
                f"`{candidate['root_stage']['absolute_path']}` is NOT_PROMOTED."
                if candidate
                else "No candidate was created."
            ),
            "",
        ]
    )


def build() -> dict[str, Any]:
    if _sha256(FROZEN_STAGE) != FROZEN_STAGE_SHA256:
        raise RuntimeError("frozen Stage SHA-256 mismatch")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    source = parse_source_urdf()
    baseline = _fresh_runs(
        stage_path=FROZEN_STAGE,
        label="baseline",
        artifact_root=ARTIFACT_ROOT,
    )
    initial = aggregate_report(
        source_urdf=source,
        runtime_records=baseline,
        candidate=None,
    )
    candidate = None
    if initial["limit_semantics_status"] == "VERIFIED_USD_LIMIT_DEFECT":
        candidate = _write_candidate(source)
        candidate_records = _fresh_runs(
            stage_path=CANDIDATE_STAGE,
            label="candidate",
            artifact_root=ARTIFACT_ROOT,
        )
        candidate["fresh_process_records"] = candidate_records
        invariant_fields = {
            "dof_order": (
                baseline[0]["runtime_readback"]["dof_order"]
                == candidate_records[0]["runtime_readback"]["dof_order"]
            ),
            "self_collision": (
                baseline[0]["runtime_readback"]["self_collision"]
                == candidate_records[0]["runtime_readback"]["self_collision"]
            ),
            "mimic_api": (
                baseline[0]["composed_usd"]["mimic_api"]
                == candidate_records[0]["composed_usd"]["mimic_api"]
            ),
            "filtered_pair_inventory": (
                baseline[0]["filtered_pair_inventory"]
                == candidate_records[0]["filtered_pair_inventory"]
            ),
        }
        candidate["non_limit_invariants"] = {
            "status": (
                "PASS" if all(invariant_fields.values()) else "FAIL"
            ),
            "fields": invariant_fields,
        }
        candidate["verification_status"] = (
            "PASS"
            if all(
                _limits_match(
                    source["limits"],
                    record["runtime_readback"]["dof_limits"],
                )
                for record in candidate_records
            )
            and len(
                {
                    record["deterministic_signature"]
                    for record in candidate_records
                }
            )
            == 1
            and all(invariant_fields.values())
            else "FAIL"
        )
    report = aggregate_report(
        source_urdf=source,
        runtime_records=baseline,
        candidate=candidate,
    )
    report["frozen_stage"] = {
        "absolute_path": str(FROZEN_STAGE.resolve()),
        "sha256": _sha256(FROZEN_STAGE),
        "modified": False,
    }
    report["local_5_1_api_evidence"] = {
        "single_articulation": str(
            (
                ROOT
                / ".venv_issac/lib/python3.11/site-packages/isaacsim/exts/"
                "isaacsim.core.prims/isaacsim/core/prims/impl/"
                "single_articulation.py"
            ).resolve(strict=True)
        ),
        "filtered_pair_helper": str(
            (
                ROOT
                / ".venv_issac/lib/python3.11/site-packages/isaacsim/"
                "extscache/omni.physx-107.3.26+107.3.3.lx64.r.cp311.u353/"
                "omni/physx/scripts/utils.py"
            ).resolve(strict=True)
        ),
    }
    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUTPUT_MD.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-output", type=Path)
    parser.add_argument("--validate-session-runtime", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--session-layer", type=Path)
    parser.add_argument("--session-layer-sha256")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.validate_session_runtime is not None:
        if args.summary_output is None or args.stage is None:
            raise ValueError(
                "--validate-session-runtime requires --summary-output and --stage"
            )
        if args.session_layer is None or args.session_layer_sha256 is None:
            raise ValueError(
                "--validate-session-runtime requires a frozen session layer"
            )
        actual_session_layer_sha256 = _sha256(args.session_layer)
        if actual_session_layer_sha256 != args.session_layer_sha256:
            raise RuntimeError(
                "session layer SHA-256 mismatch during summary generation"
            )
        runtime_path = args.validate_session_runtime.resolve(strict=True)
        source = parse_source_urdf()
        runtime_record = json.loads(runtime_path.read_text(encoding="utf-8"))
        summary = validate_session_layer_probe(
            record=runtime_record,
            source_limits=source["limits"],
            expected_stage_sha256=_sha256(args.stage),
            expected_layer_path=str(args.session_layer.resolve(strict=True)),
        )
        summary["inputs"] = {
            "runtime_output": {
                "absolute_path": str(runtime_path),
                "sha256": _sha256(runtime_path),
            },
            "source_urdf": source,
            "stage": {
                "absolute_path": str(args.stage.resolve(strict=True)),
                "sha256": _sha256(args.stage),
            },
            "session_layer": {
                "absolute_path": str(args.session_layer.resolve(strict=True)),
                "sha256": actual_session_layer_sha256,
                "expected_sha256": args.session_layer_sha256,
            },
        }
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["status"] == "PASS" else 1
    if args.runtime_output is not None:
        if args.stage is None:
            raise ValueError("--runtime-output requires --stage")
        return _run_runtime(
            args.stage,
            args.runtime_output,
            session_layer_path=args.session_layer,
            session_layer_sha256=args.session_layer_sha256,
        )
    report = build()
    print(
        json.dumps(
            {
                "status": report["status"],
                "limit_semantics_status": report[
                    "limit_semantics_status"
                ],
                "pair_collision_support_status": report[
                    "pair_collision_support_status"
                ],
                "candidate_created": report["candidate_created"],
                "output": str(OUTPUT_JSON.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
