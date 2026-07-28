"""Pure analysis and invariant helpers for ALOHA1 gripper hold diagnosis v2."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import math
from pathlib import Path
from typing import Any

import yaml

CONTACT_SEMANTICS_STATUSES = {
    "VERIFIED_PHYSICAL_CONTACT",
    "CONTACT_ENVELOPE_DOMINATED",
    "REPORT_INTERPRETATION_ERROR",
    "INCONCLUSIVE",
}
ROOT_CAUSE_CATEGORIES = {
    "contact_envelope_or_offset",
    "insufficient_drive_preload",
    "insufficient_max_force",
    "material_binding_or_combine",
    "insufficient_friction",
    "solver_time_resolution",
    "contact_normal_or_geometry",
    "bottle_rotational_instability",
    "multiple_contributing_causes",
    "inconclusive",
}


def finite_or_none(value: Any) -> float | None:
    """Return a JSON-safe finite float, or ``None`` when unavailable."""

    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_force_diagnosis_config(path: Path, project_root: Path) -> dict[str, Any]:
    """Load, validate, and hash the immutable inputs for this diagnostic."""

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or int(document.get("schema_version", 0)) != 1:
        raise ValueError("unsupported gripper force diagnosis config")
    expected_environment = {
        "isaac_sim": "5.1.0.0",
        "kit": "107.3.3",
        "physx": "107.3.26",
        "python": "3.11.13",
    }
    if document.get("environment") != expected_environment:
        raise ValueError("diagnostic environment must remain pinned to Isaac Sim 5.1")
    frozen = document.get("frozen", {})
    required_frozen = {
        "approximation": "convexHull",
        "friction": 0.7,
        "restitution": 0.0,
        "bottle_mass_kg": 0.020,
        "bottle_diameter_m": 0.065,
        "physics_frequency_hz": 60,
        "solve_articulation_contact_last": True,
        "control_mode": "current_mimic",
    }
    for key, expected in required_frozen.items():
        if frozen.get(key) != expected:
            raise ValueError(f"frozen diagnostic value changed: {key}")
    if document.get("preload", {}).get("delta_m") != [
        0.0,
        0.0005,
        0.001,
        0.0015,
        0.002,
    ]:
        raise ValueError("preload delta grid changed")
    if int(document["preload"].get("repeats", 0)) < 10:
        raise ValueError("preload requires at least 10 fresh resets")
    if document.get("friction_scan", {}).get("mu") != [0.3, 0.5, 0.7, 1.0]:
        raise ValueError("friction scan grid changed")
    if int(document["friction_scan"].get("repeats", 0)) < 20:
        raise ValueError("friction scan requires at least 20 fresh resets")
    if document.get("task8") != "NOT_RUN":
        raise ValueError("Task 8 must remain NOT_RUN")
    if document.get("default_asset_collider_modified") is not False:
        raise ValueError("default collider must remain unchanged")

    root = project_root.resolve(strict=True)
    readback = []
    for item in document.get("protected_baseline", []):
        candidate = (root / item["path"]).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"protected path leaves project root: {item['path']}")
        actual = sha256_file(candidate) if candidate.is_file() else None
        readback.append(
            {
                "path": item["path"],
                "expected_sha256": item["sha256"],
                "actual_sha256": actual,
                "match": actual == item["sha256"],
            }
        )
    if not readback or not all(item["match"] for item in readback):
        failed = [item["path"] for item in readback if not item["match"]]
        raise RuntimeError(f"protected baseline changed: {failed}")
    result = deepcopy(document)
    result["protected_baseline_readback"] = readback
    return result


def _normalized_event_type(value: Any) -> str:
    name = str(value).upper()
    if "FOUND" in name:
        return "CONTACT_FOUND"
    if "PERSIST" in name:
        return "CONTACT_PERSIST"
    if "LOST" in name:
        return "CONTACT_LOST"
    return name


def build_contact_frame_states(
    events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate point-level contact events into deterministic per-frame states."""

    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[int(event["frame"])].append(event)
    states = []
    active = False
    for frame, frame_events in sorted(grouped.items()):
        event_types = sorted({_normalized_event_type(event["type"]) for event in frame_events})
        contacts = [contact for event in frame_events for contact in event.get("contacts", [])]
        if "CONTACT_LOST" in event_types and not contacts:
            active = False
            state = "LOST"
        elif "CONTACT_FOUND" in event_types:
            active = True
            state = "FOUND"
        elif "CONTACT_PERSIST" in event_types or contacts or active:
            active = True
            state = "PERSISTS"
        else:
            state = "INACTIVE"
        separations = [float(contact["separation"]) for contact in contacts]
        states.append(
            {
                "frame": frame,
                "event_types": event_types,
                "state": state,
                "contact_point_count": len(contacts),
                "minimum_separation_m": min(separations) if separations else None,
            }
        )
    return states


def select_contact_event_at_frame(
    events: Sequence[Mapping[str, Any]],
    *,
    frame: int,
) -> Mapping[str, Any] | None:
    """Select contact evidence from the exact independent-probe frame."""

    candidates = [event for event in events if int(event["frame"]) == int(frame) and event.get("contacts")]
    return candidates[0] if candidates else None


def finite_cylinder_signed_distance(
    *,
    point_xyz: Sequence[float],
    center_xyz: Sequence[float],
    radius_m: float,
    half_height_m: float,
) -> float:
    """Exact signed distance to an axis-Z finite cylinder."""

    px, py, pz = (float(value) for value in point_xyz)
    cx, cy, cz = (float(value) for value in center_xyz)
    radial = math.hypot(px - cx, py - cy) - float(radius_m)
    axial = abs(pz - cz) - float(half_height_m)
    outside = math.hypot(max(radial, 0.0), max(axial, 0.0))
    inside = min(max(radial, axial), 0.0)
    return outside + inside


def classify_contact_semantics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Classify contact-report values against an independent surface distance."""

    required = (
        "report_first_separation_m",
        "independent_first_surface_distance_m",
        "independent_distance_error_bound_m",
    )
    if any(metrics.get(key) is None for key in required):
        status = "INCONCLUSIVE"
        reason = "required_distance_measurement_missing"
    elif not bool(metrics.get("finger_only_pairs", False)):
        status = "INCONCLUSIVE"
        reason = "contact_pairs_include_non_finger_colliders"
    else:
        report = float(metrics["report_first_separation_m"])
        independent = float(metrics["independent_first_surface_distance_m"])
        error = abs(float(metrics["independent_distance_error_bound_m"]))
        if report > error and independent > error:
            status = "CONTACT_ENVELOPE_DOMINATED"
            reason = "contact_found_while_independent_surfaces_remain_separated"
        elif independent <= error and report >= -error:
            status = "VERIFIED_PHYSICAL_CONTACT"
            reason = "independent_surface_distance_is_zero_within_error"
        elif report < -error and independent > error:
            status = "REPORT_INTERPRETATION_ERROR"
            reason = "report_and_independent_distance_have_conflicting_signs"
        else:
            status = "INCONCLUSIVE"
            reason = "distance_evidence_does_not_select_one_semantics"
    if status not in CONTACT_SEMANTICS_STATUSES:
        raise AssertionError(status)
    return {
        "CONTACT_SEMANTICS_STATUS": status,
        "reason": reason,
        "input_metrics": dict(metrics),
    }


def required_normal_force_each(
    *,
    mass_kg: float,
    friction: float,
    gravity_m_s2: float = 9.81,
) -> float:
    """Two-sided Coulomb reference N = mg / (2 mu), not a calibration."""

    if mass_kg <= 0.0 or friction <= 0.0 or gravity_m_s2 <= 0.0:
        raise ValueError("mass, friction, and gravity must be positive")
    return float(mass_kg) * float(gravity_m_s2) / (2.0 * float(friction))


def _finite_floats(values: Sequence[Any]) -> list[float]:
    result = [float(value) for value in values]
    if any(not math.isfinite(value) for value in result):
        raise ValueError("force curve contains a non-finite value")
    return result


def summarize_preload_trials(
    trials: Sequence[Mapping[str, Any]],
    *,
    minimum_repeats: int,
) -> dict[str, Any]:
    if minimum_repeats <= 0:
        raise ValueError("minimum_repeats must be positive")
    if not trials:
        return {
            "delta_m": None,
            "trial_count": 0,
            "complete": False,
            "observable": False,
        }
    deltas = {float(trial["delta_m"]) for trial in trials}
    if len(deltas) != 1:
        raise ValueError("one preload summary may contain only one delta")
    successful = [trial for trial in trials if trial.get("status") == "PASS"]
    left = [value for trial in successful for value in _finite_floats(trial.get("left_stable_normal_force_n", []))]
    right = [value for trial in successful for value in _finite_floats(trial.get("right_stable_normal_force_n", []))]
    observable = bool(left and right)

    def side_summary(values: Sequence[float]) -> dict[str, Any]:
        return {
            "sample_count": len(values),
            "mean_normal_force_n": (sum(values) / len(values) if values else None),
            "minimum_stable_normal_force_n": min(values) if values else None,
            "maximum_normal_force_n": max(values) if values else None,
        }

    left_summary = side_summary(left)
    right_summary = side_summary(right)
    left_mean = left_summary["minimum_stable_normal_force_n"]
    right_mean = right_summary["minimum_stable_normal_force_n"]
    asymmetry = None
    if left_mean is not None and right_mean is not None:
        larger = max(float(left_mean), float(right_mean))
        asymmetry = min(float(left_mean), float(right_mean)) / larger if larger > 0.0 else 1.0
    return {
        "delta_m": next(iter(deltas)),
        "trial_count": len(trials),
        "successful_trial_count": len(successful),
        "failed_trial_count": len(trials) - len(successful),
        "complete": (
            len(trials) >= minimum_repeats
            and len(successful) == len(trials)
            and all(bool(trial.get("finite", False)) for trial in trials)
            and all(bool(trial.get("fresh_reset", False)) for trial in trials)
        ),
        "observable": observable,
        "left": left_summary,
        "right": right_summary,
        "left_right_asymmetry_ratio": asymmetry,
        "all_finite": all(bool(trial.get("finite", False)) for trial in trials),
        "all_fresh_resets": all(bool(trial.get("fresh_reset", False)) for trial in trials),
    }


def select_lowest_sufficient_preload(
    curves: Sequence[Mapping[str, Any]],
    required_each_n: float,
) -> float | None:
    sufficient = []
    for curve in curves:
        left = curve.get("left", {}).get("minimum_stable_normal_force_n")
        right = curve.get("right", {}).get("minimum_stable_normal_force_n")
        if (
            bool(curve.get("complete", False))
            and left is not None
            and right is not None
            and float(left) >= required_each_n
            and float(right) >= required_each_n
        ):
            sufficient.append(float(curve["delta_m"]))
    return min(sufficient) if sufficient else None


def classify_normal_force(
    curves: Sequence[Mapping[str, Any]],
    *,
    required_each_n: float,
) -> dict[str, Any]:
    if not curves or not any(bool(curve.get("observable", True)) for curve in curves):
        return {
            "NORMAL_FORCE_STATUS": "NOT_OBSERVABLE",
            "lowest_sufficient_preload_m": None,
            "required_each_n": float(required_each_n),
        }
    selected = select_lowest_sufficient_preload(curves, required_each_n)
    if selected is not None:
        status = "SUFFICIENT"
        reason = "both_sides_meet_theoretical_threshold_in_complete_trials"
    elif all(bool(curve.get("complete", False)) for curve in curves):
        status = "INSUFFICIENT"
        reason = "no_tested_preload_meets_threshold_on_both_sides"
    else:
        status = "INCONCLUSIVE"
        reason = "preload_repeats_incomplete"
    return {
        "NORMAL_FORCE_STATUS": status,
        "lowest_sufficient_preload_m": selected,
        "required_each_n": float(required_each_n),
        "reason": reason,
    }


def combine_material_value(value0: float, value1: float, mode: str) -> float:
    values = (float(value0), float(value1))
    if mode == "average":
        return sum(values) / 2.0
    if mode == "min":
        return min(values)
    if mode == "multiply":
        return values[0] * values[1]
    if mode == "max":
        return max(values)
    raise ValueError(f"unsupported PhysX combine mode: {mode}")


def audit_material_pair(
    finger: Mapping[str, Any],
    bottle: Mapping[str, Any],
    *,
    expected_friction: float | None = None,
    expected_restitution: float | None = None,
    contact_materials: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    material_applied = bool(finger.get("material_path") and bottle.get("material_path"))
    friction_modes = (
        finger.get("friction_combine_mode"),
        bottle.get("friction_combine_mode"),
    )
    restitution_modes = (
        finger.get("restitution_combine_mode"),
        bottle.get("restitution_combine_mode"),
    )
    modes_consistent = (
        friction_modes[0] is not None
        and friction_modes[0] == friction_modes[1]
        and restitution_modes[0] is not None
        and restitution_modes[0] == restitution_modes[1]
    )
    result = {
        "material_applied": material_applied,
        "combine_mode_consistent": modes_consistent,
        "effective_static_friction": None,
        "effective_dynamic_friction": None,
        "effective_restitution": None,
        "expected_values_match": None,
        "contact_materials_match_binding": None,
        "finger": dict(finger),
        "bottle": dict(bottle),
    }
    if material_applied and modes_consistent:
        result.update(
            {
                "effective_static_friction": combine_material_value(
                    finger["static_friction"],
                    bottle["static_friction"],
                    str(friction_modes[0]),
                ),
                "effective_dynamic_friction": combine_material_value(
                    finger["dynamic_friction"],
                    bottle["dynamic_friction"],
                    str(friction_modes[0]),
                ),
                "effective_restitution": combine_material_value(
                    finger["restitution"],
                    bottle["restitution"],
                    str(restitution_modes[0]),
                ),
            }
        )
    if expected_friction is not None and expected_restitution is not None:
        tolerance = 1.0e-6
        result["expected_values_match"] = bool(
            result["effective_static_friction"] is not None
            and math.isclose(
                float(result["effective_static_friction"]),
                float(expected_friction),
                abs_tol=tolerance,
            )
            and math.isclose(
                float(result["effective_dynamic_friction"]),
                float(expected_friction),
                abs_tol=tolerance,
            )
            and math.isclose(
                float(result["effective_restitution"]),
                float(expected_restitution),
                abs_tol=tolerance,
            )
        )
    if contact_materials is not None:
        reported = {
            contact_materials.get("material0"),
            contact_materials.get("material1"),
        }
        bound = {finger.get("material_path"), bottle.get("material_path")}
        result["contact_materials_match_binding"] = reported == bound
    return result


def friction_scan_gate(normal_force: Mapping[str, Any]) -> dict[str, Any]:
    if normal_force.get("NORMAL_FORCE_STATUS") == "SUFFICIENT":
        return {"run": True, "status": "READY", "reason": "normal_force_gate_passed"}
    return {
        "run": False,
        "status": "PARTIAL",
        "reason": "stable_sufficient_normal_force_not_confirmed",
    }


def classify_hold_failure_mode(metrics: Mapping[str, Any]) -> dict[str, Any]:
    drop = float(metrics.get("drop_m", math.inf))
    gate = float(metrics.get("drop_gate_m", 0.010))
    release_speed = float(metrics.get("release_linear_speed_m_s", 0.0))
    ejection_threshold = float(metrics.get("release_ejection_threshold_m_s", 0.1))
    if bool(metrics.get("persistent_penetration", False)) or (release_speed > ejection_threshold):
        mode = "NUMERICAL_EJECTION_OR_RELEASE_TRANSIENT"
    elif math.isfinite(drop) and 0.0 <= drop <= gate:
        return {"pass": True, "mode": "STATIC_HOLD_PASS"}
    else:
        contact_loss = metrics.get("contact_loss_frame")
        drop_crossing = metrics.get("drop_gate_crossing_frame")
        if contact_loss is not None and (drop_crossing is None or int(contact_loss) <= int(drop_crossing)):
            mode = "CONTACT_LOSS_THEN_FALL"
        elif float(metrics.get("normal_force_decay_ratio", 1.0)) < 0.5:
            mode = "NORMAL_FORCE_DECAY"
        elif float(metrics.get("maximum_angular_speed_rad_s", 0.0)) >= 1.0:
            mode = "ROTATION_THEN_DROP"
        elif bool(metrics.get("contacts_persist_to_end", False)):
            mode = "PERSISTENT_CONTACT_SLIDING"
        else:
            mode = "UNCLASSIFIED_DROP"
    return {"pass": False, "mode": mode}


def has_consecutive_true(
    values: Sequence[bool],
    *,
    required: int,
) -> bool:
    """Return true only when a condition persists for the required run."""

    if required <= 0:
        raise ValueError("required must be positive")
    run = 0
    for value in values:
        run = run + 1 if value else 0
        if run >= required:
            return True
    return False


def classify_solver_sensitivity(
    frequency_results: Sequence[Mapping[str, Any]],
    iteration_results: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not frequency_results:
        return {
            "SOLVER_STATUS": "INCONCLUSIVE",
            "run": False,
            "reason": "solver_scan_not_run",
        }
    if not all(
        bool(item.get("invariant_pass", False))
        for item in frequency_results
    ) or (
        iteration_results
        and not all(
            bool(item.get("invariant_pass", False))
            for item in iteration_results
        )
    ):
        return {
            "SOLVER_STATUS": "INCONCLUSIVE",
            "run": True,
            "reason": "single_variable_invariant_failed",
        }
    by_frequency = {int(item["frequency_hz"]): float(item["hold_success_rate"]) for item in frequency_results}
    if by_frequency.get(60, 0.0) == 1.0:
        status = "STABLE_AT_60HZ"
    elif by_frequency.get(60, 0.0) == 0.0 and any(
        rate == 1.0 for frequency, rate in by_frequency.items() if frequency > 60
    ):
        status = "REQUIRES_HIGHER_RATE"
    elif iteration_results and any(float(item.get("hold_success_rate", 0.0)) == 1.0 for item in iteration_results):
        status = "ITERATION_LIMITED"
    elif len(set(by_frequency.values())) == 1:
        status = "NO_MEANINGFUL_EFFECT"
    else:
        status = "INCONCLUSIVE"
    return {"SOLVER_STATUS": status, "run": True}


def select_solver_iteration_frequency(
    frequency_results: Sequence[Mapping[str, Any]],
    *,
    baseline_frequency_hz: int,
) -> dict[str, Any]:
    """Select one completed frequency to freeze for later iteration scans."""

    eligible = [
        item
        for item in frequency_results
        if bool(item.get("invariant_pass", False))
        and int(item.get("trial_count", 0)) > 0
        and int(item.get("successful_trial_count", 0))
        == int(item.get("trial_count", 0))
    ]
    if not eligible:
        return {
            "status": "INCONCLUSIVE",
            "selected_frequency_hz": None,
            "reason": "no_complete_invariant_frequency_group",
        }
    best_rate = max(float(item["hold_success_rate"]) for item in eligible)
    best = [
        item
        for item in eligible
        if math.isclose(float(item["hold_success_rate"]), best_rate)
    ]
    baseline = next(
        (
            item
            for item in best
            if int(item["frequency_hz"]) == int(baseline_frequency_hz)
        ),
        None,
    )
    selected = baseline or min(best, key=lambda item: int(item["frequency_hz"]))
    return {
        "status": "PASS",
        "selected_frequency_hz": int(selected["frequency_hz"]),
        "reason": (
            "baseline_wins_equal_best_rate"
            if baseline is not None
            else "lowest_frequency_with_best_hold_rate"
        ),
        "best_hold_success_rate": best_rate,
    }


def classify_root_cause_v2(evidence: Mapping[str, Any]) -> dict[str, Any]:
    causes = []
    unresolved = []
    if evidence.get("contact_semantics") == "CONTACT_ENVELOPE_DOMINATED":
        causes.append("contact_envelope_or_offset")
    if evidence.get("normal_force") == "INSUFFICIENT":
        if not bool(evidence.get("max_force_observable", False)):
            unresolved.append("drive_vs_max_force_not_observable")
        else:
            causes.append(
                "insufficient_max_force"
                if bool(evidence.get("max_force_saturated", False))
                else "insufficient_drive_preload"
            )
    if evidence.get("material") in {
        "MATERIAL_NOT_APPLIED",
        "COMBINE_MODE_UNEXPECTED",
    }:
        causes.append("material_binding_or_combine")
    if evidence.get("friction") == "INSUFFICIENT":
        causes.append("insufficient_friction")
    if evidence.get("solver") in {
        "REQUIRES_HIGHER_RATE",
        "ITERATION_LIMITED",
    }:
        causes.append("solver_time_resolution")
    if evidence.get("hold_failure_mode") == "ROTATION_THEN_DROP":
        causes.append("bottle_rotational_instability")
    if evidence.get("contact_normal_quality") == "FAIL":
        causes.append("contact_normal_or_geometry")
    causes = list(dict.fromkeys(causes))
    if evidence.get("hold_failure_mode") == "NUMERICAL_EJECTION_OR_RELEASE_TRANSIENT":
        unresolved.append("kinematic_to_dynamic_release_transient")
    if unresolved or not causes:
        root = "inconclusive"
    elif len(causes) == 1:
        root = causes[0]
    else:
        root = "multiple_contributing_causes"
    if root not in ROOT_CAUSE_CATEGORIES:
        raise AssertionError(root)
    return {
        "root_cause": root,
        "contributing_causes": causes,
        "unresolved_observations": unresolved,
        "allowed_categories": sorted(ROOT_CAUSE_CATEGORIES),
    }


def verify_solver_trial_invariants(
    trials: Sequence[Mapping[str, Any]],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the serialized one-variable-at-a-time solver manifest."""

    def read_path(value: Mapping[str, Any], field: str) -> Any:
        current: Any = value
        for token in field.split("."):
            if not isinstance(current, Mapping) or token not in current:
                return None
            current = current[token]
        return current

    mismatches = []
    for trial_index, trial in enumerate(trials):
        for field, expected_value in expected.items():
            actual = read_path(trial, field)
            if actual != expected_value:
                mismatches.append(
                    {
                        "trial_index": trial_index,
                        "field": field,
                        "expected": expected_value,
                        "actual": actual,
                    }
                )
    return {
        "pass": bool(trials) and not mismatches,
        "trial_count": len(trials),
        "expected": dict(expected),
        "mismatches": mismatches,
    }
