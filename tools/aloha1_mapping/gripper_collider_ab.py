"""Pure manifest, invariant, and classification helpers for gripper collider A/B."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

ALLOWED_APPROXIMATIONS = {"convexHull", "convexDecomposition"}
ALLOWED_CONTROL_MODES = {"current_mimic", "explicit_symmetric"}

_REQUIRED_TRIAL_GATES = (
    "bilateral_contact_before_release",
    "impulses_finite",
    "persistent_penetration",
    "unexpected_gripper_collision",
    "held_for_required_steps",
    "finite_state",
)
_INVERTED_TRIAL_GATES = {
    "persistent_penetration",
    "unexpected_gripper_collision",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_collision_profiles(path: Path, project_root: Path) -> dict[str, Any]:
    """Load and validate the frozen profile, expanding common fields per variant."""

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("collision profile manifest must be a mapping")
    if int(document.get("schema_version", 0)) != 1:
        raise ValueError("unsupported collision profile schema")
    repeats = int(document.get("experiment", {}).get("repeats_per_robot", 0))
    if repeats < 20:
        raise ValueError("repeats_per_robot must be at least 20")
    frozen = document.get("frozen")
    if not isinstance(frozen, dict):
        raise ValueError("frozen profile must be a mapping")
    profiles = document.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != {
        "convex_hull",
        "convex_decomposition",
    }:
        raise ValueError("exactly convex_hull and convex_decomposition profiles are required")

    expanded: dict[str, dict[str, Any]] = {}
    for name, variant in profiles.items():
        if not isinstance(variant, dict):
            raise ValueError(f"profile {name} must be a mapping")
        approximation = variant.get("approximation")
        if approximation not in ALLOWED_APPROXIMATIONS:
            raise ValueError(f"unsupported approximation: {approximation}")
        item = deepcopy(frozen)
        item.update(deepcopy(variant))
        expanded[name] = item
    document["profiles"] = expanded

    controls = document.get("control_modes", {})
    if set(controls) != ALLOWED_CONTROL_MODES:
        raise ValueError("control modes must be current_mimic and explicit_symmetric")
    if (
        controls["explicit_symmetric"].get("status")
        != "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
    ):
        raise ValueError("explicit symmetric mode must remain diagnostic-only")

    root = project_root.resolve(strict=True)
    for item in document.get("protected_baseline", []):
        candidate = (root / item["path"]).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"protected path leaves project root: {item['path']}")
    return document


def assert_profile_pair_is_frozen(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    allowed_differences: Set[str],
) -> None:
    """Raise if two flattened profiles differ outside explicitly allowed keys."""

    differences = {
        key
        for key in set(first) | set(second)
        if first.get(key) != second.get(key)
    }
    forbidden = sorted(differences - set(allowed_differences))
    if forbidden:
        raise ValueError(f"non-experimental profile differences: {', '.join(forbidden)}")
    missing = set(allowed_differences) - differences
    if missing:
        raise ValueError(f"expected profile differences are absent: {', '.join(sorted(missing))}")


def canonical_signature(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _gate_passed(name: str, metrics: Mapping[str, Any]) -> bool:
    value = bool(metrics.get(name, False))
    return not value if name in _INVERTED_TRIAL_GATES else value


def trial_passes_hold_gate(trial: Mapping[str, Any]) -> bool:
    metrics = trial.get("metrics", {})
    return all(_gate_passed(name, metrics) for name in _REQUIRED_TRIAL_GATES)


def summarize_ab_trials(
    trials: Sequence[Mapping[str, Any]],
    *,
    minimum_repeats: int,
) -> dict[str, Any]:
    if minimum_repeats <= 0:
        raise ValueError("minimum_repeats must be positive")
    passes = [trial_passes_hold_gate(trial) for trial in trials]
    complete = len(trials) >= minimum_repeats
    all_pass = complete and all(passes)
    runtimes = [
        float(trial["runtime_s"])
        for trial in trials
        if math.isfinite(float(trial.get("runtime_s", math.nan)))
    ]
    signatures = [str(trial.get("deterministic_signature", "")) for trial in trials]
    return {
        "status": "PASS" if all_pass else ("FAIL" if complete else "PARTIAL"),
        "trial_count": len(trials),
        "minimum_repeats": minimum_repeats,
        "complete": complete,
        "hold_success_count": sum(passes),
        "hold_success_rate": (sum(passes) / len(passes) if passes else 0.0),
        "all_trials_pass_hold_gate": all_pass,
        "runtime_mean_s": (sum(runtimes) / len(runtimes) if runtimes else None),
        "deterministic_signatures": signatures,
        "exact_signature_repeat": bool(signatures) and len(set(signatures)) == 1,
    }


def _resolved(values: Sequence[bool], minimum_repeats: int) -> bool | None:
    if len(values) < minimum_repeats:
        return None
    return all(bool(value) for value in values)


def classify_root_cause(
    groups: Mapping[str, Sequence[bool]],
    *,
    minimum_repeats: int,
) -> dict[str, Any]:
    expected = {
        "hull_current",
        "decomposition_current",
        "hull_explicit",
        "decomposition_explicit",
    }
    if set(groups) != expected:
        return {
            "status": "PARTIAL",
            "classification": "inconclusive",
            "reason": "missing_or_extra_groups",
        }
    states = {name: _resolved(groups[name], minimum_repeats) for name in expected}
    if any(value is None for value in states.values()):
        return {
            "status": "PARTIAL",
            "classification": "inconclusive",
            "reason": "insufficient_repeats",
            "resolved_groups": states,
        }

    hc = bool(states["hull_current"])
    dc = bool(states["decomposition_current"])
    he = bool(states["hull_explicit"])
    de = bool(states["decomposition_explicit"])
    if not hc and dc and not he and de:
        classification = "collider_primary"
    elif not hc and not dc and he and de:
        classification = "mimic_primary"
    elif not hc and not dc and not he and de:
        classification = "collider_and_mimic"
    elif not any((hc, dc, he, de)):
        classification = "neither_resolved"
    else:
        classification = "inconclusive"
    return {
        "status": "PASS" if classification != "inconclusive" else "PARTIAL",
        "classification": classification,
        "resolved_groups": states,
        "semantics": "resolved means every fresh-reset trial passed the unchanged hold gate",
    }


def classify_decomposition_status(
    hull: Sequence[bool],
    decomposition: Sequence[bool],
    *,
    minimum_repeats: int,
) -> dict[str, Any]:
    if len(hull) < minimum_repeats or len(decomposition) < minimum_repeats:
        return {
            "status": "INCONCLUSIVE",
            "reason": "insufficient_repeats",
        }
    hull_rate = sum(bool(value) for value in hull) / len(hull)
    decomposition_rate = sum(bool(value) for value in decomposition) / len(decomposition)
    hull_all = all(hull)
    decomposition_all = all(decomposition)
    if decomposition_all and not hull_all:
        status = "IMPROVES_HOLD"
    elif hull_all and not decomposition_all:
        status = "WORSENS_CONTACT"
    elif list(hull) == list(decomposition) or math.isclose(
        hull_rate,
        decomposition_rate,
        abs_tol=0.0,
    ):
        status = "NO_MEANINGFUL_EFFECT"
    else:
        status = "INCONCLUSIVE"
    return {
        "status": status,
        "hull_success_rate": hull_rate,
        "decomposition_success_rate": decomposition_rate,
        "gate_semantics": "all trials must pass; mixed success is not promoted to a pass",
    }
