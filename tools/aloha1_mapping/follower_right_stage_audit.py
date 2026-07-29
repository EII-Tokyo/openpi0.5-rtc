"""Pure evidence classification for follower-right USD Stage candidates."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def _is_independent_right_root(path: str) -> bool:
    lower = path.lower()
    return "follower_right" in lower or "vx300s_right" in lower


def _has_mesh_signature(
    signatures: Sequence[Mapping[str, Any]],
    *,
    point_count: int,
    face_count: int,
) -> bool:
    return any(
        int(item.get("point_count", -1)) == point_count
        and int(item.get("face_count", -1)) == face_count
        for item in signatures
    )


def classify_candidate(
    candidate: Mapping[str, Any],
    supplier_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Classify one candidate using composed-USD evidence, not its basename.

    Paths are used only to enforce explicit rejection provenance and to identify
    forbidden ALOHA2 composition.  A current supplier-CAD classification
    requires both handed mesh source hashes and an independent right
    articulation root.
    """

    roots = [str(path) for path in candidate.get("articulation_roots", [])]
    layers = [str(path) for path in candidate.get("used_layers", [])]
    lower_layers = [path.lower() for path in layers]
    source_hashes = {
        str(value)
        for value in candidate.get("finger_source_hashes", [])
        if value
    }
    expected_hashes = {
        str(supplier_hashes["left"]),
        str(supplier_hashes["right"]),
    }
    signatures = list(candidate.get("finger_mesh_signatures", []))
    independent_right_roots = [
        root for root in roots if _is_independent_right_root(root)
    ]

    gates = {
        "stage_open": candidate.get("open_status") == "PASS",
        "source_immutable": (
            candidate.get("sha256_before") == candidate.get("sha256_after")
        ),
        "supplier_hash_pair": expected_hashes.issubset(source_hashes),
        "independent_right_articulation": (
            len(roots) == 1 and len(independent_right_roots) == 1
        ),
        "not_rejected_provenance": not any(
            "rejected_phantom_right_branch" in path
            or "failed_attempt" in path
            for path in lower_layers
        ),
        "not_aloha2": not any("aloha2" in path for path in lower_layers),
    }

    if not gates["stage_open"]:
        classification = "REJECTED_STAGE_OPEN_FAILED"
    elif not gates["source_immutable"]:
        classification = "FAIL_SOURCE_STAGE_MUTATED_DURING_AUDIT"
    elif not gates["not_rejected_provenance"]:
        classification = "REJECTED_PHANTOM_RIGHT_BRANCH"
    elif not gates["not_aloha2"]:
        classification = "REJECTED_ALOHA2_OR_LEGACY_REBUILD"
    elif not gates["independent_right_articulation"]:
        classification = (
            "REJECTED_NOT_INDEPENDENT_FOLLOWER_RIGHT_ARTICULATION"
        )
    elif gates["supplier_hash_pair"]:
        classification = "ELIGIBLE_CURRENT_SUPPLIER_CAD"
    elif _has_mesh_signature(
        signatures,
        point_count=2568,
        face_count=856,
    ):
        classification = "REJECTED_GENERIC_FINGER"
    elif _has_mesh_signature(
        signatures,
        point_count=4998,
        face_count=1666,
    ):
        classification = "HISTORICAL_GYM_ALOHA_NOT_CURRENT_SUPPLIER_CAD"
    elif source_hashes & expected_hashes:
        classification = "INCOMPLETE_SUPPLIER_FINGER_PAIR"
    else:
        classification = "UNKNOWN_FINGER_PROVENANCE"

    return {
        **candidate,
        "finger_source_hashes": sorted(source_hashes),
        "classification": classification,
        "eligible": classification == "ELIGIBLE_CURRENT_SUPPLIER_CAD",
        "gates": gates,
    }


def summarize_audit(
    records: Sequence[Mapping[str, Any]],
    *,
    cad_identity_classification: str = "INCONCLUSIVE",
    workcell_placement_verified: bool = False,
) -> dict[str, Any]:
    """Summarize candidates without confusing Stage absence with CAD absence."""

    counts = Counter(str(record["classification"]) for record in records)
    eligible_count = sum(bool(record.get("eligible")) for record in records)
    hard_blockers = []
    if not eligible_count:
        hard_blockers.append(
            "HARD_BLOCKER_NO_CURRENT_SUPPLIER_CAD_FOLLOWER_RIGHT_STAGE"
        )
    if not workcell_placement_verified:
        hard_blockers.append(
            "HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM"
        )
    cad_verified = cad_identity_classification.startswith("VERIFIED_")
    return {
        "status": "PASS" if eligible_count else "PARTIAL",
        "candidate_count": len(records),
        "eligible_count": eligible_count,
        "classification_counts": dict(sorted(counts.items())),
        "cad_availability": cad_identity_classification,
        "next_action": (
            "GENERATE_ROBOT_LOCAL_FOLLOWER_RIGHT_DIAGNOSTIC_STAGE"
            if cad_verified and not eligible_count
            else (
                "VALIDATE_EXISTING_FOLLOWER_RIGHT_STAGE"
                if eligible_count
                else "RESOLVE_CAD_IDENTITY"
            )
        ),
        "hard_blockers": hard_blockers,
        "blocker_definitions": {
            "HARD_BLOCKER_NO_CURRENT_SUPPLIER_CAD_FOLLOWER_RIGHT_STAGE": {
                "meaning": (
                    "No already-generated and validated supplier-CAD "
                    "follower_right Stage exists yet."
                ),
                "does_not_mean": (
                    "Supplier CAD lacks the right-arm robot product."
                ),
                "resolution": (
                    "Generate and validate an isolated robot-local Stage "
                    "from the verified reusable product."
                ),
            },
            "HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM": {
                "meaning": (
                    "The complete right-follower workcell placement matrix "
                    "is not verified by supplier CAD or calibration."
                ),
                "does_not_mean": (
                    "Robot-local follower_right validation is blocked."
                ),
            },
        },
    }
