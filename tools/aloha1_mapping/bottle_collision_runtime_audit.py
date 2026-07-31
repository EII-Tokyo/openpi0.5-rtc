"""Pure machine gates for the Bottle500 collision runtime diagnostic."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from typing import Any


def measure_aabb_registration(
    *,
    visual_minimum: list[float],
    visual_maximum: list[float],
    collider_minimum: list[float],
    collider_maximum: list[float],
) -> dict[str, Any]:
    """Compare visual and collider AABBs expressed in the same frame."""

    if not all(
        len(values) == 3
        for values in (
            visual_minimum,
            visual_maximum,
            collider_minimum,
            collider_maximum,
        )
    ):
        raise ValueError("AABB bounds must contain exactly three coordinates")
    visual_min = [float(value) for value in visual_minimum]
    visual_max = [float(value) for value in visual_maximum]
    collider_min = [float(value) for value in collider_minimum]
    collider_max = [float(value) for value in collider_maximum]
    flattened = visual_min + visual_max + collider_min + collider_max
    if not all(math.isfinite(value) for value in flattened):
        raise ValueError("AABB bounds must be finite")
    if any(lower > upper for lower, upper in zip(visual_min, visual_max, strict=True)):
        raise ValueError("visual AABB minimum exceeds maximum")
    if any(lower > upper for lower, upper in zip(collider_min, collider_max, strict=True)):
        raise ValueError("collider AABB minimum exceeds maximum")

    visual_center = [
        (lower + upper) * 0.5
        for lower, upper in zip(visual_min, visual_max, strict=True)
    ]
    collider_center = [
        (lower + upper) * 0.5
        for lower, upper in zip(collider_min, collider_max, strict=True)
    ]
    center_delta = [
        collider - visual
        for visual, collider in zip(visual_center, collider_center, strict=True)
    ]
    surface_deltas = [
        collider - visual
        for visual, collider in zip(
            visual_min + visual_max,
            collider_min + collider_max,
            strict=True,
        )
    ]
    return {
        "visual_center_m": visual_center,
        "collider_center_m": collider_center,
        "center_delta_collider_minus_visual_m": center_delta,
        "maximum_center_residual_m": max(abs(value) for value in center_delta),
        "minimum_surface_delta_collider_minus_visual_m": [
            collider - visual
            for visual, collider in zip(visual_min, collider_min, strict=True)
        ],
        "maximum_surface_delta_collider_minus_visual_m": [
            collider - visual
            for visual, collider in zip(visual_max, collider_max, strict=True)
        ],
        "maximum_surface_gap_m": max(abs(value) for value in surface_deltas),
        "visual_size_m": [
            upper - lower
            for lower, upper in zip(visual_min, visual_max, strict=True)
        ],
        "collider_size_m": [
            upper - lower
            for lower, upper in zip(collider_min, collider_max, strict=True)
        ],
    }


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(value[key])
            for key in sorted(value)
            if str(key) not in {"normal_path", "overlay_path"}
        }
    if isinstance(value, list | tuple):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return round(value, 12)
    return value


def canonical_probe_signature(probe: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _canonicalize(probe),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def classify_collision_root_cause(gates: Mapping[str, bool], probe_kind: str) -> str:
    if not gates.get("colliders_present_and_enabled", False):
        return "BOTTLE_COLLISION_MISSING_OR_DISABLED"
    if not gates.get("collision_unfiltered", False):
        return "COLLISION_FILTERING_OR_MASK"
    if not gates.get("bottle_visual_collider_registration", False):
        return "BOTTLE_VISUAL_COLLIDER_MISREGISTRATION"
    if probe_kind == "SUPPLIER_CAD_FOLLOWER_FINGER" and not gates.get(
        "probe_visual_collider_registration",
        False,
    ):
        return "FINGER_VISUAL_COLLIDER_MISREGISTRATION"
    if not gates.get("rigid_body_dynamic", False):
        return "BOTTLE_RIGID_BODY_CONFIGURATION"
    if not gates.get("trajectory_intersects_collision_envelope", False):
        return "INCONCLUSIVE"
    if gates.get("physical_contact", False) and not gates.get(
        "expected_direction_response",
        False,
    ):
        return "BOTTLE_RIGID_BODY_CONFIGURATION"
    if not gates.get("physical_contact", False):
        return "SOLVER_OR_TUNNELING_SUSPECTED"
    if not gates.get("capture_pair_synchronization", False):
        return "VIDEO_PHYSICS_FRAME_MISMATCH"
    if not gates.get("forbidden_helpers_absent", False):
        return "INCONCLUSIVE"
    if all(bool(value) for value in gates.values()):
        return "COLLISION_PIPELINE_VERIFIED"
    return "INCONCLUSIVE"


def evaluate_collision_probe(probe: Mapping[str, Any]) -> dict[str, Any]:
    rigid_body = probe["rigid_body"]
    colliders = probe["colliders"]
    registration = probe["registration"]
    contacts = probe["contacts"]
    response = probe["response"]
    captures = probe["captures"]
    forbidden = probe["forbidden"]
    limits = probe["limits"]
    probe_kind = str(probe["probe_kind"])

    maximum_transform_residual = float(limits["maximum_transform_residual_m"])
    maximum_aabb_gap = float(limits["maximum_aabb_surface_gap_m"])
    minimum_response = float(limits["minimum_response_m"])

    physical_contacts = [
        contact
        for contact in contacts
        if bool(contact.get("physical"))
        and math.isfinite(float(contact.get("impulse_ns", math.nan)))
        and math.isfinite(float(contact.get("separation_m", math.nan)))
        and float(contact["impulse_ns"]) >= 0.0
        and float(contact["separation_m"]) <= 0.0
        and bool(contact.get("collider0_path"))
        and bool(contact.get("collider1_path"))
    ]
    push_direction = [float(value) for value in response["push_direction_world"]]
    displacement = [float(value) for value in response["bottle_displacement_world_m"]]
    push_norm = math.sqrt(sum(value * value for value in push_direction))
    signed_response = (
        sum(a * b for a, b in zip(push_direction, displacement, strict=True)) / push_norm
        if push_norm > 0.0
        else -math.inf
    )

    required_phases = set(captures["required_phases"])
    records_by_phase = {str(record["phase"]): record for record in captures["paired_records"]}
    synchronized = required_phases.issubset(records_by_phase) and all(
        bool(records_by_phase[phase].get("normal_path"))
        and bool(records_by_phase[phase].get("overlay_path"))
        and bool(records_by_phase[phase].get("same_camera_pose"))
        and bool(records_by_phase[phase].get("same_physics_frame"))
        for phase in required_phases
    )

    bottle_transform_residual = float(registration["bottle_max_transform_residual_m"])
    bottle_aabb_gap = float(registration["bottle_max_aabb_surface_gap_m"])
    probe_transform_residual = float(registration["probe_max_transform_residual_m"])
    probe_aabb_gap = float(registration["probe_max_aabb_surface_gap_m"])
    mass = float(rigid_body["mass_kg"])

    gates = {
        "frozen_inputs": bool(probe["frozen_inputs_verified"]) and str(probe["explicit_product_prim"]) == "/Bottle500",
        "rigid_body_dynamic": bool(rigid_body["enabled"])
        and not bool(rigid_body["kinematic_during_push"])
        and bool(rigid_body["gravity_enabled"])
        and math.isfinite(mass)
        and mass > 0.0,
        "colliders_present_and_enabled": int(colliders["count"]) > 0
        and bool(colliders["all_enabled"])
        and bool(colliders["approximation_tokens"]),
        "collision_unfiltered": not bool(colliders["filtered_pair_with_probe"]),
        "bottle_visual_collider_registration": math.isfinite(bottle_transform_residual)
        and math.isfinite(bottle_aabb_gap)
        and bottle_transform_residual <= maximum_transform_residual
        and bottle_aabb_gap <= maximum_aabb_gap,
        "probe_visual_collider_registration": math.isfinite(probe_transform_residual)
        and math.isfinite(probe_aabb_gap)
        and probe_transform_residual <= maximum_transform_residual
        and probe_aabb_gap <= maximum_aabb_gap,
        "physical_contact": bool(physical_contacts),
        "trajectory_intersects_collision_envelope": bool(response["trajectory_intersects_collision_envelope"]),
        "expected_direction_response": math.isfinite(signed_response) and signed_response >= minimum_response,
        "capture_pair_synchronization": synchronized,
        "forbidden_helpers_absent": not any(bool(value) for value in forbidden.values()),
    }
    root_cause = classify_collision_root_cause(gates, probe_kind)
    physical_gate_names = {
        "frozen_inputs",
        "rigid_body_dynamic",
        "colliders_present_and_enabled",
        "collision_unfiltered",
        "bottle_visual_collider_registration",
        "probe_visual_collider_registration",
        "physical_contact",
        "trajectory_intersects_collision_envelope",
        "expected_direction_response",
        "forbidden_helpers_absent",
    }
    physical_pass = all(gates[name] for name in physical_gate_names)
    status = "PASS" if physical_pass and synchronized else "PARTIAL" if physical_pass else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "root_cause": root_cause,
        "probe_kind": probe_kind,
        "gates": gates,
        "metrics": {
            "physical_contact_count": len(physical_contacts),
            "signed_response_m": signed_response,
            "bottle_max_transform_residual_m": bottle_transform_residual,
            "bottle_max_aabb_surface_gap_m": bottle_aabb_gap,
            "probe_max_transform_residual_m": probe_transform_residual,
            "probe_max_aabb_surface_gap_m": probe_aabb_gap,
        },
        "deterministic_signature": canonical_probe_signature(probe),
    }


def evaluate_follower_finger_collision_probe(
    probe: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the isolated follower-finger/Bottle500 collision replay."""

    maximum_gap = float(probe["maximum_registration_gap_m"])
    finger_colliders = probe["finger_colliders"]

    def physical_contacts(side: str) -> list[Mapping[str, Any]]:
        return [
            contact
            for contact in probe["contacts"][side]
            if bool(contact.get("physical"))
            and bool(contact.get("collider0_path"))
            and bool(contact.get("collider1_path"))
            and math.isfinite(float(contact.get("impulse_ns", math.nan)))
            and float(contact["impulse_ns"]) >= 0.0
            and math.isfinite(float(contact.get("separation_m", math.nan)))
            and float(contact["separation_m"]) <= 0.0
        ]

    left_contacts = physical_contacts("left")
    right_contacts = physical_contacts("right")
    required_phases = set(probe["captures"]["required_phases"])
    records_by_phase: dict[str, list[Mapping[str, Any]]] = {}
    for record in probe["captures"]["paired_records"]:
        records_by_phase.setdefault(str(record["phase"]), []).append(record)
    synchronized = required_phases.issubset(records_by_phase) and all(
        any(
            bool(record.get("normal_path"))
            and bool(record.get("overlay_path"))
            and bool(record.get("same_camera_pose"))
            and bool(record.get("same_physics_frame"))
            for record in records_by_phase[phase]
        )
        for phase in required_phases
    )
    collider_gates = {
        side: (
            bool(finger_colliders[side]["enabled"])
            and str(finger_colliders[side]["approximation"])
            in {"convexHull", "convexDecomposition"}
            and math.isfinite(
                float(finger_colliders[side]["maximum_registration_gap_m"])
            )
            and float(finger_colliders[side]["maximum_registration_gap_m"])
            <= maximum_gap
        )
        for side in ("left", "right")
    }
    maximum_displacement = float(
        probe["bottle_response"]["maximum_displacement_m"]
    )
    minimum_displacement = float(
        probe["bottle_response"]["minimum_required_displacement_m"]
    )
    gates = {
        "frozen_inputs": bool(probe["frozen_inputs_verified"]),
        "finger_colliders_enabled_and_registered": all(collider_gates.values()),
        "collision_unfiltered": not bool(probe["filtered_pair_with_bottle"]),
        "left_physical_contact": bool(left_contacts),
        "right_physical_contact": bool(right_contacts),
        "finite_bottle_response": math.isfinite(maximum_displacement)
        and maximum_displacement >= minimum_displacement,
        "capture_pair_synchronization": synchronized,
        "forbidden_helpers_absent": bool(probe["forbidden_helpers_absent"]),
    }
    if not gates["finger_colliders_enabled_and_registered"]:
        classification = "FINGER_VISUAL_COLLIDER_MISREGISTRATION"
    elif not gates["collision_unfiltered"]:
        classification = "COLLISION_FILTERING_OR_MASK"
    elif not gates["left_physical_contact"] or not gates["right_physical_contact"]:
        classification = "BILATERAL_FINGER_CONTACT_NOT_ESTABLISHED"
    elif not gates["finite_bottle_response"]:
        classification = "BOTTLE_RESPONSE_NOT_OBSERVED"
    elif not gates["capture_pair_synchronization"]:
        classification = "VIDEO_PHYSICS_FRAME_MISMATCH"
    elif all(gates.values()):
        classification = "FINGER_COLLISION_PIPELINE_VERIFIED"
    else:
        classification = "INCONCLUSIVE"
    physical_names = {
        "frozen_inputs",
        "finger_colliders_enabled_and_registered",
        "collision_unfiltered",
        "left_physical_contact",
        "right_physical_contact",
        "finite_bottle_response",
        "forbidden_helpers_absent",
    }
    physical_pass = all(gates[name] for name in physical_names)
    status = "PASS" if physical_pass and synchronized else "PARTIAL" if physical_pass else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "classification": classification,
        "gates": gates,
        "metrics": {
            "left_physical_contact_count": len(left_contacts),
            "right_physical_contact_count": len(right_contacts),
            "maximum_bottle_displacement_m": maximum_displacement,
        },
        "deterministic_signature": canonical_probe_signature(probe),
    }
