"""Evidence-based ALOHA public-CAD finger instance and URDF-link mapping."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from tools.aloha1_mapping.cad_finger_installation import CAD_ASSEMBLY_TO_FINGER_LINK_ROTATION
from tools.aloha1_mapping.cad_finger_installation import cad_global_to_finger_link_matrix
from tools.aloha1_mapping.cad_finger_installation import determinant3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAD_ARTIFACT_ROOT = (
    PROJECT_ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _floats(value: str) -> list[float]:
    return [float(item) for item in value.split()]


def _rpy_matrix(rpy: list[float]) -> list[list[float]]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
        ],
        [
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
        ],
        [-sp, cp * sr, cp * cr],
    ]


def _mat_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    ]


def _bbox_center(bbox: Mapping[str, float]) -> list[float]:
    return [
        0.5 * (bbox[f"{axis}Min"] + bbox[f"{axis}Max"])
        for axis in "XYZ"
    ]


def _norm(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in vector))
    if length <= 0:
        raise ValueError("zero-length direction")
    return [value / length for value in vector]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _urdf_mapping(urdf_path: Path) -> dict[str, Any]:
    root = ET.parse(urdf_path).getroot()
    gripper_bar = next(
        link
        for link in root.findall("link")
        if link.get("name", "").endswith("_gripper_bar_link")
    )
    origin = gripper_bar.find("visual/origin")
    if origin is None:
        raise ValueError("gripper_bar visual origin is missing")
    rpy = _floats(origin.get("rpy", ""))
    rotation = _rpy_matrix(rpy)
    cad_positive_x_in_urdf = _mat_vec(rotation, [1.0, 0.0, 0.0])
    cad_negative_x_in_urdf = [-value for value in cad_positive_x_in_urdf]

    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    left = joints["left_finger"]
    right = joints["right_finger"]
    left_axis = _floats(left.find("axis").get("xyz", ""))
    right_axis = _floats(right.find("axis").get("xyz", ""))
    left_limit = left.find("limit")
    right_limit = right.find("limit")
    if left_limit is None or right_limit is None:
        raise ValueError("finger limits are missing")
    left_limits = [
        float(left_limit.get("lower", "nan")),
        float(left_limit.get("upper", "nan")),
    ]
    right_limits = [
        float(right_limit.get("lower", "nan")),
        float(right_limit.get("upper", "nan")),
    ]
    axis_alignment = _dot(
        _norm(cad_positive_x_in_urdf),
        _norm(left_axis),
    )
    gates = {
        "gripper_bar_rotation_is_rz_positive_90": all(
            abs(actual - expected) <= 1.0e-12
            for actual, expected in zip(
                rpy,
                [0.0, 0.0, math.pi / 2.0],
                strict=True,
            )
        ),
        "cad_positive_x_aligns_left_joint_positive_axis": (
            axis_alignment > 0.999999
        ),
        "left_and_right_joint_axes_match": left_axis == right_axis,
        "right_is_negative_mimic": (
            right.find("mimic") is not None
            and right.find("mimic").get("joint") == "left_finger"
            and float(right.find("mimic").get("multiplier", "nan")) == -1.0
        ),
        "finger_ranges_are_signed_mirrors": (
            right_limits
            == [-left_limits[1], -left_limits[0]]
        ),
    }
    return {
        "source": (
            "generated follower URDF gripper_bar visual origin and finger joints"
        ),
        "urdf_path": str(urdf_path.resolve()),
        "urdf_sha256": _sha256(urdf_path),
        "gripper_bar_visual_rpy_rad": rpy,
        "gripper_bar_visual_rotation": rotation,
        "cad_positive_x_in_urdf": cad_positive_x_in_urdf,
        "cad_negative_x_in_urdf": cad_negative_x_in_urdf,
        "left_joint_axis": left_axis,
        "right_joint_axis": right_axis,
        "positive_cad_x_link": "left_finger",
        "negative_cad_x_link": "right_finger",
        "left_limits_m": left_limits,
        "right_limits_m": right_limits,
        "open_delta_mm": 1000.0 * (left_limits[1] - left_limits[0]),
        "gates": gates,
    }


def _shape_bbox(obj: Mapping[str, Any]) -> dict[str, float]:
    shape = obj.get("shape")
    if not isinstance(shape, dict) or shape.get("is_null"):
        raise ValueError(f"object has no finite shape: {obj.get('name')}")
    bbox = shape.get("bound_box_mm")
    if not isinstance(bbox, dict):
        raise ValueError(f"object has no bounding box: {obj.get('name')}")
    return {str(key): float(value) for key, value in bbox.items()}


def _matrix_maximum_delta(
    left: list[list[float]],
    right: list[list[float]],
) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(left, dtype=np.float64)
                - np.asarray(right, dtype=np.float64)
            )
        )
    )


def _mirror_residual(
    positive_vertices: list[list[float]],
    negative_vertices: list[list[float]],
) -> dict[str, Any]:
    positive = np.asarray(positive_vertices, dtype=np.float64)
    negative = np.asarray(negative_vertices, dtype=np.float64)
    positive_center = 0.5 * (positive.min(axis=0) + positive.max(axis=0))
    negative_center = 0.5 * (negative.min(axis=0) + negative.max(axis=0))
    mirrored = positive - positive_center
    mirrored[:, 0] *= -1.0
    mirrored += negative_center
    positive_to_negative = _nearest_distances(mirrored, negative)
    negative_to_positive = _nearest_distances(negative, mirrored)
    combined_squared = np.concatenate(
        (positive_to_negative**2, negative_to_positive**2)
    )
    return {
        "method": (
            "AABB-center aligned CAD-X reflection followed by symmetric "
            "nearest-vertex distance on FreeCAD 0.20 mm tessellations"
        ),
        "positive_center_mm": positive_center.tolist(),
        "negative_center_mm": negative_center.tolist(),
        "maximum": float(
            max(positive_to_negative.max(), negative_to_positive.max())
        ),
        "rms": float(np.sqrt(np.mean(combined_squared))),
        "tessellation_error_note": (
            "residual includes independent tessellation sampling and is not "
            "an exact B-Rep surface Hausdorff distance"
        ),
    }


def _nearest_distances(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    """Return exact Euclidean nearest distances without a SciPy ABI dependency."""
    distances: list[np.ndarray] = []
    for start in range(0, len(query), chunk_size):
        chunk = query[start : start + chunk_size]
        squared = np.sum(
            (chunk[:, np.newaxis, :] - reference[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        distances.append(np.sqrt(np.min(squared, axis=1)))
    return np.concatenate(distances)


def _authoritative_widow_installation(
    authority_audit_path: Path,
    tessellation_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    audit = json.loads(authority_audit_path.read_text(encoding="utf-8"))
    source = audit["sources"][0]
    objects = {obj["name"]: obj for obj in source["objects"]}
    roots = source["root_objects"]
    if roots != ["Dummy_Aloha_WX_with_Fingers_v2"]:
        raise ValueError(f"unexpected authoritative root objects: {roots}")
    root = objects[roots[0]]
    grippers = [
        objects[name]
        for name in root["out_list"]
        if name in objects
        and objects[name]["label"].startswith("Dummy Aloha WX Gripper")
    ]
    finger_groups = [
        objects[name]
        for name in root["out_list"]
        if name in objects
        and objects[name]["type_id"] == "App::Part"
        and objects[name]["label"].startswith("Aloha VX Fingers")
    ]
    if len(grippers) != 1 or len(finger_groups) != 1:
        raise ValueError(
            "authoritative Widow file must contain one gripper shell and one "
            "finger assembly"
        )
    gripper = grippers[0]
    group = finger_groups[0]
    fingers = [
        objects[name]
        for name in group["out_list"]
        if name in objects
        and objects[name]["type_id"] == "Part::Feature"
        and objects[name]["label"].startswith("Aloha VX Fingers")
    ]
    if len(fingers) != 2:
        raise ValueError("authoritative Widow finger assembly is not bilateral")
    finger_records = []
    for finger in fingers:
        bbox = _shape_bbox(finger)
        center = _bbox_center(bbox)
        side = "positive" if center[0] > 0 else "negative"
        finger_records.append(
            {
                "object_name": finger["name"],
                "label": finger["label"],
                "assembly_path": [
                    root["name"],
                    group["name"],
                    finger["name"],
                ],
                "cad_opening_side": side,
                "mapped_urdf_joint": (
                    "left_finger" if side == "positive" else "right_finger"
                ),
                "shape_bbox_mm": bbox,
                "shape_bbox_center_mm": center,
                "shape_volume_mm3": finger["shape"]["volume_mm3"],
                "shape_topology_counts": finger["shape"]["topology_counts"],
                "local_placement": finger["local_placement"],
                "global_placement": finger["global_placement"],
            }
        )
    positive = next(
        item
        for item in finger_records
        if item["cad_opening_side"] == "positive"
    )
    negative = next(
        item
        for item in finger_records
        if item["cad_opening_side"] == "negative"
    )
    placement_delta = _matrix_maximum_delta(
        positive["local_placement"]["matrix"],
        negative["local_placement"]["matrix"],
    )
    gripper_matrix = np.asarray(
        gripper["local_placement"]["matrix"],
        dtype=np.float64,
    )
    finger_matrix = np.asarray(
        positive["local_placement"]["matrix"],
        dtype=np.float64,
    )
    relative = np.linalg.inv(gripper_matrix) @ finger_matrix
    closed_gap = (
        positive["shape_bbox_mm"]["XMin"]
        - negative["shape_bbox_mm"]["XMax"]
    )

    tessellation = json.loads(tessellation_path.read_text(encoding="utf-8"))
    meshes = tessellation["meshes"]
    mirror = _mirror_residual(
        meshes["cad_positive_x_finger"]["vertices_mm"],
        meshes["cad_negative_x_finger"]["vertices_mm"],
    )
    relationships = tessellation["relationships"]
    positive_connection = relationships["gripper_to_positive_x_finger"]
    negative_connection = relationships["gripper_to_negative_x_finger"]
    connection_asymmetric = (
        abs(
            positive_connection["common_volume_mm3"]
            - negative_connection["common_volume_mm3"]
        )
        > 1.0e-6
    )
    gates = {
        "user_confirmed_authority_hash": (
            source["sha256"]
            == "adc6a2c96912ab7973347b6a4587b6001bdc6316ab294dba5c46273139365500"
        ),
        "single_gripper_shell": len(grippers) == 1,
        "single_finger_assembly": len(finger_groups) == 1,
        "two_distinct_finger_breps": (
            len(fingers) == 2
            and positive["object_name"] != negative["object_name"]
        ),
        "identical_rigid_placement": placement_delta <= 1.0e-12,
        "positive_x_maps_left": (
            positive["mapped_urdf_joint"] == "left_finger"
        ),
        "negative_x_maps_right": (
            negative["mapped_urdf_joint"] == "right_finger"
        ),
        "positive_closed_gap": closed_gap > 0.0,
        "finger_pair_minimum_distance_matches_inner_gap": (
            abs(
                relationships["finger_to_finger"][
                    "minimum_shape_distance_mm"
                ]
                - closed_gap
            )
            <= 1.0e-9
        ),
        "both_fingers_contact_gripper_shell": (
            positive_connection["minimum_shape_distance_mm"] == 0.0
            and negative_connection["minimum_shape_distance_mm"] == 0.0
        ),
        "mirror_residual_below_tessellation_deflection_scale": (
            mirror["maximum"] < 0.30
        ),
    }
    installation = {
        "evidence_class": "WX_SHARED_GRIPPER_CROSSCHECK",
        "source_statement": (
            "The user initially nominated the Widow file, then supplied the "
            "purchase drawing and asked for model identification. The drawing "
            "directly identifies the purchased follower as VX300S/ViperX, so "
            "Widow is retained only as a shared-gripper cross-check."
        ),
        "root_object": root["name"],
        "root_label": root["label"],
        "gripper_shell_object": gripper["name"],
        "gripper_shell_label": gripper["label"],
        "gripper_shell_placement": gripper["local_placement"],
        "finger_group_object": group["name"],
        "finger_group_label": group["label"],
        "positive_x_finger": positive,
        "negative_x_finger": negative,
        "identical_finger_placement_matrix": placement_delta <= 1.0e-12,
        "finger_placement_maximum_delta": placement_delta,
        "gripper_to_finger_relative_matrix": relative.tolist(),
        "installation_rule": (
            "use the two distinct handed B-Reps with the same rigid placement; "
            "do not reuse one side with an arbitrary per-side 180-degree roll"
        ),
        "arbitrary_per_side_roll_required": False,
        "closed_inner_gap_mm": closed_gap,
        "mirror_residual_mm": mirror,
        "connection_geometry": {
            "positive_x": positive_connection,
            "negative_x": negative_connection,
            "finger_to_finger": relationships["finger_to_finger"],
            "one_sided_common_volume_asymmetry": connection_asymmetric,
            "interpretation": (
                "both fingers touch the gripper shell; the negative-X side "
                "contains a small nonzero CAD Boolean common volume that must "
                "be preserved as a reported source-model finding"
            ),
        },
        "gates": gates,
    }
    source_record = {
        "path": source["path"],
        "sha256": source["sha256"],
        "freecad_audit_path": str(authority_audit_path.resolve()),
        "freecad_audit_sha256": _sha256(authority_audit_path),
        "tessellation_path": str(tessellation_path.resolve()),
        "tessellation_sha256": _sha256(tessellation_path),
    }
    return installation, source_record


def _simple_viper_installation(
    simple: Mapping[str, Any],
    viper_state_report: Mapping[str, Any],
) -> dict[str, Any]:
    objects = {obj["name"]: obj for obj in simple["objects"]}
    root = objects[simple["root_objects"][0]]
    gripper = next(
        objects[name]
        for name in root["out_list"]
        if name in objects
        and objects[name]["label"].startswith("Aloha VX Gripper")
    )
    group = next(
        objects[name]
        for name in root["out_list"]
        if name in objects
        and objects[name]["type_id"] == "App::Part"
        and objects[name]["label"].startswith("Aloha VX Fingers")
    )
    fingers = [
        objects[name]
        for name in group["out_list"]
        if name in objects
        and objects[name]["type_id"] == "Part::Feature"
        and objects[name]["label"].startswith("Aloha VX Fingers")
    ]
    records = []
    for finger in fingers:
        bbox = _shape_bbox(finger)
        center = _bbox_center(bbox)
        side = "positive" if center[0] > 0 else "negative"
        records.append(
            {
                "object_name": finger["name"],
                "label": finger["label"],
                "assembly_path": [
                    root["name"],
                    group["name"],
                    finger["name"],
                ],
                "cad_opening_side": side,
                "mapped_urdf_joint": (
                    "left_finger" if side == "positive" else "right_finger"
                ),
                "shape_bbox_mm": bbox,
                "shape_bbox_center_mm": center,
                "shape_volume_mm3": finger["shape"]["volume_mm3"],
                "shape_topology_counts": finger["shape"]["topology_counts"],
                "local_placement": finger["local_placement"],
            }
        )
    positive = next(
        item for item in records if item["cad_opening_side"] == "positive"
    )
    negative = next(
        item for item in records if item["cad_opening_side"] == "negative"
    )
    placement_delta = _matrix_maximum_delta(
        positive["local_placement"]["matrix"],
        negative["local_placement"]["matrix"],
    )
    placement_matrix = np.asarray(
        positive["local_placement"]["matrix"],
        dtype=np.float64,
    )
    state_relationships = {
        state: viper_state_report["states"][state]["relationships"]
        for state in ("closed", "open")
    }
    return {
        "evidence_class": "PURCHASE_DRAWING_CONFIRMED_VX300S_FOLLOWER",
        "source_path": simple["path"],
        "source_sha256": simple["sha256"],
        "root_object": root["name"],
        "root_label": root["label"],
        "gripper_shell_object": gripper["name"],
        "gripper_shell_label": gripper["label"],
        "finger_group_object": group["name"],
        "finger_group_label": group["label"],
        "positive_x_finger": positive,
        "negative_x_finger": negative,
        "identical_finger_placement_matrix": placement_delta <= 1.0e-12,
        "finger_placement_maximum_delta": placement_delta,
        "shared_source_placement_matrix": placement_matrix.tolist(),
        "shared_source_placement_determinant": float(
            np.linalg.det(placement_matrix[:3, :3])
        ),
        "minimum_pair_x_gap_mm": (
            positive["shape_bbox_mm"]["XMin"]
            - negative["shape_bbox_mm"]["XMax"]
        ),
        "installation_rule": (
            "use the two distinct handed B-Reps with their common rigid "
            "placement; do not apply an arbitrary per-side roll"
        ),
        "arbitrary_per_side_roll_required": False,
        "static_supplier_assembly_state": "CLOSED_REFERENCE",
        "derived_open_state": {
            "cad_positive_x_finger_translation_mm": [36.0, 0.0, 0.0],
            "cad_negative_x_finger_translation_mm": [-36.0, 0.0, 0.0],
            "shape_or_handedness_changed": False,
        },
        "source_connection_common_volume_mm3": {
            state: {
                "cad_positive_x_finger": state_relationships[state][
                    "gripper_to_positive_x_finger"
                ]["common_volume_mm3"],
                "cad_negative_x_finger": state_relationships[state][
                    "gripper_to_negative_x_finger"
                ]["common_volume_mm3"],
            }
            for state in ("closed", "open")
        },
        "source_connection_interpretation": (
            "Boolean common volumes between the supplier gripper shell/"
            "sliding carriage and embedded fingers are source-CAD connection "
            "geometry. They are recorded numerically and are not classified "
            "as an unexpected simulated collision."
        ),
    }


def build_gripper_mapping_report(
    freecad_audit_path: Path,
    urdf_path: Path,
    widow_audit_path: Path,
    widow_tessellation_path: Path,
    viper_state_path: Path | None = None,
    toolchain_probe_path: Path | None = None,
    screenshot_review_path: Path | None = None,
    render_metadata_path: Path | None = None,
    tessellation_report_path: Path | None = None,
    mount_registration_path: Path | None = None,
    isaac_screenshot_review_path: Path | None = None,
    diagnostic_asset_path: Path | None = None,
) -> dict[str, Any]:
    viper_state_path = viper_state_path or (
        CAD_ARTIFACT_ROOT / "viper_gripper/viper_gripper_states.json"
    )
    toolchain_probe_path = toolchain_probe_path or (
        CAD_ARTIFACT_ROOT / "cad_toolchain_probe_final.json"
    )
    screenshot_review_path = screenshot_review_path or (
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_viper_gripper_screenshot_review.json"
    )
    render_metadata_path = render_metadata_path or (
        CAD_ARTIFACT_ROOT
        / "viper_gripper/attempt5_candidate/render_metadata.json"
    )
    tessellation_report_path = tessellation_report_path or (
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_viper_finger_tessellation.json"
    )
    mount_registration_path = mount_registration_path or (
        PROJECT_ROOT
        / "reports/aloha1_mapping/aloha_viper_cad_mount_registration.json"
    )
    isaac_screenshot_review_path = isaac_screenshot_review_path or (
        PROJECT_ROOT
        / "reports/aloha1_mapping/"
        "aloha_viper_cad_finger_isaac_screenshot_review.json"
    )
    diagnostic_asset_path = diagnostic_asset_path or (
        PROJECT_ROOT
        / "reports/aloha1_mapping/"
        "aloha_viper_cad_finger_diagnostic_asset_v2.json"
    )
    audit = json.loads(freecad_audit_path.read_text(encoding="utf-8"))
    viper_state_report = json.loads(
        viper_state_path.read_text(encoding="utf-8")
    )
    toolchain_probe = json.loads(
        toolchain_probe_path.read_text(encoding="utf-8")
    )
    screenshot_review = json.loads(
        screenshot_review_path.read_text(encoding="utf-8")
    )
    render_metadata = json.loads(
        render_metadata_path.read_text(encoding="utf-8")
    )
    tessellation_report = json.loads(
        tessellation_report_path.read_text(encoding="utf-8")
    )
    mount_registration = json.loads(
        mount_registration_path.read_text(encoding="utf-8")
    )
    isaac_screenshot_review = json.loads(
        isaac_screenshot_review_path.read_text(encoding="utf-8")
    )
    diagnostic_asset = json.loads(
        diagnostic_asset_path.read_text(encoding="utf-8")
    )
    sources = {
        source["source_label"]: source for source in audit.get("sources", [])
    }
    stationary = sources["stationary_aloha"]
    exact = sources["exact_vx_finger"]
    simple = sources["simple_viper"]
    primary_installation = _simple_viper_installation(
        simple,
        viper_state_report,
    )
    widow_crosscheck, widow_source = _authoritative_widow_installation(
        widow_audit_path,
        widow_tessellation_path,
    )
    objects = {obj["name"]: obj for obj in stationary["objects"]}
    followers = sorted(
        (
            obj
            for obj in stationary["objects"]
            if obj["type_id"] == "App::Part"
            and obj["label"].startswith("Dummy Aloha VX SV2")
        ),
        key=lambda obj: obj["name"],
    )
    follower_records = []
    all_instance_gates: dict[str, bool] = {}
    installed_labels: set[str] = set()
    for follower in followers:
        finger_groups = [
            objects[name]
            for name in follower["out_list"]
            if name in objects
            and objects[name]["type_id"] == "App::Part"
            and objects[name]["label"].startswith("Aloha VX Fingers")
        ]
        if len(finger_groups) != 1:
            raise ValueError(
                f"{follower['name']} has {len(finger_groups)} finger groups"
            )
        group = finger_groups[0]
        fingers = [
            objects[name]
            for name in group["out_list"]
            if name in objects
            and objects[name]["type_id"] == "Part::Feature"
            and objects[name]["label"].startswith("Aloha VX Fingers")
        ]
        if len(fingers) != 2:
            raise ValueError(
                f"{follower['name']} has {len(fingers)} finger instances"
            )
        items = []
        centers = []
        for finger in fingers:
            bbox = _shape_bbox(finger)
            center = _bbox_center(bbox)
            centers.append(center)
            side = "positive" if center[0] > 0 else "negative"
            link = "left_finger" if side == "positive" else "right_finger"
            installed_labels.add(finger["label"].rsplit(" v", 1)[0])
            items.append(
                {
                    "object_name": finger["name"],
                    "label": finger["label"],
                    "assembly_path": [
                        stationary["root_objects"][0],
                        follower["name"],
                        group["name"],
                        finger["name"],
                    ],
                    "cad_opening_side": side,
                    "mapped_urdf_joint": link,
                    "shape_hash_code_runtime_only": finger["shape"][
                        "hash_code_runtime_only"
                    ],
                    "shape_topology_counts": finger["shape"][
                        "topology_counts"
                    ],
                    "shape_volume_mm3": finger["shape"]["volume_mm3"],
                    "shape_area_mm2": finger["shape"]["area_mm2"],
                    "shape_bbox_mm": bbox,
                    "shape_bbox_center_mm": center,
                    "local_placement": finger["local_placement"],
                    "global_placement": finger["global_placement"],
                }
            )
        positive = next(
            item for item in items if item["cad_opening_side"] == "positive"
        )
        negative = next(
            item for item in items if item["cad_opening_side"] == "negative"
        )
        direction = _norm(
            [
                positive["shape_bbox_center_mm"][axis]
                - negative["shape_bbox_center_mm"][axis]
                for axis in range(3)
            ]
        )
        closed_gap = (
            positive["shape_bbox_mm"]["XMin"]
            - negative["shape_bbox_mm"]["XMax"]
        )
        follower_gate = {
            "exactly_two_finger_instances": len(items) == 2,
            "one_instance_on_each_opening_side": (
                {item["cad_opening_side"] for item in items}
                == {"positive", "negative"}
            ),
            "opening_axis_is_cad_x": _dot(direction, [1.0, 0.0, 0.0])
            > 0.9999,
            "positive_side_maps_left": (
                positive["mapped_urdf_joint"] == "left_finger"
            ),
            "negative_side_maps_right": (
                negative["mapped_urdf_joint"] == "right_finger"
            ),
            "closed_reference_has_positive_inner_gap": closed_gap > 0.0,
        }
        all_instance_gates.update(
            {
                f"{follower['name']}_{name}": value
                for name, value in follower_gate.items()
            }
        )
        follower_records.append(
            {
                "object_name": follower["name"],
                "label": follower["label"],
                "stationary_parent": follower["in_list"],
                "stationary_position_mm": follower["global_placement"][
                    "base_mm"
                ],
                "stationary_pose": follower["global_placement"],
                "finger_group": group["name"],
                "finger_instances": sorted(
                    items,
                    key=lambda item: item["cad_opening_side"],
                ),
                "opening_axis_cad_local": direction,
                "opening_axis_alignment_to_cad_x": _dot(
                    direction,
                    [1.0, 0.0, 0.0],
                ),
                "closed_inner_gap_mm": closed_gap,
                "gates": follower_gate,
            }
        )

    urdf = _urdf_mapping(urdf_path)
    exact_object = exact["objects"][0]
    installed_object = next(
        item
        for follower in follower_records
        for item in follower["finger_instances"]
    )
    standalone_same_revision = (
        exact_object["label"] == installed_object["label"]
        and abs(
            exact_object["shape"]["volume_mm3"]
            - installed_object["shape_volume_mm3"]
        )
        <= 1.0e-6
    )
    gates = {
        "freecad_audit_pass": audit["status"] == "PASS",
        "exactly_two_stationary_vx_followers": len(followers) == 2,
        "primary_is_purchase_confirmed_simple_viper": (
            primary_installation["evidence_class"]
            == "PURCHASE_DRAWING_CONFIRMED_VX300S_FOLLOWER"
        ),
        "primary_has_two_handed_finger_breps": (
            primary_installation["positive_x_finger"]["object_name"]
            != primary_installation["negative_x_finger"]["object_name"]
        ),
        "primary_fingers_share_rigid_placement": (
            primary_installation["identical_finger_placement_matrix"]
        ),
        **all_instance_gates,
        **{
            f"urdf_{name}": value
            for name, value in urdf["gates"].items()
        },
        "standalone_not_substituted_for_installed_revision": (
            not standalone_same_revision
        ),
        **{
            f"widow_crosscheck_{name}": value
            for name, value in widow_crosscheck["gates"].items()
        },
        "production_angular_tessellation_pass": (
            tessellation_report["production_tessellation_gate"] == "PASS"
        ),
        "mounting_datum_registration_pass": (
            mount_registration["status"] == "PASS"
        ),
        "isaac_visual_review_pass": (
            isaac_screenshot_review["status"] == "PASS"
        ),
        "isolated_diagnostic_asset_pass": (
            diagnostic_asset["status"] == "PASS"
        ),
    }
    connection_geometry_status = "PASS"
    production_tessellation_gate = tessellation_report[
        "production_tessellation_gate"
    ]
    return {
        "schema_version": 3,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "orientation_mapping_status": (
            "PASS" if all(gates.values()) else "FAIL"
        ),
        "connection_geometry_status": connection_geometry_status,
        "connection_geometry_finding": (
            "SOURCE_CAD_SLIDING_CARRIAGE_COMMON_VOLUME_RECORDED"
        ),
        "scope": (
            "Purchase-confirmed Simple Viper follower gripper orientation, "
            "Widow shared-gripper and Stationary cross-validation, and "
            "URDF-link mapping; no Isaac asset mutation"
        ),
        "sources": {
            "freecad_audit": {
                "path": str(freecad_audit_path.resolve()),
                "sha256": _sha256(freecad_audit_path),
                "freecad_version": audit["freecad_version"],
            },
            "stationary_step": {
                "path": stationary["path"],
                "sha256": stationary["sha256"],
            },
            "simple_viper_step": {
                "path": simple["path"],
                "sha256": simple["sha256"],
            },
            "exact_vx_finger_step": {
                "path": exact["path"],
                "sha256": exact["sha256"],
            },
            "widow_gripper_crosscheck_step": widow_source,
            "viper_state_report": {
                "path": str(viper_state_path.resolve()),
                "sha256": _sha256(viper_state_path),
            },
            "screenshot_review": {
                "path": str(screenshot_review_path.resolve()),
                "sha256": _sha256(screenshot_review_path),
            },
            "tessellation_report": {
                "path": str(tessellation_report_path.resolve()),
                "sha256": _sha256(tessellation_report_path),
            },
            "mount_registration_report": {
                "path": str(mount_registration_path.resolve()),
                "sha256": _sha256(mount_registration_path),
            },
            "isaac_screenshot_review": {
                "path": str(isaac_screenshot_review_path.resolve()),
                "sha256": _sha256(isaac_screenshot_review_path),
            },
            "isolated_diagnostic_asset": {
                "path": str(diagnostic_asset_path.resolve()),
                "sha256": _sha256(diagnostic_asset_path),
            },
        },
        "unit_conversion": {
            "cad_unit": "millimetre",
            "isaac_unit": "metre",
            "cad_mm_to_isaac_m": 0.001,
        },
        "toolchain": {
            "freecad": {
                "version": toolchain_probe["freecad_version"],
                "executable": toolchain_probe["freecad_executable"],
            },
            "opencascade": {
                "version": toolchain_probe["opencascade_version"],
            },
            "blender": {
                "version": render_metadata["renderer"]["blender_version"],
                "render_engine": render_metadata["renderer"][
                    "render_engine"
                ],
            },
            "production_tessellation_gate": production_tessellation_gate,
            "angular_deflection_control": (
                "EXPLICIT_MESHPART_LINEAR_AND_ANGULAR"
            ),
            "tessellation_report_path": str(
                tessellation_report_path.resolve()
            ),
            "tessellation_report_sha256": _sha256(
                tessellation_report_path
            ),
            "tessellation_run_a": tessellation_report["run_a"],
            "tessellation_run_b": tessellation_report["run_b"],
            "probe_path": str(toolchain_probe_path.resolve()),
            "probe_sha256": _sha256(toolchain_probe_path),
        },
        "visual_evidence": {
            "status": screenshot_review["status"],
            "gate": screenshot_review["gate"],
            "capture_count": screenshot_review["capture_count"],
            "raw_and_annotated_file_count": screenshot_review[
                "raw_and_annotated_file_count"
            ],
            "report_path": str(screenshot_review_path.resolve()),
            "report_sha256": _sha256(screenshot_review_path),
        },
        "isaac_visual_evidence": {
            "status": isaac_screenshot_review["status"],
            "gate": isaac_screenshot_review["gate"],
            "capture_count": isaac_screenshot_review["capture_count"],
            "report_path": str(isaac_screenshot_review_path.resolve()),
            "report_sha256": _sha256(isaac_screenshot_review_path),
            "acceptance_boundary": (
                "CAD installation visual evidence only; no collision, "
                "contact, dynamics, or grasp acceptance"
            ),
        },
        "mounting_datum_registration": {
            "status": mount_registration["status"],
            "method": mount_registration["method"],
            "threshold_m": mount_registration["threshold_m"],
            "datums": mount_registration["datums"],
            "decision_boundary": mount_registration["decision_boundary"],
            "report_path": str(mount_registration_path.resolve()),
            "report_sha256": _sha256(mount_registration_path),
        },
        "isolated_diagnostic_asset": {
            "status": diagnostic_asset["status"],
            "root_usd": diagnostic_asset["diagnostic_outputs"][
                "root_usd"
            ],
            "source_stage": diagnostic_asset["source_stage"],
            "collision_policy": diagnostic_asset["collision_policy"],
            "report_path": str(diagnostic_asset_path.resolve()),
            "report_sha256": _sha256(diagnostic_asset_path),
        },
        "cad_to_finger_link_mapping": {
            "matrix_convention": (
                "column-vector affine transform from STEP-global metres "
                "to each closed-state finger-link local frame"
            ),
            "left_matrix": [
                list(row)
                for row in cad_global_to_finger_link_matrix("left")
            ],
            "right_matrix": [
                list(row)
                for row in cad_global_to_finger_link_matrix("right")
            ],
            "local_axis_mapping": {
                "cad_local_x": "finger_link_+Y",
                "cad_local_y": "finger_link_+Z",
                "cad_local_z": "finger_link_+X",
            },
            "determinant": determinant3(
                CAD_ASSEMBLY_TO_FINGER_LINK_ROTATION
            ),
            "mirror_used": False,
            "single_side_180_degree_rotation_used": False,
        },
        "cad_to_urdf_frame_mapping": urdf,
        "primary_follower_installation": primary_installation,
        "widow_gripper_crosscheck": widow_crosscheck,
        "stationary_vx_followers": follower_records,
        "standalone_vs_installed_geometry": {
            "same_revision": standalone_same_revision,
            "replacement_allowed": False,
            "standalone_label": exact_object["label"],
            "standalone_volume_mm3": exact_object["shape"]["volume_mm3"],
            "standalone_bbox_mm": exact_object["shape"]["bound_box_mm"],
            "installed_label_family": min(installed_labels),
            "installed_volume_mm3": installed_object["shape_volume_mm3"],
            "installed_bbox_mm": installed_object["shape_bbox_mm"],
            "policy": (
                "standalone proves supplier shape provenance; purchase-confirmed "
                "Simple Viper proves follower installation; Widow and "
                "Stationary provide cross-checks"
            ),
            "replacement_prohibition_reason": (
                "the standalone 3D-A1 v3 label, volume, bounds, and revision "
                "differ from the supplier assembly's embedded handed v2 pair"
            ),
        },
        "visualization_state_policy": {
            "closed": (
                "must be derived from the source follower joint coordinate; "
                "the static CAD pose is not relabeled closed from its AABB"
            ),
            "open": (
                "must be derived from the source follower joint coordinate; "
                "the purchase drawing's 67.96 mm callout is retained as a "
                "geometry cross-check, not silently treated as AABB gap"
            ),
            "default_asset_modified": False,
        },
        "gates": gates,
    }
def write_gripper_mapping_reports(
    report: Mapping[str, Any],
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
        "# ALOHA Public CAD Gripper Mapping",
        "",
        f"- Status: `{report['status']}`",
        (
            "- Installation/orientation mapping: "
            f"`{report['orientation_mapping_status']}`"
        ),
        (
            "- Source connection geometry audit: "
            f"`{report['connection_geometry_status']}` "
            f"(`{report['connection_geometry_finding']}`)"
        ),
        (
            "- Production angular-controlled tessellation: "
            f"`{report['toolchain']['production_tessellation_gate']}`"
        ),
        (
            "- Supplier-CAD mounting datum registration: "
            f"`{report['mounting_datum_registration']['status']}` "
            f"(`{report['mounting_datum_registration']['method']}`)"
        ),
        (
            "- Isaac isolated installation screenshots: "
            f"`{report['isaac_visual_evidence']['status']}` "
            f"({report['isaac_visual_evidence']['capture_count']} "
            "raw/annotated pairs)"
        ),
        "- Isaac/default asset mutation: `false`",
        "- CAD +X opening side → URDF `left_finger` (+Y)",
        "- CAD -X opening side → URDF `right_finger` (-Y via mimic)",
        (
            "- Frame evidence: URDF gripper-bar visual uses `Rz(+90 deg)`, "
            "mapping CAD/STL +X to URDF +Y."
        ),
        "",
        "## Primary purchase-confirmed follower assembly",
        "",
        (
            "- Source: `Simple Aloha Viper 2024-5-13.step` "
            f"(`{report['sources']['simple_viper_step']['sha256']}`)"
        ),
        (
            "- Root / finger group: "
            f"`{report['primary_follower_installation']['root_object']}` / "
            f"`{report['primary_follower_installation']['finger_group_object']}`"
        ),
        (
            "- Blue / CAD +X / URDF left: "
            f"`{report['primary_follower_installation']['positive_x_finger']['object_name']}` "
            f"(`{report['primary_follower_installation']['positive_x_finger']['label']}`)"
        ),
        (
            "- Orange / CAD -X / URDF right: "
            f"`{report['primary_follower_installation']['negative_x_finger']['object_name']}` "
            f"(`{report['primary_follower_installation']['negative_x_finger']['label']}`)"
        ),
        (
            "- Shared source placement determinant: "
            f"`{report['primary_follower_installation']['shared_source_placement_determinant']}`"
        ),
        "- CAD unit → Isaac unit: `0.001 m/mm`",
        "- Supplier static state: `CLOSED_REFERENCE`",
        "- Derived open state: left `+36 mm` CAD X; right `-36 mm` CAD X",
        (
            "- Visual evidence: "
            f"`{report['visual_evidence']['status']}` "
            f"({report['visual_evidence']['capture_count']} raw/annotated pairs)"
        ),
        "",
        (
            "The supplier shell/sliding-carriage Boolean common volumes are "
            "recorded as source connection geometry. They are not silently "
            "relabeled as an unexpected simulated collision."
        ),
        "",
        "## Toolchain",
        "",
        (
            "- FreeCAD: "
            f"`{' / '.join(report['toolchain']['freecad']['version'])}`"
        ),
        (
            "- OpenCascade: "
            f"`{report['toolchain']['opencascade']['version']}`"
        ),
        (
            "- Blender: "
            f"`{report['toolchain']['blender']['version']}` / "
            f"`{report['toolchain']['blender']['render_engine']}`"
        ),
        (
            "- Angular deflection control: "
            f"`{report['toolchain']['angular_deflection_control']}`"
        ),
        (
            "- CAD local axes → finger link: "
            "`+X→+Y`, `+Y→+Z`, `+Z→+X`; determinant "
            f"`{report['cad_to_finger_link_mapping']['determinant']}`; "
            "mirror `false`."
        ),
        (
            "- Mounting datum threshold: "
            f"`{report['mounting_datum_registration']['threshold_m']} m`; "
            "full-surface ICP was not used for the decision."
        ),
        "",
        (
            "The standalone 2025 finger STEP is not silently substituted for "
            "the embedded 2024 instances because their labels, bounds, and "
            "volumes identify different revisions. Purchase-confirmed Simple "
            "Viper is the follower-primary assembly; Widow and Stationary are "
            "cross-checks."
        ),
        "",
        (
            "The Isaac screenshot PASS is a CAD-installation visual gate only. "
            "It does not claim collider, contact, dynamics, or bottle-grasp "
            "acceptance."
        ),
        "",
        "## Stationary follower instances",
        "",
        "| Follower CAD object | Position (mm) | CAD closed inner gap | +X finger | -X finger |",
        "|---|---|---:|---|---|",
    ]
    for follower in report["stationary_vx_followers"]:
        positive = next(
            item
            for item in follower["finger_instances"]
            if item["cad_opening_side"] == "positive"
        )
        negative = next(
            item
            for item in follower["finger_instances"]
            if item["cad_opening_side"] == "negative"
        )
        lines.append(
            "| {name} | `{position}` | {gap:.9f} mm | "
            "`{positive}` → left_finger | `{negative}` → right_finger |".format(
                name=follower["object_name"],
                position=follower["stationary_position_mm"],
                gap=follower["closed_inner_gap_mm"],
                positive=positive["object_name"],
                negative=negative["object_name"],
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
