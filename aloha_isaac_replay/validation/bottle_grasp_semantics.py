from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


BOTTLE_LENGTH_M = 0.206
BOTTLE_RADIUS_M = 0.034
BOTTLE_LONG_AXIS_OBJECT = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
BOTTLE_SIDE_APPROACH_OBJECT = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
WORLD_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}
ALOHA_LEFT_GRIPPER_APPROACH_AXIS = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
ALOHA_LEFT_GRIPPER_CLOSING_AXIS = np.asarray(
    [-8.425203887821992e-07, 0.9999996832263347, 0.0007959563558607434],
    dtype=np.float64,
)
ALOHA_LEFT_GRIPPER_FINGER_MIDPOINT = np.asarray(
    [0.056161419262861445, -4.948896925776984e-06, 0.014180531657344238],
    dtype=np.float64,
)


def _normalize(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError(f"cannot normalize near-zero vector: {value}")
    return np.asarray(value, dtype=np.float64) / norm


def quat_wxyz_to_matrix(quat: list[float] | np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    norm = float(np.linalg.norm([w, x, y, z]))
    if norm < 1e-12:
        raise ValueError("grasp quaternion is near zero")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _iter_grasps(data: dict[str, Any]):
    grasps = data.get("grasps") or {}
    if isinstance(grasps, dict):
        for name, grasp in grasps.items():
            row = dict(grasp)
            row["name"] = name
            yield row
        return
    for grasp in grasps:
        yield grasp


def load_grasps(path: str | Path) -> dict[str, Any]:
    grasp_path = Path(path)
    data = yaml.safe_load(grasp_path.read_text(encoding="utf-8")) or {}
    return data


def evaluate_grasp_semantics(
    grasp: dict[str, Any],
    *,
    bottle_length_m: float = BOTTLE_LENGTH_M,
    bottle_radius_m: float = BOTTLE_RADIUS_M,
    rear_fraction_target: float = 0.25,
    rear_fraction_tolerance: float = 0.07,
    max_closing_long_axis_dot: float = 0.20,
    min_side_approach_dot: float = 0.70,
) -> dict[str, Any]:
    position = np.asarray(grasp["position"], dtype=np.float64)
    quat = np.asarray([grasp["orientation"]["w"], *grasp["orientation"]["xyz"]], dtype=np.float64)
    rotation_object_gripper = quat_wxyz_to_matrix(quat)
    approach_object = _normalize(rotation_object_gripper @ _normalize(ALOHA_LEFT_GRIPPER_APPROACH_AXIS))
    closing_object = _normalize(rotation_object_gripper @ _normalize(ALOHA_LEFT_GRIPPER_CLOSING_AXIS))
    finger_midpoint_object = position + rotation_object_gripper @ ALOHA_LEFT_GRIPPER_FINGER_MIDPOINT
    long_axis = _normalize(BOTTLE_LONG_AXIS_OBJECT)
    side_approach = _normalize(BOTTLE_SIDE_APPROACH_OBJECT)

    rear_fraction = float(finger_midpoint_object[2] / bottle_length_m)
    rear_error = abs(rear_fraction - float(rear_fraction_target))
    closing_long_axis_dot_abs = abs(float(np.dot(closing_object, long_axis)))
    approach_long_axis_dot_abs = abs(float(np.dot(approach_object, long_axis)))
    side_approach_dot = float(np.dot(approach_object, side_approach))
    finger_midpoint_radial_distance = float(np.linalg.norm(finger_midpoint_object[:2]))

    rear_quarter_ok = rear_error <= float(rear_fraction_tolerance)
    closing_perpendicular_ok = closing_long_axis_dot_abs <= float(max_closing_long_axis_dot)
    approach_side_ok = side_approach_dot >= float(min_side_approach_dot)
    approach_perpendicular_ok = approach_long_axis_dot_abs <= float(max_closing_long_axis_dot)
    finger_midpoint_near_centerline_ok = finger_midpoint_radial_distance <= bottle_radius_m * 0.25
    pass_gate = bool(
        rear_quarter_ok
        and closing_perpendicular_ok
        and approach_side_ok
        and approach_perpendicular_ok
        and finger_midpoint_near_centerline_ok
    )
    return {
        "name": grasp.get("name"),
        "position_m": position.tolist(),
        "finger_midpoint_object_m": finger_midpoint_object.tolist(),
        "rear_fraction_from_bottom": rear_fraction,
        "rear_fraction_target": float(rear_fraction_target),
        "rear_fraction_tolerance": float(rear_fraction_tolerance),
        "rear_quarter_ok": rear_quarter_ok,
        "closing_axis_object": closing_object.tolist(),
        "closing_long_axis_dot_abs": closing_long_axis_dot_abs,
        "closing_perpendicular_ok": closing_perpendicular_ok,
        "approach_axis_object": approach_object.tolist(),
        "approach_long_axis_dot_abs": approach_long_axis_dot_abs,
        "approach_side_dot": side_approach_dot,
        "approach_side_ok": approach_side_ok,
        "approach_perpendicular_ok": approach_perpendicular_ok,
        "finger_midpoint_radial_distance_m": finger_midpoint_radial_distance,
        "bottle_radius_m": float(bottle_radius_m),
        "finger_midpoint_near_centerline_ok": finger_midpoint_near_centerline_ok,
        "pass": pass_gate,
    }


def evaluate_grasp_file(path: str | Path, *, selected_grasp: str | None = None) -> dict[str, Any]:
    data = load_grasps(path)
    rows = [evaluate_grasp_semantics(grasp) for grasp in _iter_grasps(data)]
    selected_row = None
    if selected_grasp is not None:
        selected_row = next((row for row in rows if row["name"] == selected_grasp), None)
        if selected_row is None:
            raise ValueError(f"selected grasp {selected_grasp!r} not found in {path}")
    return {
        "path": str(Path(path).resolve()),
        "object_frame": data.get("object_frame"),
        "gripper_frame": data.get("gripper_frame"),
        "bottle_semantics": {
            "long_axis_object": BOTTLE_LONG_AXIS_OBJECT.tolist(),
            "bottom_z_m": 0.0,
            "mouth_z_m": BOTTLE_LENGTH_M,
            "rear_quarter_target_fraction_from_bottom": 0.25,
            "radius_m": BOTTLE_RADIUS_M,
            "aloha_left_gripper_finger_midpoint_m": ALOHA_LEFT_GRIPPER_FINGER_MIDPOINT.tolist(),
        },
        "selected_grasp": selected_grasp,
        "selected_grasp_pass": selected_row["pass"] if selected_row is not None else None,
        "all_grasps": rows,
        "pass": all(row["pass"] for row in rows) if selected_grasp is None else bool(selected_row["pass"]),
    }


def evaluate_axis_aligned_finger_rear_quarter(
    *,
    finger_contact_center_world: list[float] | np.ndarray,
    object_bbox: dict[str, Any],
    object_axis: str,
    finger_gap_axis: str,
    finger_gap_axis_vector_world: list[float] | np.ndarray | None = None,
    rear_fraction_target: float = 0.25,
    rear_fraction_tolerance: float = 0.07,
    max_closing_long_axis_dot: float = 0.20,
) -> dict[str, Any]:
    """Validate a collision-aware ALOHA grasp placement from runtime fingertip bboxes.

    This gate is intentionally different from a Grasp Editor transform check.
    It validates the placement used by the dynamic replay smoke tests: the
    midpoint between the two finger collision proxies should land on the bottle
    body near the rear quarter, while the closing/gap axis should be
    perpendicular to the bottle long axis.
    """

    axis_name = object_axis.upper()
    gap_axis_name = finger_gap_axis.upper()
    if axis_name not in WORLD_AXIS_INDEX:
        raise ValueError(f"unsupported object axis: {object_axis!r}")
    if gap_axis_name not in WORLD_AXIS_INDEX:
        raise ValueError(f"unsupported finger gap axis: {finger_gap_axis!r}")
    if not object_bbox.get("bbox_valid"):
        return {
            "pass": False,
            "status": "FAIL_INVALID_OBJECT_BBOX",
            "object_axis": axis_name,
            "finger_gap_axis": gap_axis_name,
        }

    axis = WORLD_AXIS_INDEX[axis_name]
    bbox_min = np.asarray(object_bbox["min"], dtype=np.float64)
    bbox_max = np.asarray(object_bbox["max"], dtype=np.float64)
    contact = np.asarray(finger_contact_center_world, dtype=np.float64)
    length = float(bbox_max[axis] - bbox_min[axis])
    if length <= 1e-9 or not np.isfinite(length):
        return {
            "pass": False,
            "status": "FAIL_INVALID_OBJECT_AXIS_LENGTH",
            "object_axis": axis_name,
            "finger_gap_axis": gap_axis_name,
            "object_axis_length_m": length,
        }

    fraction_from_axis_min = float((contact[axis] - bbox_min[axis]) / length)
    rear_error = abs(fraction_from_axis_min - float(rear_fraction_target))
    rear_quarter_ok = bool(rear_error <= float(rear_fraction_tolerance))
    object_axis_vector = np.zeros(3, dtype=np.float64)
    object_axis_vector[axis] = 1.0
    if finger_gap_axis_vector_world is None:
        gap_axis = WORLD_AXIS_INDEX[gap_axis_name]
        finger_gap_vector = np.zeros(3, dtype=np.float64)
        finger_gap_vector[gap_axis] = 1.0
        gap_vector_source = "axis_name_fallback"
    else:
        finger_gap_vector = _normalize(np.asarray(finger_gap_axis_vector_world, dtype=np.float64))
        gap_vector_source = "finger_center_vector"
    closing_long_axis_dot_abs = abs(float(np.dot(finger_gap_vector, object_axis_vector)))
    closing_perpendicular_ok = bool(closing_long_axis_dot_abs <= float(max_closing_long_axis_dot))
    pass_gate = bool(rear_quarter_ok and closing_perpendicular_ok)
    return {
        "pass": pass_gate,
        "status": "PASS_FINGER_REAR_QUARTER_PLACEMENT"
        if pass_gate
        else "FAIL_FINGER_REAR_QUARTER_PLACEMENT",
        "placement_semantics": "finger_gap_center_on_bottle_rear_quarter",
        "object_axis": axis_name,
        "object_axis_index": axis,
        "finger_gap_axis": gap_axis_name,
        "finger_gap_axis_vector_world": finger_gap_vector.tolist(),
        "finger_gap_axis_vector_source": gap_vector_source,
        "finger_contact_center_world_m": contact.tolist(),
        "object_axis_min_m": float(bbox_min[axis]),
        "object_axis_max_m": float(bbox_max[axis]),
        "object_axis_length_m": length,
        "fraction_from_axis_min": fraction_from_axis_min,
        "rear_fraction_target": float(rear_fraction_target),
        "rear_fraction_tolerance": float(rear_fraction_tolerance),
        "rear_quarter_ok": rear_quarter_ok,
        "closing_long_axis_dot_abs": closing_long_axis_dot_abs,
        "max_closing_long_axis_dot": float(max_closing_long_axis_dot),
        "closing_perpendicular_ok": closing_perpendicular_ok,
    }


def write_semantic_report(result: dict[str, Any], output_json: str | Path, output_md: str | Path) -> None:
    json_path = Path(output_json)
    md_path = Path(output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Bottle Grasp Semantics Gate",
        "",
        f"- Grasp file: `{result['path']}`",
        f"- Object frame: `{result.get('object_frame')}`",
        f"- Gripper frame: `{result.get('gripper_frame')}`",
        f"- Selected grasp: `{result.get('selected_grasp')}`",
        f"- PASS: `{result['pass']}`",
        "",
        "## Checks",
        "",
        "The bottle local `+Z` axis is the bottle long axis from bottom toward mouth. "
        "The expected replay grasp places the ALOHA fingertip midpoint near the rear quarter of the bottle body, "
        "and the gripper closing axis must be perpendicular to the bottle long axis.",
        "",
        "| grasp | rear fraction | closing dot long-axis | approach side dot | pass |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in result["all_grasps"]:
        lines.append(
            f"| `{row['name']}` | {row['rear_fraction_from_bottom']:.3f} | "
            f"{row['closing_long_axis_dot_abs']:.3f} | {row['approach_side_dot']:.3f} | `{row['pass']}` |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
