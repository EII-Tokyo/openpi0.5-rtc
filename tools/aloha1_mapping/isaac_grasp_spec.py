from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from .task_frames import rigid_transform
from .task_frames import validate_rigid_transform

_TOP_LEVEL_FIELDS = {
    "format",
    "format_version",
    "object_frame",
    "gripper_frame",
    "grasps",
}
_GRASP_FIELDS = {
    "confidence",
    "position",
    "orientation",
    "cspace_position",
    "pregrasp_cspace_position",
}
_ORIENTATION_FIELDS = {"w", "xyz"}
_FINGER_NAMES = {"left_finger", "right_finger"}


class _StrictSafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader,
    node: yaml.MappingNode,
    *,
    deep: bool = False,
) -> dict[Any, Any]:
    keys: list[Any] = []
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=False)
        if key in keys:
            raise ValueError(f"duplicate YAML key: {key!r}")
        keys.append(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _require_mapping(value: Any, *, label: str) -> Mapping[Any, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_exact_fields(
    value: Any,
    expected: set[str],
    *,
    label: str,
) -> Mapping[Any, Any]:
    mapping = _require_mapping(value, label=label)
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected, key=str)
        raise ValueError(f"{label} fields must be exact; missing={missing}, unknown={unknown}")
    return mapping


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _finite_vector(value: Any, *, length: int, label: str) -> list[float]:
    if not isinstance(value, list | tuple) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} finite numbers")
    return [_finite_float(item, label=f"{label}[{index}]") for index, item in enumerate(value)]


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _finger_state(value: Any, *, label: str) -> dict[str, float]:
    mapping = _require_mapping(value, label=label)
    if set(mapping) != _FINGER_NAMES:
        raise ValueError(f"{label} requires exact left_finger/right_finger states")
    return {
        "left_finger": _finite_float(mapping["left_finger"], label=f"{label}.left_finger"),
        "right_finger": _finite_float(mapping["right_finger"], label=f"{label}.right_finger"),
    }


@dataclass(frozen=True)
class IsaacGrasp:
    name: str
    confidence: float
    object_from_gripper: np.ndarray
    cspace_position: dict[str, float]
    pregrasp_cspace_position: dict[str, float]


@dataclass(frozen=True)
class IsaacGraspFile:
    object_frame: str
    gripper_frame: str
    grasps: dict[str, IsaacGrasp]

    @classmethod
    def load(cls, path: Path) -> IsaacGraspFile:
        try:
            data = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_StrictSafeLoader)
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid Isaac grasp YAML: {exc}") from exc
        top_level = _require_exact_fields(data, _TOP_LEVEL_FIELDS, label="top-level")
        if top_level["format"] != "isaac_grasp":
            raise ValueError("expected isaac_grasp format")
        version = _finite_float(top_level["format_version"], label="format_version")
        if version != 1.0:
            raise ValueError("expected isaac_grasp format_version 1.0")

        object_frame = _nonempty_string(top_level["object_frame"], label="object_frame")
        gripper_frame = _nonempty_string(top_level["gripper_frame"], label="gripper_frame")
        grasp_records = _require_mapping(top_level["grasps"], label="grasps")
        if "horizontal_body_grasp" not in grasp_records:
            raise ValueError("missing horizontal_body_grasp")

        grasps: dict[str, IsaacGrasp] = {}
        for raw_name, raw_record in grasp_records.items():
            name = _nonempty_string(raw_name, label="grasp name")
            record = _require_exact_fields(raw_record, _GRASP_FIELDS, label=f"{name} fields")
            confidence = _finite_float(record["confidence"], label=f"{name}.confidence")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"{name}.confidence must be between 0 and 1")
            position = _finite_vector(record["position"], length=3, label=f"{name}.position")
            orientation = _require_exact_fields(
                record["orientation"],
                _ORIENTATION_FIELDS,
                label="orientation fields",
            )
            quaternion_xyzw = [
                *_finite_vector(orientation["xyz"], length=3, label=f"{name}.orientation.xyz"),
                _finite_float(orientation["w"], label=f"{name}.orientation.w"),
            ]
            if not np.isclose(np.linalg.norm(quaternion_xyzw), 1.0, rtol=0.0, atol=1e-10):
                raise ValueError(f"{name}.orientation must be a unit quaternion")
            object_from_gripper = rigid_transform(
                Rotation.from_quat(quaternion_xyzw).as_matrix(),
                position,
            )
            grasps[name] = IsaacGrasp(
                name=name,
                confidence=confidence,
                object_from_gripper=object_from_gripper,
                cspace_position=_finger_state(record["cspace_position"], label=f"{name}.cspace_position"),
                pregrasp_cspace_position=_finger_state(
                    record["pregrasp_cspace_position"],
                    label=f"{name}.pregrasp_cspace_position",
                ),
            )
        return cls(
            object_frame=object_frame,
            gripper_frame=gripper_frame,
            grasps=grasps,
        )

    def grasp(self, name: str) -> IsaacGrasp:
        try:
            return self.grasps[name]
        except KeyError as exc:
            raise KeyError(f"unknown grasp: {name}") from exc

    def to_dict(self) -> dict[str, Any]:
        if "horizontal_body_grasp" not in self.grasps:
            raise ValueError("missing horizontal_body_grasp")
        records: dict[str, Any] = {}
        for name in sorted(self.grasps):
            grasp = self.grasps[name]
            matrix = validate_rigid_transform(grasp.object_from_gripper)
            quaternion_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat(canonical=True)
            confidence = _finite_float(grasp.confidence, label=f"{name}.confidence")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"{name}.confidence must be between 0 and 1")
            records[name] = {
                "confidence": confidence,
                "position": [float(value) for value in matrix[:3, 3]],
                "orientation": {
                    "w": float(quaternion_xyzw[3]),
                    "xyz": [float(value) for value in quaternion_xyzw[:3]],
                },
                "cspace_position": _finger_state(
                    grasp.cspace_position,
                    label=f"{name}.cspace_position",
                ),
                "pregrasp_cspace_position": _finger_state(
                    grasp.pregrasp_cspace_position,
                    label=f"{name}.pregrasp_cspace_position",
                ),
            }
        return {
            "format": "isaac_grasp",
            "format_version": 1.0,
            "object_frame": _nonempty_string(self.object_frame, label="object_frame"),
            "gripper_frame": _nonempty_string(self.gripper_frame, label="gripper_frame"),
            "grasps": records,
        }

    def write(self, path: Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(
            self.to_dict(),
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=False,
            line_break="\n",
        )
        output_path.write_text(text, encoding="utf-8", newline="\n")
