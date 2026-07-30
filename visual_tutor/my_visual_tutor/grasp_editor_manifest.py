from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any

import yaml

SCHEMA_VERSION = "aloha1-grasp-editor-live-manifest/v1"
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "isaac",
        "stage",
        "prims",
        "bottle",
        "variant_b",
        "output",
        "status",
    }
)
_SECTION_KEYS = {
    "isaac": frozenset(
        {
            "version",
            "kit",
            "physx",
            "grasp_editor_extension",
            "grasp_editor_version",
        }
    ),
    "stage": frozenset({"path", "sha256"}),
    "prims": frozenset({"articulation", "gripper_frame", "object"}),
    "bottle": frozenset({"usd_path", "sha256", "body_coordinate_mm"}),
    "variant_b": frozenset(
        {
            "active_joint",
            "observer_joint",
            "open_position_m",
            "closed_position_m",
            "observer_setup_position_m",
            "max_speed_m_s",
            "max_effort_n",
        }
    ),
    "output": frozenset({"root"}),
    "status": frozenset({"ik", "task8"}),
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ManifestError(ValueError):
    """Raised when the approved Grasp Editor manifest contract is violated."""


@dataclass(frozen=True)
class IsaacConfig:
    version: str
    kit: str
    physx: str
    grasp_editor_extension: str
    grasp_editor_version: str


@dataclass(frozen=True)
class StageConfig:
    path: Path
    sha256: str


@dataclass(frozen=True)
class PrimConfig:
    articulation: str
    gripper_frame: str
    object: str


@dataclass(frozen=True)
class BottleConfig:
    usd_path: Path
    sha256: str
    body_coordinate_mm: float


@dataclass(frozen=True)
class VariantBConfig:
    active_joint: str
    observer_joint: str
    open_position_m: float
    closed_position_m: float
    observer_setup_position_m: float
    max_speed_m_s: float
    max_effort_n: float


@dataclass(frozen=True)
class OutputConfig:
    root: Path


@dataclass(frozen=True)
class StatusConfig:
    ik: str
    task8: str


@dataclass(frozen=True)
class ApprovedGraspEditorManifest:
    isaac: IsaacConfig
    stage: StageConfig
    prims: PrimConfig
    bottle: BottleConfig
    variant_b: VariantBConfig
    output: OutputConfig
    status: StatusConfig

    @property
    def stage_path(self) -> Path:
        return self.stage.path

    @property
    def stage_sha256(self) -> str:
        return self.stage.sha256

    @property
    def bottle_usd_path(self) -> Path:
        return self.bottle.usd_path

    @property
    def bottle_sha256(self) -> str:
        return self.bottle.sha256

    @property
    def active_joint(self) -> str:
        return self.variant_b.active_joint

    @property
    def observer_joint(self) -> str:
        return self.variant_b.observer_joint

    @property
    def open_position_m(self) -> float:
        return self.variant_b.open_position_m

    @property
    def closed_position_m(self) -> float:
        return self.variant_b.closed_position_m

    @property
    def max_speed_m_s(self) -> float:
        return self.variant_b.max_speed_m_s

    @property
    def max_effort_n(self) -> float:
        return self.variant_b.max_effort_n

    @property
    def output_root(self) -> Path:
        return self.output.root

    @property
    def ik_status(self) -> str:
        return self.status.ik

    def verify_stage(self, path: Path | None = None) -> Path:
        return _verify_hash(
            path if path is not None else self.stage.path,
            self.stage.path,
            self.stage.sha256,
            "Stage",
        )

    def verify_bottle(self, path: Path | None = None) -> Path:
        return _verify_hash(
            path if path is not None else self.bottle.usd_path,
            self.bottle.usd_path,
            self.bottle.sha256,
            "Bottle",
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_new_output_path(output: Path, approved_root: Path) -> Path:
    """Validate a local export target without creating it.

    This is a preflight check, not a defense against concurrent TOCTOU changes.
    Task 2 must call it again immediately before export while the target still
    does not exist.
    """
    if not approved_root.is_absolute():
        raise ManifestError("approved_root must be absolute")
    if approved_root.is_symlink() or approved_root.resolve() != approved_root:
        raise ManifestError("approved_root must not be a symlink")
    if not approved_root.is_dir():
        raise ManifestError("approved_root must be an existing directory")
    if not output.is_absolute():
        raise ManifestError("output must be absolute")
    try:
        output.relative_to(approved_root)
    except ValueError as error:
        raise ManifestError(f"Output path is outside approved root: {output}") from error
    if output.parent != approved_root:
        raise ManifestError("Output path must be directly inside approved_root")
    if output.parent.is_symlink() or output.parent.resolve() != approved_root:
        raise ManifestError("Output parent must not be a symlink")
    if output.exists() or output.is_symlink():
        raise ManifestError(f"Output path already exists: {output}")
    return output


def load_approved_manifest(path: Path) -> ApprovedGraspEditorManifest:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ManifestError(f"Unable to read manifest {path}: {error}") from error

    data = _require_mapping(raw, "manifest")
    unknown = set(data) - _TOP_LEVEL_KEYS
    if unknown:
        rendered = ", ".join(sorted(repr(key) for key in unknown))
        raise ManifestError(f"Unknown top-level manifest keys: {rendered}")
    missing = _TOP_LEVEL_KEYS - set(data)
    if missing:
        rendered = ", ".join(sorted(missing))
        raise ManifestError(f"Missing top-level manifest keys: {rendered}")
    if data["schema_version"] != SCHEMA_VERSION:
        raise ManifestError(
            f"Unsupported manifest schema: {data['schema_version']!r}"
        )

    isaac = _require_section(data, "isaac")
    stage = _require_section(data, "stage")
    prims = _require_section(data, "prims")
    bottle = _require_section(data, "bottle")
    variant_b = _require_section(data, "variant_b")
    output = _require_section(data, "output")
    status = _require_section(data, "status")
    return ApprovedGraspEditorManifest(
        isaac=IsaacConfig(
            version=_require_text(isaac, "isaac", "version"),
            kit=_require_text(isaac, "isaac", "kit"),
            physx=_require_text(isaac, "isaac", "physx"),
            grasp_editor_extension=_require_text(
                isaac,
                "isaac",
                "grasp_editor_extension",
            ),
            grasp_editor_version=_require_text(
                isaac,
                "isaac",
                "grasp_editor_version",
            ),
        ),
        stage=StageConfig(
            path=_require_absolute_path(stage, "stage", "path"),
            sha256=_require_sha256(stage, "stage", "sha256"),
        ),
        prims=PrimConfig(
            articulation=_require_text(prims, "prims", "articulation"),
            gripper_frame=_require_text(prims, "prims", "gripper_frame"),
            object=_require_text(prims, "prims", "object"),
        ),
        bottle=BottleConfig(
            usd_path=_require_absolute_path(bottle, "bottle", "usd_path"),
            sha256=_require_sha256(bottle, "bottle", "sha256"),
            body_coordinate_mm=_require_finite_number(
                bottle,
                "bottle",
                "body_coordinate_mm",
            ),
        ),
        variant_b=VariantBConfig(
            active_joint=_require_text(variant_b, "variant_b", "active_joint"),
            observer_joint=_require_text(
                variant_b,
                "variant_b",
                "observer_joint",
            ),
            open_position_m=_require_finite_number(
                variant_b,
                "variant_b",
                "open_position_m",
            ),
            closed_position_m=_require_finite_number(
                variant_b,
                "variant_b",
                "closed_position_m",
            ),
            observer_setup_position_m=_require_finite_number(
                variant_b,
                "variant_b",
                "observer_setup_position_m",
            ),
            max_speed_m_s=_require_finite_number(
                variant_b,
                "variant_b",
                "max_speed_m_s",
                positive=True,
            ),
            max_effort_n=_require_finite_number(
                variant_b,
                "variant_b",
                "max_effort_n",
                positive=True,
            ),
        ),
        output=OutputConfig(
            root=_require_absolute_path(output, "output", "root"),
        ),
        status=StatusConfig(
            ik=_require_not_run(status, "status", "ik"),
            task8=_require_not_run(status, "status", "task8"),
        ),
    )


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{context} must be a mapping")
    return value


def _require_section(
    data: Mapping[str, Any],
    section: str,
) -> Mapping[str, Any]:
    value = _require_mapping(data[section], section)
    expected = _SECTION_KEYS[section]
    unknown = set(value) - expected
    if unknown:
        field = sorted(str(key) for key in unknown)[0]
        raise ManifestError(f"{section}.{field}: unknown field")
    missing = expected - set(value)
    if missing:
        field = sorted(missing)[0]
        raise ManifestError(f"{section}.{field}: missing field")
    return value


def _require_text(
    data: Mapping[str, Any],
    section: str,
    field: str,
) -> str:
    value = data[field]
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{section}.{field}: expected nonempty string")
    return value


def _require_absolute_path(
    data: Mapping[str, Any],
    section: str,
    field: str,
) -> Path:
    value = _require_text(data, section, field)
    path = Path(value)
    if not path.is_absolute():
        raise ManifestError(f"{section}.{field}: expected absolute path")
    return path


def _require_sha256(
    data: Mapping[str, Any],
    section: str,
    field: str,
) -> str:
    value = _require_text(data, section, field)
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ManifestError(f"{section}.{field}: expected lowercase SHA-256")
    return value


def _require_finite_number(
    data: Mapping[str, Any],
    section: str,
    field: str,
    *,
    positive: bool = False,
) -> float:
    value = data[field]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ManifestError(f"{section}.{field}: expected number, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ManifestError(f"{section}.{field}: expected finite number")
    if positive and result <= 0:
        raise ManifestError(f"{section}.{field}: expected positive number")
    return result


def _require_not_run(
    data: Mapping[str, Any],
    section: str,
    field: str,
) -> str:
    value = _require_text(data, section, field)
    if value != "NOT_RUN":
        raise ManifestError(f"{section}.{field}: expected NOT_RUN")
    return value


def _verify_hash(
    path: Path,
    approved_path: Path,
    expected: str,
    label: str,
) -> Path:
    if (
        not path.is_absolute()
        or path != approved_path
        or path.is_symlink()
        or path.resolve() != approved_path
    ):
        raise ManifestError(
            f"{label} exact approved path mismatch: expected {approved_path}, got {path}"
        )
    try:
        actual = sha256_file(approved_path)
    except OSError as error:
        raise ManifestError(
            f"Unable to hash {label} at {approved_path}: {error}"
        ) from error
    if actual != expected:
        raise ManifestError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return approved_path
