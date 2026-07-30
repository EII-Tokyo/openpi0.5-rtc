from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
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
        return _verify_hash(path or self.stage.path, self.stage.sha256, "Stage")

    def verify_bottle(self, path: Path | None = None) -> Path:
        return _verify_hash(path or self.bottle.usd_path, self.bottle.sha256, "Bottle")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_new_output_path(output: Path, approved_root: Path) -> Path:
    resolved_root = approved_root.resolve()
    resolved_output = output.resolve()
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError as error:
        raise ManifestError(
            f"Output path is outside approved root: {resolved_output}"
        ) from error
    if resolved_output.exists():
        raise ManifestError(f"Output path already exists: {resolved_output}")
    return resolved_output


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

    try:
        isaac = _require_mapping(data["isaac"], "isaac")
        stage = _require_mapping(data["stage"], "stage")
        prims = _require_mapping(data["prims"], "prims")
        bottle = _require_mapping(data["bottle"], "bottle")
        variant_b = _require_mapping(data["variant_b"], "variant_b")
        output = _require_mapping(data["output"], "output")
        status = _require_mapping(data["status"], "status")
        return ApprovedGraspEditorManifest(
            isaac=IsaacConfig(
                version=str(isaac["version"]),
                kit=str(isaac["kit"]),
                physx=str(isaac["physx"]),
                grasp_editor_extension=str(isaac["grasp_editor_extension"]),
                grasp_editor_version=str(isaac["grasp_editor_version"]),
            ),
            stage=StageConfig(
                path=Path(str(stage["path"])),
                sha256=str(stage["sha256"]),
            ),
            prims=PrimConfig(
                articulation=str(prims["articulation"]),
                gripper_frame=str(prims["gripper_frame"]),
                object=str(prims["object"]),
            ),
            bottle=BottleConfig(
                usd_path=Path(str(bottle["usd_path"])),
                sha256=str(bottle["sha256"]),
                body_coordinate_mm=float(bottle["body_coordinate_mm"]),
            ),
            variant_b=VariantBConfig(
                active_joint=str(variant_b["active_joint"]),
                observer_joint=str(variant_b["observer_joint"]),
                open_position_m=float(variant_b["open_position_m"]),
                closed_position_m=float(variant_b["closed_position_m"]),
                observer_setup_position_m=float(
                    variant_b["observer_setup_position_m"]
                ),
                max_speed_m_s=float(variant_b["max_speed_m_s"]),
                max_effort_n=float(variant_b["max_effort_n"]),
            ),
            output=OutputConfig(root=Path(str(output["root"]))),
            status=StatusConfig(
                ik=str(status["ik"]),
                task8=str(status["task8"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ManifestError(f"Invalid manifest value: {error}") from error


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{context} must be a mapping")
    return value


def _verify_hash(path: Path, expected: str, label: str) -> Path:
    resolved = path.resolve()
    try:
        actual = sha256_file(resolved)
    except OSError as error:
        raise ManifestError(f"Unable to hash {label} at {resolved}: {error}") from error
    if actual != expected:
        raise ManifestError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return resolved
