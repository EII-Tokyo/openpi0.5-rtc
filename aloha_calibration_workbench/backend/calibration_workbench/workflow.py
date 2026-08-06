from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from .models import FactoryIntrinsics
from .models import ProductionProfile


class WorkflowGateError(RuntimeError):
    """A fail-closed calibration acceptance gate did not pass."""


class TransformRecord(BaseModel):
    """A matrix that maps homogeneous points from source_frame to target_frame."""

    source_frame: str
    target_frame: str
    matrix: list[list[float]]
    length_unit: Literal["meter"] = "meter"
    matrix_order: Literal["row-major"] = "row-major"
    vector_convention: Literal["column-vector"] = "column-vector"
    quaternion_order: Literal["wxyz"] = "wxyz"

    @field_validator("matrix")
    @classmethod
    def validate_rigid_matrix(cls, value: list[list[float]]) -> list[list[float]]:
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("transform must be a finite 4x4 matrix")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError("transform must have homogeneous final row [0,0,0,1]")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("transform rotation must be orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
            raise ValueError("transform rotation must be right-handed")
        return matrix.tolist()

    def array(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=np.float64)


class TagPoseSample(BaseModel):
    camera_from_tag: TransformRecord
    reprojection_rms_px: float = Field(ge=0.0)
    frame_id: str | None = None
    device_timestamp_ms: float | None = None
    image_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class WorldOriginResult(BaseModel):
    status: Literal["WORLD_ORIGIN_SOLVED"]
    world_from_camera: TransformRecord
    accepted_frames: int
    total_frames: int
    median_reprojection_rms_px: float
    p95_reprojection_rms_px: float
    translation_jitter_m: float
    rotation_jitter_deg: float


class TagPoseStabilityResult(BaseModel):
    camera_from_tag: TransformRecord
    accepted_frames: int
    total_frames: int
    median_reprojection_rms_px: float
    p95_reprojection_rms_px: float
    translation_jitter_m: float
    rotation_jitter_deg: float


class DotObservation(BaseModel):
    id: str
    color: Literal["blue", "magenta", "lime"]
    partition: Literal["SOLVE", "HELD_OUT"]
    world_xyz_m: tuple[float, float, float]
    image_uv_px: tuple[float, float]
    repeated_measurement_delta_m: float = Field(ge=0.0)
    operator_confirmed: bool = True


class TablePointMeasurement(BaseModel):
    id: str
    color: Literal["blue", "magenta", "lime"]
    measurement_1_xy_m: tuple[float, float]
    measurement_2_xy_m: tuple[float, float]


class TablePointContractRequest(BaseModel):
    contract_id: str = Field(pattern=r"^[a-z0-9-]{3,64}$")
    revision: int = Field(ge=1)
    measurement_method: Literal["steel-ruler-and-square"]
    points: list[TablePointMeasurement]


class TablePointTruth(BaseModel):
    id: str
    color: Literal["blue", "magenta", "lime"]
    partition: Literal["SOLVE", "HELD_OUT"]
    world_xyz_m: tuple[float, float, float]
    repeated_measurement_delta_m: float


class FrozenTablePointContract(BaseModel):
    status: Literal["TABLE_POINT_CONTRACT_FROZEN"] = "TABLE_POINT_CONTRACT_FROZEN"
    contract_id: str
    revision: int
    measurement_method: str
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    points: list[TablePointTruth]


class DotPixelObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    image_uv_px: tuple[float, float]
    operator_confirmed: bool


class TableObservationsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: list[DotPixelObservation]


class TableRegistrationResult(BaseModel):
    status: Literal["WORLD_REGISTRATION_VALIDATED"]
    validation_scope: Literal["tabletop-xy-cross-validation"]
    world_from_camera: TransformRecord
    solve_point_ids: list[str]
    held_out_point_ids: list[str]
    solve_reprojection_rms_px: float
    initial_reprojection_rms_px: float
    refinement_translation_m: float
    refinement_rotation_deg: float
    held_out_rms_m: float
    held_out_max_m: float


class BottleFixture(BaseModel):
    measured_length_m: float = Field(gt=0.0)
    measured_diameter_m: float = Field(gt=0.0)
    tag_from_bottle: TransformRecord
    asset_from_task: TransformRecord


class BottleFixtureContractRequest(BaseModel):
    fixture_id: str = Field(pattern=r"^[a-z0-9-]{3,64}$")
    revision: int = Field(ge=1)
    measured_length_m: float = Field(gt=0.0)
    measured_diameter_m: float = Field(gt=0.0)
    tag_from_bottle: TransformRecord
    task_from_asset: TransformRecord
    block_height_m: float = Field(gt=0.0, le=0.20)
    measurement_method: Literal["steel-ruler-square-and-rigid-stops"]
    repeated_installation_delta_m: float = Field(ge=0.0)


class FrozenBottleFixtureContract(BaseModel):
    status: Literal["BOTTLE_FIXTURE_CONTRACT_FROZEN"] = "BOTTLE_FIXTURE_CONTRACT_FROZEN"
    fixture_id: str
    revision: int
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    measured_length_m: float
    measured_diameter_m: float
    tag_from_bottle: TransformRecord
    task_from_asset: TransformRecord
    block_height_m: float
    measurement_method: str
    repeated_installation_delta_m: float


class BottleTrialObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Literal["B-A", "B-B", "B-C"]
    camera_from_tag: TransformRecord


class BottleTagCaptureRequest(BaseModel):
    tag_size_m: float = Field(gt=0.01, le=0.30)
    frame_count: int = Field(default=150, ge=150, le=300)


class BottleTrialCaptureResult(BaseModel):
    observation: BottleTrialObservation
    stability: TagPoseStabilityResult


class BottleTrialInput(BaseModel):
    id: Literal["B-A", "B-B", "B-C"]
    expected_world_from_bottle: TransformRecord
    camera_from_tag: TransformRecord
    support_height_m: float = Field(ge=0.0)


class BottleTrialMetric(BaseModel):
    id: str
    expected_world_from_bottle: TransformRecord
    estimated_world_from_bottle: TransformRecord
    center_error_m: float
    long_axis_error_deg: float
    support_residual_m: float


class BottleValidationResult(BaseModel):
    status: Literal["TAGGED_FIXTURE_TRANSFER_PASS"]
    claim_scope: Literal["tagged-rigid-fixture-transfer-only"]
    center_rms_m: float
    center_max_m: float
    long_axis_rms_deg: float
    long_axis_max_deg: float
    support_max_abs_m: float
    trials: list[BottleTrialMetric]


class StageContract(BaseModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FactoryCameraSnapshot(BaseModel):
    role: Literal["cam_high", "cam_low", "wrist_left", "wrist_right"]
    serial: str
    firmware: str | None = None
    profile: ProductionProfile
    intrinsics: FactoryIntrinsics
    depth_stream_started: Literal[False] = False


class FactorySnapshotBundle(BaseModel):
    status: Literal["FACTORY_INTRINSICS_FROZEN"] = "FACTORY_INTRINSICS_FROZEN"
    cameras: list[FactoryCameraSnapshot]

    @field_validator("cameras")
    @classmethod
    def validate_complete_registry(cls, value: list[FactoryCameraSnapshot]) -> list[FactoryCameraSnapshot]:
        expected = {"cam_high", "cam_low", "wrist_left", "wrist_right"}
        if len(value) != 4 or {camera.role for camera in value} != expected:
            raise ValueError("factory bundle requires exactly the four configured camera roles")
        if len({camera.serial for camera in value}) != 4:
            raise ValueError("factory bundle camera serials must be unique")
        return sorted(value, key=lambda camera: camera.role)


class WorldOriginSolveRequest(BaseModel):
    samples: list[TagPoseSample]
    world_from_tag: TransformRecord
    total_frames: int | None = Field(default=None, ge=1)


class WorldOriginCaptureRequest(BaseModel):
    session_id: str
    tag_size_m: float = Field(gt=0.01, le=0.30)
    tag_plane_height_m: float = Field(ge=0.0, le=0.05)
    frame_count: int = Field(default=200, ge=150, le=300)


class WorldOriginPhysicalRequest(BaseModel):
    tag_size_m: float = Field(gt=0.01, le=0.30)
    tag_plane_height_m: float = Field(ge=0.0, le=0.05)
    frame_count: int = Field(default=200, ge=150, le=300)


class WorldOriginCaptureBatch(BaseModel):
    samples: list[TagPoseSample]
    world_from_tag: TransformRecord
    total_frames: int
    detected_frames: int


class TableSolveRequest(BaseModel):
    points: list[DotObservation]
    intrinsics: FactoryIntrinsics


class TablePointsRequest(BaseModel):
    points: list[DotObservation]


class BottleValidationRequest(BaseModel):
    world_from_camera: TransformRecord
    fixture: BottleFixture
    trials: list[BottleTrialInput]


class BottleTrialsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: list[BottleTrialObservation]


class ExportRequest(BaseModel):
    output_dir: str
    stage: StageContract
    world_from_camera: TransformRecord
    bottle_asset_path: str
    bottle_asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bottle_asset_prim: str = "/Bottle500"


class ExportBundleRequest(BaseModel):
    stage: StageContract
    bottle_asset_path: str
    bottle_asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bottle_asset_prim: str = "/Bottle500"


class ExportResult(BaseModel):
    calibration_json: str
    calibration_layer: str
    review_stage: str
    source_stage_sha256: str


def _rotation_mean(rotations: list[np.ndarray]) -> np.ndarray:
    accumulator = np.sum(np.stack(rotations), axis=0)
    u, _, vt = np.linalg.svd(accumulator)
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


def _rotation_distance_deg(left: np.ndarray, right: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(left.T @ right) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _record(matrix: np.ndarray, source: str, target: str) -> TransformRecord:
    return TransformRecord(source_frame=source, target_frame=target, matrix=matrix.tolist())


def _camera_matrix(intrinsics: FactoryIntrinsics) -> np.ndarray:
    return np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _normalized_points(
    image_points: np.ndarray,
    intrinsics: FactoryIntrinsics,
) -> np.ndarray:
    points = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    if intrinsics.distortion_model in {"none", "brown_conrady"}:
        distortion = None
        if intrinsics.distortion_model == "brown_conrady":
            distortion = np.asarray(intrinsics.distortion_coefficients, dtype=np.float64)
        return cv2.undistortPoints(points, _camera_matrix(intrinsics), distortion).reshape(-1, 2)
    if intrinsics.distortion_model == "inverse_brown_conrady":
        try:
            import pyrealsense2 as rs
        except ImportError as exc:  # pragma: no cover - production dependency
            raise WorkflowGateError("pyrealsense2 is required for inverse_brown_conrady") from exc
        rs_intrinsics = rs.intrinsics()
        rs_intrinsics.width = intrinsics.width
        rs_intrinsics.height = intrinsics.height
        rs_intrinsics.fx = intrinsics.fx
        rs_intrinsics.fy = intrinsics.fy
        rs_intrinsics.ppx = intrinsics.cx
        rs_intrinsics.ppy = intrinsics.cy
        rs_intrinsics.model = rs.distortion.inverse_brown_conrady
        coefficients = list(intrinsics.distortion_coefficients)[:5]
        coefficients.extend([0.0] * (5 - len(coefficients)))
        rs_intrinsics.coeffs = coefficients
        rays = [rs.rs2_deproject_pixel_to_point(rs_intrinsics, point.tolist(), 1.0) for point in points[:, 0]]
        return np.asarray([[ray[0] / ray[2], ray[1] / ray[2]] for ray in rays], dtype=np.float64)
    raise WorkflowGateError(f"unsupported distortion model: {intrinsics.distortion_model}")


def _project_pixels(points_camera: np.ndarray, intrinsics: FactoryIntrinsics) -> np.ndarray:
    normalized = points_camera[:, :2] / points_camera[:, 2:3]
    if intrinsics.distortion_model == "none":
        return np.column_stack(
            (
                intrinsics.fx * normalized[:, 0] + intrinsics.cx,
                intrinsics.fy * normalized[:, 1] + intrinsics.cy,
            )
        )
    if intrinsics.distortion_model == "brown_conrady":
        pixels, _ = cv2.projectPoints(
            points_camera,
            np.zeros(3),
            np.zeros(3),
            _camera_matrix(intrinsics),
            np.asarray(intrinsics.distortion_coefficients, dtype=np.float64),
        )
        return pixels.reshape(-1, 2)
    if intrinsics.distortion_model == "inverse_brown_conrady":
        # Projection of an inverse Brown model is not closed-form in OpenCV.
        # RealSense accepts the calibrated model and iterates internally.
        import pyrealsense2 as rs

        rs_intrinsics = rs.intrinsics()
        rs_intrinsics.width = intrinsics.width
        rs_intrinsics.height = intrinsics.height
        rs_intrinsics.fx = intrinsics.fx
        rs_intrinsics.fy = intrinsics.fy
        rs_intrinsics.ppx = intrinsics.cx
        rs_intrinsics.ppy = intrinsics.cy
        rs_intrinsics.model = rs.distortion.inverse_brown_conrady
        coefficients = list(intrinsics.distortion_coefficients)[:5]
        coefficients.extend([0.0] * (5 - len(coefficients)))
        rs_intrinsics.coeffs = coefficients
        return np.asarray(
            [rs.rs2_project_point_to_pixel(rs_intrinsics, point.tolist()) for point in points_camera],
            dtype=np.float64,
        )
    raise WorkflowGateError(f"unsupported distortion model: {intrinsics.distortion_model}")


class CalibrationWorkflow:
    def freeze_table_point_contract(
        self,
        request: TablePointContractRequest,
    ) -> FrozenTablePointContract:
        expected_ids = {f"P{row}{column}" for row in range(1, 4) for column in range(1, 4)}
        if len(request.points) != 9 or {point.id for point in request.points} != expected_ids:
            raise WorkflowGateError("table point contract requires exactly P11 through P33")
        expected_colors = {
            **{f"P1{column}": "blue" for column in range(1, 4)},
            **{f"P2{column}": "magenta" for column in range(1, 4)},
            **{f"P3{column}": "lime" for column in range(1, 4)},
        }
        held_out = {"P11", "P23", "P32"}
        truths: list[TablePointTruth] = []
        for point in sorted(request.points, key=lambda item: item.id):
            if point.color != expected_colors[point.id]:
                raise WorkflowGateError(f"table point {point.id} has the wrong row color")
            first = np.asarray(point.measurement_1_xy_m, dtype=np.float64)
            second = np.asarray(point.measurement_2_xy_m, dtype=np.float64)
            delta = float(np.linalg.norm(first - second))
            if delta > 0.002:
                raise WorkflowGateError(f"table point {point.id} repeated measurement differs by more than 2 mm")
            xy = (first + second) / 2.0
            truths.append(
                TablePointTruth(
                    id=point.id,
                    color=point.color,
                    partition="HELD_OUT" if point.id in held_out else "SOLVE",
                    world_xyz_m=(float(xy[0]), float(xy[1]), 0.0),
                    repeated_measurement_delta_m=delta,
                )
            )
        canonical = {
            "contract_id": request.contract_id,
            "revision": request.revision,
            "measurement_method": request.measurement_method,
            "points": [truth.model_dump(mode="json") for truth in truths],
        }
        digest = hashlib.sha256(
            json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return FrozenTablePointContract(contract_sha256=digest, **canonical)

    def observations_from_contract(
        self,
        *,
        contract: FrozenTablePointContract,
        observations: list[DotPixelObservation],
    ) -> list[DotObservation]:
        if len(observations) != 9 or {item.id for item in observations} != {point.id for point in contract.points}:
            raise WorkflowGateError("table observations must contain each frozen point exactly once")
        by_id = {observation.id: observation for observation in observations}
        return [
            DotObservation(
                id=truth.id,
                color=truth.color,
                partition=truth.partition,
                world_xyz_m=truth.world_xyz_m,
                image_uv_px=by_id[truth.id].image_uv_px,
                repeated_measurement_delta_m=truth.repeated_measurement_delta_m,
                operator_confirmed=by_id[truth.id].operator_confirmed,
            )
            for truth in contract.points
        ]

    def freeze_bottle_fixture_contract(
        self,
        request: BottleFixtureContractRequest,
    ) -> FrozenBottleFixtureContract:
        if abs(request.measured_length_m - 0.206) > 0.005:
            raise WorkflowGateError("BOTTLE_ASSET_MISMATCH: length differs by more than 5 mm")
        if abs(request.measured_diameter_m - 0.068) > 0.003:
            raise WorkflowGateError("BOTTLE_ASSET_MISMATCH: diameter differs by more than 3 mm")
        if request.repeated_installation_delta_m > 0.002:
            raise WorkflowGateError("fixture repeated installation differs by more than 2 mm")
        if (
            request.tag_from_bottle.source_frame != "bottle_task"
            or request.tag_from_bottle.target_frame != "tag"
        ):
            raise WorkflowGateError("tag_from_bottle frame contract is bottle_task -> tag")
        if (
            request.task_from_asset.source_frame != "bottle_asset"
            or request.task_from_asset.target_frame != "bottle_task"
        ):
            raise WorkflowGateError("task_from_asset frame contract is bottle_asset -> bottle_task")
        canonical = request.model_dump(mode="json")
        digest = hashlib.sha256(
            json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return FrozenBottleFixtureContract(contract_sha256=digest, **canonical)

    def expected_bottle_poses(
        self,
        *,
        fixture: FrozenBottleFixtureContract,
        table_contract: FrozenTablePointContract,
    ) -> dict[str, TransformRecord]:
        points = {point.id: point for point in table_contract.points}
        specification = {
            "B-A": ("P22", 0.0, 0.0),
            "B-B": ("P23", math.pi / 2.0, 0.0),
            "B-C": ("P11", math.pi, fixture.block_height_m),
        }
        expected: dict[str, TransformRecord] = {}
        for trial_id, (point_id, yaw, support_height) in specification.items():
            if point_id not in points:
                raise WorkflowGateError(f"table contract is missing placement point {point_id}")
            cosine, sine = math.cos(yaw), math.sin(yaw)
            pose = np.eye(4)
            pose[:3, :3] = np.array(
                [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
            )
            pose[:2, 3] = points[point_id].world_xyz_m[:2]
            pose[2, 3] = support_height + fixture.measured_diameter_m / 2.0
            expected[trial_id] = _record(pose, "bottle_task", "table_world")
        return expected

    def validate_bottle_observations(
        self,
        *,
        world_from_camera: TransformRecord,
        fixture: FrozenBottleFixtureContract,
        table_contract: FrozenTablePointContract,
        observations: list[BottleTrialObservation],
    ) -> BottleValidationResult:
        if len(observations) != 3 or {item.id for item in observations} != {"B-A", "B-B", "B-C"}:
            raise WorkflowGateError("bottle observations require exactly B-A, B-B, B-C")
        expected = self.expected_bottle_poses(fixture=fixture, table_contract=table_contract)
        supports = {"B-A": 0.0, "B-B": 0.0, "B-C": fixture.block_height_m}
        trials = [
            BottleTrialInput(
                id=observation.id,
                expected_world_from_bottle=expected[observation.id],
                camera_from_tag=observation.camera_from_tag,
                support_height_m=supports[observation.id],
            )
            for observation in observations
        ]
        return self.validate_bottle_trials(
            world_from_camera=world_from_camera,
            fixture=fixture,
            trials=trials,
        )

    def solve_world_origin(
        self,
        *,
        samples: list[TagPoseSample],
        world_from_tag: TransformRecord,
        total_frames: int | None = None,
    ) -> WorldOriginResult:
        valid = [sample for sample in samples if sample.reprojection_rms_px <= 2.0]
        if len(valid) < 150:
            raise WorkflowGateError(f"world origin requires at least 150 accepted frames; got {len(valid)}")
        if any(
            sample.frame_id is None
            or sample.device_timestamp_ms is None
            or sample.image_sha256 is None
            for sample in valid
        ):
            raise WorkflowGateError("world origin samples require frame, device timestamp, and image evidence")
        frame_ids = [sample.frame_id for sample in valid]
        image_hashes = [sample.image_sha256 for sample in valid]
        if len(set(frame_ids)) != len(frame_ids) or len(set(image_hashes)) != len(image_hashes):
            raise WorkflowGateError("world origin samples must be distinct immutable frames")
        timestamps = np.asarray([sample.device_timestamp_ms for sample in valid], dtype=np.float64)
        if float(np.max(timestamps) - np.min(timestamps)) < 2000.0:
            raise WorkflowGateError("world origin capture must span at least 2 seconds of device time")
        if world_from_tag.source_frame != "tag" or world_from_tag.target_frame != "table_world":
            raise WorkflowGateError("world_from_tag frame contract is tag -> table_world")
        reprojection = np.asarray([sample.reprojection_rms_px for sample in valid], dtype=np.float64)
        median_reprojection = float(np.median(reprojection))
        p95_reprojection = float(np.percentile(reprojection, 95))
        if median_reprojection > 1.0 or p95_reprojection > 2.0:
            raise WorkflowGateError("world origin reprojection gate failed")

        matrices = [sample.camera_from_tag.array() for sample in valid]
        if any(
            sample.camera_from_tag.source_frame != "tag"
            or sample.camera_from_tag.target_frame != "camera_high_optical"
            for sample in valid
        ):
            raise WorkflowGateError("camera_from_tag frame contract is tag -> camera_high_optical")
        rotation = _rotation_mean([matrix[:3, :3] for matrix in matrices])
        translation = np.median(np.stack([matrix[:3, 3] for matrix in matrices]), axis=0)
        aggregate = np.eye(4)
        aggregate[:3, :3] = rotation
        aggregate[:3, 3] = translation
        translation_jitter = float(
            np.percentile([np.linalg.norm(matrix[:3, 3] - translation) for matrix in matrices], 95)
        )
        rotation_jitter = float(
            np.percentile([_rotation_distance_deg(matrix[:3, :3], rotation) for matrix in matrices], 95)
        )
        # The 80 mm tag occupies only about 37 pixels in cam_high. Keep the
        # strict 2 mm translation gate, but allow the observed planar-PnP
        # tilt uncertainty here; the following 9-dot table solve independently
        # refines the full camera rotation and validates three held-out points.
        if translation_jitter > 0.002 or rotation_jitter > 1.5:
            raise WorkflowGateError("world origin pose jitter gate failed")

        world_from_camera = world_from_tag.array() @ np.linalg.inv(aggregate)
        return WorldOriginResult(
            status="WORLD_ORIGIN_SOLVED",
            world_from_camera=_record(world_from_camera, "camera_high_optical", "table_world"),
            accepted_frames=len(valid),
            total_frames=total_frames if total_frames is not None else len(samples),
            median_reprojection_rms_px=median_reprojection,
            p95_reprojection_rms_px=p95_reprojection,
            translation_jitter_m=translation_jitter,
            rotation_jitter_deg=rotation_jitter,
        )

    def aggregate_tag_pose(
        self,
        *,
        samples: list[TagPoseSample],
        minimum_frames: int = 150,
        total_frames: int | None = None,
    ) -> TagPoseStabilityResult:
        valid = [sample for sample in samples if sample.reprojection_rms_px <= 2.0]
        if len(valid) < minimum_frames:
            raise WorkflowGateError(
                f"tag pose requires at least {minimum_frames} accepted frames; got {len(valid)}"
            )
        if any(
            sample.frame_id is None
            or sample.device_timestamp_ms is None
            or sample.image_sha256 is None
            for sample in valid
        ):
            raise WorkflowGateError("tag pose samples require frame, device timestamp, and image evidence")
        if len({sample.frame_id for sample in valid}) != len(valid):
            raise WorkflowGateError("tag pose frame IDs must be unique")
        if len({sample.image_sha256 for sample in valid}) != len(valid):
            raise WorkflowGateError("tag pose image evidence must be unique")
        timestamps = np.asarray([sample.device_timestamp_ms for sample in valid], dtype=np.float64)
        if float(np.max(timestamps) - np.min(timestamps)) < 2000.0:
            raise WorkflowGateError("tag pose capture must span at least 2 seconds of device time")
        if any(
            sample.camera_from_tag.source_frame != "tag"
            or sample.camera_from_tag.target_frame != "camera_high_optical"
            for sample in valid
        ):
            raise WorkflowGateError("camera_from_tag frame contract is tag -> camera_high_optical")
        reprojection = np.asarray([sample.reprojection_rms_px for sample in valid], dtype=np.float64)
        median_reprojection = float(np.median(reprojection))
        p95_reprojection = float(np.percentile(reprojection, 95))
        if median_reprojection > 1.0 or p95_reprojection > 2.0:
            raise WorkflowGateError("tag pose reprojection gate failed")
        matrices = [sample.camera_from_tag.array() for sample in valid]
        rotation = _rotation_mean([matrix[:3, :3] for matrix in matrices])
        translation = np.median(np.stack([matrix[:3, 3] for matrix in matrices]), axis=0)
        aggregate = np.eye(4)
        aggregate[:3, :3] = rotation
        aggregate[:3, 3] = translation
        translation_jitter = float(
            np.percentile([np.linalg.norm(matrix[:3, 3] - translation) for matrix in matrices], 95)
        )
        rotation_jitter = float(
            np.percentile([_rotation_distance_deg(matrix[:3, :3], rotation) for matrix in matrices], 95)
        )
        if translation_jitter > 0.002 or rotation_jitter > 0.5:
            raise WorkflowGateError("tag pose jitter gate failed")
        return TagPoseStabilityResult(
            camera_from_tag=_record(aggregate, "tag", "camera_high_optical"),
            accepted_frames=len(valid),
            total_frames=total_frames if total_frames is not None else len(samples),
            median_reprojection_rms_px=median_reprojection,
            p95_reprojection_rms_px=p95_reprojection,
            translation_jitter_m=translation_jitter,
            rotation_jitter_deg=rotation_jitter,
        )

    def solve_table_registration(
        self,
        *,
        points: list[DotObservation],
        intrinsics: FactoryIntrinsics,
        initial_world_from_camera: TransformRecord,
    ) -> TableRegistrationResult:
        excessive = [point.id for point in points if point.repeated_measurement_delta_m > 0.002]
        if excessive:
            raise WorkflowGateError(f"repeated measurement differs by more than 2 mm: {excessive}")
        unconfirmed = [point.id for point in points if not point.operator_confirmed]
        if unconfirmed:
            raise WorkflowGateError(f"operator confirmation missing for dots: {unconfirmed}")
        expected_ids = {f"P{row}{column}" for row in range(1, 4) for column in range(1, 4)}
        if {point.id for point in points} != expected_ids or len(points) != 9:
            raise WorkflowGateError("table registration requires exactly P11 through P33")
        expected_held_out = {"P11", "P23", "P32"}
        actual_held_out = {point.id for point in points if point.partition == "HELD_OUT"}
        if actual_held_out != expected_held_out:
            raise WorkflowGateError("held-out partition is fixed to non-collinear P11, P23, P32")
        solve = sorted((point for point in points if point.partition == "SOLVE"), key=lambda item: item.id)
        held_out = sorted((point for point in points if point.partition == "HELD_OUT"), key=lambda item: item.id)
        if len(solve) != 6 or len(held_out) != 3:
            raise WorkflowGateError("table registration requires 6 solve and 3 held-out dots")

        if (
            initial_world_from_camera.source_frame != "camera_high_optical"
            or initial_world_from_camera.target_frame != "table_world"
        ):
            raise WorkflowGateError(
                "initial world anchor must map camera_high_optical -> table_world"
            )
        all_object_points = np.asarray([point.world_xyz_m for point in points], dtype=np.float64)
        initial_camera_from_world = np.linalg.inv(initial_world_from_camera.array())
        initial_camera_points = (
            initial_camera_from_world[:3, :3] @ all_object_points.T
        ).T + initial_camera_from_world[:3, 3]
        if np.any(initial_camera_points[:, 2] <= 0):
            raise WorkflowGateError("world anchor places table points behind the camera")
        initial_projected = _project_pixels(initial_camera_points, intrinsics)
        all_observed = np.asarray([point.image_uv_px for point in points], dtype=np.float64)
        initial_rms = float(
            np.sqrt(np.mean(np.sum((initial_projected - all_observed) ** 2, axis=1)))
        )
        if initial_rms > 20.0:
            raise WorkflowGateError(
                f"world anchor is inconsistent with table observations: {initial_rms:.3f}px"
            )

        object_points = np.asarray([point.world_xyz_m for point in solve], dtype=np.float64)
        normalized = _normalized_points(np.asarray([point.image_uv_px for point in solve]), intrinsics)
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            normalized,
            np.eye(3, dtype=np.float64),
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            raise WorkflowGateError("table solvePnP failed")
        rotation, _ = cv2.Rodrigues(rvec)
        camera_from_world = np.eye(4)
        camera_from_world[:3, :3] = rotation
        camera_from_world[:3, 3] = tvec.reshape(3)
        solve_camera = (rotation @ object_points.T).T + tvec.reshape(1, 3)
        if np.any(solve_camera[:, 2] <= 0):
            raise WorkflowGateError("table solution placed points behind the camera")
        projected = _project_pixels(solve_camera, intrinsics)
        observed = np.asarray([point.image_uv_px for point in solve], dtype=np.float64)
        solve_rms = float(np.sqrt(np.mean(np.sum((projected - observed) ** 2, axis=1))))
        if solve_rms > 1.5:
            raise WorkflowGateError(f"solve reprojection RMS {solve_rms:.3f}px exceeds 1.5px")

        world_from_camera = np.linalg.inv(camera_from_world)
        refinement_translation = float(
            np.linalg.norm(world_from_camera[:3, 3] - initial_world_from_camera.array()[:3, 3])
        )
        refinement_rotation = _rotation_distance_deg(
            initial_world_from_camera.array()[:3, :3], world_from_camera[:3, :3]
        )
        if refinement_translation > 0.030 or refinement_rotation > 3.0:
            raise WorkflowGateError(
                "table refinement differs too far from the frozen world anchor"
            )
        origin_world = world_from_camera[:3, 3]
        held_out_errors: list[float] = []
        for point in held_out:
            normalized_point = _normalized_points(np.asarray([point.image_uv_px]), intrinsics)[0]
            ray_camera = np.array([normalized_point[0], normalized_point[1], 1.0])
            ray_world = world_from_camera[:3, :3] @ ray_camera
            if abs(ray_world[2]) < 1e-9:
                raise WorkflowGateError(f"held-out ray for {point.id} is parallel to the table")
            scale = -origin_world[2] / ray_world[2]
            if scale <= 0:
                raise WorkflowGateError(f"held-out ray for {point.id} intersects behind the camera")
            recovered = origin_world + scale * ray_world
            held_out_errors.append(float(np.linalg.norm(recovered[:2] - np.asarray(point.world_xyz_m[:2]))))
        held_out_rms = float(np.sqrt(np.mean(np.square(held_out_errors))))
        held_out_max = float(max(held_out_errors))
        if held_out_rms > 0.005 or held_out_max > 0.010:
            raise WorkflowGateError("held-out table metric gate failed")

        return TableRegistrationResult(
            status="WORLD_REGISTRATION_VALIDATED",
            validation_scope="tabletop-xy-cross-validation",
            world_from_camera=_record(world_from_camera, "camera_high_optical", "table_world"),
            solve_point_ids=[point.id for point in solve],
            held_out_point_ids=[point.id for point in held_out],
            solve_reprojection_rms_px=solve_rms,
            initial_reprojection_rms_px=initial_rms,
            refinement_translation_m=refinement_translation,
            refinement_rotation_deg=refinement_rotation,
            held_out_rms_m=held_out_rms,
            held_out_max_m=held_out_max,
        )

    def validate_bottle_trials(
        self,
        *,
        world_from_camera: TransformRecord,
        fixture: BottleFixture | FrozenBottleFixtureContract,
        trials: list[BottleTrialInput],
    ) -> BottleValidationResult:
        if abs(fixture.measured_length_m - 0.206) > 0.005:
            raise WorkflowGateError("BOTTLE_ASSET_MISMATCH: length differs by more than 5 mm")
        if abs(fixture.measured_diameter_m - 0.068) > 0.003:
            raise WorkflowGateError("BOTTLE_ASSET_MISMATCH: diameter differs by more than 3 mm")
        if {trial.id for trial in trials} != {"B-A", "B-B", "B-C"} or len(trials) != 3:
            raise WorkflowGateError("bottle validation requires exactly B-A, B-B, B-C")
        if world_from_camera.source_frame != "camera_high_optical" or world_from_camera.target_frame != "table_world":
            raise WorkflowGateError("world_from_camera frame contract is camera_high_optical -> table_world")
        if fixture.tag_from_bottle.source_frame != "bottle_task" or fixture.tag_from_bottle.target_frame != "tag":
            raise WorkflowGateError("tag_from_bottle frame contract is bottle_task -> tag")

        metrics: list[BottleTrialMetric] = []
        for trial in sorted(trials, key=lambda item: item.id):
            if trial.camera_from_tag.source_frame != "tag" or trial.camera_from_tag.target_frame != "camera_high_optical":
                raise WorkflowGateError("camera_from_tag frame contract is tag -> camera_high_optical")
            estimated = world_from_camera.array() @ trial.camera_from_tag.array() @ fixture.tag_from_bottle.array()
            expected = trial.expected_world_from_bottle.array()
            center_error = float(np.linalg.norm(estimated[:3, 3] - expected[:3, 3]))
            axis_estimated = estimated[:3, 0]
            axis_expected = expected[:3, 0]
            axis_cosine = float(np.clip(np.dot(axis_estimated, axis_expected), -1.0, 1.0))
            axis_error = math.degrees(math.acos(axis_cosine))
            support_residual = float(
                estimated[2, 3] - fixture.measured_diameter_m / 2.0 - trial.support_height_m
            )
            metrics.append(
                BottleTrialMetric(
                    id=trial.id,
                    expected_world_from_bottle=trial.expected_world_from_bottle,
                    estimated_world_from_bottle=_record(estimated, "bottle_task", "table_world"),
                    center_error_m=center_error,
                    long_axis_error_deg=axis_error,
                    support_residual_m=support_residual,
                )
            )
        center_errors = np.asarray([metric.center_error_m for metric in metrics])
        angle_errors = np.asarray([metric.long_axis_error_deg for metric in metrics])
        support_errors = np.asarray([metric.support_residual_m for metric in metrics])
        center_rms = float(np.sqrt(np.mean(center_errors**2)))
        center_max = float(np.max(center_errors))
        angle_rms = float(np.sqrt(np.mean(angle_errors**2)))
        angle_max = float(np.max(angle_errors))
        support_max = float(np.max(np.abs(support_errors)))
        if center_rms > 0.008 or center_max > 0.015:
            raise WorkflowGateError("bottle center metric gate failed")
        if angle_rms > 2.0 or angle_max > 4.0:
            raise WorkflowGateError("bottle long-axis metric gate failed")
        if support_max > 0.005:
            raise WorkflowGateError("bottle support penetration/hover gate failed")
        return BottleValidationResult(
            status="TAGGED_FIXTURE_TRANSFER_PASS",
            claim_scope="tagged-rigid-fixture-transfer-only",
            center_rms_m=center_rms,
            center_max_m=center_max,
            long_axis_rms_deg=angle_rms,
            long_axis_max_deg=angle_max,
            support_max_abs_m=support_max,
            trials=metrics,
        )

    def export_calibration_bundle(
        self,
        *,
        output_dir: Path,
        stage: StageContract,
        world_from_camera: TransformRecord,
        bottle_asset_path: str,
        bottle_asset_sha256: str,
        bottle_asset_prim: str,
        bottle_validation: BottleValidationResult | None = None,
        task_from_asset: TransformRecord | None = None,
    ) -> ExportResult:
        if (
            world_from_camera.source_frame != "camera_high_optical"
            or world_from_camera.target_frame != "table_world"
        ):
            raise WorkflowGateError(
                "export world_from_camera must map camera_high_optical -> table_world"
            )
        source = Path(stage.path).expanduser().resolve()
        if not source.is_file():
            raise WorkflowGateError(f"source Stage does not exist: {source}")
        source_stat_before = source.stat()
        actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual_hash != stage.sha256:
            raise WorkflowGateError(
                f"source Stage hash mismatch: expected {stage.sha256}, got {actual_hash}"
            )
        if bottle_asset_prim != "/Bottle500":
            raise WorkflowGateError("Bottle500 reference must use explicit prim path /Bottle500")
        if (bottle_validation is None) != (task_from_asset is None):
            raise WorkflowGateError("bottle validation and task_from_asset must be exported together")
        if task_from_asset is not None and (
            task_from_asset.source_frame != "bottle_asset"
            or task_from_asset.target_frame != "bottle_task"
        ):
            raise WorkflowGateError("task_from_asset frame contract is bottle_asset -> bottle_task")
        bottle_asset = Path(bottle_asset_path).expanduser().resolve()
        if not bottle_asset.is_file():
            raise WorkflowGateError(f"Bottle500 asset does not exist: {bottle_asset}")
        actual_bottle_hash = hashlib.sha256(bottle_asset.read_bytes()).hexdigest()
        if actual_bottle_hash != bottle_asset_sha256:
            raise WorkflowGateError(
                f"Bottle500 asset hash mismatch: expected {bottle_asset_sha256}, got {actual_bottle_hash}"
            )
        if bottle_asset.suffix.lower() == ".usda":
            bottle_text = bottle_asset.read_text(encoding="utf-8")
            required_tokens = ('"Bottle500"', '"VIS_Bottle"', '"Collisions"')
            if any(token not in bottle_text for token in required_tokens):
                raise WorkflowGateError("Bottle500 USDA is missing required prim declarations")
        if output_dir.exists():
            raise WorkflowGateError(f"output directory already exists: {output_dir}")
        output_dir.mkdir(parents=True, mode=0o700)
        calibration_json = output_dir / "calibration.json"
        calibration_layer = output_dir / "calibration.usda"
        review_stage = output_dir / "calibrated_review.usda"
        payload = {
            "schema_version": 1,
            "claim_scope": "table-world-camera-calibration-with-bottle-reference-probe",
            "source_stage": {
                "path": str(source),
                "sha256_before": actual_hash,
                "sha256_after": None,
                "size_before": source_stat_before.st_size,
                "size_after": None,
                "mtime_ns_before": source_stat_before.st_mtime_ns,
                "mtime_ns_after": None,
                "source_stage_modified": None,
            },
            "bottle_asset": {
                "path": str(bottle_asset),
                "sha256": actual_bottle_hash,
                "prim_path": bottle_asset_prim,
                "composition_validation": "pending-isaac-runtime-readback",
            },
            "bottle_trials": None
            if bottle_validation is None
            else bottle_validation.model_dump(mode="json"),
            "world_from_camera_high_optical": world_from_camera.model_dump(mode="json"),
            "opencv_optical_to_usd_camera": {
                "axis_conversion": "rotate_180_degrees_about_x",
                "matrix": [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            },
        }
        matrix = world_from_camera.array()
        optical_gf = matrix.T
        rx_pi = np.diag([1.0, -1.0, -1.0, 1.0])
        world_from_usd_camera = matrix @ rx_pi
        camera_gf = world_from_usd_camera.T
        optical_matrix_rows = ", ".join(
            "(" + ", ".join(f"{value:.12g}" for value in row) + ")" for row in optical_gf
        )
        camera_matrix_rows = ", ".join(
            "(" + ", ".join(f"{value:.12g}" for value in row) + ")" for row in camera_gf
        )
        if bottle_validation is None:
            bottle_prims = (
                '        def Xform "BottleReferenceProbe" (\n'
                f"            prepend references = @{bottle_asset.as_posix()}@<{bottle_asset_prim}>\n"
                "        )\n        {\n        }\n"
            )
        else:
            assert task_from_asset is not None
            authored: list[str] = []
            for trial in bottle_validation.trials:
                world_from_asset = trial.estimated_world_from_bottle.array() @ task_from_asset.array()
                gf_matrix = world_from_asset.T
                rows = ", ".join(
                    "(" + ", ".join(f"{value:.12g}" for value in row) + ")" for row in gf_matrix
                )
                prim_name = trial.id.replace("-", "_")
                authored.append(
                    f'        def Xform "Bottle_{prim_name}" (\n'
                    f"            prepend references = @{bottle_asset.as_posix()}@<{bottle_asset_prim}>\n"
                    "        )\n        {\n"
                    f"            matrix4d xformOp:transform = ({rows})\n"
                    '            uniform token[] xformOpOrder = ["xformOp:transform"]\n'
                    "        }\n"
                )
            bottle_prims = "".join(authored)
        calibration_layer.write_text(
            "#usda 1.0\n\n"
            'over "World"\n{\n'
            '    def Scope "Calibration"\n    {\n'
            f"        custom matrix4d cameraHighWorldFromOpticalColumnVector = ({optical_matrix_rows})\n"
            '        custom string cameraAxisConversion = "OpenCV optical to USD Camera: Rx(180 deg)"\n'
            '        custom string matrixBoundary = "JSON column-vector; USD matrix is transposed for Gf row-vector semantics"\n'
            + bottle_prims
            + "    }\n"
            '    def Camera "CameraHigh"\n    {\n'
            f"        matrix4d xformOp:transform = ({camera_matrix_rows})\n"
            '        uniform token[] xformOpOrder = ["xformOp:transform"]\n'
            "    }\n}\n",
            encoding="utf-8",
        )
        review_stage.write_text(
            "#usda 1.0\n(\n"
            '    defaultPrim = "World"\n'
            "    subLayers = [\n"
            "        @./calibration.usda@,\n"
            f"        @{source.as_posix()}@\n"
            "    ]\n)\n",
            encoding="utf-8",
        )
        source_stat_after = source.stat()
        after_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        source_unchanged = (
            after_hash == actual_hash
            and source_stat_after.st_size == source_stat_before.st_size
            and source_stat_after.st_mtime_ns == source_stat_before.st_mtime_ns
        )
        if not source_unchanged:
            raise WorkflowGateError("source Stage changed while exporting calibration bundle")
        payload["source_stage"].update(
            {
                "sha256_after": after_hash,
                "size_after": source_stat_after.st_size,
                "mtime_ns_after": source_stat_after.st_mtime_ns,
                "source_stage_modified": False,
            }
        )
        calibration_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return ExportResult(
            calibration_json=str(calibration_json),
            calibration_layer=str(calibration_layer),
            review_stage=str(review_stage),
            source_stage_sha256=actual_hash,
        )
