from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from calibration_workbench.models import FactoryIntrinsics
from calibration_workbench.workflow import BottleFixtureContractRequest
from calibration_workbench.workflow import BottleTagCaptureRequest
from calibration_workbench.workflow import BottleTrialObservation
from calibration_workbench.workflow import CalibrationWorkflow
from calibration_workbench.workflow import DotObservation
from calibration_workbench.workflow import StageContract
from calibration_workbench.workflow import TagPoseSample
from calibration_workbench.workflow import TablePointContractRequest
from calibration_workbench.workflow import TransformRecord
from calibration_workbench.workflow import WorkflowGateError


def _transform(matrix: np.ndarray, source: str, target: str) -> TransformRecord:
    return TransformRecord(source_frame=source, target_frame=target, matrix=matrix.tolist())


def _bottle_task_from_asset() -> TransformRecord:
    matrix = np.array(
        [
            [0.0, 0.0, 1.0, -0.103],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return _transform(matrix, "bottle_asset", "bottle_task")


def _intrinsics() -> FactoryIntrinsics:
    return FactoryIntrinsics(
        width=640,
        height=480,
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        distortion_model="none",
        distortion_coefficients=[0.0] * 5,
    )


def _world_from_camera() -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
    pose[:3, 3] = [0.0, 0.0, 1.0]
    return pose


def _project(world_point: tuple[float, float, float]) -> tuple[float, float]:
    camera_from_world = np.linalg.inv(_world_from_camera())
    point_camera = camera_from_world @ np.array([*world_point, 1.0])
    return (
        600.0 * point_camera[0] / point_camera[2] + 320.0,
        600.0 * point_camera[1] / point_camera[2] + 240.0,
    )


def test_world_origin_requires_independent_quality_gates_and_solves_world_from_camera():
    workflow = CalibrationWorkflow()
    camera_from_tag = np.eye(4)
    camera_from_tag[2, 3] = 0.8
    samples = [
        TagPoseSample(
            camera_from_tag=_transform(camera_from_tag, "tag", "camera_high_optical"),
            reprojection_rms_px=0.4,
            frame_id=f"F-{index:03d}",
            device_timestamp_ms=index * 16.7,
            image_sha256=hashlib.sha256(f"frame-{index}".encode()).hexdigest(),
        )
        for index in range(160)
    ]
    world_from_tag = np.eye(4)
    world_from_tag[2, 3] = 0.003

    result = workflow.solve_world_origin(
        samples=samples,
        world_from_tag=_transform(world_from_tag, "tag", "table_world"),
    )

    assert result.status == "WORLD_ORIGIN_SOLVED"
    assert result.accepted_frames == 160
    assert result.median_reprojection_rms_px == pytest.approx(0.4)
    assert np.asarray(result.world_from_camera.matrix)[:3, 3] == pytest.approx([0.0, 0.0, -0.797])


def test_world_origin_rejects_too_few_frames():
    workflow = CalibrationWorkflow()
    identity = np.eye(4)
    with pytest.raises(WorkflowGateError, match="150"):
        workflow.solve_world_origin(
            samples=[
                TagPoseSample(
                    camera_from_tag=_transform(identity, "tag", "camera_high_optical"),
                    reprojection_rms_px=0.5,
                )
                for _ in range(149)
            ],
            world_from_tag=_transform(identity, "tag", "table_world"),
        )


def test_world_origin_allows_small_tag_tilt_jitter_for_later_table_refinement():
    workflow = CalibrationWorkflow()
    samples = []
    for index, angle_deg in enumerate(np.linspace(-1.2, 1.2, 160)):
        angle = np.deg2rad(angle_deg)
        camera_from_tag = np.eye(4)
        camera_from_tag[:3, :3] = np.array(
            [
                [np.cos(angle), 0.0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        camera_from_tag[2, 3] = 0.84
        samples.append(
            TagPoseSample(
                camera_from_tag=_transform(camera_from_tag, "tag", "camera_high_optical"),
                reprojection_rms_px=0.1,
                frame_id=f"F-{index:03d}",
                device_timestamp_ms=index * 33.4,
                image_sha256=hashlib.sha256(f"tilt-frame-{index}".encode()).hexdigest(),
            )
        )

    result = workflow.solve_world_origin(
        samples=samples,
        world_from_tag=_transform(np.eye(4), "tag", "table_world"),
    )

    assert 0.5 < result.rotation_jitter_deg < 1.5


def test_table_solver_keeps_six_solve_points_and_three_held_out_points_separate():
    workflow = CalibrationWorkflow()
    a, b = 0.35, 0.18
    points = []
    held_out = {"P11", "P23", "P32"}
    for row, y in enumerate((b, 0.0, -b), start=1):
        for column, x in enumerate((-a, 0.0, a), start=1):
            point_id = f"P{row}{column}"
            world = (x, y, 0.0)
            points.append(
                DotObservation(
                    id=point_id,
                    color=("blue", "magenta", "lime")[row - 1],
                    partition="HELD_OUT" if point_id in held_out else "SOLVE",
                    world_xyz_m=world,
                    image_uv_px=_project(world),
                    repeated_measurement_delta_m=0.001,
                )
            )

    result = workflow.solve_table_registration(
        points=points,
        intrinsics=_intrinsics(),
        initial_world_from_camera=_transform(
            _world_from_camera(), "camera_high_optical", "table_world"
        ),
    )

    assert result.status == "WORLD_REGISTRATION_VALIDATED"
    assert result.solve_point_ids == ["P12", "P13", "P21", "P22", "P31", "P33"]
    assert result.held_out_point_ids == ["P11", "P23", "P32"]
    assert result.validation_scope == "tabletop-xy-cross-validation"
    assert result.held_out_rms_m < 1e-6
    assert result.held_out_max_m < 1e-6


def test_table_solver_rejects_measurement_repeatability_over_two_mm():
    workflow = CalibrationWorkflow()
    with pytest.raises(WorkflowGateError, match="2 mm"):
        workflow.solve_table_registration(
            points=[
                DotObservation(
                    id="P11",
                    color="blue",
                    partition="SOLVE",
                    world_xyz_m=(-0.35, 0.18, 0.0),
                    image_uv_px=(110.0, 130.0),
                    repeated_measurement_delta_m=0.0021,
                )
            ],
            intrinsics=_intrinsics(),
            initial_world_from_camera=_transform(
                _world_from_camera(), "camera_high_optical", "table_world"
            ),
        )


def test_table_solver_rejects_observations_inconsistent_with_world_anchor():
    workflow = CalibrationWorkflow()
    a, b = 0.35, 0.18
    held_out = {"P11", "P23", "P32"}
    points = []
    for row, y in enumerate((b, 0.0, -b), start=1):
        for column, x in enumerate((-a, 0.0, a), start=1):
            point_id = f"P{row}{column}"
            world = (x, y, 0.0)
            points.append(
                DotObservation(
                    id=point_id,
                    color=("blue", "magenta", "lime")[row - 1],
                    partition="HELD_OUT" if point_id in held_out else "SOLVE",
                    world_xyz_m=world,
                    image_uv_px=_project(world),
                    repeated_measurement_delta_m=0.001,
                )
            )
    wrong_anchor = _world_from_camera()
    wrong_anchor[0, 3] += 0.20

    with pytest.raises(WorkflowGateError, match="world anchor"):
        workflow.solve_table_registration(
            points=points,
            intrinsics=_intrinsics(),
            initial_world_from_camera=_transform(
                wrong_anchor, "camera_high_optical", "table_world"
            ),
        )


def test_three_bottle_trials_prove_tagged_fixture_transfer_only():
    workflow = CalibrationWorkflow()
    table_contract = workflow.freeze_table_point_contract(
        TablePointContractRequest(
            contract_id="table-dots-test",
            revision=1,
            measurement_method="steel-ruler-and-square",
            points=[
                {
                    "id": f"P{row}{column}",
                    "color": ("blue", "magenta", "lime")[row - 1],
                    "measurement_1_xy_m": (x, y),
                    "measurement_2_xy_m": (x + 0.001, y),
                }
                for row, y in enumerate((0.18, 0.0, -0.18), start=1)
                for column, x in enumerate((-0.35, 0.0, 0.35), start=1)
            ],
        )
    )
    fixture = workflow.freeze_bottle_fixture_contract(BottleFixtureContractRequest(
        fixture_id="bottle500-v-block-test",
        revision=1,
        measured_length_m=0.206,
        measured_diameter_m=0.068,
        tag_id=1,
        tag_size_m=0.080,
        tag_from_bottle=_transform(np.eye(4), "bottle_task", "tag"),
        task_from_asset=_bottle_task_from_asset(),
        block_height_m=0.050,
        measurement_method="steel-ruler-square-and-rigid-stops",
        repeated_installation_delta_m=0.001,
    ))
    identity = _transform(np.eye(4), "camera_high_optical", "table_world")
    expected = workflow.expected_bottle_poses(fixture=fixture, table_contract=table_contract)
    trials = [
        BottleTrialObservation(
            id=trial_id,
            camera_from_tag=_transform(pose.array(), "tag", "camera_high_optical"),
        )
        for trial_id, pose in expected.items()
    ]

    report = workflow.validate_bottle_observations(
        world_from_camera=identity,
        fixture=fixture,
        table_contract=table_contract,
        observations=trials,
    )

    assert report.status == "TAGGED_FIXTURE_TRANSFER_PASS"
    assert report.claim_scope == "tagged-rigid-fixture-transfer-only"
    assert report.center_rms_m == pytest.approx(0.0)


def test_bottle_contract_freezes_non_world_tag_and_rejects_missing_asset_center_offset():
    workflow = CalibrationWorkflow()
    base = {
        "fixture_id": "bottle500-v-block-test",
        "revision": 1,
        "measured_length_m": 0.206,
        "measured_diameter_m": 0.068,
        "tag_id": 1,
        "tag_size_m": 0.080,
        "tag_from_bottle": _transform(np.eye(4), "bottle_task", "tag"),
        "block_height_m": 0.050,
        "measurement_method": "steel-ruler-square-and-rigid-stops",
        "repeated_installation_delta_m": 0.001,
    }

    frozen = workflow.freeze_bottle_fixture_contract(
        BottleFixtureContractRequest(**base, task_from_asset=_bottle_task_from_asset())
    )
    assert frozen.tag_id == 1
    assert frozen.tag_size_m == pytest.approx(0.080)

    with pytest.raises(WorkflowGateError, match="103 mm center offset"):
        workflow.freeze_bottle_fixture_contract(
            BottleFixtureContractRequest(
                **base,
                task_from_asset=_transform(
                    np.array(
                        [
                            [0.0, 0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    ),
                    "bottle_asset",
                    "bottle_task",
                ),
            )
        )

    with pytest.raises(ValueError, match="greater than or equal to 1"):
        BottleFixtureContractRequest(
            **{**base, "tag_id": 0},
            task_from_asset=_bottle_task_from_asset(),
        )

    assert BottleTagCaptureRequest().frame_count == 150


def test_export_refuses_stage_hash_mismatch_and_writes_independent_layers(tmp_path: Path):
    source_stage = tmp_path / "source.usda"
    source_stage.write_text('#usda 1.0\n(defaultPrim = "World")\n\ndef Xform "World" {}\n', encoding="utf-8")
    actual_hash = hashlib.sha256(source_stage.read_bytes()).hexdigest()
    bottle_asset = tmp_path / "bottle.usda"
    bottle_asset.write_text(
        '#usda 1.0\n(defaultPrim = "Bottle500")\n\n'
        'def Xform "Bottle500" {\n'
        '    def Xform "Visuals" { def Mesh "VIS_Bottle" {} }\n'
        '    def Scope "Collisions" {}\n'
        '}\n',
        encoding="utf-8",
    )
    bottle_hash = hashlib.sha256(bottle_asset.read_bytes()).hexdigest()
    workflow = CalibrationWorkflow()
    world_from_camera = np.eye(4)
    world_from_camera[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    world_from_camera[:3, 3] = [0.31, -0.22, 1.14]

    with pytest.raises(WorkflowGateError, match="hash"):
        workflow.export_calibration_bundle(
            output_dir=tmp_path / "bad",
            stage=StageContract(path=str(source_stage), sha256="0" * 64),
            world_from_camera=_transform(world_from_camera, "camera_high_optical", "table_world"),
            bottle_asset_path=str(bottle_asset),
            bottle_asset_sha256=bottle_hash,
            bottle_asset_prim="/Bottle500",
        )

    result = workflow.export_calibration_bundle(
        output_dir=tmp_path / "good",
        stage=StageContract(path=str(source_stage), sha256=actual_hash),
        world_from_camera=_transform(world_from_camera, "camera_high_optical", "table_world"),
        bottle_asset_path=str(bottle_asset),
        bottle_asset_sha256=bottle_hash,
        bottle_asset_prim="/Bottle500",
    )

    calibration = Path(result.calibration_layer)
    wrapper = Path(result.review_stage)
    assert calibration.is_file() and wrapper.is_file()
    wrapper_text = wrapper.read_text(encoding="utf-8")
    assert wrapper_text.index("@./calibration.usda@") < wrapper_text.index(f"@{source_stage}@")
    calibration_text = calibration.read_text(encoding="utf-8")
    assert "def Camera \"CameraHigh\"" in calibration_text
    assert "cameraHighWorldFromOpticalColumnVector" in calibration_text
    assert f"@{bottle_asset}@</Bottle500>" in calibration_text
    # GfMatrix4d uses row-vector transform semantics, so the column-vector
    # translation is authored in the final row rather than final column.
    assert "(0.31, -0.22, 1.14, 1)" in calibration_text
    manifest = Path(result.calibration_json).read_text(encoding="utf-8")
    assert '"vector_convention": "column-vector"' in manifest
    assert '"source_stage_modified": false' in manifest
    assert source_stage.read_text(encoding="utf-8") == '#usda 1.0\n(defaultPrim = "World")\n\ndef Xform "World" {}\n'


def test_export_rejects_missing_bottle_asset_and_wrong_frame_contract(tmp_path: Path):
    source_stage = tmp_path / "source.usda"
    source_stage.write_text('#usda 1.0\n(defaultPrim = "World")\ndef Xform "World" {}\n', encoding="utf-8")
    source_hash = hashlib.sha256(source_stage.read_bytes()).hexdigest()
    workflow = CalibrationWorkflow()

    with pytest.raises(WorkflowGateError, match="camera_high_optical"):
        workflow.export_calibration_bundle(
            output_dir=tmp_path / "bad-frame",
            stage=StageContract(path=str(source_stage), sha256=source_hash),
            world_from_camera=_transform(np.eye(4), "some_camera", "table_world"),
            bottle_asset_path=str(tmp_path / "missing.usd"),
            bottle_asset_sha256="0" * 64,
            bottle_asset_prim="/Bottle500",
        )
