from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
from fastapi.testclient import TestClient

from calibration_workbench.models import FactoryIntrinsics
from calibration_workbench.intrinsics_capture import RgbSnapshot
from calibration_workbench.orchestrator_api import create_orchestrator_app
from calibration_workbench.sessions import SessionStore
from calibration_workbench.workflow import FactoryCameraSnapshot
from calibration_workbench.workflow import TransformRecord

from test_api_contract import FakeCaptureClient


class WorkflowCaptureClient(FakeCaptureClient):
    def snapshot_factory_intrinsics(self) -> list[FactoryCameraSnapshot]:
        roles = {
            "cam_high": "130322270656",
            "cam_low": "218622270440",
            "wrist_left": "130322272542",
            "wrist_right": "218622278936",
        }
        return [
            FactoryCameraSnapshot(
                role=role,
                serial=serial,
                firmware="5.16.0.1",
                profile={"width": 640, "height": 480, "fps": 60, "format": "rgb8"},
                intrinsics=FactoryIntrinsics(
                    width=640,
                    height=480,
                    fx=600.0,
                    fy=600.0,
                    cx=320.0,
                    cy=240.0,
                    distortion_model="none",
                    distortion_coefficients=[0.0] * 5,
                ),
            )
            for role, serial in roles.items()
        ]

    def capture_table_snapshot(self, session_id: str) -> RgbSnapshot:
        return RgbSnapshot(
            jpeg=b"jpeg-evidence",
            attempt_id="A-001",
            frame_number=42,
            device_timestamp_ms=1234.5,
            image_sha256="b" * 64,
        )


def _identity(source: str, target: str) -> dict:
    return TransformRecord(source_frame=source, target_frame=target, matrix=np.eye(4).tolist()).model_dump(mode="json")


def test_workflow_api_freezes_factory_bundle_then_solves_world_origin(tmp_path: Path):
    store = SessionStore(tmp_path)
    client = TestClient(create_orchestrator_app(WorkflowCaptureClient(), store))
    session_id = client.post("/api/preflight-session").json()["id"]

    frozen = client.post(f"/api/sessions/{session_id}/actions/factory/freeze")
    assert frozen.status_code == 200
    assert len(frozen.json()["cameras"]) == 4
    assert store.get(session_id).state == "FACTORY_INTRINSICS_FROZEN"

    samples = [
        {
            "camera_from_tag": _identity("tag", "camera_high_optical"),
            "reprojection_rms_px": 0.4,
            "frame_id": f"F-{index:03d}",
            "device_timestamp_ms": index * 16.7,
            "image_sha256": hashlib.sha256(f"frame-{index}".encode()).hexdigest(),
        }
        for index in range(160)
    ]
    solved = client.post(
        f"/api/sessions/{session_id}/actions/world-origin/solve",
        json={
            "samples": samples,
            "world_from_tag": _identity("tag", "table_world"),
        },
    )

    assert solved.status_code == 200
    assert solved.json()["status"] == "WORLD_ORIGIN_SOLVED"
    assert store.get(session_id).state == "WORLD_ORIGIN_SOLVED"

    cached = client.post(
        f"/api/sessions/{session_id}/actions/world-origin/capture-solve",
        json={"tag_size_m": 0.080, "tag_plane_height_m": 0.0, "frame_count": 200},
    )
    assert cached.status_code == 200
    assert cached.json() == solved.json()

    snapshot = client.post(f"/api/sessions/{session_id}/actions/table/snapshot")
    assert snapshot.status_code == 200
    assert snapshot.content == b"jpeg-evidence"
    assert snapshot.headers["X-Attempt-Id"] == "A-001"
    assert snapshot.headers["X-Image-Sha256"] == "b" * 64
    assert (tmp_path / session_id / "artifacts/factory_intrinsics.json").is_file()
    assert (tmp_path / session_id / "artifacts/world_origin.json").is_file()

    table_points = []
    for row, y in enumerate((0.18, 0.0, -0.18), start=1):
        for column, x in enumerate((-0.35, 0.0, 0.35), start=1):
            table_points.append(
                {
                    "id": f"P{row}{column}",
                    "color": ("blue", "magenta", "lime")[row - 1],
                    "measurement_1_xy_m": [x, y],
                    "measurement_2_xy_m": [x + 0.001, y],
                }
            )
    contract = client.post(
        f"/api/sessions/{session_id}/actions/table-contract/freeze",
        json={
            "contract_id": "table-dots-20260805",
            "revision": 1,
            "measurement_method": "steel-ruler-and-square",
            "points": table_points,
        },
    )
    assert contract.status_code == 200
    assert contract.json()["status"] == "TABLE_POINT_CONTRACT_FROZEN"
    assert store.get(session_id).state == "TABLE_POINT_CONTRACT_FROZEN"
    assert (tmp_path / session_id / "artifacts/table_point_contract.json").is_file()

    forbidden_truth = client.post(
        f"/api/sessions/{session_id}/actions/table/solve",
        json={
            "observations": [
                {
                    "id": "P11",
                    "image_uv_px": [100.0, 100.0],
                    "operator_confirmed": True,
                    "world_xyz_m": [-0.35, 0.18, 0.0],
                }
            ]
        },
    )
    assert forbidden_truth.status_code == 422

    cached_payload = {
        "status": "WORLD_REGISTRATION_VALIDATED",
        "validation_scope": "tabletop-xy-cross-validation",
        "world_from_camera": _identity("camera_high_optical", "table_world"),
        "solve_point_ids": ["P12", "P13", "P21", "P22", "P31", "P33"],
        "held_out_point_ids": ["P11", "P23", "P32"],
        "solve_reprojection_rms_px": 1.8,
        "initial_reprojection_rms_px": 3.0,
        "refinement_translation_m": 0.031,
        "refinement_rotation_deg": 2.0,
        "held_out_rms_m": 0.007,
        "held_out_max_m": 0.009,
        "quality_gate_passed": False,
        "operator_override": {"reason": "explicit test approval"},
    }
    store.record_workflow_artifact(
        session_id,
        expected_state="TABLE_POINT_CONTRACT_FROZEN",
        next_state="WORLD_REGISTRATION_VALIDATED",
        artifact_name="table_registration.json",
        payload=cached_payload,
    )
    cached = client.post(
        f"/api/sessions/{session_id}/actions/table/solve",
        json={"observations": []},
    )
    assert cached.status_code == 200
    assert cached.json()["status"] == "WORLD_REGISTRATION_VALIDATED"
    assert cached.json()["solve_reprojection_rms_px"] == 1.8


def test_workflow_api_rejects_skipping_factory_freeze(tmp_path: Path):
    client = TestClient(create_orchestrator_app(WorkflowCaptureClient(), SessionStore(tmp_path)))
    session_id = client.post("/api/preflight-session").json()["id"]
    samples = [
        {
            "camera_from_tag": _identity("tag", "camera_high_optical"),
            "reprojection_rms_px": 0.4,
            "frame_id": f"F-{index:03d}",
            "device_timestamp_ms": index * 16.7,
            "image_sha256": hashlib.sha256(f"frame-{index}".encode()).hexdigest(),
        }
        for index in range(160)
    ]

    response = client.post(
        f"/api/sessions/{session_id}/actions/world-origin/solve",
        json={"samples": samples, "world_from_tag": _identity("tag", "table_world")},
    )

    assert response.status_code == 409
    assert "FACTORY_INTRINSICS_FROZEN" in response.json()["detail"]
