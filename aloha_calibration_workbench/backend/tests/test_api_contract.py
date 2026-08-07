from pathlib import Path

from calibration_workbench.capture_api import create_capture_app
from calibration_workbench.models import CaptureStatus
from calibration_workbench.models import FactoryIntrinsics
from calibration_workbench.models import PreflightReport
from calibration_workbench.models import PreflightStatus
from calibration_workbench.orchestrator_api import create_orchestrator_app
from calibration_workbench.sessions import SessionStore
from calibration_workbench.workflow import TagPoseCaptureBatch
from fastapi.testclient import TestClient


def _report() -> PreflightReport:
    return PreflightReport(
        status=PreflightStatus.READY,
        registry_source="/project/aloha_stationary.yaml",
        registry_sha256="a" * 64,
        cameras=[],
        issues=[],
    )


class FakePreflightService:
    def run(self) -> PreflightReport:
        return _report()


class FakeBottleCaptureService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def capture_bottle_tag_batch(
        self,
        *,
        session_id: str,
        tag_id: int,
        tag_size_m: float,
        frame_count: int,
    ) -> TagPoseCaptureBatch:
        self.calls.append(
            {
                "session_id": session_id,
                "tag_id": tag_id,
                "tag_size_m": tag_size_m,
                "frame_count": frame_count,
            }
        )
        return TagPoseCaptureBatch(
            tag_id=tag_id,
            samples=[],
            total_frames=frame_count,
            detected_frames=0,
        )


class FakeCaptureClient:
    def run_preflight(self) -> PreflightReport:
        return _report()

    def start_intrinsics(self, session_id: str, role: str) -> CaptureStatus:
        assert role == "cam_high"
        return CaptureStatus(
            state="STREAMING",
            session_id=session_id,
            role=role,
            serial="130322270656",
            profile={"width": 640, "height": 480, "fps": 60, "format": "rgb8"},
            pipeline_started=True,
            factory_intrinsics=FactoryIntrinsics(
                width=640,
                height=480,
                fx=600.0,
                fy=601.0,
                cx=320.0,
                cy=240.0,
                distortion_model="brown_conrady",
                distortion_coefficients=[0.0] * 5,
            ),
        )


class FailingCaptureClient:
    def run_preflight(self) -> PreflightReport:
        raise ConnectionError("capture agent offline")


def test_capture_agent_exposes_read_only_safety_contract():
    client = TestClient(create_capture_app(FakePreflightService()))

    health = client.get("/health")
    response = client.post("/api/preflight")

    assert health.status_code == 200
    assert health.json() == {
        "service": "capture-agent",
        "status": "ok",
        "bind_policy": "localhost-only",
        "robot_command_api": False,
        "capture_pipeline_api": False,
    }
    assert response.status_code == 200
    assert response.json()["status"] == "READY"
    assert response.json()["pipeline_started"] is False
    assert response.json()["hardware_reset_called"] is False


def test_capture_agent_has_a_separate_bottle_tag_endpoint():
    capture = FakeBottleCaptureService()
    client = TestClient(create_capture_app(FakePreflightService(), capture))

    response = client.post(
        "/api/bottle-tag/capture",
        json={
            "session_id": "cal-20260805T120000-1234abcd",
            "tag_id": 1,
            "tag_size_m": 0.080,
            "frame_count": 150,
        },
    )

    assert response.status_code == 200
    assert response.json()["tag_id"] == 1
    assert capture.calls[0]["tag_id"] == 1

    rejected = client.post(
        "/api/bottle-tag/capture",
        json={
            "session_id": "cal-20260805T120000-1234abcd",
            "tag_id": 0,
            "tag_size_m": 0.080,
            "frame_count": 150,
        },
    )
    assert rejected.status_code == 422


def test_orchestrator_creates_session_and_persists_preflight_artifact(tmp_path: Path):
    store = SessionStore(tmp_path)
    client = TestClient(create_orchestrator_app(FakeCaptureClient(), store))

    created = client.post("/api/sessions", json={"name": "first-camera-preflight"})
    assert created.status_code == 201
    session_id = created.json()["id"]

    preflight = client.post(f"/api/sessions/{session_id}/actions/preflight")
    fetched = client.get(f"/api/sessions/{session_id}")

    assert preflight.status_code == 200
    assert preflight.json()["status"] == "READY"
    assert fetched.status_code == 200
    assert fetched.json()["state"] == "PREFLIGHT_READY"
    assert fetched.json()["latest_preflight_sha256"]
    assert (tmp_path / session_id / "session.json").is_file()
    assert (tmp_path / session_id / "artifacts/preflight.json").is_file()


def test_orchestrator_rejects_unknown_session(tmp_path: Path):
    client = TestClient(create_orchestrator_app(FakeCaptureClient(), SessionStore(tmp_path)))

    response = client.post("/api/sessions/not-a-session/actions/preflight")

    assert response.status_code == 404


def test_orchestrator_maps_capture_agent_failure_without_writing_false_evidence(tmp_path: Path):
    store = SessionStore(tmp_path)
    client = TestClient(create_orchestrator_app(FailingCaptureClient(), store))
    session_id = client.post("/api/sessions", json={"name": "offline-capture-agent"}).json()["id"]

    response = client.post(f"/api/sessions/{session_id}/actions/preflight")

    assert response.status_code == 502
    assert response.json()["detail"] == "Capture agent preflight failed"
    assert store.get(session_id).state == "SETUP"
    assert not (tmp_path / session_id / "artifacts/preflight.json").exists()


def test_preflight_artifact_cannot_be_overwritten(tmp_path: Path):
    store = SessionStore(tmp_path)
    client = TestClient(create_orchestrator_app(FakeCaptureClient(), store))
    session_id = client.post("/api/sessions", json={"name": "immutable-preflight"}).json()["id"]

    assert client.post(f"/api/sessions/{session_id}/actions/preflight").status_code == 200
    second = client.post(f"/api/sessions/{session_id}/actions/preflight")

    assert second.status_code == 409
    assert "create a new session" in second.json()["detail"]


def test_orchestrator_starts_intrinsics_only_after_ready_preflight(tmp_path: Path):
    store = SessionStore(tmp_path)
    client = TestClient(create_orchestrator_app(FakeCaptureClient(), store))
    session_id = client.post("/api/sessions", json={"name": "cam-high-intrinsics"}).json()["id"]

    blocked = client.post(f"/api/sessions/{session_id}/actions/intrinsics/start", json={"role": "cam_high"})
    assert blocked.status_code == 409

    assert client.post(f"/api/sessions/{session_id}/actions/preflight").status_code == 200
    started = client.post(f"/api/sessions/{session_id}/actions/intrinsics/start", json={"role": "cam_high"})

    assert started.status_code == 200
    assert started.json()["state"] == "STREAMING"
    assert started.json()["profile"] == {
        "stream": "color",
        "width": 640,
        "height": 480,
        "fps": 60,
        "format": "rgb8",
    }
    assert store.get(session_id).state == "INTRINSICS_CAPTURING"
    assert (tmp_path / session_id / "artifacts/intrinsics_start.json").is_file()
