from __future__ import annotations

from typing import Protocol

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .models import CaptureStatus
from .models import CreateSessionRequest
from .models import IntrinsicsRoleRequest
from .models import PreflightReport
from .models import SampleRecord
from .models import SessionRecord
from .sessions import SessionNotFoundError
from .sessions import SessionStore
from .sessions import SessionTransitionError


class CaptureClient(Protocol):
    def run_preflight(self) -> PreflightReport: ...

    def start_intrinsics(self, session_id: str, role: str) -> CaptureStatus: ...

    def intrinsics_status(self) -> CaptureStatus: ...

    def preview_jpeg(self) -> bytes: ...

    def capture_sample(self) -> SampleRecord: ...

    def stop_intrinsics(self) -> CaptureStatus: ...


def create_orchestrator_app(capture_client: CaptureClient, store: SessionStore) -> FastAPI:
    app = FastAPI(title="ALOHA Calibration Orchestrator")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:4173", "http://localhost:4173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health() -> dict[str, str | bool]:
        return {"service": "orchestrator", "status": "ok", "robot_command_api": False}

    @app.post("/api/sessions", response_model=SessionRecord, status_code=201)
    def create_session(request: CreateSessionRequest) -> SessionRecord:
        try:
            return store.create(request.name)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}", response_model=SessionRecord)
    def get_session(session_id: str) -> SessionRecord:
        try:
            return store.get(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Calibration session not found") from exc

    @app.post("/api/sessions/{session_id}/actions/preflight", response_model=PreflightReport)
    def run_preflight(session_id: str) -> PreflightReport:
        try:
            store.get(session_id)
            report = capture_client.run_preflight()
            store.record_preflight(session_id, report)
            return report
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Calibration session not found") from exc
        except SessionTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Capture agent preflight failed") from exc

    @app.post("/api/preflight-session", response_model=SessionRecord, status_code=201)
    def create_and_run_preflight() -> SessionRecord:
        record = store.create("camera-preflight")
        try:
            report = capture_client.run_preflight()
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Capture agent preflight failed") from exc
        return store.record_preflight(record.id, report)

    @app.post("/api/sessions/{session_id}/actions/intrinsics/start", response_model=CaptureStatus)
    def start_intrinsics(session_id: str, request: IntrinsicsRoleRequest) -> CaptureStatus:
        try:
            store.assert_intrinsics_start_allowed(session_id)
            status = capture_client.start_intrinsics(session_id, request.role)
            try:
                store.record_intrinsics_start(session_id, status)
            except Exception:
                capture_client.stop_intrinsics()
                raise
            return status
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Calibration session not found") from exc
        except SessionTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Capture agent intrinsics start failed") from exc

    @app.get("/api/intrinsics/status", response_model=CaptureStatus)
    def intrinsics_status() -> CaptureStatus:
        try:
            return capture_client.intrinsics_status()
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Capture agent status failed") from exc

    @app.get("/api/intrinsics/preview.jpg")
    def intrinsics_preview() -> Response:
        try:
            return Response(
                content=capture_client.preview_jpeg(),
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Capture agent preview failed") from exc

    @app.post("/api/intrinsics/sample", response_model=SampleRecord)
    def capture_sample() -> SampleRecord:
        try:
            return capture_client.capture_sample()
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Capture agent sample failed") from exc

    @app.post("/api/intrinsics/stop", response_model=CaptureStatus)
    def stop_intrinsics() -> CaptureStatus:
        try:
            return capture_client.stop_intrinsics()
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Capture agent stop failed") from exc

    return app
