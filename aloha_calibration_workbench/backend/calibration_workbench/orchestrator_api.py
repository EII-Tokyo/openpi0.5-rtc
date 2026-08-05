from __future__ import annotations

from typing import Protocol

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import CreateSessionRequest
from .models import PreflightReport
from .models import SessionRecord
from .sessions import SessionNotFoundError
from .sessions import SessionStore
from .sessions import SessionTransitionError


class CaptureClient(Protocol):
    def run_preflight(self) -> PreflightReport: ...


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

    return app
