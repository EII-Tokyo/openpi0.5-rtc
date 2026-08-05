from __future__ import annotations

from typing import Protocol

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import Response

from .intrinsics_capture import CaptureConflictError
from .intrinsics_capture import CaptureNotRunningError
from .intrinsics_capture import IntrinsicsCaptureService
from .models import CaptureStatus
from .models import IntrinsicsStartRequest
from .models import PreflightReport
from .models import SampleRecord


class PreflightRunner(Protocol):
    def run(self) -> PreflightReport: ...


def create_capture_app(
    service: PreflightRunner,
    capture: IntrinsicsCaptureService | None = None,
) -> FastAPI:
    app = FastAPI(title="ALOHA Calibration Capture Agent")

    @app.get("/health")
    def health() -> dict[str, str | bool]:
        return {
            "service": "capture-agent",
            "status": "ok",
            "bind_policy": "localhost-only",
            "robot_command_api": False,
            "capture_pipeline_api": capture is not None,
        }

    @app.post("/api/preflight", response_model=PreflightReport)
    def run_preflight() -> PreflightReport:
        return service.run()

    @app.get("/api/intrinsics/status", response_model=CaptureStatus)
    def intrinsics_status() -> CaptureStatus:
        if capture is None:
            return CaptureStatus(state="UNAVAILABLE")
        return capture.status()

    @app.post("/api/intrinsics/start", response_model=CaptureStatus)
    def start_intrinsics(request: IntrinsicsStartRequest) -> CaptureStatus:
        if capture is None:
            raise HTTPException(status_code=503, detail="Capture pipeline is unavailable")
        try:
            return capture.start(session_id=request.session_id, role=request.role)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except CaptureConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/intrinsics/preview.jpg")
    def intrinsics_preview() -> Response:
        if capture is None:
            raise HTTPException(status_code=503, detail="Capture pipeline is unavailable")
        try:
            preview = capture.preview()
        except CaptureNotRunningError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return Response(
            content=preview.jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store", "X-Frame-Number": str(preview.observation.frame_number)},
        )

    @app.post("/api/intrinsics/sample", response_model=SampleRecord)
    def capture_intrinsics_sample() -> SampleRecord:
        if capture is None:
            raise HTTPException(status_code=503, detail="Capture pipeline is unavailable")
        try:
            return capture.capture_sample()
        except CaptureNotRunningError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/intrinsics/stop", response_model=CaptureStatus)
    def stop_intrinsics() -> CaptureStatus:
        if capture is None:
            raise HTTPException(status_code=503, detail="Capture pipeline is unavailable")
        return capture.stop()

    return app
