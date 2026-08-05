from __future__ import annotations

from typing import Protocol

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import Response

from .intrinsics_capture import CaptureConflictError
from .intrinsics_capture import CaptureNotRunningError
from .intrinsics_capture import IntrinsicsCaptureService
from .intrinsics_capture import RgbSnapshot
from .models import CaptureStatus
from .models import IntrinsicsStartRequest
from .models import PreflightReport
from .models import SampleRecord
from .workflow import FactoryCameraSnapshot
from .workflow import WorldOriginCaptureBatch
from .workflow import WorldOriginCaptureRequest


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

    @app.post("/api/factory-intrinsics/snapshot", response_model=list[FactoryCameraSnapshot])
    def snapshot_factory_intrinsics() -> list[FactoryCameraSnapshot]:
        if capture is None:
            raise HTTPException(status_code=503, detail="Capture pipeline is unavailable")
        try:
            return capture.snapshot_factory_intrinsics()
        except CaptureConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/world-origin/capture", response_model=WorldOriginCaptureBatch)
    def capture_world_origin(request: WorldOriginCaptureRequest) -> WorldOriginCaptureBatch:
        if capture is None:
            raise HTTPException(status_code=503, detail="Capture pipeline is unavailable")
        try:
            return capture.capture_world_origin_batch(
                session_id=request.session_id,
                tag_size_m=request.tag_size_m,
                tag_plane_height_m=request.tag_plane_height_m,
                frame_count=request.frame_count,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except CaptureConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/table/snapshot")
    def capture_table_snapshot(session_id: str) -> Response:
        if capture is None:
            raise HTTPException(status_code=503, detail="Capture pipeline is unavailable")
        try:
            snapshot: RgbSnapshot = capture.capture_table_snapshot(session_id=session_id)
            return Response(
                content=snapshot.jpeg,
                media_type="image/jpeg",
                headers={
                    "Cache-Control": "no-store",
                    "X-Attempt-Id": snapshot.attempt_id,
                    "X-Frame-Number": str(snapshot.frame_number),
                    "X-Device-Timestamp-Ms": str(snapshot.device_timestamp_ms),
                    "X-Image-Sha256": snapshot.image_sha256,
                },
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except CaptureConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

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
