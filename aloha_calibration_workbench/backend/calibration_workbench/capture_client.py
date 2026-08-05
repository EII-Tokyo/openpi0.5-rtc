from __future__ import annotations

import httpx

from .intrinsics_capture import RgbSnapshot
from .models import CaptureStatus
from .models import PreflightReport
from .models import SampleRecord
from .workflow import FactoryCameraSnapshot
from .workflow import WorldOriginCaptureBatch


class CaptureAgentClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8017", timeout_seconds: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def run_preflight(self) -> PreflightReport:
        response = httpx.post(f"{self._base_url}/api/preflight", timeout=self._timeout_seconds)
        response.raise_for_status()
        return PreflightReport.model_validate(response.json())

    def snapshot_factory_intrinsics(self) -> list[FactoryCameraSnapshot]:
        response = httpx.post(
            f"{self._base_url}/api/factory-intrinsics/snapshot",
            timeout=max(self._timeout_seconds, 60.0),
        )
        response.raise_for_status()
        return [FactoryCameraSnapshot.model_validate(item) for item in response.json()]

    def capture_world_origin(
        self,
        session_id: str,
        *,
        tag_size_m: float,
        tag_plane_height_m: float,
        frame_count: int,
    ) -> WorldOriginCaptureBatch:
        response = httpx.post(
            f"{self._base_url}/api/world-origin/capture",
            json={
                "session_id": session_id,
                "tag_size_m": tag_size_m,
                "tag_plane_height_m": tag_plane_height_m,
                "frame_count": frame_count,
            },
            timeout=max(self._timeout_seconds, 60.0),
        )
        response.raise_for_status()
        return WorldOriginCaptureBatch.model_validate(response.json())

    def capture_table_snapshot(self, session_id: str) -> RgbSnapshot:
        response = httpx.post(
            f"{self._base_url}/api/table/snapshot",
            params={"session_id": session_id},
            timeout=max(self._timeout_seconds, 30.0),
        )
        response.raise_for_status()
        return RgbSnapshot(
            jpeg=response.content,
            attempt_id=response.headers["X-Attempt-Id"],
            frame_number=int(response.headers["X-Frame-Number"]),
            device_timestamp_ms=float(response.headers["X-Device-Timestamp-Ms"]),
            image_sha256=response.headers["X-Image-Sha256"],
        )

    def start_intrinsics(self, session_id: str, role: str) -> CaptureStatus:
        response = httpx.post(
            f"{self._base_url}/api/intrinsics/start",
            json={"session_id": session_id, "role": role},
            timeout=max(self._timeout_seconds, 20.0),
        )
        response.raise_for_status()
        return CaptureStatus.model_validate(response.json())

    def intrinsics_status(self) -> CaptureStatus:
        response = httpx.get(f"{self._base_url}/api/intrinsics/status", timeout=self._timeout_seconds)
        response.raise_for_status()
        return CaptureStatus.model_validate(response.json())

    def preview_jpeg(self) -> bytes:
        response = httpx.get(f"{self._base_url}/api/intrinsics/preview.jpg", timeout=self._timeout_seconds)
        response.raise_for_status()
        return response.content

    def capture_sample(self) -> SampleRecord:
        response = httpx.post(f"{self._base_url}/api/intrinsics/sample", timeout=self._timeout_seconds)
        response.raise_for_status()
        return SampleRecord.model_validate(response.json())

    def stop_intrinsics(self) -> CaptureStatus:
        response = httpx.post(f"{self._base_url}/api/intrinsics/stop", timeout=self._timeout_seconds)
        response.raise_for_status()
        return CaptureStatus.model_validate(response.json())
