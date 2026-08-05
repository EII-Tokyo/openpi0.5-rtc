from __future__ import annotations

import httpx

from .models import CaptureStatus
from .models import PreflightReport
from .models import SampleRecord


class CaptureAgentClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8017", timeout_seconds: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def run_preflight(self) -> PreflightReport:
        response = httpx.post(f"{self._base_url}/api/preflight", timeout=self._timeout_seconds)
        response.raise_for_status()
        return PreflightReport.model_validate(response.json())

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
