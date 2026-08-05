from __future__ import annotations

import httpx

from .models import PreflightReport


class CaptureAgentClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8017", timeout_seconds: float = 10.0):
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def run_preflight(self) -> PreflightReport:
        response = httpx.post(f"{self._base_url}/api/preflight", timeout=self._timeout_seconds)
        response.raise_for_status()
        return PreflightReport.model_validate(response.json())
