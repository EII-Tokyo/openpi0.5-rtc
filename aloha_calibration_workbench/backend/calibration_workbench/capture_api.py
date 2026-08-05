from __future__ import annotations

from typing import Protocol

from fastapi import FastAPI

from .models import PreflightReport


class PreflightRunner(Protocol):
    def run(self) -> PreflightReport: ...


def create_capture_app(service: PreflightRunner) -> FastAPI:
    app = FastAPI(title="ALOHA Calibration Capture Agent")

    @app.get("/health")
    def health() -> dict[str, str | bool]:
        return {
            "service": "capture-agent",
            "status": "ok",
            "bind_policy": "localhost-only",
            "robot_command_api": False,
            "capture_pipeline_api": False,
        }

    @app.post("/api/preflight", response_model=PreflightReport)
    def run_preflight() -> PreflightReport:
        return service.run()

    return app
