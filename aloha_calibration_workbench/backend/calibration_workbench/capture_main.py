from __future__ import annotations

from pathlib import Path

import uvicorn

from .capture_api import create_capture_app
from .preflight import PreflightService
from .registry import load_candidate_registry
from .rs_cli_probe import RsEnumerateCliProbe

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROBOT_CONFIG = PROJECT_ROOT / "third_party/aloha_collection/config/robot/aloha_stationary.yaml"
registry = load_candidate_registry(ROBOT_CONFIG)
app = create_capture_app(PreflightService(registry=registry, probe=RsEnumerateCliProbe(registry.profile)))


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8017)
