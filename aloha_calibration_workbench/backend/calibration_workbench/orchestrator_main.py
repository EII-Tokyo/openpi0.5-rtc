from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from .capture_client import CaptureAgentClient
from .orchestrator_api import create_orchestrator_app
from .sessions import SessionStore

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SESSION_ROOT = Path(os.environ.get("ALOHA_CALIBRATION_SESSION_ROOT", PROJECT_ROOT / ".calibration_sessions"))
CAPTURE_AGENT_URL = os.environ.get("ALOHA_CALIBRATION_CAPTURE_AGENT_URL", "http://127.0.0.1:8017")
app = create_orchestrator_app(CaptureAgentClient(CAPTURE_AGENT_URL), SessionStore(SESSION_ROOT))


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8016)
