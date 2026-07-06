from __future__ import annotations

import os
from pathlib import Path
import tempfile


_TEST_STATE_DIR = Path(tempfile.mkdtemp(prefix="openpi-rlt-backend-test-"))

os.environ.setdefault("RLT_SEGMENT_DB_PATH", str(_TEST_STATE_DIR / "segments.sqlite3"))
os.environ.setdefault("RLT_STATE_PATH", str(_TEST_STATE_DIR / "rlt_control_state.json"))
os.environ.setdefault("EII_PILOT_ENABLE_ROS", "0")
