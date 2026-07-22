#!/usr/bin/env python3
"""One-shot, no-step A20 Isaac runtime articulation discovery probe."""

from __future__ import annotations

import argparse
from datetime import UTC
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

from isaacsim import SimulationApp
import yaml

MARKER = "A20_RUNTIME_DISCOVERY_JSON="
CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml")
SAFETY = {
    "physics_stepped": False,
    "actions_applied": False,
    "targets_written": False,
    "stage_saved": False,
}


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invocation-id", required=True)
    args = parser.parse_args()
    started = _now()
    app = SimulationApp({"headless": True})
    payload: dict[str, object]
    try:
        from pxr import Usd
        from pxr import UsdPhysics

        from aloha_isaac_rebuild.scripts.audit_a20_usd_dof_metadata import expected_dof_records

        repo = Path.cwd().resolve()
        config_path = (repo / CONFIG).resolve()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        outputs = config["outputs"]
        stage_path = (repo / outputs["a19_clean_articulation_candidate"]).resolve()
        mapping_path = (repo / outputs["a17_clean_articulation_mapping_plan_json"]).resolve()
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        records = expected_dof_records(mapping)
        stage = Usd.Stage.Open(str(stage_path))
        roots = [str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
        status = "BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION"
        payload = {
            "status": status,
            "probe_returncode": 0,
            "invocation_id": args.invocation_id,
            "pid": os.getpid(),
            "started_at": started,
            "finished_at": _now(),
            "isaac_sim_version": importlib.metadata.version("isaacsim"),
            "inputs": {
                "config": {"path": str(config_path), "sha256": _digest(config_path)},
                "mapping": {"path": str(mapping_path), "sha256": _digest(mapping_path)},
                "stage": {"path": str(stage_path), "sha256": _digest(stage_path)},
            },
            "articulation_root": roots[0] if len(roots) == 1 else None,
            "articulation_count": len(roots),
            "dof_count": len(records),
            "valid_handle": False,
            "records": records,
            "requires_unapproved_initialization": True,
            "initialization_operations": ["timeline Play", "physics simulation step"],
            **SAFETY,
        }
        if len(roots) != 1 or len(records) != 16:
            payload["status"] = "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY"
            payload["requires_unapproved_initialization"] = False
    except Exception as exc:
        payload = {
            "status": "FAIL_A20_RUNTIME_ARTICULATION_DISCOVERY",
            "probe_returncode": 1,
            "invocation_id": args.invocation_id,
            "pid": os.getpid(),
            "started_at": started,
            "finished_at": _now(),
            "isaac_sim_version": importlib.metadata.version("isaacsim"),
            "inputs": {},
            "articulation_root": None,
            "articulation_count": 0,
            "dof_count": 0,
            "valid_handle": False,
            "records": [],
            "requires_unapproved_initialization": False,
            "initialization_operations": [],
            "errors": [{"code": "probe_error", "message": str(exc)}],
            **SAFETY,
        }
    finally:
        print(
            MARKER
            + json.dumps(payload, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        app.close()
    return 0 if payload["status"] == "BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
