#!/usr/bin/env python3
"""Open the A19 candidate stage in Isaac runtime and record a bounded smoke audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


DEFAULT_CONFIG = Path("aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml")


def _start_isaac_headless():
    from isaacsim import SimulationApp

    return SimulationApp({"headless": True})


def runtime_open_audit(config_path: Path) -> dict:
    import omni.usd

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    outputs = {key: Path(value) for key, value in config["outputs"].items()}
    stage_path = outputs["a19_clean_articulation_candidate"].resolve()
    output_path = outputs["a19_clean_articulation_candidate_audit_json"].with_name(
        "a19_clean_articulation_candidate_runtime_open_audit.json"
    )

    context = omni.usd.get_context()
    opened = bool(context.open_stage(str(stage_path)))
    stage = context.get_stage()
    type_counts: dict[str, int] = {}
    api_counts: dict[str, int] = {}
    if opened and stage is not None:
        for prim in stage.Traverse():
            prim_type = prim.GetTypeName() or "Typeless"
            type_counts[prim_type] = type_counts.get(prim_type, 0) + 1
            for schema in prim.GetAppliedSchemas():
                api_counts[schema] = api_counts.get(schema, 0) + 1

    result = {
        "ok": opened
        and stage is not None
        and str(stage.GetDefaultPrim().GetPath()) == "/aloha"
        and type_counts.get("PhysicsRevoluteJoint", 0) == 12
        and type_counts.get("PhysicsPrismaticJoint", 0) == 4
        and type_counts.get("PhysicsFixedJoint", 0) == 5
        and api_counts.get("PhysicsArticulationRootAPI", 0) == 1
        and api_counts.get("PhysicsCollisionAPI", 0) == 0,
        "status": "PASS_A19_SINGLE_ROOT_ISAAC_RUNTIME_OPEN_STAGE_SMOKE"
        if opened
        and stage is not None
        and str(stage.GetDefaultPrim().GetPath()) == "/aloha"
        and type_counts.get("PhysicsRevoluteJoint", 0) == 12
        and type_counts.get("PhysicsPrismaticJoint", 0) == 4
        and type_counts.get("PhysicsFixedJoint", 0) == 5
        and api_counts.get("PhysicsArticulationRootAPI", 0) == 1
        and api_counts.get("PhysicsCollisionAPI", 0) == 0
        else "FAIL_A19_SINGLE_ROOT_ISAAC_RUNTIME_OPEN_STAGE_SMOKE",
        "stage_path": str(stage_path),
        "opened": opened,
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if opened and stage and stage.GetDefaultPrim() else None,
        "type_counts": type_counts,
        "api_counts": api_counts,
        "physics_stepped": False,
        "control_ready": False,
        "replay_ready": False,
        "training_eligible": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    app = _start_isaac_headless()
    try:
        result = runtime_open_audit(args.config)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"type_counts", "api_counts"}
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(0 if result["ok"] else 1)
    finally:
        app.close()


if __name__ == "__main__":
    main()
