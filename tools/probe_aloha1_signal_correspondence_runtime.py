#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Read the dual-follower Stage through the Isaac Sim 5.1 runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any

from tools.aloha1_mapping.signal_correspondence import RUNTIME_SPECS
from tools.aloha1_mapping.signal_correspondence import build_signal_mapping_plan
from tools.aloha1_mapping.signal_correspondence import canonical_dof_name

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE = (
    ROOT / "assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda"
)
DEFAULT_OUTPUT = ROOT / "reports/aloha1_mapping/aloha1_signal_correspondence_runtime_inventory.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dof_record(properties: Any, index: int, name: str) -> dict[str, Any]:
    fields = properties.dtype.names or ()
    integer_fields = {"type", "driveMode"}
    return {
        "index": index,
        "runtime_name": name,
        **{
            key: (
                bool(properties[index][key])
                if key == "hasLimits"
                else int(properties[index][key])
                if key in integer_fields
                else float(properties[index][key])
            )
            for key in (
                "type",
                "hasLimits",
                "lower",
                "upper",
                "driveMode",
                "maxVelocity",
                "maxEffort",
                "stiffness",
                "damping",
            )
            if key in fields
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    stage_path = args.stage.resolve(strict=True)
    stage_hash_before = _sha256(stage_path)

    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.core.utils.xforms import get_world_pose

    if not open_stage(str(stage_path)):
        raise RuntimeError(f"failed to open Stage: {stage_path}")
    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
    )
    world.get_physics_context().set_solve_articulation_contact_last(True)
    articulations = {}
    for robot, spec in RUNTIME_SPECS.items():
        articulation = SingleArticulation(
            prim_path=spec["articulation_path"],
            name=f"signal_inventory_{robot}",
            reset_xform_properties=False,
        )
        world.scene.add(articulation)
        articulations[robot] = articulation
    world.reset()

    plan = build_signal_mapping_plan(ROOT)
    robots = {}
    for robot, articulation in articulations.items():
        actual_order = list(articulation.dof_names)
        expected_order = RUNTIME_SPECS[robot]["runtime_expected_order"]
        eef_position, eef_orientation = get_world_pose(RUNTIME_SPECS[robot]["end_effector_path"])
        base_position, base_orientation = get_world_pose(RUNTIME_SPECS[robot]["base_link_path"])
        properties = articulation.dof_properties.copy()
        robots[robot] = {
            "status": "PASS" if actual_order == expected_order else "FAIL",
            "articulation_path": RUNTIME_SPECS[robot]["articulation_path"],
            "articulation_count": 1,
            "num_dof": int(articulation.num_dof),
            "num_bodies": int(articulation.num_bodies),
            # Isaac Sim 5.1 Articulation does not expose body_names publicly.
            "body_names": list(articulation._articulation_view.body_names),  # noqa: SLF001
            "expected_dof_order": expected_order,
            "runtime_dof_order": actual_order,
            "runtime_canonical_order": [canonical_dof_name(robot, name) for name in actual_order],
            "dof_properties": [_dof_record(properties, index, name) for index, name in enumerate(actual_order)],
            "initial_qpos": [float(value) for value in articulation.get_joint_positions()],
            "end_effector_path": RUNTIME_SPECS[robot]["end_effector_path"],
            "base_link_path": RUNTIME_SPECS[robot]["base_link_path"],
            "base_link_position_m": [float(value) for value in base_position],
            "base_link_orientation_wxyz": [float(value) for value in base_orientation],
            "end_effector_position_m": [float(value) for value in eef_position],
            "end_effector_orientation_wxyz": [float(value) for value in eef_orientation],
            "mapping_plan": plan["robots"][robot],
        }

    stage_hash_after = _sha256(stage_path)
    stage_immutable = stage_hash_before == stage_hash_after
    status = "PASS" if stage_immutable and all(record["status"] == "PASS" for record in robots.values()) else "FAIL"
    report = {
        "schema_version": 1,
        "status": status,
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
            "physics_frequency_hz": 60,
            "solve_articulation_contact_last": True,
        },
        "stage": {
            "absolute_path": str(stage_path),
            "sha256_before": stage_hash_before,
            "sha256_after": stage_hash_after,
            "immutable": stage_immutable,
            "root_prim": "/World",
        },
        "articulation_count": len(articulations),
        "robots": robots,
        "task_8": "NOT_RUN",
        "real_robot_connected": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": status,
                "articulation_count": len(articulations),
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if status == "PASS" else 1


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "create_new_stage": False,
            "disable_viewport_updates": True,
        }
    )
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
