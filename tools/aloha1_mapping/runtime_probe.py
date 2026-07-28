"""Isaac Sim 5.1 runtime articulation inventory for ALOHA 1 assets."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any


def build_probe_targets(
    project_root: Path,
    *,
    enable_leaders: bool,
) -> list[dict[str, str]]:
    root = project_root.resolve(strict=True)
    specifications = [
        ("follower_left", "follower_vx300s"),
        ("follower_right", "follower_vx300s"),
    ]
    if enable_leaders:
        specifications.extend(
            [
                ("leader_left", "leader_wx250s"),
                ("leader_right", "leader_wx250s"),
            ]
        )
    targets: list[dict[str, str]] = []
    for name, family in specifications:
        usd = root / "assets/Trossen/ALOHA1/1.0" / family / name / f"{name}.usd"
        if not usd.is_file():
            raise FileNotFoundError(f"USD asset is unavailable: {usd}")
        targets.append(
            {
                "name": name,
                "usd": str(usd.resolve()),
                "stage_prim": f"/World/{name}",
                "articulation_prim": f"/World/{name}/root_joint",
            }
        )
    return targets


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def probe_runtime(
    targets: Sequence[dict[str, str]],
    *,
    report_path: Path,
) -> dict[str, Any]:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage

        world = World(stage_units_in_meters=1.0, backend="numpy", device="cpu")
        articulations = []
        for target in targets:
            add_reference_to_stage(
                usd_path=target["usd"],
                prim_path=target["stage_prim"],
            )
            articulation = SingleArticulation(
                prim_path=target["articulation_prim"],
                name=target["name"],
                reset_xform_properties=False,
            )
            world.scene.add(articulation)
            articulations.append((target, articulation))
        world.reset()

        robots: list[dict[str, Any]] = []
        for target, articulation in articulations:
            properties = articulation.dof_properties
            positions = articulation.get_joint_positions()
            robots.append(
                {
                    "name": target["name"],
                    "usd": target["usd"],
                    "articulation_prim": target["articulation_prim"],
                    "num_dof": articulation.num_dof,
                    "num_bodies": articulation.num_bodies,
                    "dof_order": list(articulation.dof_names),
                    "dofs": [
                        {
                            "index": index,
                            "name": name,
                            "type": int(properties[index]["type"]),
                            "has_limits": bool(properties[index]["hasLimits"]),
                            "lower": float(properties[index]["lower"]),
                            "upper": float(properties[index]["upper"]),
                            "drive_mode": int(properties[index]["driveMode"]),
                            "max_velocity": float(
                                properties[index]["maxVelocity"]
                            ),
                            "max_effort": float(properties[index]["maxEffort"]),
                            "stiffness": float(properties[index]["stiffness"]),
                            "damping": float(properties[index]["damping"]),
                            "initial_position": float(positions[index]),
                        }
                        for index, name in enumerate(articulation.dof_names)
                    ],
                }
            )
        report = {
            "schema_version": 1,
            "status": "PASS",
            "isaac_sim": "5.1.0.0",
            "robots": robots,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finally:
        app.close()
    return report
