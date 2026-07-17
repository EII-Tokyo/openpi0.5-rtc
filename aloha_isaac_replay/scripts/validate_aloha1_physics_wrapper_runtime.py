from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

from aloha_isaac_replay.runtime.isaac_light_app import LIGHTWEIGHT_SIMULATION_APP_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEFT_USD = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase17_physics_layer_wrapper_20260718/aloha1_left_physics_layer_wrapper.usda"
DEFAULT_RIGHT_USD = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase17_physics_layer_wrapper_20260718/aloha1_right_physics_layer_wrapper.usda"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase18_runtime_articulation_20260718"


def _rel(path: str | Path | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    try:
        return value.tolist()
    except Exception:
        return repr(value)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _articulation_payload(art: Any) -> dict[str, Any]:
    view = art._articulation_view
    dof_names = list(art.dof_names)
    body_names = list(view.body_names)
    return {
        "prim_path": art.prim_path,
        "num_dof": int(art.num_dof),
        "num_bodies": int(art.num_bodies),
        "dof_names": dof_names,
        "body_names": body_names,
        "ee_body_candidates": [name for name in body_names if "ee" in name or "gripper" in name],
        "positions": _json_safe(art.get_joint_positions()),
        "velocities": _json_safe(art.get_joint_velocities()),
    }


def _gate(side_payload: dict[str, Any]) -> dict[str, bool]:
    expected_core = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    names = side_payload.get("dof_names", [])
    return {
        "has_dofs": side_payload.get("num_dof", 0) >= 6,
        "has_bodies": side_payload.get("num_bodies", 0) > 0,
        "has_core_dofs": all(name in names for name in expected_core),
        "has_ee_candidate": bool(side_payload.get("ee_body_candidates")),
    }


def _find_articulation_roots(stage: Any, prefix: str) -> list[str]:
    roots = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(prefix):
            continue
        schemas = [str(item) for item in prim.GetAppliedSchemas()]
        if "ArticulationRootAPI" in schemas or "PhysicsArticulationRootAPI" in schemas:
            roots.append(path)
    return roots


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate repaired ALOHA1 physics wrapper runtime articulations.")
    parser.add_argument("--left-usd", default=str(DEFAULT_LEFT_USD))
    parser.add_argument("--right-usd", default=str(DEFAULT_RIGHT_USD))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    json_path = output_dir / "physics_wrapper_runtime_articulation.json"
    md_path = output_dir / "physics_wrapper_runtime_articulation.md"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "left_usd": _rel(args.left_usd),
        "right_usd": _rel(args.right_usd),
        "status": "STARTED",
    }
    _write(json_path, payload)

    from isaacsim import SimulationApp

    app = SimulationApp(dict(LIGHTWEIGHT_SIMULATION_APP_CONFIG))
    try:
        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from pxr import Usd

        World.clear_instance()
        stage_utils.create_new_stage()
        world = World(stage_units_in_meters=1.0, backend="numpy", device="cpu")
        stage_utils.add_reference_to_stage(usd_path=str(Path(args.left_usd).resolve()), prim_path="/World/left")
        stage_utils.add_reference_to_stage(usd_path=str(Path(args.right_usd).resolve()), prim_path="/World/right")
        stage = world.stage
        articulation_roots = {
            "left": _find_articulation_roots(stage, "/World/left"),
            "right": _find_articulation_roots(stage, "/World/right"),
        }
        prim_checks = {side: len(paths) == 1 for side, paths in articulation_roots.items()}
        payload.update({"status": "REFERENCED", "articulation_roots": articulation_roots, "required_prim_checks": prim_checks})
        _write(json_path, payload)
        if not all(prim_checks.values()):
            payload.update({"status": "FAILED", "failure": "required articulation prim path missing"})
            _write(json_path, payload)
            print(json.dumps({"json": _rel(json_path), "status": payload["status"], "failure": payload["failure"]}))
            return 2

        left = world.scene.add(SingleArticulation(prim_path=articulation_roots["left"][0], name="left_vx300s"))
        right = world.scene.add(SingleArticulation(prim_path=articulation_roots["right"][0], name="right_vx300s"))
        payload["status"] = "ARTICULATIONS_ADDED"
        _write(json_path, payload)

        world.reset()
        side_payloads = {"left": _articulation_payload(left), "right": _articulation_payload(right)}
        gates = {side: _gate(row) for side, row in side_payloads.items()}
        overall_pass = all(all(gate.values()) for gate in gates.values())
        payload.update(
            {
                "status": "PASS" if overall_pass else "FAILED",
                "sides": side_payloads,
                "gates": gates,
                "overall_pass": overall_pass,
            }
        )
        _write(json_path, payload)
        lines = [
            "# Phase 18 Runtime Articulation Validation",
            "",
            "| Side | DOFs | Bodies | Core DOFs | EE candidate | Gate |",
            "| --- | ---: | ---: | --- | --- | --- |",
        ]
        for side, row in side_payloads.items():
            gate = gates[side]
            lines.append(
                f"| {side} | {row['num_dof']} | {row['num_bodies']} | {gate['has_core_dofs']} | "
                f"{gate['has_ee_candidate']} | {'PASS' if all(gate.values()) else 'FAIL'} |"
            )
        lines.extend(
            [
                "",
                "## DOF Names",
                "",
                f"- left: `{side_payloads['left']['dof_names']}`",
                f"- right: `{side_payloads['right']['dof_names']}`",
                "",
                "## Artifacts",
                "",
                f"- JSON: `{_rel(json_path)}`",
                f"- Markdown: `{_rel(md_path)}`",
            ]
        )
        md_path.write_text("\n".join(lines) + "\n")
        print(json.dumps({"json": _rel(json_path), "markdown": _rel(md_path), "status": payload["status"]}, ensure_ascii=False), flush=True)
        return 0 if overall_pass else 3
    except Exception as exc:
        payload.update(
            {
                "status": "EXCEPTION",
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-20:],
            }
        )
        _write(json_path, payload)
        print(json.dumps({"json": _rel(json_path), "status": "EXCEPTION", "exception": payload["exception"]}, ensure_ascii=False), flush=True)
        return 1
    finally:
        app.close(skip_cleanup=True)


if __name__ == "__main__":
    raise SystemExit(main())
