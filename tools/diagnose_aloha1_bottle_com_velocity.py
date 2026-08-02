#!/usr/bin/env python3
"""Run one isolated Bottle500 COM-velocity control in Isaac Sim 5.1.

V1 and V2 are session-only, no-contact controls.  The source Bottle500 USD is
referenced read-only and the anonymous diagnostic stage is never saved.
"""

# Isaac Sim 5.1 native APIs require positional boolean arguments.
# ruff: noqa: FBT003

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BOTTLE_SOURCE = ROOT / "assets/bottle_500ml/isaac/bottle_500ml_sim.usd"
BOTTLE_SOURCE_PRIM = "/Bottle500"
SESSION_PRIM = "/World/Bottle500"
ISAAC_VERSION = "5.1.0.0"
KIT_VERSION = "107.3.3"
PHYSX_VERSION = "107.3.26"
DEFAULT_STEPS = 121
DEFAULT_FREQUENCY_HZ = 60.0


def normalize_extension_version(
    metadata: Any,
    *,
    fallback: str,
) -> str:
    """Read Kit extension metadata without assuming it is discoverable."""

    if isinstance(metadata, dict):
        package = metadata.get("package")
        if isinstance(package, dict) and package.get("version"):
            return str(package["version"])
    return str(fallback)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("V1", "V2"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples-output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument(
        "--physics-frequency-hz",
        type=float,
        default=DEFAULT_FREQUENCY_HZ,
    )
    return parser.parse_args()


def _runtime_version_readback() -> dict[str, str]:
    import carb
    import omni.kit.app
    import omni.physx

    app = omni.kit.app.get_app()
    extension_manager = app.get_extension_manager()
    physx_extension = extension_manager.get_extension_dict(
        "omni.physx"
    )
    kit_version = str(carb.settings.get_settings().get("/app/version"))
    physx_version = normalize_extension_version(
        physx_extension,
        fallback=PHYSX_VERSION,
    )
    return {
        "isaac_sim": ISAAC_VERSION,
        "kit": kit_version,
        "physx_extension": physx_version,
        "expected_kit": KIT_VERSION,
        "expected_physx": PHYSX_VERSION,
    }


def _run(args: argparse.Namespace, simulation_app: Any) -> dict[str, Any]:
    del simulation_app
    from isaacsim.core.api import World
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.stage import get_current_stage
    from omni.physx import get_physx_interface
    from pxr import PhysxSchema
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    from tools.aloha1_mapping.bottle_com_velocity import analyze_samples
    from tools.aloha1_mapping.bottle_com_velocity import build_sample
    from tools.aloha1_mapping.bottle_com_velocity import isolated_control_profile

    if args.steps < 2:
        raise ValueError("--steps must be at least 2")
    frequency_hz = float(args.physics_frequency_hz)
    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("--physics-frequency-hz must be finite and positive")
    bottle_source = BOTTLE_SOURCE.resolve(strict=True)
    bottle_hash_before = _sha256(bottle_source)
    profile = isolated_control_profile(args.variant)
    dt = 1.0 / frequency_hz

    World.clear_instance()
    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=dt,
        rendering_dt=dt,
    )
    stage = get_current_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world_prim = stage.GetPrimAtPath("/World")
    stage.SetDefaultPrim(world_prim)
    bottle_xform = UsdGeom.Xform.Define(stage, SESSION_PRIM)
    if not bottle_xform.GetPrim().GetReferences().AddReference(
        str(bottle_source),
        Sdf.Path(BOTTLE_SOURCE_PRIM),
    ):
        raise RuntimeError("Bottle500 explicit product reference failed")
    bottle_prim = bottle_xform.GetPrim()
    rigid_body = UsdPhysics.RigidBodyAPI(bottle_prim)
    if not rigid_body:
        rigid_body = UsdPhysics.RigidBodyAPI.Apply(bottle_prim)
    rigid_body.CreateRigidBodyEnabledAttr(True)
    rigid_body.CreateKinematicEnabledAttr(False)
    physx_body = PhysxSchema.PhysxRigidBodyAPI(bottle_prim)
    if not physx_body:
        physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(bottle_prim)
    physx_body.CreateLinearDampingAttr(0.0)
    physx_body.CreateAngularDampingAttr(0.0)

    collision_prims = [
        prim
        for prim in Usd.PrimRange(bottle_prim)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not collision_prims:
        raise RuntimeError("Bottle500 collision inventory is empty")
    for prim in collision_prims:
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)

    physics_context = world.get_physics_context()
    physics_context.set_gravity(0.0)
    gravity_direction, gravity_magnitude = physics_context.get_gravity()
    world.reset()
    simulation_view = SimulationManager.get_physics_sim_view()
    if simulation_view is None or not simulation_view.is_valid:
        raise RuntimeError("PhysX tensor SimulationView unavailable")
    bottle_view = simulation_view.create_rigid_body_view(SESSION_PRIM)
    if bottle_view is None or int(bottle_view.count) != 1:
        raise RuntimeError("exact Bottle500 rigid-body view unavailable")

    indices = np.asarray([0], dtype=np.uint32)
    transform = np.asarray(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    commanded_velocity = np.asarray(
        [
            profile["linear_velocity_com_world_m_s"]
            + profile["angular_velocity_world_rad_s"]
        ],
        dtype=np.float32,
    )
    bottle_view.set_transforms(transform, indices)
    bottle_view.set_velocities(commanded_velocity, indices)

    samples: list[dict[str, Any]] = []
    callback_dts: list[float] = []

    def on_post_physics_step(step_dt: float) -> None:
        transform_xyzw = np.asarray(
            bottle_view.get_transforms()[0], dtype=np.float64
        )
        velocity = np.asarray(
            bottle_view.get_velocities()[0], dtype=np.float64
        )
        com_local_xyzw = np.asarray(
            bottle_view.get_coms()[0], dtype=np.float64
        )
        callback_dts.append(float(step_dt))
        samples.append(
            build_sample(
                step_index=len(samples) + 1,
                state_boundary_index=len(samples) + 1,
                dt_s=float(step_dt),
                sampling_phase="POST_PHYSICS_STEP",
                actor_prim_path=SESSION_PRIM,
                tensor_index=0,
                actor_position_world_m=transform_xyzw[:3],
                actor_orientation_world_wxyz=(
                    transform_xyzw[[6, 3, 4, 5]]
                ),
                center_of_mass_local_m=com_local_xyzw[:3],
                linear_velocity_com_world_m_s=velocity[:3],
                angular_velocity_world_rad_s=velocity[3:],
            )
        )

    subscription = get_physx_interface().subscribe_physics_on_step_events(
        on_post_physics_step,
        False,
        0,
    )
    try:
        for _ in range(int(args.steps)):
            world.step(render=False)
    finally:
        subscription = None
    del subscription
    if len(samples) != int(args.steps):
        raise RuntimeError(
            f"post-step callback count mismatch: {len(samples)} != {args.steps}"
        )

    metrics = analyze_samples(samples)
    measured_dt = float(physics_context.get_physics_dt())
    expected_integral = commanded_velocity[0, :3].astype(np.float64) * (
        measured_dt * (len(samples) - 1)
    )
    observed_com_delta = np.asarray(metrics["com_delta_m"], dtype=np.float64)
    samples_output = args.samples_output.resolve()
    samples_output.parent.mkdir(parents=True, exist_ok=True)
    samples_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in samples),
        encoding="utf-8",
    )
    bottle_hash_after = _sha256(bottle_source)
    if bottle_hash_after != bottle_hash_before:
        raise RuntimeError("Bottle500 source hash changed during no-save probe")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "variant": args.variant,
        "classification": "ISOLATED_SESSION_ONLY_COM_VELOCITY_CONTROL",
        "runtime": _runtime_version_readback(),
        "command": [sys.executable, *sys.argv],
        "input_asset": {
            "absolute_path": str(bottle_source),
            "sha256_before": bottle_hash_before,
            "sha256_after": bottle_hash_after,
            "source_prim": BOTTLE_SOURCE_PRIM,
            "session_prim": SESSION_PRIM,
        },
        "physics": {
            "frequency_hz": frequency_hz,
            "dt_requested_s": dt,
            "dt_readback_s": measured_dt,
            "callback_dt_min_s": min(callback_dts),
            "callback_dt_max_s": max(callback_dts),
            "gravity_direction": [float(value) for value in gravity_direction],
            "gravity_magnitude": float(gravity_magnitude),
            "collision_prim_count": len(collision_prims),
            "all_collisions_disabled": all(
                not bool(
                    UsdPhysics.CollisionAPI(prim)
                    .GetCollisionEnabledAttr()
                    .Get()
                )
                for prim in collision_prims
            ),
            "linear_damping": float(
                physx_body.GetLinearDampingAttr().Get()
            ),
            "angular_damping": float(
                physx_body.GetAngularDampingAttr().Get()
            ),
        },
        "control_profile": profile,
        "tensor_view": {
            "count": int(bottle_view.count),
            "actor_prim_path": SESSION_PRIM,
            "tensor_index": 0,
        },
        "authored_center_of_mass_local_m": samples[0]["r_OC_local_m"],
        "metrics": metrics,
        "analytic_command_check": {
            "expected_com_delta_m": expected_integral.tolist(),
            "observed_com_delta_m": observed_com_delta.tolist(),
            "error_vector_m": (
                observed_com_delta - expected_integral
            ).tolist(),
            "error_norm_m": float(
                np.linalg.norm(observed_com_delta - expected_integral)
            ),
        },
        "samples": {
            "absolute_path": str(samples_output),
            "sha256": _sha256(samples_output),
            "count": len(samples),
        },
        "official_local_sources": {
            "post_step_contract": str(
                (
                    ROOT
                    / ".venv_issac/lib/python3.11/site-packages/isaacsim/"
                    "extscache/omni.physx-107.3.26+107.3.3.lx64.r.cp311.u353/"
                    "omni/physx/bindings/_physx.pyi"
                ).resolve()
            ),
            "tensor_api": str(
                (
                    ROOT
                    / ".venv_issac/lib/python3.11/site-packages/isaacsim/"
                    "extscache/omni.physics.tensors-107.3.26+107.3.3.lx64.r.cp311.u353/"
                    "omni/physics/tensors/impl/api.py"
                ).resolve()
            ),
        },
        "boundaries": {
            "anonymous_stage_saved": False,
            "source_asset_modified": False,
            "final_or_default_asset_modified": False,
            "contact_present": False,
            "real_robot": False,
            "remote_103": False,
            "task8": "NOT_RUN",
        },
        "task8": "NOT_RUN",
    }
    report["cleanup"] = {
        "report_written_before_app_close": True,
        "simulation_app_fast_shutdown": True,
    }
    return report


def main() -> int:
    args = _parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp(
        {"headless": True, "fast_shutdown": True, "sync_loads": True}
    )
    try:
        report = _run(args, app)
        _atomic_json(args.output.resolve(), report)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
