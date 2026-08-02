#!/usr/bin/env python3
"""Run fresh-process negative controls for the ALOHA finger safety gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import traceback
from typing import Any

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import yaml

from tools.aloha1_mapping.convex_geometry_audit import convex_pair_relation
from tools.aloha1_mapping.grasp_initialization_contract import evaluate_finger_initialization

ROOT = Path(__file__).resolve().parents[1]
ISAAC_PYTHON = ROOT / ".venv_issac/bin/python"
GUI_LAUNCHER = ROOT / "tools/run_aloha1_grasp_20cm_gui.py"
RUNTIME_CONFIG = ROOT / "configs/aloha1_grasp_20cm_gui_cad_derived_colliders.yaml"
FORMAL_CONFIG = ROOT / "configs/aloha1_grasp_20cm_five_pose_cad_derived_colliders.yaml"
PREFLIGHT = ROOT / "reports/aloha1_mapping/aloha1_cad_derived_five_pose_runtime_preflight.json"
FROZEN_STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_derived_full_body_colliders/1.0/"
    "aloha1_cad_derived_full_body_collider_gripper_decomposition_"
    "tabletop_zero_z_up_meters_diagnostic.usda"
)
CANDIDATE_STAGE = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "finger_limit_pair_collision_candidate/1.0/"
    "aloha1_finger_source_limit_candidate.usda"
)
ARTIFACT_ROOT = (
    ROOT
    / ".codex/artifacts/20260802-aloha1-five-pose-finger-safety/"
    "negative_controls_attempt3"
)
OUTPUT_JSON = (
    ROOT
    / "reports/aloha1_mapping/"
    "aloha1_grasp_initialization_negative_controls.json"
)
OUTPUT_MD = OUTPUT_JSON.with_suffix(".md")
ARTICULATION_PRIM = "/World/follower_left/vx300s_left/root_joint"
ROBOT_ROOT = "/World/follower_left/vx300s_left"
SCENARIOS = (
    "STATIC_LOAD_WITHOUT_RESET",
    "ILLEGAL_Q_ZERO",
    "LEGAL_OPEN_CLOSE_SWEEP",
    "SAMPLE_02_ENVIRONMENT_INTERFERENCE",
)
EXPECTED = {
    "STATIC_LOAD_WITHOUT_RESET": "FAIL_INITIALIZATION_CONTRACT",
    "ILLEGAL_Q_ZERO": "FINGER_PAIR_OVERLAP",
    "LEGAL_OPEN_CLOSE_SWEEP": None,
    "SAMPLE_02_ENVIRONMENT_INTERFERENCE": "FINGER_LIMIT_VIOLATION",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def aggregate_controls(
    records: list[dict[str, Any]],
    *,
    require_visual_review: bool = False,
) -> dict[str, Any]:
    by_name = {str(record.get("scenario")): record for record in records}
    if list(by_name) != list(SCENARIOS):
        missing = sorted(set(SCENARIOS) - set(by_name))
        unexpected = sorted(set(by_name) - set(SCENARIOS))
        raise ValueError(
            f"negative-control inventory mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    failed = []
    gates: dict[str, dict[str, bool]] = {}
    for scenario in SCENARIOS:
        record = by_name[scenario]
        expected = EXPECTED[scenario]
        observed = [str(code) for code in record["observed_failure_codes"]]
        scenario_gates = {
            "fresh_process": record.get("fresh_process") is True,
            "stage_immutable": record.get("stage_immutable") is True,
            "expected_classification": (
                record.get("status") == "PASS"
                if expected is None
                else record.get("status") == "EXPECTED_FAIL_OBSERVED"
                and expected in observed
            ),
            "raw_screenshot_recorded": bool(record.get("raw_screenshot")),
            "annotated_screenshot_recorded": bool(
                record.get("annotated_screenshot")
            ),
        }
        if require_visual_review:
            scenario_gates["visual_model_review_pass"] = (
                record.get("visual_model_review") == "PASS"
            )
        gates[scenario] = scenario_gates
        if not all(scenario_gates.values()):
            failed.append(scenario)
    return {
        "schema_version": 1,
        "status": "PASS" if not failed else "FAIL",
        "control_count": len(records),
        "failed_controls": failed,
        "all_fresh_processes": all(
            record.get("fresh_process") is True for record in records
        ),
        "gates": gates,
        "controls": records,
        "source_or_final_asset_modified": False,
        "task8": "NOT_RUN",
    }


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        size,
    )


def _annotate_capture(
    *,
    overview: Path,
    closeup: Path,
    destination: Path,
    record: dict[str, Any],
) -> None:
    overview_image = Image.open(overview).convert("RGB")
    closeup_image = Image.open(closeup).convert("RGB")
    target_height = 540
    images = []
    for source in (overview_image, closeup_image):
        scale = target_height / source.height
        images.append(
            source.resize(
                (int(source.width * scale), target_height),
                Image.Resampling.LANCZOS,
            )
        )
    panel_width = 560
    canvas = Image.new(
        "RGB",
        (sum(image.width for image in images) + panel_width, target_height),
        (20, 22, 27),
    )
    x_offset = 0
    for label, image in zip(("FULL ARM", "FINGER CLOSEUP"), images, strict=True):
        canvas.paste(image, (x_offset, 0))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(
            (x_offset, 0, x_offset + image.width, 42),
            fill=(0, 0, 0),
        )
        draw.text(
            (x_offset + 12, 8),
            label,
            fill=(255, 255, 255),
            font=_font(22),
        )
        x_offset += image.width
    draw = ImageDraw.Draw(canvas)
    panel_x = x_offset + 20
    status_color = (
        (80, 230, 120)
        if record["scenario"] == "LEGAL_OPEN_CLOSE_SWEEP"
        else (255, 90, 90)
    )
    draw.text(
        (panel_x, 18),
        record["scenario"],
        fill=status_color,
        font=_font(22),
    )
    y = 58
    lines = [
        "Isaac Sim 5.1.0.0 / Kit 107.3.3",
        "PhysX 107.3.26",
        f"Stage SHA: {record['stage_sha256'][:20]}...",
        f"Expected: {record['expected_failure_code']}",
        "Observed: " + ", ".join(record["observed_failure_codes"]),
        f"Target q (m): {record.get('target_m')}",
        f"Readback q (m): {record.get('readback_m')}",
        f"Pair overlap (m^3): {record.get('pair_overlap_volume_m3')}",
        f"Frame/phase: {record.get('frame')} / {record.get('phase')}",
        "Collision display: session visual clones + runtime data",
        "Closeup scope: actual finger collider clones only",
        "Cyan = left_finger; yellow/orange = right_finger",
        "Final/default asset modified: NO",
        "Task 8: NOT_RUN",
    ]
    for line in lines:
        for wrapped in textwrap.wrap(str(line), width=50) or [""]:
            draw.text(
                (panel_x, y),
                wrapped,
                fill=(235, 235, 235),
                font=_font(17),
            )
            y += 25
        y += 3
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=False)


def _finger_world_points(stage: Any, path: str) -> np.ndarray:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"missing finger collider mesh: {path}")
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    return np.asarray(
        [
            list(transform.Transform(point))
            for point in UsdGeom.Mesh(prim).GetPointsAttr().Get()
        ],
        dtype=np.float64,
    )


def _pair_geometry(stage: Any, config: dict[str, Any]) -> dict[str, Any]:
    colliders = config["evidence"]["collider_overlay"]["finger_colliders"]
    left = _finger_world_points(stage, str(colliders["left"]["collider"]))
    right = _finger_world_points(stage, str(colliders["right"]["collider"]))
    return convex_pair_relation(left, right)


def _camera_clipping_range(view: str) -> tuple[float, float]:
    """Return an explicit evidence-camera clipping range in Stage metres."""

    if view not in {"overview", "closeup"}:
        raise ValueError(f"unsupported evidence view: {view}")
    return (0.005, 10.0) if view == "closeup" else (0.01, 20.0)


def _camera_visual_scope(view: str) -> str:
    """Name the evidence geometry shown by each camera view."""

    scopes = {
        "overview": "FULL_STAGE_WITH_COLLIDER_OVERLAY",
        "closeup": "FINGER_COLLIDERS_ONLY",
    }
    try:
        return scopes[view]
    except KeyError as error:
        raise ValueError(f"unsupported evidence view: {view}") from error


def _closeup_camera_distance(finger_aabb_diagonal_m: float) -> float:
    """Frame both fingers with a deterministic margin at maximum aperture."""

    span = float(finger_aabb_diagonal_m)
    if not np.isfinite(span) or span <= 0.0:
        raise ValueError("finger AABB diagonal must be finite and positive")
    return max(4.0 * span, 0.75)


def _capture_views(
    *,
    app: Any,
    stage: Any,
    output_root: Path,
) -> dict[str, Any]:
    from isaacsim.sensors.camera import Camera
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Gf
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdLux

    from tools.aloha1_mapping.isaac_screenshot import look_at_orientation_wxyz
    from tools.capture_aloha1_task7_virtual_helper_failure import _build_overlay
    from tools.capture_aloha1_task7_virtual_helper_failure import _visual_points
    from tools.capture_aloha_viper_cad_finger_task5_numeric_pass_viewport import _capture_viewport_png

    stage.SetEditTarget(stage.GetSessionLayer())
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        dome = UsdLux.DomeLight.Define(stage, "/FingerSafetyEvidence/Dome")
        dome.CreateIntensityAttr(900.0)
        key = UsdLux.DistantLight.Define(stage, "/FingerSafetyEvidence/Key")
        key.CreateIntensityAttr(1700.0)
        key.AddRotateXYZOp().Set(Gf.Vec3f(30.0, -35.0, -20.0))
        overlay_records, groups = _build_overlay(stage, ROBOT_ROOT)
    visual = _visual_points(stage, ROBOT_ROOT)
    full = np.concatenate([visual, groups["all"]])
    config = yaml.safe_load(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    collider_paths = config["evidence"]["collider_overlay"][
        "finger_colliders"
    ]
    finger_sources = {
        str(collider_paths["left"]["collider"]): {
            "side": "left_finger",
            "color": Gf.Vec3f(0.05, 0.45, 1.0),
        },
        str(collider_paths["right"]["collider"]): {
            "side": "right_finger",
            "color": Gf.Vec3f(1.0, 0.42, 0.02),
        },
    }
    left_finger = _finger_world_points(
        stage,
        str(collider_paths["left"]["collider"]),
    )
    right_finger = _finger_world_points(
        stage,
        str(collider_paths["right"]["collider"]),
    )
    finger = np.concatenate([left_finger, right_finger])
    direction = np.asarray([0.75, 1.0, 0.62], dtype=np.float64)
    direction /= np.linalg.norm(direction)

    def camera_spec(points: np.ndarray, minimum_distance: float) -> dict[str, np.ndarray]:
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        center = (minimum + maximum) / 2.0
        span = float(np.linalg.norm(maximum - minimum))
        return {
            "target": center,
            "position": center + direction * max(3.0 * span, minimum_distance),
        }

    specs = {
        "overview": camera_spec(full, 1.8),
    }
    finger_minimum = finger.min(axis=0)
    finger_maximum = finger.max(axis=0)
    finger_center = (finger_minimum + finger_maximum) / 2.0
    base_points = groups["base"]
    base_center = (
        (base_points.min(axis=0) + base_points.max(axis=0)) / 2.0
        if len(base_points)
        else np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    )
    outward = finger_center - base_center
    outward[2] += 0.15
    outward /= np.linalg.norm(outward)
    finger_span = float(np.linalg.norm(finger_maximum - finger_minimum))
    specs["closeup"] = {
        "target": finger_center,
        "position": finger_center
        + outward * _closeup_camera_distance(finger_span),
    }
    camera = Camera(
        prim_path="/FingerSafetyEvidence/Camera",
        name="finger_safety_evidence_camera",
        resolution=(1280, 720),
        frequency=60,
    )
    camera.initialize()
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("active viewport unavailable")
    output_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"camera_metadata": {}}
    for label, spec in specs.items():
        if label == "closeup":
            with Usd.EditContext(stage, stage.GetSessionLayer()):
                world_prim = stage.GetPrimAtPath("/World")
                if not world_prim.IsValid():
                    raise RuntimeError("closeup scope requires /World")
                UsdGeom.Imageable(world_prim).CreateVisibilityAttr().Set(
                    UsdGeom.Tokens.invisible
                )
                for overlay_record in overlay_records:
                    clone_prim = stage.GetPrimAtPath(
                        overlay_record["clone_prim"]
                    )
                    source = str(overlay_record["source_prim"])
                    source_record = finger_sources.get(source)
                    clone_imageable = UsdGeom.Imageable(clone_prim)
                    if source_record is None:
                        clone_imageable.CreateVisibilityAttr().Set(
                            UsdGeom.Tokens.invisible
                        )
                        continue
                    clone_imageable.CreateVisibilityAttr().Set(
                        UsdGeom.Tokens.inherited
                    )
                    UsdGeom.Gprim(clone_prim).CreateDisplayColorAttr(
                        [source_record["color"]]
                    )
                    UsdGeom.Gprim(clone_prim).CreateDisplayOpacityAttr(
                        [0.82]
                    )
        orientation = look_at_orientation_wxyz(
            spec["position"],
            spec["target"],
            np.asarray([0.0, 0.0, 1.0]),
        )
        camera.set_world_pose(
            position=spec["position"],
            orientation=orientation,
            camera_axes="usd",
        )
        near_m, far_m = _camera_clipping_range(label)
        camera.set_clipping_range(near_m, far_m)
        viewport.camera_path = Sdf.Path(camera.prim_path)
        for _ in range(35):
            app.update()
        destination = output_root / f"{label}_raw.png"
        _capture_viewport_png(app, viewport, destination)
        result[label] = str(destination.resolve(strict=True))
        result["camera_metadata"][label] = {
            "position_world_m": spec["position"].tolist(),
            "target_world_m": spec["target"].tolist(),
            "orientation_wxyz": orientation.tolist(),
            "clipping_range_m": [near_m, far_m],
            "resolution": [1280, 720],
            "visual_scope": _camera_visual_scope(label),
        }
    return result


def _run_structural_scenario(
    *,
    scenario: str,
    output: Path,
    screenshot_root: Path,
) -> int:
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "width": 1280,
            "height": 720,
            "/app/useFabricSceneDelegate": False,
        }
    )
    exit_code = 0
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.stage import open_stage
        from omni.physx import get_physx_interface

        from tools.audit_aloha_viper_cad_finger_task5_geometry import _sync_physx_transforms_to_usd

        config = yaml.safe_load(RUNTIME_CONFIG.read_text(encoding="utf-8"))
        safety = config["finger_safety"]
        stage_path = (
            FROZEN_STAGE
            if scenario == "STATIC_LOAD_WITHOUT_RESET"
            else CANDIDATE_STAGE
        )
        before = _sha256(stage_path)
        if not open_stage(str(stage_path.resolve(strict=True))):
            raise RuntimeError(f"cannot open Stage: {stage_path}")
        stage = get_current_stage()
        target_m = [0.0, 0.0]
        readback_m = [0.0, 0.0]
        states: list[dict[str, Any]] = []
        if scenario != "STATIC_LOAD_WITHOUT_RESET":
            World.clear_instance()
            world = World(
                stage_units_in_meters=1.0,
                backend="numpy",
                device="cpu",
                physics_dt=1.0 / 60.0,
                rendering_dt=1.0 / 60.0,
            )
            articulation = SingleArticulation(
                prim_path=ARTICULATION_PRIM,
                name=f"finger_safety_{scenario.lower()}",
                reset_xform_properties=False,
            )
            world.scene.add(articulation)
            world.reset()
            order = list(articulation.dof_names)
            indices = [order.index(name) for name in safety["dof_names"]]
            base = np.asarray(
                articulation.get_joint_positions(),
                dtype=np.float64,
            )
            physx = get_physx_interface()
            targets = (
                [(0.0, 0.0)]
                if scenario == "ILLEGAL_Q_ZERO"
                else [
                    (0.057, -0.057),
                    (0.039, -0.039),
                    (0.021, -0.021),
                ]
            )
            for pair in targets:
                qpos = base.copy()
                qpos[indices] = pair
                articulation.set_joint_positions(qpos)
                articulation.set_joint_velocities(np.zeros_like(qpos))
                _sync_physx_transforms_to_usd(physx)
                geometry = _pair_geometry(stage, config)
                actual = np.asarray(
                    articulation.get_joint_positions(),
                    dtype=np.float64,
                )[indices].tolist()
                state = evaluate_finger_initialization(
                    reset_complete=True,
                    dof_order=list(safety["dof_names"]),
                    targets=list(pair),
                    readback=actual,
                    source_limits=safety["source_limits_m"],
                    overlap_volume_m3=float(
                        geometry["overlap_volume_m3"]
                    ),
                )
                state["pair_geometry"] = geometry
                states.append(state)
            target_m = list(targets[-1])
            readback_m = list(states[-1]["readback_m"])
            contract = states[-1]
        else:
            geometry = _pair_geometry(stage, config)
            contract = evaluate_finger_initialization(
                reset_complete=False,
                dof_order=list(safety["dof_names"]),
                targets=target_m,
                readback=readback_m,
                source_limits=safety["source_limits_m"],
                overlap_volume_m3=float(geometry["overlap_volume_m3"]),
            )
            contract["pair_geometry"] = geometry
            states.append(contract)
        screenshots = _capture_views(
            app=app,
            stage=stage,
            output_root=screenshot_root,
        )
        expected = EXPECTED[scenario]
        failures = sorted(
            {
                str(code)
                for state in states
                for code in state["failure_codes"]
            }
        )
        status = (
            "PASS"
            if expected is None and not failures
            else (
                "EXPECTED_FAIL_OBSERVED"
                if expected in failures
                else "FAIL"
            )
        )
        record = {
            "scenario": scenario,
            "status": status,
            "expected_failure_code": expected,
            "observed_failure_codes": failures,
            "states": states,
            "target_m": target_m,
            "readback_m": readback_m,
            "pair_overlap_volume_m3": contract[
                "pair_overlap_volume_m3"
            ],
            "frame": 0,
            "phase": "PRE_FORMAL_STEP",
            "fresh_process": True,
            "stage_absolute_path": str(stage_path.resolve()),
            "stage_sha256": before,
            "stage_immutable": _sha256(stage_path) == before,
            "raw_screenshots": screenshots,
            "source_or_final_asset_modified": False,
            "task8": "NOT_RUN",
        }
        _atomic_json(output, record)
    except Exception:
        exit_code = 1
        _atomic_json(
            output,
            {
                "scenario": scenario,
                "status": "FAIL",
                "traceback": traceback.format_exc(),
            },
        )
        print(traceback.format_exc(), file=sys.stderr, flush=True)
    finally:
        app.close()
    return exit_code


def _run_pose_capture(
    *,
    joint_q_path: Path,
    output: Path,
    screenshot_root: Path,
) -> int:
    """Render the exact failed q-state with collision clones in a fresh process."""

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "width": 1280,
            "height": 720,
            "/app/useFabricSceneDelegate": False,
        }
    )
    exit_code = 0
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.core.utils.stage import open_stage
        from omni.physx import get_physx_interface

        from tools.audit_aloha_viper_cad_finger_task5_geometry import _sync_physx_transforms_to_usd

        qpos = np.asarray(
            json.loads(joint_q_path.read_text(encoding="utf-8"))[
                "joint_readback"
            ],
            dtype=np.float64,
        )
        if qpos.shape != (9,) or not np.isfinite(qpos).all():
            raise ValueError("pose-capture q must contain nine finite values")
        before = _sha256(FROZEN_STAGE)
        if not open_stage(str(FROZEN_STAGE.resolve(strict=True))):
            raise RuntimeError(f"cannot open Stage: {FROZEN_STAGE}")
        stage = get_current_stage()
        World.clear_instance()
        world = World(
            stage_units_in_meters=1.0,
            backend="numpy",
            device="cpu",
            physics_dt=1.0 / 60.0,
            rendering_dt=1.0 / 60.0,
        )
        articulation = SingleArticulation(
            prim_path=ARTICULATION_PRIM,
            name="sample02_first_failure_pose_capture",
            reset_xform_properties=False,
        )
        world.scene.add(articulation)
        world.reset()
        articulation.set_joint_positions(qpos)
        articulation.set_joint_velocities(np.zeros_like(qpos))
        _sync_physx_transforms_to_usd(get_physx_interface())
        screenshots = _capture_views(
            app=app,
            stage=stage,
            output_root=screenshot_root,
        )
        _atomic_json(
            output,
            {
                "status": "PASS",
                "joint_readback_requested": qpos.tolist(),
                "joint_readback_actual": np.asarray(
                    articulation.get_joint_positions(),
                    dtype=np.float64,
                ).tolist(),
                "raw_screenshots": screenshots,
                "stage_sha256": before,
                "stage_immutable": _sha256(FROZEN_STAGE) == before,
                "semantics": (
                    "FRESH_PROCESS_SESSION_POSE_RECONSTRUCTION_FOR_"
                    "COLLIDER_VISUALIZATION; AUTHORITATIVE_FAILURE_"
                    "TELEMETRY_COMES_FROM_DYNAMIC_CONTROL_PROCESS"
                ),
            },
        )
    except Exception:
        exit_code = 1
        _atomic_json(
            output,
            {"status": "FAIL", "traceback": traceback.format_exc()},
        )
        print(traceback.format_exc(), file=sys.stderr, flush=True)
    finally:
        app.close()
    return exit_code


def _run_logged(command: list[str], log: Path, *, timeout_s: float) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    environment["PYTHONPATH"] = str(ROOT)
    started = time.perf_counter()
    with log.open("wb") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        process_id = int(process.pid)
        try:
            exit_code = int(process.wait(timeout=timeout_s))
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            exit_code = int(process.wait(timeout=30.0))
    return {
        "process_id": process_id,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "runtime_s": time.perf_counter() - started,
        "command": command,
        "log_absolute_path": str(log.resolve()),
    }


def _run_sample02(root: Path) -> dict[str, Any]:
    config = yaml.safe_load(FORMAL_CONFIG.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    selected = next(
        item
        for item in preflight["selected_samples"]
        if item["sample_id"] == "sample_02"
    )
    pose_path = root / "frozen_world_from_object.json"
    _atomic_json(
        pose_path,
        {
            "schema_version": 1,
            "sample_id": "sample_02",
            "world_from_object": selected["world_from_object"],
        },
    )
    runtime_root = root / "runtime"
    trajectory = config["arm_trajectory"]
    command = [
        str(ISAAC_PYTHON),
        str(GUI_LAUNCHER),
        "--config",
        str(RUNTIME_CONFIG),
        "--autorun",
        "--close-after-terminal",
        "--bottle-world-from-object-json",
        str(pose_path),
        "--initial-arm-q-rad",
        *(repr(float(value)) for value in selected["initial_arm_q_rad"]),
        "--initial-pose-hold-frames",
        str(config["runtime"]["initial_pose_hold_frames"]),
        "--arm-phase-readback-tolerance-rad",
        repr(float(config["gates"]["arm_phase_readback_tolerance_rad"])),
        "--arm-trajectory-mode",
        str(trajectory["mode"]),
        "--arm-acceleration-limits-rad-s2",
        *(repr(float(value)) for value in trajectory["acceleration_limits_rad_s2"]),
        "--artifact-root",
        str(runtime_root),
        "--skip-collider-evidence",
    ]
    stage_before = _sha256(FROZEN_STAGE)
    process = _run_logged(command, root / "runtime.log", timeout_s=1800.0)
    runtime_path = runtime_root / "aloha1_grasp_20cm_runtime.json"
    telemetry_path = runtime_root / "aloha1_grasp_20cm_telemetry.jsonl"
    if not runtime_path.is_file() or not telemetry_path.is_file():
        raise RuntimeError(
            f"sample02 control missing runtime evidence: process={process}"
        )
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    safety = runtime["runtime"]["finger_safety"]
    violation = safety["first_violation"]
    if not violation:
        raise RuntimeError("sample02 control did not produce a first violation")
    frame = int(violation["frame"])
    attempt = runtime_root / "video_attempt_001/frames"
    overview = attempt / "overview" / f"{frame:06d}.png"
    closeup = attempt / "gripper_closeup" / f"{frame:06d}.png"
    if not overview.is_file() or not closeup.is_file():
        raise RuntimeError(
            f"sample02 violation frame screenshots missing: frame={frame}"
        )
    telemetry = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    frame_record = next(item for item in telemetry if int(item["frame"]) == frame)
    pose_capture_q = root / "first_failure_joint_readback.json"
    _atomic_json(
        pose_capture_q,
        {"joint_readback": frame_record["joint_readback"]},
    )
    pose_capture_output = root / "pose_capture_record.json"
    pose_capture_process = _run_logged(
        [
            str(ISAAC_PYTHON),
            str(Path(__file__).resolve()),
            "--pose-capture-q-json",
            str(pose_capture_q),
            "--runtime-output",
            str(pose_capture_output),
            "--screenshot-root",
            str(root / "screenshots_raw_collision_visible"),
        ],
        root / "pose_capture.log",
        timeout_s=900.0,
    )
    if (
        pose_capture_process["exit_code"] != 0
        or not pose_capture_output.is_file()
    ):
        raise RuntimeError(
            f"sample02 failed-pose capture failed: {pose_capture_process}"
        )
    pose_capture = json.loads(
        pose_capture_output.read_text(encoding="utf-8")
    )
    if pose_capture.get("status") != "PASS":
        raise RuntimeError(f"sample02 pose capture is not PASS: {pose_capture}")
    return {
        "scenario": "SAMPLE_02_ENVIRONMENT_INTERFERENCE",
        "status": (
            "EXPECTED_FAIL_OBSERVED"
            if EXPECTED["SAMPLE_02_ENVIRONMENT_INTERFERENCE"]
            in violation["failure_codes"]
            else "FAIL"
        ),
        "expected_failure_code": EXPECTED[
            "SAMPLE_02_ENVIRONMENT_INTERFERENCE"
        ],
        "observed_failure_codes": violation["failure_codes"],
        "target_m": frame_record["joint_target"][7:9],
        "readback_m": frame_record["joint_readback"][7:9],
        "pair_overlap_volume_m3": frame_record["finger_safety"][
            "pair_overlap_volume_m3"
        ],
        "frame": frame,
        "phase": violation["phase"],
        "classification": "IMPORTED_RIGHT_FINGER_LIMIT_ESCAPE_PREEMPTS_LATER_ENVIRONMENT_CONTACT",
        "historical_secondary_failure_code": (
            "ENVIRONMENT_CONTACT_FORCED_LIMIT_VIOLATION"
        ),
        "historical_secondary_failure_status": (
            "NOT_REACHED_BECAUSE_FIRST_FRAME_FAILURE_MUST_ABORT"
        ),
        "contacts": frame_record["contacts"],
        "fresh_process": True,
        "process": process,
        "pose_capture_process": pose_capture_process,
        "stage_absolute_path": str(FROZEN_STAGE.resolve()),
        "stage_sha256": stage_before,
        "stage_immutable": _sha256(FROZEN_STAGE) == stage_before,
        "dynamic_failure_raw_screenshots": {
            "overview": str(overview.resolve()),
            "closeup": str(closeup.resolve()),
        },
        "raw_screenshots": pose_capture["raw_screenshots"],
        "pose_capture_semantics": pose_capture["semantics"],
        "runtime_report_absolute_path": str(runtime_path.resolve()),
        "telemetry_absolute_path": str(telemetry_path.resolve()),
        "source_or_final_asset_modified": False,
        "task8": "NOT_RUN",
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 grasp initialization negative controls",
        "",
        f"- Status: `{report['status']}`",
        f"- Fresh processes: `{report['all_fresh_processes']}`",
        f"- Task 8: `{report['task8']}`",
        "",
        "| Scenario | Result | Expected | Observed | Annotated evidence |",
        "|---|---|---|---|---|",
    ]
    lines.extend(
        "| {scenario} | {status} | {expected} | {observed} | `{image}` |".format(
                scenario=record["scenario"],
                status=record["status"],
                expected=record["expected_failure_code"],
                observed=",".join(record["observed_failure_codes"]),
                image=record["annotated_screenshot"],
            )
        for record in report["controls"]
    )
    lines.extend(
        [
            "",
            "Expected failures count as control PASS only when the exact machine "
            "failure code is observed. Screenshots are auxiliary evidence; qpos, "
            "overlap, contacts, Stage hash, and runtime telemetry are authoritative.",
            "",
        ]
    )
    return "\n".join(lines)


def build() -> dict[str, Any]:
    if ARTIFACT_ROOT.exists():
        raise FileExistsError(f"fresh artifact root already exists: {ARTIFACT_ROOT}")
    ARTIFACT_ROOT.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for scenario in SCENARIOS[:3]:
        scenario_root = ARTIFACT_ROOT / scenario.lower()
        scenario_root.mkdir()
        output = scenario_root / "machine_record.json"
        command = [
            str(ISAAC_PYTHON),
            str(Path(__file__).resolve()),
            "--runtime-scenario",
            scenario,
            "--runtime-output",
            str(output),
            "--screenshot-root",
            str(scenario_root / "screenshots_raw"),
        ]
        process = _run_logged(
            command,
            scenario_root / "runtime.log",
            timeout_s=900.0,
        )
        if process["exit_code"] != 0 or not output.is_file():
            raise RuntimeError(
                f"structural negative control failed: {scenario}: {process}"
            )
        record = json.loads(output.read_text(encoding="utf-8"))
        record["process"] = process
        records.append(record)
    sample_root = ARTIFACT_ROOT / "sample_02_environment_interference"
    sample_root.mkdir()
    records.append(_run_sample02(sample_root))
    for record in records:
        raw = record["raw_screenshots"]
        annotated = (
            ARTIFACT_ROOT
            / str(record["scenario"]).lower()
            / "screenshots_annotated/failure_or_control_annotated.png"
        )
        _annotate_capture(
            overview=Path(raw["overview"]),
            closeup=Path(raw["closeup"]),
            destination=annotated,
            record=record,
        )
        record["raw_screenshot"] = raw["overview"]
        record["raw_screenshot_sha256"] = _sha256(Path(raw["overview"]))
        record["annotated_screenshot"] = str(annotated.resolve(strict=True))
        record["annotated_screenshot_sha256"] = _sha256(annotated)
        record["visual_model_review"] = "PENDING"
    report = aggregate_controls(records)
    report["status"] = "PARTIAL" if report["status"] == "PASS" else "FAIL"
    report["reason"] = "PENDING_PER_IMAGE_VISUAL_MODEL_REVIEW"
    _atomic_json(OUTPUT_JSON, report)
    OUTPUT_MD.write_text(_render_markdown(report), encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-scenario", choices=SCENARIOS[:3])
    parser.add_argument("--runtime-output", type=Path)
    parser.add_argument("--screenshot-root", type=Path)
    parser.add_argument("--pose-capture-q-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.runtime_scenario:
        if args.runtime_output is None or args.screenshot_root is None:
            raise ValueError(
                "runtime scenario requires output and screenshot root"
            )
        return _run_structural_scenario(
            scenario=args.runtime_scenario,
            output=args.runtime_output,
            screenshot_root=args.screenshot_root,
        )
    if args.pose_capture_q_json is not None:
        if args.runtime_output is None or args.screenshot_root is None:
            raise ValueError("pose capture requires output and screenshot root")
        return _run_pose_capture(
            joint_q_path=args.pose_capture_q_json.resolve(strict=True),
            output=args.runtime_output,
            screenshot_root=args.screenshot_root,
        )
    report = build()
    print(
        json.dumps(
            {
                "status": report["status"],
                "reason": report["reason"],
                "output": str(OUTPUT_JSON.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
