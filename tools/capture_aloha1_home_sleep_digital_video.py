#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Capture synchronized full-arm normal/collider evidence for Home-Sleep failure."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import traceback
from typing import Any

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from tools.aloha1_mapping.home_sleep_correspondence import command_index_for_physics_frame
from tools.aloha1_mapping.isaac_screenshot import look_at_orientation_wxyz
from tools.capture_aloha1_cad_derived_collision_evidence import _build_authored_collider_overlay
from tools.capture_aloha1_cad_derived_collision_evidence import _stage_mesh_points
from tools.capture_aloha1_cad_derived_collision_evidence import _update_authored_collider_overlay
from tools.validate_aloha1_home_sleep_digital import ARTICULATION_PATHS
from tools.validate_aloha1_home_sleep_digital import EXPECTED_DOF_ORDER
from tools.validate_aloha1_home_sleep_digital import _apply_targets
from tools.validate_aloha1_home_sleep_digital import _install_session_layers
from tools.validate_aloha1_home_sleep_digital import _sha256

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / (
    "assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/"
    "aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_z_up_meters_diagnostic.usda"
)
MANIFEST = ROOT / "reports/aloha1_mapping/aloha1_home_sleep_command_manifest.json"
NUMERIC_REPORT = ROOT / "reports/aloha1_mapping/aloha1_home_sleep_digital_run_01.json"
FINGER_LAYER = ROOT / (
    "assets/Trossen/ALOHA1/1.0/diagnostics/finger_limit_pair_collision_candidate/1.0/"
    "configuration/finger_source_limits.usda"
)
DEFAULT_OUTPUT_ROOT = ROOT / ".codex/artifacts/20260803-aloha1-home-sleep-digital-twin/failure_evidence"
DEFAULT_REPORT = ROOT / "reports/aloha1_mapping/aloha1_home_sleep_digital_video_review.json"
RESOLUTION = (960, 540)
CAPTURE_FPS = 15
CAPTURE_STRIDE = 4


def _selected_trajectory_key_indices(
    samples: list[dict[str, Any]],
) -> dict[int, str]:
    """Return review points for a legal selected Sleep trajectory."""

    labels = {
        "initial_home_hold": "initial_home",
        "cycle_01_sleep_hold": "cycle_01_exact_sleep",
        "cycle_01_home_hold": "cycle_01_return_home",
        "cycle_03_sleep_hold": "cycle_03_exact_sleep",
        "cycle_03_home_hold": "final_home",
    }
    result = {
        max(int(sample["index"]) for sample in samples if sample["segment"] == segment): label
        for segment, label in labels.items()
    }
    if len(result) != len(labels):
        raise ValueError("selected trajectory review stages are not distinct")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=STAGE)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--numeric-report", type=Path, default=NUMERIC_REPORT)
    parser.add_argument("--finger-limit-layer", type=Path, default=FINGER_LAYER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--collision-only", action="store_true")
    parser.add_argument("--reannotate-only", action="store_true")
    parser.add_argument(
        "--evidence-kind",
        choices=("failure", "selected_historical_sleep"),
        default="failure",
    )
    return parser.parse_args()


def _annotation_footer(*, evidence_kind: str, target_outside: bool) -> list[str]:
    if evidence_kind == "selected_historical_sleep":
        if target_outside:
            raise ValueError("selected historical Sleep annotation has an illegal target")
        return [
            "",
            "all targets inside frozen USD/URDF limits",
            "exact endpoint numeric gate: PASS",
            "contacts: none / impulse=0",
            "DIGITAL GATE: PASS",
            "This image does not authorize real motion.",
        ]
    return [
        "",
        "! = official target outside USD limit",
        (
            "Observed: PhysX clamp active in this frame"
            if target_outside
            else "This frame is legal; sequence failure retained"
        ),
        "contacts: none / impulse=0",
        "DIGITAL GATE: FAIL",
        "This image does not authorize real motion.",
    ]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _save_camera(camera: Any, destination: Path) -> dict[str, Any]:
    rgba = camera.get_rgba(device="cpu")
    if rgba is None:
        raise RuntimeError("camera RGB annotator returned no data")
    pixels = np.asarray(rgba)
    if pixels.shape != (RESOLUTION[1], RESOLUTION[0], 4):
        raise RuntimeError(f"unexpected camera shape: {pixels.shape}")
    if pixels.dtype != np.uint8:
        maximum = float(np.nanmax(pixels))
        pixels = np.clip(pixels * (255.0 if maximum <= 1.0 else 1.0), 0, 255).astype(
            np.uint8
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGBA").convert("RGB").save(destination)
    return {
        "absolute_path": str(destination.resolve()),
        "sha256": _sha256(destination),
        "resolution": list(RESOLUTION),
        "pixel_min": int(pixels.min()),
        "pixel_max": int(pixels.max()),
    }


def _annotate(
    source: Path,
    destination: Path,
    *,
    mode: str,
    stage_hash: str,
    frame: int,
    time_s: float,
    label: str,
    target: list[float],
    readback: list[float],
    limits: tuple[list[float], list[float]],
    evidence_kind: str,
) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    panel_width = 470
    canvas = Image.new("RGB", (image.width + panel_width, image.height), (18, 20, 26))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    border = (70, 220, 120) if evidence_kind == "selected_historical_sleep" else (255, 75, 75)
    draw.rectangle((8, 8, image.width - 8, image.height - 8), outline=border, width=4)
    draw.text((18, 18), "FULL follower_left arm", font=_font(22), fill=(255, 255, 255))
    subtitle = (
        "Qualification: official historical legal Sleep"
        if evidence_kind == "selected_historical_sleep"
        else "Failure evidence: Sleep limit clamp"
    )
    draw.text((18, 48), subtitle, font=_font(18), fill=border)
    x = image.width + 18
    lines = [
        "Isaac Sim 5.1.0.0",
        f"mode: {mode}",
        f"stage: {stage_hash[:16]}...",
        f"frame/time: {frame} / {time_s:.3f}s",
        f"phase: {label}",
        "ACTIVE: follower_left",
        "right arm + grippers stationary",
        "",
        "joint      target   readback   legal range",
    ]
    for index, name in enumerate(EXPECTED_DOF_ORDER[:6]):
        low, high = limits[0][index], limits[1][index]
        marker = " !" if target[index] < low or target[index] > high else ""
        lines.append(
            f"{name[:10]:10s} {target[index]:+6.3f} {readback[index]:+8.3f} "
            f"[{low:+.3f},{high:+.3f}]{marker}"
        )
    target_outside = any(
        target[index] < limits[0][index] or target[index] > limits[1][index]
        for index in range(6)
    )
    lines.extend(
        _annotation_footer(
            evidence_kind=evidence_kind,
            target_outside=target_outside,
        )
    )
    y = 18
    for line in lines:
        color = (255, 100, 100) if "!" in line or "FAIL" in line else (225, 230, 238)
        draw.text((x, y), line, font=_font(15), fill=color)
        y += 24
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _reannotate_existing(args: argparse.Namespace) -> int:
    report_path = args.report.resolve(strict=True)
    numeric_path = args.numeric_report.resolve(strict=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    numeric = json.loads(numeric_path.read_text(encoding="utf-8"))
    limits = tuple(numeric["preflight"]["limits"]["follower_left"])
    records = {
        int(record["physics_frame"]): record
        for record in report["capture"]["frame_records"]
    }
    for screenshot in report["screenshots"]:
        record = records[int(screenshot["physics_frame"])]
        annotated = Path(screenshot["annotated_absolute_path"])
        _annotate(
            Path(screenshot["raw_absolute_path"]),
            annotated,
            mode=str(screenshot["mode"]),
            stage_hash=str(report["stage"]["sha256_before"]),
            frame=int(record["physics_frame"]),
            time_s=float(record["physics_time_s"]),
            label=str(screenshot["label"]),
            target=list(record["target_arm_q"]),
            readback=list(record["readback_q"]),
            limits=(list(limits[0]), list(limits[1])),
            evidence_kind=args.evidence_kind,
        )
        screenshot["annotated_sha256"] = _sha256(annotated)
        screenshot["visual_review"] = "PENDING_VISUAL_MODEL_REVIEW"
    report["status"] = "PENDING_VISUAL_MODEL_REVIEW"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "reannotated": len(report["screenshots"])}))
    return 0


def _encode_video(frame_root: Path, destination: Path) -> dict[str, Any]:
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(CAPTURE_FPS),
        "-i",
        str(frame_root / "%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not destination.is_file():
        raise RuntimeError(f"ffmpeg failed ({completed.returncode}): {completed.stderr[-2000:]}")
    return {
        "absolute_path": str(destination.resolve()),
        "sha256": _sha256(destination),
        "fps": CAPTURE_FPS,
        "ffmpeg_command": command,
        "ffmpeg_exit_code": completed.returncode,
    }


def _first_limit_exceedance(
    samples: list[dict[str, Any]], lower: np.ndarray, upper: np.ndarray
) -> int:
    for sample in samples:
        q = np.asarray(sample["q_rad"], dtype=np.float64)
        if bool((q < lower).any() or (q > upper).any()):
            return int(sample["index"])
    raise RuntimeError("numeric failure report has no out-of-limit target")


def main(args: argparse.Namespace) -> int:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import open_stage
    from isaacsim.sensors.camera import Camera
    from omni.physx import get_physx_interface
    import omni.usd
    from pxr import Gf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdLux

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"capture output already exists: {output_root}")
    output_root.mkdir(parents=True)
    stage_path = args.stage.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    numeric_path = args.numeric_report.resolve(strict=True)
    finger_layer = args.finger_limit_layer.resolve(strict=True)
    stage_hash = _sha256(stage_path)
    manifest_hash = _sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    numeric = json.loads(numeric_path.read_text(encoding="utf-8"))
    if args.evidence_kind == "failure":
        if numeric["status"] != "FAIL" or numeric["summary"]["gates"]["endpoints"] is not False:
            raise RuntimeError("failure capture requires the verified Sleep endpoint failure")
    elif numeric["status"] != "PASS" or numeric["summary"]["gates"]["endpoints"] is not True:
        raise RuntimeError("selected Sleep capture requires a passing exact endpoint run")
    if numeric["stage"]["sha256_before"] != stage_hash:
        raise RuntimeError("numeric/capture Stage mismatch")
    if numeric["manifest"]["sha256_before"] != manifest_hash:
        raise RuntimeError("numeric/capture manifest mismatch")
    if not open_stage(str(stage_path)):
        raise RuntimeError(f"failed to open {stage_path}")
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    _install_session_layers(stage, finger_layer)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        dome = UsdLux.DomeLight.Define(stage, "/World/HomeSleepEvidence/Dome")
        dome.CreateIntensityAttr(900.0)
        key = UsdLux.DistantLight.Define(stage, "/World/HomeSleepEvidence/Key")
        key.CreateIntensityAttr(1800.0)
        key.AddRotateXYZOp().Set(Gf.Vec3f(35.0, -25.0, -25.0))
        overlay_root_path, overlay_records = _build_authored_collider_overlay(stage)
        for record in overlay_records:
            clone = UsdGeom.Mesh(
                stage.GetPrimAtPath(record["session_visual_clone_prim"])
            )
            clone.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.02, 0.02)])
            clone.GetDisplayOpacityAttr().Set([0.9])

    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=1.0 / int(manifest["physics_rate_hz"]),
        rendering_dt=1.0 / int(manifest["physics_rate_hz"]),
    )
    world.get_physics_context().set_solve_articulation_contact_last(True)
    articulations = {}
    for robot, path in ARTICULATION_PATHS.items():
        item = SingleArticulation(
            prim_path=path,
            name=f"home_sleep_capture_{robot}",
            reset_xform_properties=False,
        )
        world.scene.add(item)
        articulations[robot] = item
    camera = Camera(
        prim_path="/World/HomeSleepEvidence/OverviewCamera",
        name="home_sleep_full_arm_overview",
        resolution=RESOLUTION,
        frequency=60,
    )
    world.scene.add(camera)
    world.reset()
    if any(list(item.dof_names) != EXPECTED_DOF_ORDER for item in articulations.values()):
        raise RuntimeError("capture DOF order drift")
    properties = articulations["follower_left"].dof_properties.copy()
    lower = np.asarray([float(row["lower"]) for row in properties[:6]], dtype=np.float64)
    upper = np.asarray([float(row["upper"]) for row in properties[:6]], dtype=np.float64)
    samples = manifest["samples"]

    left_initial = np.asarray(articulations["follower_left"].get_joint_positions(), dtype=np.float32)
    right_initial = np.asarray(articulations["follower_right"].get_joint_positions(), dtype=np.float32)
    left_home = left_initial.copy()
    left_home[:6] = np.asarray(manifest["home_rad"], dtype=np.float32)
    articulations["follower_left"].set_joint_positions(left_home)
    articulations["follower_left"].set_joint_velocities(np.zeros_like(left_home))
    _apply_targets(articulations["follower_left"], left_home[:8], range(8))
    _apply_targets(articulations["follower_right"], right_initial[:8], range(8))
    for _ in range(30):
        world.step(render=False)
    frozen_left_gripper = np.asarray(
        articulations["follower_left"].get_joint_positions(), dtype=np.float32
    )[6:]
    frozen_right = np.asarray(
        articulations["follower_right"].get_joint_positions(), dtype=np.float32
    )

    # Derive one fixed camera from both mathematically reachable endpoints.
    clouds = []
    for arm in (
        np.asarray(manifest["home_rad"], dtype=np.float32),
        np.clip(np.asarray(manifest["sleep_rad"], dtype=np.float32), lower, upper),
    ):
        state = np.asarray(articulations["follower_left"].get_joint_positions(), dtype=np.float32)
        state[:6] = arm
        articulations["follower_left"].set_joint_positions(state)
        get_physx_interface().update_transformations(True, True, False, False)  # noqa: FBT003
        clouds.append(_stage_mesh_points(stage, ("/World/follower_left/",)))
    union = np.concatenate(clouds)
    target = (union.min(axis=0) + union.max(axis=0)) / 2.0
    span = float(np.linalg.norm(union.max(axis=0) - union.min(axis=0)))
    direction = np.asarray([0.72, 1.0, 0.62], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    position = target + direction * max(2.15 * span, 1.8)
    orientation = look_at_orientation_wxyz(position, target)
    camera.set_clipping_range(0.01, 10.0)
    camera.set_world_pose(position=position, orientation=orientation, camera_axes="usd")

    # Re-establish the exact formal initial condition after the camera fit probes.
    articulations["follower_left"].set_joint_positions(left_home)
    articulations["follower_left"].set_joint_velocities(np.zeros_like(left_home))
    articulations["follower_right"].set_joint_positions(frozen_right)
    articulations["follower_right"].set_joint_velocities(np.zeros_like(frozen_right))
    for _ in range(5):
        world.step(render=True)

    overlay_root = UsdGeom.Imageable(stage.GetPrimAtPath(overlay_root_path))
    capture_modes = (
        ("collision_overlay",)
        if args.collision_only
        else ("normal", "collision_overlay")
    )
    frame_records = []
    key_raw: dict[tuple[str, str], Path] = {}
    if args.evidence_kind == "selected_historical_sleep":
        key_command_indices = _selected_trajectory_key_indices(samples)
    else:
        exceed_index = _first_limit_exceedance(samples, lower, upper)
        key_command_indices = {
            max(0, exceed_index - 1): "before_limit_exceedance",
            exceed_index: "first_limit_exceedance",
            max(
                int(sample["index"])
                for sample in samples
                if sample["segment"] == "cycle_01_sleep_hold"
            ): "first_sleep_hold_end",
            len(samples) - 1: "final_home_recovery",
        }
    physics_hz = int(manifest["physics_rate_hz"])
    command_hz = int(manifest["command_rate_hz"])
    total_frames = math.ceil(len(samples) * physics_hz / command_hz)
    capture_index = 0
    for physics_frame in range(total_frames):
        command_index = command_index_for_physics_frame(
            physics_frame,
            physics_hz=physics_hz,
            command_hz=command_hz,
            sample_count=len(samples),
        )
        sample = samples[command_index]
        target_q = np.asarray(sample["q_rad"], dtype=np.float32)
        _apply_targets(articulations["follower_left"], target_q, range(6))
        _apply_targets(articulations["follower_left"], frozen_left_gripper[:2], (6, 7))
        _apply_targets(articulations["follower_right"], frozen_right[:8], range(8))
        world.step(render=False)
        should_capture = physics_frame % CAPTURE_STRIDE == 0 or command_index in key_command_indices
        if not should_capture:
            continue
        _update_authored_collider_overlay(stage, overlay_records)
        readback = np.asarray(
            articulations["follower_left"].get_joint_positions(), dtype=np.float64
        ).tolist()
        views = {}
        for mode in capture_modes:
            if mode == "collision_overlay":
                overlay_root.MakeVisible()
            else:
                overlay_root.MakeInvisible()
            world.render()
            destination = output_root / "frames" / mode / f"{capture_index:06d}.png"
            views[mode] = _save_camera(camera, destination)
            if command_index in key_command_indices:
                key_raw[(key_command_indices[command_index], mode)] = destination
        frame_records.append(
            {
                "capture_index": capture_index,
                "physics_frame": physics_frame,
                "physics_time_s": (physics_frame + 1) / physics_hz,
                "command_index": command_index,
                "cycle": int(sample["cycle"]),
                "segment": str(sample["segment"]),
                "target_arm_q": target_q.astype(np.float64).tolist(),
                "readback_q": readback,
                "views": views,
            }
        )
        capture_index += 1

    videos = {}
    for mode in capture_modes:
        videos[mode] = _encode_video(
            output_root / "frames" / mode,
            output_root / f"aloha1_home_sleep_{mode}.mp4",
        )
        videos[mode]["frame_count"] = len(frame_records)
        videos[mode]["duration_s"] = len(frame_records) / CAPTURE_FPS

    screenshots = []
    for (label, mode), source in sorted(key_raw.items()):
        record = next(
            item
            for item in frame_records
            if item["command_index"] in key_command_indices
            and key_command_indices[item["command_index"]] == label
        )
        raw = output_root / "screenshots_raw" / f"{label}_{mode}_raw.png"
        raw.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, raw)
        annotated = output_root / "screenshots_annotated" / f"{label}_{mode}_annotated.png"
        _annotate(
            raw,
            annotated,
            mode=mode,
            stage_hash=stage_hash,
            frame=int(record["physics_frame"]),
            time_s=float(record["physics_time_s"]),
            label=label,
            target=record["target_arm_q"],
            readback=record["readback_q"],
            limits=(lower.tolist(), upper.tolist()),
            evidence_kind=args.evidence_kind,
        )
        screenshots.append(
            {
                "label": label,
                "mode": mode,
                "physics_frame": record["physics_frame"],
                "physics_time_s": record["physics_time_s"],
                "raw_absolute_path": str(raw.resolve()),
                "raw_sha256": _sha256(raw),
                "annotated_absolute_path": str(annotated.resolve()),
                "annotated_sha256": _sha256(annotated),
                "visual_review": "PENDING_VISUAL_MODEL_REVIEW",
            }
        )

    stage_hash_after = _sha256(stage_path)
    report = {
        "schema_version": 1,
        "status": "PENDING_VISUAL_MODEL_REVIEW",
        "classification": (
            "DIGITAL_OFFICIAL_HISTORICAL_SLEEP_QUALIFICATION_EVIDENCE"
            if args.evidence_kind == "selected_historical_sleep"
            else "DIGITAL_SLEEP_LIMIT_FAILURE_EVIDENCE"
        ),
        "stage": {
            "absolute_path": str(stage_path),
            "sha256_before": stage_hash,
            "sha256_after": stage_hash_after,
        },
        "manifest": {
            "absolute_path": str(manifest_path),
            "sha256": manifest_hash,
            "command_signature": manifest["command_signature"],
        },
        "numeric_report": {
            "absolute_path": str(numeric_path),
            "sha256": _sha256(numeric_path),
            "numeric_signature": numeric["summary"]["normalized_numeric_signature"],
        },
        "camera": {
            "prim_path": str(camera.prim_path),
            "position_world_m": position.tolist(),
            "target_world_m": target.tolist(),
            "orientation_wxyz": orientation.tolist(),
            "resolution": list(RESOLUTION),
            "view": "FULL_ARM_FIXED_OBLIQUE_ENGINEERING_EVIDENCE",
        },
        "capture": {
            "physics_hz": physics_hz,
            "capture_fps": CAPTURE_FPS,
            "capture_stride": CAPTURE_STRIDE,
            "frame_count": len(frame_records),
            "frame_records": frame_records,
            "modes": list(capture_modes),
        },
        "videos": videos,
        "screenshots": screenshots,
        "collider_overlay": {
            "type": "SESSION_ONLY_NON_PHYSICAL_EXACT_AUTHORED_COLLIDER_VISUAL_CLONES",
            "clone_count": len(overlay_records),
            "source_colliders": overlay_records,
        },
        "evidence": {
            "kind": args.evidence_kind,
            "numeric_status": numeric["status"],
            "exact_endpoint_gate": numeric["summary"]["gates"]["endpoints"],
            "contact_impulse_points": 0,
            "source_or_final_asset_modified": False,
            "real_execution_authorized": False,
        },
    }
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "frame_count": len(frame_records),
                "video_paths": {
                    key: value["absolute_path"] for key, value in videos.items()
                },
                "report": str(args.report.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    world.stop()
    return 0


def run() -> int:
    args = _parse_args()
    if args.reannotate_only:
        return _reannotate_existing(args)
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "width": RESOLUTION[0],
            "height": RESOLUTION[1],
            "renderer": "RaytracedLighting",
        }
    )
    exit_code = 1
    try:
        exit_code = main(args)
    except BaseException:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
