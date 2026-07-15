from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import h5py
import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
GYM_ALOHA_ASSETS = REPO_ROOT / ".venv/lib/python3.11/site-packages/gym_aloha/assets"
DEFAULT_HDF5 = Path(
    "/home/eii/data/openpi0.5-rtc-reward-learning/rollouts/key_regions/"
    "twist_off_the_bottle_cap/2026-07-08/warmup/"
    "key_region_00d1891748c74b5d8556a9ab153c8ec6/episode.hdf5"
)


ARM_JOINTS = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
)


def _parse_xyz(value: str) -> tuple[float, float, float]:
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected x,y,z, got {value!r}")
    return tuple(parts)  # type: ignore[return-value]


def _copy_gym_aloha_assets(model_dir: Path) -> Path:
    asset_dir = model_dir / "gym_aloha_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    for src in GYM_ALOHA_ASSETS.iterdir():
        for dst in (asset_dir / src.name, model_dir / src.name):
            if dst.exists():
                continue
            if src.is_file():
                try:
                    dst.symlink_to(src)
                except OSError:
                    shutil.copy2(src, dst)
    return asset_dir


def _axis_geom(name: str, fromto: str, rgba: str) -> str:
    return f'<geom name="{name}" type="cylinder" fromto="{fromto}" size="0.006" rgba="{rgba}"/>'


def _make_model_xml(
    model_dir: Path,
    pipe_start: tuple[float, float, float],
    pipe_end: tuple[float, float, float],
    *,
    offwidth: int,
    offheight: int,
) -> Path:
    _copy_gym_aloha_assets(model_dir)
    pipe_fromto = " ".join(f"{v:.4f}" for v in (*pipe_start, *pipe_end))

    actuators = []
    for side in ("left", "right"):
        for joint, kp in zip(ARM_JOINTS, (800, 1600, 800, 10, 50, 20), strict=True):
            actuators.append(f'<position joint="vx300s_{side}/{joint}" kp="{kp}" />')
        actuators.append(f'<position joint="vx300s_{side}/left_finger" kp="200" />')
        actuators.append(f'<position joint="vx300s_{side}/right_finger" kp="200" />')

    xml = f"""<mujoco model="aloha_workcell_minimal">
  <include file="gym_aloha_assets/scene.xml"/>
  <include file="gym_aloha_assets/vx300s_dependencies.xml"/>

  <visual>
    <global offwidth="{offwidth}" offheight="{offheight}"/>
  </visual>

  <worldbody>
    <include file="gym_aloha_assets/vx300s_left.xml"/>
    <include file="gym_aloha_assets/vx300s_right.xml"/>

    <body name="table_frame_T" pos="0 0.35 0.035">
      <geom name="T_origin" type="sphere" size="0.018" rgba="0.05 0.20 1.0 1"/>
      {_axis_geom("T_x_axis", "0 0 0 0.25 0 0", "1 0 0 1")}
      {_axis_geom("T_y_axis", "0 0 0 0 0.25 0", "0 0.8 0 1")}
      {_axis_geom("T_z_axis", "0 0 0 0 0 0.25", "0.1 0.2 1 1")}
    </body>

    <body name="placeholder_pipe">
      <geom name="pipe_axis_placeholder" type="cylinder" fromto="{pipe_fromto}" size="0.018" rgba="0.9 0.05 0.05 0.55"/>
      <geom name="pipe_inlet_placeholder" type="sphere" pos="{pipe_end[0]:.4f} {pipe_end[1]:.4f} {pipe_end[2]:.4f}" size="0.032" rgba="1 0 0 1"/>
    </body>
  </worldbody>

  <actuator>
    {"".join(actuators)}
  </actuator>
</mujoco>
"""
    xml_path = model_dir / "workcell.xml"
    xml_path.write_text(xml, encoding="utf-8")
    return xml_path


def _load_hdf5_qpos(path: Path | None) -> np.ndarray:
    if path is None:
        return np.array([[0, -0.96, 1.16, 1.57, 0, -1.57, 0.0, 0, -0.96, 1.16, 0, 0, 0, 0.0]], dtype=np.float32)
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        qpos = np.asarray(f["observations/qpos"][:], dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 14:
        raise ValueError(f"expected HDF5 observations/qpos shape [T,14], got {qpos.shape}")
    return qpos


def _gripper_norm_to_fingers(value: float) -> tuple[float, float]:
    value = float(np.clip(value, 0.0, 1.0))
    opening = 0.021 + value * (0.057 - 0.021)
    return opening, -opening


def _apply_aloha_qpos(data: mujoco.MjData, qpos14: np.ndarray) -> None:
    data.qpos[0:6] = qpos14[0:6]
    data.qpos[6:8] = _gripper_norm_to_fingers(float(qpos14[6]))
    data.qpos[8:14] = qpos14[7:13]
    data.qpos[14:16] = _gripper_norm_to_fingers(float(qpos14[13]))


def _render_video(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    out_path: Path,
    *,
    camera: str,
    width: int,
    height: int,
    fps: int,
    stride: int,
    hdf5_path: Path | None,
) -> None:
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    raw_path = out_path.with_name(out_path.stem + "_mp4v.mp4")
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {raw_path}")

    total = int(math.ceil(len(qpos) / stride))
    for out_i, frame_i in enumerate(range(0, len(qpos), stride)):
        _apply_aloha_qpos(data, qpos[frame_i])
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        lines = [
            "Minimal ALOHA workcell viewer",
            f"frame {frame_i + 1}/{len(qpos)}  rendered {out_i + 1}/{total}",
            "blue/green/red axes: table frame T; translucent red: placeholder pipe axis",
        ]
        if hdf5_path is not None:
            lines.append(f"HDF5: {hdf5_path.parent.name}/episode.hdf5")
        for row, text in enumerate(lines):
            cv2.putText(bgr, text, (18, 30 + row * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(bgr, text, (18, 30 + row * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
        writer.write(bgr)

    writer.release()
    close = getattr(renderer, "close", None)
    if close is not None:
        close()

    if shutil.which("ffmpeg"):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(raw_path),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out_path),
            ],
            check=True,
        )
    else:
        raw_path.replace(out_path)


def _write_index(output_dir: Path, video_path: Path, xml_path: Path) -> None:
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Minimal ALOHA Workcell</title>
  <style>
    body {{ margin: 0; background: #111827; color: #e5e7eb; font-family: system-ui, sans-serif; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    video {{ width: 100%; background: #000; border: 1px solid #374151; }}
    code {{ color: #bae6fd; }}
  </style>
</head>
<body>
  <main>
    <h1>Minimal ALOHA Workcell</h1>
    <video controls autoplay muted loop preload="auto">
      <source src="{video_path.name}" type="video/mp4">
    </video>
    <p>Model XML: <code>{xml_path}</code></p>
    <p>Blue/green/red axes show table frame T. The translucent red line is the placeholder pipe axis.</p>
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a minimal ALOHA MuJoCo workcell and optional HDF5 qpos replay.")
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5 if DEFAULT_HDF5.exists() else None)
    parser.add_argument("--output-dir", type=Path, default=Path("local_eval_assets/aloha_workcell_minimal"))
    parser.add_argument("--camera", default="angle", help="MuJoCo camera name, e.g. angle, top, front_close")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--stride", type=int, default=2, help="Render every Nth HDF5 qpos frame.")
    parser.add_argument("--pipe-start", type=_parse_xyz, default=(0.45, 0.58, 0.36))
    parser.add_argument("--pipe-end", type=_parse_xyz, default=(0.25, 0.45, 0.22))
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    xml_path = _make_model_xml(
        model_dir,
        args.pipe_start,
        args.pipe_end,
        offwidth=args.width,
        offheight=args.height,
    )
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    qpos = _load_hdf5_qpos(args.hdf5)

    video_path = args.output_dir / "aloha_workcell_replay.mp4"
    _render_video(
        model,
        qpos,
        video_path,
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        stride=max(1, args.stride),
        hdf5_path=args.hdf5,
    )
    _write_index(args.output_dir, video_path, xml_path)

    print(f"model_xml={xml_path}")
    print(f"html={args.output_dir / 'index.html'}")
    print(f"video={video_path}")
    print(f"qpos_frames={len(qpos)}")


if __name__ == "__main__":
    main()
