#!/usr/bin/env python3
"""Run PI05 subtask decoding on Aloha HDF5 episodes and write an MP4 (quad + caption, like eval exports).

Reads ``episode_*.hdf5`` under a directory (e.g. aloha-2.0 ``.../2026_03.16_twist_many``),
subsamples in time (default: 50 Hz source → 5 Hz → every 10th frame), runs ``infer_subtask_batch``,
and encodes each frame as the same white-header + 2×2 camera layout as
``eval_subtask_accuracy_train.py`` quad PNGs.

Writes **one MP4 per episode**: ``{output_dir}/{episode_stem}_subtask_overlay.mp4``.

Example:
  uv run python scripts/hdf5_subtask_overlay_video.py \\
    --data-dir /path/to/aloha-2.0/aloha_data/aloha_stationary/2026_03.16_twist_many \\
    --output-dir .
"""

from __future__ import annotations

import importlib.util
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import tyro
from PIL import Image

from openpi.evaluation import subtask_obs_utils as _obs
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

_EPISODE_RE = re.compile(r"^episode_(\d+)\.hdf5$", re.IGNORECASE)

_QUAD_MODEL_KEYS: tuple[str, ...] = (
    "cam_high",
    "cam_low",
    "cam_left_wrist",
    "cam_right_wrist",
)


def _load_eval_quad_builder():
    """Reuse quad renderer from eval script (same pixels as ``__quad.png``)."""
    path = Path(__file__).resolve().parent / "eval_subtask_accuracy_train.py"
    spec = importlib.util.spec_from_file_location("eval_subtask_accuracy_train", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod._build_quad_preview_image


_BUILD_QUAD = _load_eval_quad_builder()


def _reencode_mp4_h264(path: Path) -> None:
    """Replace file in place with H.264 / yuv420p + faststart (broad player support)."""
    tmp = path.with_name(path.name + ".h264.tmp.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(tmp),
            ],
            check=True,
        )
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _decode_hdf5_image(arr: np.ndarray, *, compressed: bool) -> np.ndarray:
    """Return RGB uint8 HWC."""
    if compressed:
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("cv2.imdecode failed")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = np.asarray(arr)
    if x.ndim == 3 and x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    return np.ascontiguousarray(x.astype(np.uint8))


def _hwc_rgb_to_chw(img: np.ndarray) -> np.ndarray:
    return np.transpose(img, (2, 0, 1)).astype(np.uint8, copy=False)


def _list_episode_hdf5(data_dir: Path) -> list[Path]:
    cands = [p for p in data_dir.glob("*.hdf5") if _EPISODE_RE.match(p.name)]

    def sort_key(p: Path) -> tuple[int, str]:
        m = _EPISODE_RE.match(p.name)
        return (int(m.group(1)), p.name.lower()) if m else (10**9, p.name.lower())

    return sorted(cands, key=sort_key)


def _open_hdf5_cameras(h5: h5py.File) -> tuple[bool, list[str], h5py.Group]:
    if "/observations/images" not in h5:
        raise ValueError("HDF5 missing /observations/images")
    grp = h5["/observations/images"]
    cams = sorted(grp.keys())
    if len(cams) < 4:
        raise ValueError(f"Need 4 camera datasets, found {len(cams)}: {cams}")
    compressed = bool(h5.attrs.get("compress", False))
    return compressed, cams[:4], grp


def _map_cameras_to_model(
    h5_cam_names: list[str],
    cam_map: tuple[str, ...] | None,
) -> dict[str, str]:
    """HDF5 dataset name -> model key suffix (cam_high, ...)."""
    expected = _QUAD_MODEL_KEYS
    if cam_map is not None and len(cam_map) == 4:
        return dict(zip(cam_map, expected, strict=True))
    if set(h5_cam_names) == set(expected):
        return {n: n for n in expected}
    return dict(zip(h5_cam_names, expected, strict=True))


def _clip_or_pad(vec: np.ndarray, dim: int) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.shape[0] >= dim:
        return v[:dim].copy()
    out = np.zeros(dim, dtype=np.float32)
    out[: v.shape[0]] = v
    return out


def _action_vector(h5: h5py.File, t: int, state_dim: int) -> np.ndarray:
    for key in ("/action", "action", "/observations/action"):
        if key in h5:
            return _clip_or_pad(np.asarray(h5[key][t], dtype=np.float32), state_dim)
    return np.zeros(state_dim, dtype=np.float32)


@dataclass(frozen=True)
class _EpMeta:
    compressed: bool
    h5_to_model: dict[str, str]
    n_frames: int


def _infer_obs_from_h5(
    h5: h5py.File,
    t: int,
    meta: _EpMeta,
    *,
    prompt: str,
    state_dim: int,
    action_horizon: int,
) -> dict:
    images_grp = h5["/observations/images"]
    qpos_ds = h5["/observations/qpos"]
    qpos = _clip_or_pad(np.asarray(qpos_ds[t], dtype=np.float32), state_dim)
    act = _action_vector(h5, t, state_dim)
    imgs: dict[str, np.ndarray] = {}
    for h5_name, model_suffix in meta.h5_to_model.items():
        raw = images_grp[h5_name][t]
        rgb = _decode_hdf5_image(raw, compressed=meta.compressed)
        imgs[f"observation.images.{model_suffix}"] = _hwc_rgb_to_chw(rgb)
    row = {
        "task": prompt,
        **imgs,
        "observation.state": qpos,
        "action": act,
    }
    return _obs.lerobot_row_to_subtask_infer_obs(row, action_horizon=action_horizon, pop_subtask=True)


def _scan_episodes(
    eps: list[Path],
    cam_map: tuple[str, ...] | None,
) -> dict[Path, _EpMeta]:
    ep_meta: dict[Path, _EpMeta] = {}
    for ep_path in eps:
        try:
            with h5py.File(ep_path, "r") as h5:
                compressed, h5_cams, images_grp = _open_hdf5_cameras(h5)
                h5_to_model = _map_cameras_to_model(h5_cams, cam_map)
                n = int(images_grp[h5_cams[0]].shape[0])
                ep_meta[ep_path] = _EpMeta(compressed, h5_to_model, n)
        except OSError as e:
            logging.warning("Skipping unreadable HDF5 %s: %s", ep_path.name, e)
    return ep_meta


def _dummy_obs_for_layout() -> dict:
    return {f"observation.images.{k}": np.zeros((3, 256, 256), dtype=np.uint8) for k in _QUAD_MODEL_KEYS}


@dataclass
class Args:
    data_dir: Path
    """Directory containing ``episode_*.hdf5`` files."""

    output_dir: Path = Path(".")
    """Directory for per-episode videos: ``{episode_stem}_subtask_overlay.mp4``."""

    config_name: str = "twist_and_static_mixture_full_finetune"
    checkpoint: Path = Path(
        "checkpoints/twist_and_static_mixture_full_finetune/"
        "twist_and_static_mixture_full_finetune_vast_20260405_100600/39999"
    )

    source_fps: float = 50.0
    """HDF5 / robot recording rate."""

    target_fps: float = 5.0
    """Video frame rate (each written frame is one subsampled HDF5 step)."""

    prompt: str = "Process all bottles"
    """LeRobot-style task string (``task`` → ``prompt`` in policy)."""

    batch_size: int = 8
    temperature: float = 0.0

    state_dim: int = 14
    """Truncate or zero-pad ``qpos`` / ``action`` to this length (dual-arm Aloha)."""

    cam_map: tuple[str, ...] | None = None
    """Optional four HDF5 camera dataset names in order: cam_high, cam_low, left_wrist, right_wrist."""

    skip_ffmpeg_h264: bool = False
    """If true, keep OpenCV ``mp4v`` output only (may not play in some players)."""


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Not a directory: {data_dir}")

    eps = _list_episode_hdf5(data_dir)
    if not eps:
        raise SystemExit(f"No episode_*.hdf5 under {data_dir}")

    step = max(1, int(round(args.source_fps / args.target_fps)))
    logging.info(
        "Episodes: %d | subsample every %d frame(s) (~%.2f Hz → %.2f Hz)",
        len(eps),
        step,
        args.source_fps,
        args.target_fps,
    )

    ep_meta = _scan_episodes(eps, args.cam_map)
    if not ep_meta:
        raise SystemExit("No readable episode HDF5 files.")

    out_root = args.output_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    train_cfg = _config.get_config(args.config_name)
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    logging.info("Loading policy from %s", args.checkpoint)
    policy = _policy_config.create_trained_policy(
        train_cfg,
        args.checkpoint,
        repack_transforms=data_config.repack_transforms,
    )
    ah = train_cfg.model.action_horizon

    infer_bs = max(1, args.batch_size)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dummy = _dummy_obs_for_layout()

    for ep_path in eps:
        if ep_path not in ep_meta:
            continue
        meta = ep_meta[ep_path]
        indices_ep = [(ep_path, t) for t in range(0, meta.n_frames, step)]
        if not indices_ep:
            logging.warning("No subsampled frames for %s, skip.", ep_path.name)
            continue

        logging.info(
            "Episode %s: %d subsampled frame(s) — infer_subtask_batch …",
            ep_path.name,
            len(indices_ep),
        )
        texts_ep: list[str] = []
        for start in range(0, len(indices_ep), infer_bs):
            chunk = indices_ep[start : start + infer_bs]
            handles: dict[Path, h5py.File] = {}
            try:
                for ep, _ in chunk:
                    if ep not in handles:
                        handles[ep] = h5py.File(ep, "r")
                obs_list = [
                    _infer_obs_from_h5(
                        handles[ep],
                        t,
                        ep_meta[ep],
                        prompt=args.prompt,
                        state_dim=args.state_dim,
                        action_horizon=ah,
                    )
                    for ep, t in chunk
                ]
                outs = policy.infer_subtask_batch(
                    obs_list,
                    batch_size=infer_bs,
                    temperature=args.temperature,
                )
                if len(outs) != len(chunk):
                    raise RuntimeError(f"infer_subtask_batch: expected {len(chunk)} outputs, got {len(outs)}")
                for out in outs:
                    texts_ep.append(str(out.get("subtask_text") or ""))
            finally:
                for h in handles.values():
                    h.close()

        captions_ep: list[str] = []
        for (ep_p, t), subtask_text in zip(indices_ep, texts_ep, strict=True):
            raw = subtask_text[:500]
            captions_ep.append(
                "\n".join(
                    [
                        f"episode={ep_p.name}",
                        f"frame_idx={t}",
                        f"subtask_text={subtask_text}",
                        f"raw_subtask_text={raw}",
                    ]
                )
            )

        max_w, max_h = 0, 0
        for cap in captions_ep:
            sz = _BUILD_QUAD(dummy, caption_text=cap).size
            max_w = max(max_w, sz[0])
            max_h = max(max_h, sz[1])

        out_path = out_root / f"{ep_path.stem}_subtask_overlay.mp4"
        writer = cv2.VideoWriter(str(out_path), fourcc, float(args.target_fps), (max_w, max_h))
        if not writer.isOpened():
            raise SystemExit(f"Failed to open VideoWriter for {out_path}")

        fh: h5py.File | None = None
        try:
            fh = h5py.File(ep_path, "r")
            for (_ep_p, t), cap in zip(indices_ep, captions_ep, strict=True):
                obs = _infer_obs_from_h5(
                    fh,
                    t,
                    meta,
                    prompt=args.prompt,
                    state_dim=args.state_dim,
                    action_horizon=ah,
                )
                quad = _BUILD_QUAD(obs, caption_text=cap)
                canvas = Image.new("RGB", (max_w, max_h), (255, 255, 255))
                canvas.paste(quad, (0, 0))
                bgr = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
                writer.write(bgr)
        finally:
            if fh is not None:
                fh.close()
            writer.release()

        if not args.skip_ffmpeg_h264:
            logging.info("Re-encoding %s to H.264 …", out_path.name)
            _reencode_mp4_h264(out_path)

        logging.info("Wrote %s (%d frames, %dx%d)", out_path, len(indices_ep), max_w, max_h)


if __name__ == "__main__":
    try:
        main(tyro.cli(Args))
    except KeyboardInterrupt:
        sys.exit(130)
