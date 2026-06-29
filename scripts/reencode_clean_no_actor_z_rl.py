from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_REPLAY_ROOT = Path("/home/eii/data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions_clean")
DEFAULT_ROLLOUT_ROOT = Path("/home/eii/data/openpi0.5-rtc-reward-learning/rollouts/key_regions")
DEFAULT_OUTPUT_ROOT = Path("/home/eii/data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions_clean_z512_cam4")
DEFAULT_CHECKPOINT = Path(
    "checkpoints/eii_rinse_11repo_cam4_fullft_rl_token_small_query/"
    "rinse_11repo_rl_token_small_query_512_from_9000_20260615/9999"
)
DEFAULT_CONFIG = "eii_rinse_11repo_cam4_fullft_rl_token_small_query"
REPLAY_KEYS = (
    "z_rl",
    "proprio",
    "action",
    "reference_action",
    "reward_seq",
    "next_z_rl",
    "next_proprio",
    "next_reference_action",
    "done",
)


@dataclasses.dataclass(frozen=True)
class ReencodeArgs:
    replay_root: Path = DEFAULT_REPLAY_ROOT
    rollout_root: Path = DEFAULT_ROLLOUT_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    config_name: str = DEFAULT_CONFIG
    no_actor_threshold: float = 1e-6
    limit: int | None = None
    execute: bool = False
    probe_only: bool = False
    overwrite: bool = False
    convert_bgr_to_rgb: bool = False
    prompt: str = "Twist off the bottle cap."
    dedupe: bool = True
    require_camera: tuple[str, ...] = ("cam_low",)


@dataclasses.dataclass(frozen=True)
class ReencodeSummary:
    planned: int
    converted: int
    skipped: dict[str, int]
    output_root: Path


def load_manifest_from_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        if "manifest" not in data:
            raise ValueError(f"{path} does not contain manifest")
        raw = data["manifest"]
        value = raw.item() if raw.shape == () else raw.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def is_no_actor_shard(path: Path, *, threshold: float = 1e-6) -> bool:
    with np.load(path, allow_pickle=False) as data:
        if "action" not in data or "reference_action" not in data:
            return False
        action = np.asarray(data["action"], dtype=np.float32)
        reference = np.asarray(data["reference_action"], dtype=np.float32)
    if action.shape != reference.shape:
        return False
    return float(np.max(np.abs(action - reference))) <= float(threshold)


def compute_replay_frame_indices(
    manifest: dict[str, Any],
    *,
    clean_rows: int,
    episode_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map clean replay rows back to original episode frame indices.

    The recorder stores source replay transitions using transition starts from the
    original episode records. A clean shard keeps a contiguous slice of those
    replay rows, so crop_start_sample is a source replay row index, not a video
    frame index.
    """
    train_horizon = int(manifest.get("train_horizon") or manifest.get("train_chunk_horizon") or 10)
    chunk_stride = int(manifest.get("chunk_stride") or 2)
    crop_start = int(manifest.get("crop_start_sample", 0))
    if train_horizon <= 0:
        raise ValueError("train_horizon must be positive")
    if chunk_stride <= 0:
        raise ValueError("chunk_stride must be positive")
    last_start = int(episode_frames) - (2 * train_horizon)
    if last_start < 0:
        raise ValueError(f"episode is too short for horizon {train_horizon}: {episode_frames} frames")
    starts = list(range(0, last_start + 1, chunk_stride))
    if starts and starts[-1] != last_start:
        starts.append(last_start)
    expected = manifest.get("crop_original_num_replay_transitions")
    if expected is not None and int(expected) != len(starts):
        logging.warning(
            "source replay transition count mismatch: manifest=%s computed=%s",
            expected,
            len(starts),
        )
    source_rows = np.arange(crop_start, crop_start + int(clean_rows), dtype=np.int64)
    if source_rows[-1] >= len(starts):
        raise ValueError(
            f"clean rows [{source_rows[0]}, {source_rows[-1]}] exceed source replay rows {len(starts)}"
        )
    current = np.asarray([starts[int(row)] for row in source_rows], dtype=np.int64)
    nxt = current + train_horizon
    if int(nxt[-1]) >= int(episode_frames):
        raise ValueError(f"next frame {int(nxt[-1])} exceeds episode frame count {episode_frames}")
    return current, nxt


def rewrite_shard_z_rl(
    input_path: Path,
    output_path: Path,
    *,
    z_rl: np.ndarray,
    next_z_rl: np.ndarray,
    checkpoint_path: Path,
    config_name: str,
) -> None:
    with np.load(input_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in REPLAY_KEYS if key in data}
        manifest = load_manifest_from_npz(input_path)
    if "z_rl" not in arrays or "next_z_rl" not in arrays:
        raise ValueError(f"{input_path} is missing z_rl or next_z_rl")
    if z_rl.ndim != 2 or next_z_rl.ndim != 2:
        raise ValueError("new z_rl and next_z_rl must have shape [N, z_dim]")
    if z_rl.shape != next_z_rl.shape:
        raise ValueError(f"z_rl shape {z_rl.shape} must match next_z_rl shape {next_z_rl.shape}")
    if int(z_rl.shape[0]) != int(arrays["z_rl"].shape[0]):
        raise ValueError(f"new z rows {z_rl.shape[0]} do not match shard rows {arrays['z_rl'].shape[0]}")

    previous_shape = list(arrays["z_rl"].shape)
    arrays["z_rl"] = np.asarray(z_rl, dtype=np.float32)
    arrays["next_z_rl"] = np.asarray(next_z_rl, dtype=np.float32)
    manifest.update(
        {
            "z_rl_source": "rl_token_reencoded",
            "z_rl_dim": int(z_rl.shape[-1]),
            "previous_z_rl_shape": previous_shape,
            "rl_token_checkpoint_path": str(checkpoint_path),
            "rl_token_config_name": config_name,
        }
    )
    arrays["manifest"] = np.asarray(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    tmp_path.replace(output_path)


def discover_no_actor_shards(replay_root: Path, *, threshold: float) -> list[Path]:
    shards = []
    for path in sorted(replay_root.rglob("*.npz")):
        try:
            if is_no_actor_shard(path, threshold=threshold):
                shards.append(path)
        except Exception as exc:  # pragma: no cover - CLI diagnostics.
            logging.warning("skip unreadable shard %s: %s", path, exc)
    return shards


def dedupe_no_actor_shards(shards: list[Path]) -> list[Path]:
    """Keep one clean shard per independent source trajectory.

    Key region editing can produce several crop files for the same source replay,
    and some records can exist both under manual/ and the task/date tree. Training
    should see each independent no-actor key region once.
    """
    selected: dict[str, Path] = {}
    selected_rank: dict[str, tuple[int, int, str]] = {}
    for path in shards:
        try:
            manifest = load_manifest_from_npz(path)
        except Exception as exc:  # pragma: no cover - CLI diagnostics.
            logging.warning("skip shard with unreadable manifest %s: %s", path, exc)
            continue
        source_key = _dedupe_key(path, manifest)
        rank = _dedupe_rank(path, manifest)
        if source_key not in selected or rank > selected_rank[source_key]:
            selected[source_key] = path
            selected_rank[source_key] = rank
    return sorted(selected.values())


def _dedupe_key(path: Path, manifest: dict[str, Any]) -> str:
    source = manifest.get("source_shard_path")
    if source:
        return f"source:{source}"
    key_region_id = manifest.get("key_region_id")
    if key_region_id:
        return f"key_region:{key_region_id}"
    return f"path:{path.resolve()}"


def _dedupe_rank(path: Path, manifest: dict[str, Any]) -> tuple[int, int, str]:
    crop_end = int(manifest.get("crop_end_sample") or manifest.get("num_replay_transitions") or 0)
    manual_penalty = 0 if "manual" in path.parts else 1
    crop_count = path.name.count(".crop_")
    return crop_end, manual_penalty + crop_count, str(path)


def validate_required_cameras(actual: tuple[str, ...], required: tuple[str, ...]) -> None:
    missing = tuple(camera for camera in required if camera not in actual)
    if missing:
        raise ValueError(
            "RL Token policy training cameras are missing required cameras: "
            f"missing={missing}, actual={actual}. Use a checkpoint/config trained with these cameras."
        )


def reencode_clean_no_actor(args: ReencodeArgs) -> ReencodeSummary:
    shards = discover_no_actor_shards(args.replay_root, threshold=args.no_actor_threshold)
    raw_candidates = len(shards)
    if args.dedupe:
        shards = dedupe_no_actor_shards(shards)
        logging.info("deduped no-actor shards from %s files to %s independent sources", raw_candidates, len(shards))
    if args.limit is not None:
        shards = shards[: args.limit]
    skipped: Counter[str] = Counter()
    converted = 0
    if not args.execute:
        return ReencodeSummary(planned=len(shards), converted=0, skipped={}, output_root=args.output_root)

    encoder = RLTokenPolicyEncoder(
        config_name=args.config_name,
        checkpoint_path=args.checkpoint_path,
        prompt=args.prompt,
        convert_bgr_to_rgb=args.convert_bgr_to_rgb,
        require_camera=args.require_camera,
    )
    for shard_path in shards:
        try:
            manifest = load_manifest_from_npz(shard_path)
            rollout_dir = find_rollout_dir(args.rollout_root, manifest)
            current_z, next_z = encoder.encode_shard(rollout_dir, manifest, clean_rows=_shard_rows(shard_path))
            rel = shard_path.relative_to(args.replay_root)
            output_path = args.output_root / rel
            if output_path.exists() and not args.overwrite:
                skipped["output_exists"] += 1
                continue
            rewrite_shard_z_rl(
                shard_path,
                output_path,
                z_rl=current_z,
                next_z_rl=next_z,
                checkpoint_path=args.checkpoint_path,
                config_name=args.config_name,
            )
            converted += 1
        except Exception as exc:  # pragma: no cover - CLI diagnostics.
            skipped[type(exc).__name__] += 1
            logging.exception("failed to reencode %s: %s", shard_path, exc)
    return ReencodeSummary(planned=len(shards), converted=converted, skipped=dict(sorted(skipped.items())), output_root=args.output_root)


def find_rollout_dir(rollout_root: Path, manifest: dict[str, Any]) -> Path:
    key_region_id = manifest.get("key_region_id")
    if not key_region_id:
        raise ValueError("manifest missing key_region_id")
    task = manifest.get("task")
    date = manifest.get("date")
    phase = manifest.get("phase")
    candidates: list[Path] = []
    if task and date and phase:
        candidates.append(rollout_root / str(task) / str(date) / str(phase) / f"key_region_{key_region_id}")
    candidates.extend(rollout_root.rglob(f"key_region_{key_region_id}"))
    for candidate in candidates:
        if (candidate / "episode.hdf5").exists():
            return candidate
    raise FileNotFoundError(f"could not find rollout directory for key_region_id={key_region_id}")


def _shard_rows(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        return int(data["z_rl"].shape[0])


class RLTokenPolicyEncoder:
    def __init__(
        self,
        *,
        config_name: str,
        checkpoint_path: Path,
        prompt: str,
        convert_bgr_to_rgb: bool,
        require_camera: tuple[str, ...],
    ) -> None:
        self._prompt = prompt
        self._convert_bgr_to_rgb = convert_bgr_to_rgb
        from openpi.policies import policy_config
        from openpi.training import config as train_config

        cfg = train_config.get_config(config_name)
        data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
        image_keys = policy_config._training_image_keys(data_cfg) or ()
        validate_required_cameras(tuple(image_keys), require_camera)
        logging.info("loading RL Token policy config=%s checkpoint=%s", config_name, checkpoint_path)
        self._policy = policy_config.create_trained_policy(cfg, checkpoint_path, default_prompt=prompt)

    def probe_one(self, rollout_dir: Path, manifest: dict[str, Any]) -> np.ndarray:
        z, _ = self.encode_shard(rollout_dir, manifest, clean_rows=1)
        return z[0]

    def encode_shard(self, rollout_dir: Path, manifest: dict[str, Any], *, clean_rows: int) -> tuple[np.ndarray, np.ndarray]:
        qpos = _load_qpos(rollout_dir / "episode.hdf5")
        current_frames, next_frames = compute_replay_frame_indices(
            manifest,
            clean_rows=clean_rows,
            episode_frames=len(qpos),
        )
        video_reader = _VideoFrameReader(rollout_dir, convert_bgr_to_rgb=self._convert_bgr_to_rgb)
        current_z = [self._encode_one(video_reader, qpos, int(frame)) for frame in current_frames]
        next_z = [self._encode_one(video_reader, qpos, int(frame)) for frame in next_frames]
        video_reader.close()
        return np.stack(current_z, axis=0).astype(np.float32), np.stack(next_z, axis=0).astype(np.float32)

    def _encode_one(self, video_reader: "_VideoFrameReader", qpos: np.ndarray, frame_index: int) -> np.ndarray:
        images = video_reader.read_all(frame_index)
        obs = {
            "images": images,
            "state": np.asarray(qpos[frame_index], dtype=np.float32),
            "prompt": self._prompt,
        }
        result = self._policy.infer(obs, use_rtc=False)
        if "z_rl" not in result:
            raise RuntimeError("policy inference did not return z_rl")
        return np.asarray(result["z_rl"], dtype=np.float32)


class _VideoFrameReader:
    CAMERA_FILES = {
        "cam_high": "cam_high.mp4",
        "cam_low": "cam_low.mp4",
        "cam_left_wrist": "cam_left_wrist.mp4",
        "cam_right_wrist": "cam_right_wrist.mp4",
    }

    def __init__(self, rollout_dir: Path, *, convert_bgr_to_rgb: bool) -> None:
        import cv2

        self._cv2 = cv2
        self._convert_bgr_to_rgb = convert_bgr_to_rgb
        self._captures = {}
        for camera, filename in self.CAMERA_FILES.items():
            path = rollout_dir / filename
            if not path.exists():
                continue
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError(f"failed to open video {path}")
            self._captures[camera] = capture
        if "cam_high" not in self._captures:
            raise FileNotFoundError(f"{rollout_dir} is missing cam_high.mp4")

    def read_all(self, frame_index: int) -> dict[str, np.ndarray]:
        return {camera: self.read(camera, frame_index) for camera in self._captures}

    def read(self, camera: str, frame_index: int) -> np.ndarray:
        capture = self._captures[camera]
        capture.set(self._cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"failed to read frame {frame_index} from {camera}")
        if self._convert_bgr_to_rgb:
            frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        return np.asarray(frame, dtype=np.uint8)

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()


def _load_qpos(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as file:
        if "observations/qpos" not in file:
            raise KeyError(f"{path} missing observations/qpos")
        return np.asarray(file["observations/qpos"], dtype=np.float32)


def _print_gpu_memory(prefix: str) -> None:
    try:
        import pynvml

        pynvml.nvmlInit()
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            logging.info(
                "%s GPU%d %s memory used=%.2fGiB free=%.2fGiB total=%.2fGiB",
                prefix,
                index,
                name,
                info.used / 2**30,
                info.free / 2**30,
                info.total / 2**30,
            )
    except Exception as exc:  # pragma: no cover - optional diagnostics.
        logging.warning("could not read GPU memory with pynvml: %s", exc)


def run_probe(args: ReencodeArgs) -> None:
    shards = discover_no_actor_shards(args.replay_root, threshold=args.no_actor_threshold)
    if not shards:
        raise RuntimeError(f"no no-actor shards found under {args.replay_root}")
    shard_path = shards[0]
    manifest = load_manifest_from_npz(shard_path)
    rollout_dir = find_rollout_dir(args.rollout_root, manifest)
    logging.info("probe shard=%s", shard_path)
    logging.info("probe rollout=%s", rollout_dir)
    _print_gpu_memory("before-load")
    encoder = RLTokenPolicyEncoder(
        config_name=args.config_name,
        checkpoint_path=args.checkpoint_path,
        prompt=args.prompt,
        convert_bgr_to_rgb=args.convert_bgr_to_rgb,
        require_camera=args.require_camera,
    )
    _print_gpu_memory("after-load")
    z = encoder.probe_one(rollout_dir, manifest)
    _print_gpu_memory("after-one-infer")
    logging.info("probe z_rl shape=%s dtype=%s finite=%s", z.shape, z.dtype, bool(np.isfinite(z).all()))


def _parse_args() -> ReencodeArgs:
    parser = argparse.ArgumentParser(
        description=(
            "Re-encode clean no-actor RLT replay shards with the verified cam4 512-dim RL Token checkpoint. "
            "Default mode only prints the plan. Use --probe-only for a one-sample VRAM test, and --execute "
            "only after confirming the probe is safe."
        )
    )
    parser.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAY_ROOT)
    parser.add_argument("--rollout-root", type=Path, default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--no-actor-threshold", type=float, default=1e-6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--execute", action="store_true", help="Actually write output shards. Do not use before probe.")
    parser.add_argument("--probe-only", action="store_true", help="Load policy and encode one transition only; writes nothing.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Convert every matching crop file. Default keeps one latest clean shard per source/key_region_id.",
    )
    parser.add_argument(
        "--convert-bgr-to-rgb",
        action="store_true",
        help="Convert OpenCV-decoded BGR frames to RGB. Default is false to preserve existing LeRobot BGR-valued training semantics.",
    )
    parser.add_argument("--prompt", default="Twist off the bottle cap.")
    parser.add_argument(
        "--require-camera",
        action="append",
        default=["cam_low"],
        help="Camera key that must be present in the policy training input. Repeatable. Default: cam_low.",
    )
    parser.add_argument(
        "--allow-missing-cam-low",
        action="store_true",
        help="Disable the default cam_low safety check. Only use for experiments that intentionally use a non-cam_low checkpoint.",
    )
    ns = parser.parse_args()
    return ReencodeArgs(
        replay_root=ns.replay_root,
        rollout_root=ns.rollout_root,
        output_root=ns.output_root,
        checkpoint_path=ns.checkpoint_path,
        config_name=ns.config_name,
        no_actor_threshold=ns.no_actor_threshold,
        limit=ns.limit,
        execute=ns.execute,
        probe_only=ns.probe_only,
        overwrite=ns.overwrite,
        convert_bgr_to_rgb=ns.convert_bgr_to_rgb,
        prompt=ns.prompt,
        dedupe=not ns.no_dedupe,
        require_camera=() if ns.allow_missing_cam_low else tuple(ns.require_camera),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    if args.probe_only:
        run_probe(args)
        return
    summary = reencode_clean_no_actor(args)
    logging.info(
        "reencode summary planned=%s converted=%s skipped=%s output_root=%s execute=%s",
        summary.planned,
        summary.converted,
        summary.skipped,
        summary.output_root,
        args.execute,
    )
    if not args.execute:
        logging.info("dry-run only. Run --probe-only first, then ask for approval before using --execute.")


if __name__ == "__main__":
    main()
