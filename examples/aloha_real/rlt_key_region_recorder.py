from __future__ import annotations

from collections import deque
import dataclasses
import json
import logging
import os
import pathlib
import queue
import shutil
import subprocess
import threading
import time
from typing import Any, Callable

import h5py
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override

_REPLAY_KEYS = (
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
class StepRecord:
    step_index: int
    timestamp: float
    qpos: np.ndarray
    qvel: np.ndarray
    effort: np.ndarray
    action: np.ndarray | None
    reference_action: np.ndarray | None
    action_full: np.ndarray | None
    reference_action_full: np.ndarray | None
    z_rl: np.ndarray | None
    proprio: np.ndarray | None
    images: dict[str, np.ndarray]


@dataclasses.dataclass(frozen=True)
class KeyRegionSegment:
    key_region_id: str
    task: str
    phase: str
    start_event: dict[str, Any]
    end_event: dict[str, Any]
    score_event: dict[str, Any]
    records: list[StepRecord]


class _FfmpegMp4Writer:
    def __init__(self, path: pathlib.Path, *, fps: float, width: int, height: int, prefer_gpu: bool = True) -> None:
        encoder = "libx264"
        preset_args = ["-preset", "veryfast", "-crf", "23"]
        if prefer_gpu and _ffmpeg_supports_encoder("h264_nvenc"):
            encoder = "h264_nvenc"
            preset_args = ["-preset", "p4", "-cq", "23"]
        self._path = path
        self._process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps}",
                "-i",
                "-",
                "-an",
                "-c:v",
                encoder,
                *preset_args,
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, image_rgb: np.ndarray) -> None:
        if self._process.stdin is None:
            raise RuntimeError(f"ffmpeg stdin is closed for {self._path}")
        self._process.stdin.write(np.ascontiguousarray(image_rgb).tobytes())

    def close(self) -> None:
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        return_code = self._process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code} for {self._path}")


_FFMPEG_ENCODER_CACHE: dict[str, bool] = {}


class _RedisReplayAckPublisher:
    def __init__(self) -> None:
        self._client = None
        self._channel = os.getenv("RLT_STATE_CHANNEL", "aloha_rlt_state")
        try:
            import redis

            self._client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "0")),
                decode_responses=True,
            )
            self._client.ping()
        except Exception as exc:
            self._client = None
            logging.warning("Disabling RLT replay ack publisher: %s", exc)

    def __call__(self, payload: dict[str, Any]) -> None:
        if self._client is None:
            return
        self._client.publish(self._channel, json.dumps(payload, sort_keys=True))


def _ffmpeg_supports_encoder(encoder: str) -> bool:
    cached = _FFMPEG_ENCODER_CACHE.get(encoder)
    if cached is not None:
        return cached
    if shutil.which("ffmpeg") is None:
        _FFMPEG_ENCODER_CACHE[encoder] = False
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        _FFMPEG_ENCODER_CACHE[encoder] = False
        return False
    supported = encoder in result.stdout
    _FFMPEG_ENCODER_CACHE[encoder] = supported
    return supported


def _safe_name(value: str | None, fallback: str = "unknown") -> str:
    if not value:
        return fallback
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip().lower())
    return cleaned.strip("_") or fallback


def _as_float_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    return np.array(array, copy=True)


def _extract_action_array(action: dict, *keys: str) -> np.ndarray | None:
    for key in keys:
        if key in action:
            return _as_float_array(action[key])
    return None


class KeyRegionReplayRecorder(_subscriber.Subscriber):
    """Record manually marked RLT key regions without blocking the policy loop."""

    def __init__(
        self,
        *,
        rollout_dir: str = "/data/openpi0.5-rtc-reward-learning/rollouts/key_regions",
        replay_dir: str = "/data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions",
        rollouts_root: str | None = None,
        replay_root: str | None = None,
        fps: float = 50.0,
        pre_roll_seconds: float = 2.0,
        post_roll_seconds: float = 0.3,
        max_key_region_seconds: float = 20.0,
        action_horizon: int | None = None,
        chunk_horizon: int = 10,
        train_horizon: int | None = None,
        full_horizon: int = 50,
        chunk_stride: int = 2,
        prefer_gpu_video: bool = True,
        ack_publisher: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        del action_horizon  # Kept for older call sites; RLT training horizon is explicit.
        train_horizon = chunk_horizon if train_horizon is None else train_horizon
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if train_horizon <= 0 or full_horizon <= 0 or chunk_stride <= 0:
            raise ValueError("train_horizon, full_horizon, and chunk_stride must be > 0")
        self._rollouts_root = pathlib.Path(rollouts_root) / "key_regions" if rollouts_root else pathlib.Path(rollout_dir)
        self._replay_root = pathlib.Path(replay_root) / "rlt_key_regions" if replay_root else pathlib.Path(replay_dir)
        self._fps = fps
        self._pre_roll_steps = max(0, round(pre_roll_seconds * fps))
        self._post_roll_steps = max(0, round(post_roll_seconds * fps))
        ring_steps = round((pre_roll_seconds + post_roll_seconds + max_key_region_seconds + 5.0) * fps)
        self._ring: deque[StepRecord] = deque(maxlen=max(ring_steps, self._pre_roll_steps + 1))
        self._train_horizon = train_horizon
        self._full_horizon = full_horizon
        self._chunk_stride = chunk_stride
        self._prefer_gpu_video = prefer_gpu_video
        self._ack_publisher = ack_publisher if ack_publisher is not None else _RedisReplayAckPublisher()
        self._step_index = 0
        self._active_start_event: dict[str, Any] | None = None
        self._active_start_step: int | None = None
        self._pending_end_event: dict[str, Any] | None = None
        self._pending_records: list[StepRecord] | None = None
        self._pending_post_roll_remaining = 0
        self._lock = threading.Lock()
        self._write_queue: queue.Queue[KeyRegionSegment | None] = queue.Queue(maxsize=16)
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    @override
    def on_episode_start(self) -> None:
        pass

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        images = {}
        for name, image in observation.get("images", {}).items():
            if "_depth" in name:
                continue
            image_array = np.asarray(image)
            if image_array.ndim == 3 and image_array.shape[0] == 3:
                image_array = np.transpose(image_array, (1, 2, 0))
            if image_array.dtype != np.uint8:
                image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            images[name] = np.array(image_array, copy=True)

        record = StepRecord(
            step_index=self._step_index,
            timestamp=time.time(),
            qpos=np.array(np.asarray(observation.get("qpos", []), dtype=np.float32), copy=True),
            qvel=np.array(np.asarray(observation.get("qvel", []), dtype=np.float32), copy=True),
            effort=np.array(np.asarray(observation.get("effort", []), dtype=np.float32), copy=True),
            action=_extract_action_array(action, "actions", "action"),
            reference_action=_extract_action_array(action, "reference_actions", "reference_action", "vla_reference_action"),
            action_full=_extract_action_array(action, "action_full", "actions_full", "vla_action_full"),
            reference_action_full=_extract_action_array(
                action,
                "reference_action_full",
                "reference_actions_full",
                "vla_reference_action_full",
            ),
            z_rl=_extract_action_array(action, "z_rl", "rl_token"),
            proprio=_extract_action_array(action, "proprio", "rlt_proprio", "state"),
            images=images,
        )
        with self._lock:
            self._ring.append(record)
            self._step_index += 1
            if (
                self._pending_end_event is not None
                and self._pending_records is not None
                and self._pending_post_roll_remaining > 0
            ):
                self._pending_records.append(record)
                self._pending_post_roll_remaining -= 1

    @override
    def on_episode_end(self, episode_subdir: str | None = None) -> None:
        pass

    @override
    def on_key_region_start(self, event: dict) -> None:
        with self._lock:
            self._active_start_event = dict(event)
            self._active_start_step = self._step_index
            self._pending_end_event = None
            self._pending_records = None
        logging.info("RLT key region started: %s", event.get("key_region_id"))

    @override
    def on_key_region_end(self, event: dict) -> None:
        with self._lock:
            if self._active_start_event is None:
                logging.warning("Ignoring key-region end without a matching start.")
                return
            active_start_step = self._active_start_step if self._active_start_step is not None else self._step_index
            start_step = max(0, active_start_step - self._pre_roll_steps)
            records = [record for record in self._ring if record.step_index >= start_step]
            self._pending_end_event = dict(event)
            self._pending_records = records
            self._pending_post_roll_remaining = self._post_roll_steps
        logging.info("RLT key region ended: %s (%d buffered frames)", event.get("key_region_id"), len(records))

    @override
    def on_key_region_score(self, event: dict) -> None:
        with self._lock:
            if self._active_start_event is None or self._pending_end_event is None:
                logging.warning("Ignoring key-region score without a completed region.")
                return
            records = list(self._pending_records or [])
            current_task = self._active_start_event.get("current_task") or {}
            task = current_task.get("task_name") or self._active_start_event.get("task") or "unknown_task"
            phase = (event.get("state") or {}).get("training_phase") or "unknown_phase"
            segment = KeyRegionSegment(
                key_region_id=str(event.get("key_region_id") or self._active_start_event.get("key_region_id") or time.time_ns()),
                task=str(task),
                phase=str(phase),
                start_event=dict(self._active_start_event),
                end_event=dict(self._pending_end_event),
                score_event=dict(event),
                records=records,
            )
            self._active_start_event = None
            self._active_start_step = None
            self._pending_end_event = None
            self._pending_records = None
            self._pending_post_roll_remaining = 0

        try:
            self._write_queue.put_nowait(segment)
        except queue.Full:
            logging.error("RLT key-region writer queue is full; dropping segment %s", segment.key_region_id)

    def close(self) -> None:
        try:
            self._write_queue.put(None, timeout=5)
        except queue.Full:
            logging.warning("RLT key-region writer queue is full during close; writer may finish asynchronously.")
            return
        self._writer_thread.join(timeout=10)

    def _writer_loop(self) -> None:
        while True:
            segment = self._write_queue.get()
            try:
                if segment is None:
                    return
                self._write_segment(segment)
            except Exception:
                logging.exception("Failed to write RLT key-region segment %s", segment.key_region_id if segment else None)
            finally:
                self._write_queue.task_done()

    def _write_segment(self, segment: KeyRegionSegment) -> None:
        date = time.strftime("%Y-%m-%d", time.localtime(segment.start_event.get("timestamp", time.time())))
        task_name = _safe_name(segment.task)
        region_name = f"key_region_{segment.key_region_id}"
        rollout_dir = self._rollouts_root / task_name / date / segment.phase / region_name
        replay_dir = self._replay_root / task_name / date / "shards"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        replay_dir.mkdir(parents=True, exist_ok=True)

        replay_arrays, missing_metadata = self._build_replay_arrays(segment.records, segment.score_event)
        self._write_videos(rollout_dir, segment.records)
        self._write_hdf5(rollout_dir / "episode.hdf5", segment, missing_metadata=missing_metadata)
        manifest = self._write_manifest(rollout_dir / "manifest.json", segment, missing_metadata, replay_arrays)
        shard_path = None
        if replay_arrays is not None:
            shard_tmp = replay_dir / f"{region_name}.npz.tmp"
            shard_path = replay_dir / f"{region_name}.npz"
            with shard_tmp.open("wb") as stream:
                np.savez_compressed(stream, **replay_arrays, manifest=json.dumps(manifest))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(shard_tmp, shard_path)
            with (replay_dir.parent / "manifest.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps({**manifest, "shard_path": str(shard_path)}, ensure_ascii=False) + "\n")
        self._publish_replay_ack(manifest, shard_path=shard_path)
        logging.info("Saved RLT key region to %s", rollout_dir)

    def _write_videos(self, rollout_dir: pathlib.Path, records: list[StepRecord]) -> None:
        if not records:
            return
        camera_names = sorted({name for record in records for name in record.images})
        for camera_name in camera_names:
            first = next((record.images.get(camera_name) for record in records if camera_name in record.images), None)
            if first is None:
                continue
            height, width = first.shape[:2]
            writer = _FfmpegMp4Writer(
                rollout_dir / f"{camera_name}.mp4",
                fps=self._fps,
                width=width,
                height=height,
                prefer_gpu=self._prefer_gpu_video,
            )
            try:
                for record in records:
                    image = record.images.get(camera_name)
                    if image is not None:
                        writer.write(image)
            finally:
                writer.close()

    def _write_hdf5(self, path: pathlib.Path, segment: KeyRegionSegment, *, missing_metadata: list[str]) -> None:
        records = segment.records
        with h5py.File(path, "w") as root:
            root.attrs["sim"] = False
            root.attrs["compress"] = False
            root.attrs["images_external"] = True
            root.attrs["image_format"] = "mp4"
            root.attrs["is_key_region"] = True
            root.attrs["key_region_id"] = segment.key_region_id
            root.attrs["task"] = segment.task
            root.attrs["phase"] = segment.phase
            root.attrs["reward"] = segment.score_event.get("reward", 0)
            root.attrs["score_timeout"] = bool(segment.score_event.get("score_timeout", False))
            root.attrs["fps"] = self._fps
            root.attrs["missing_rlt_metadata"] = np.asarray(missing_metadata, dtype="S")
            root.attrs["camera_names"] = np.asarray(sorted({name for record in records for name in record.images}), dtype="S")
            obs = root.create_group("observations")
            obs.create_dataset("qpos", data=np.asarray([record.qpos for record in records], dtype=np.float32))
            obs.create_dataset("qvel", data=np.asarray([record.qvel for record in records], dtype=np.float32))
            obs.create_dataset("effort", data=np.asarray([record.effort for record in records], dtype=np.float32))
            actions = [record.action for record in records if record.action is not None]
            if len(actions) == len(records):
                root.create_dataset("action", data=np.asarray(actions, dtype=np.float32))
            root.create_dataset("timestamps", data=np.asarray([record.timestamp for record in records], dtype=np.float64))

    def _write_manifest(
        self,
        path: pathlib.Path,
        segment: KeyRegionSegment,
        missing_metadata: list[str],
        replay_arrays: dict[str, np.ndarray] | None,
    ) -> dict[str, Any]:
        manifest = {
            "key_region_id": segment.key_region_id,
            "task": segment.task,
            "phase": segment.phase,
            "reward": segment.score_event.get("reward"),
            "score_timeout": bool(segment.score_event.get("score_timeout", False)),
            "start_time": segment.start_event.get("timestamp"),
            "end_time": segment.end_event.get("timestamp"),
            "score_time": segment.score_event.get("timestamp"),
            "num_frames": len(segment.records),
            "num_replay_transitions": 0 if replay_arrays is None else len(replay_arrays["z_rl"]),
            "fps": self._fps,
            "missing_rlt_metadata": missing_metadata,
            "replay_status": _replay_status(missing_metadata, replay_arrays),
            "replay_ready": replay_arrays is not None,
            "schema_version": 1,
            "train_chunk_horizon": self._train_horizon,
            "policy_horizon": self._full_horizon,
            "action_space": "aloha_exec",
            "action_dim": 0 if replay_arrays is None else int(replay_arrays["action"].shape[-1]),
            "reward_placement": "terminal_last_train_step",
            "train_horizon": self._train_horizon,
            "full_horizon": self._full_horizon,
            "replay_array_shapes": {}
            if replay_arrays is None
            else {key: list(value.shape) for key, value in replay_arrays.items()},
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            file.write(json.dumps(manifest, indent=2, ensure_ascii=False))
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, path)
        return manifest

    def _publish_replay_ack(self, manifest: dict[str, Any], *, shard_path: pathlib.Path | None) -> None:
        payload = {
            "type": "rlt_replay_segment_written",
            "timestamp": time.time(),
            "key_region_id": manifest.get("key_region_id"),
            "task": manifest.get("task"),
            "phase": manifest.get("phase"),
            "reward": manifest.get("reward"),
            "score_timeout": bool(manifest.get("score_timeout", False)),
            "replay_ready": bool(manifest.get("replay_ready", False)),
            "replay_status": manifest.get("replay_status"),
            "num_replay_transitions": int(manifest.get("num_replay_transitions") or 0),
            "missing_rlt_metadata": list(manifest.get("missing_rlt_metadata") or []),
            "shard_path": None if shard_path is None else str(shard_path),
        }
        try:
            self._ack_publisher(payload)
        except Exception:
            logging.exception("Failed to publish RLT replay ack for key region %s", manifest.get("key_region_id"))

    def _build_replay_arrays(
        self,
        records: list[StepRecord],
        score_event: dict,
    ) -> tuple[dict[str, np.ndarray] | None, list[str]]:
        missing = []
        train_horizon = self._train_horizon
        full_horizon = self._full_horizon
        if len(records) < 2 * train_horizon:
            missing.append("not_enough_frames")
        for key, attr in (
            ("z_rl", "z_rl"),
            ("proprio", "proprio"),
            ("action_full", "action_full"),
            ("reference_action_full", "reference_action_full"),
        ):
            if any(getattr(record, attr) is None for record in records):
                missing.append(key)
        if missing:
            return None, missing

        reward = float(score_event.get("reward", 0.0))
        samples: dict[str, list[np.ndarray]] = {key: [] for key in _REPLAY_KEYS}
        last_start = len(records) - (2 * train_horizon)
        for start in range(0, max(last_start + 1, 0), self._chunk_stride):
            current = records[start]
            next_record = records[start + train_horizon]
            action_chunk = _prefix_full_chunk(current.action_full, full_horizon)
            reference_chunk = _prefix_full_chunk(current.reference_action_full, full_horizon)
            next_reference_chunk = _prefix_full_chunk(next_record.reference_action_full, full_horizon)
            if action_chunk is None or reference_chunk is None or next_reference_chunk is None:
                continue
            reward_seq = np.zeros((full_horizon,), dtype=np.float32)
            done = start == last_start
            if done:
                reward_seq[train_horizon - 1] = reward
            samples["z_rl"].append(np.asarray(current.z_rl, dtype=np.float32))
            samples["proprio"].append(np.asarray(current.proprio, dtype=np.float32))
            samples["action"].append(action_chunk)
            samples["reference_action"].append(reference_chunk)
            samples["reward_seq"].append(reward_seq)
            samples["next_z_rl"].append(np.asarray(next_record.z_rl, dtype=np.float32))
            samples["next_proprio"].append(np.asarray(next_record.proprio, dtype=np.float32))
            samples["next_reference_action"].append(next_reference_chunk)
            samples["done"].append(np.asarray(done, dtype=np.bool_))

        if not samples["z_rl"]:
            return None, ["no_valid_replay_samples"]
        arrays = {key: np.asarray(value) for key, value in samples.items()}
        return arrays, []



def _prefix_full_chunk(value: np.ndarray | None, horizon: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0 or array.shape[0] < horizon:
        return None
    return np.array(array[:horizon], copy=True)

def _replay_status(missing_metadata: list[str], replay_arrays: dict[str, np.ndarray] | None) -> str:
    if replay_arrays is not None:
        return "written"
    if any(item not in {"not_enough_frames", "no_valid_replay_samples"} for item in missing_metadata):
        return "missing_metadata"
    if "not_enough_frames" in missing_metadata or "no_valid_replay_samples" in missing_metadata:
        return "too_short"
    return "missing_metadata"


RLTKeyRegionReplayRecorder = KeyRegionReplayRecorder
