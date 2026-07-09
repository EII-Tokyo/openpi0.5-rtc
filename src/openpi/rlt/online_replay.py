from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np


DEFAULT_IMAGE_KEYS = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")


def _array_or_empty(values: list[np.ndarray], *, shape_tail: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    if not values:
        return np.empty((0, *shape_tail), dtype=dtype)
    return np.asarray(values, dtype=dtype)


def _stack_images_or_empty(values: list[np.ndarray]) -> np.ndarray:
    valid_shapes = [value.shape for value in values if value.size > 0]
    if not valid_shapes:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)
    shape = valid_shapes[0]
    normalized = [value if value.shape == shape else np.zeros(shape, dtype=np.uint8) for value in values]
    return np.asarray(normalized, dtype=np.uint8)


class _FfmpegRgbMp4Writer:
    def __init__(self, path: Path, *, fps: float, width: int, height: int) -> None:
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
                str(float(max(1, int(round(fps))))),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame_rgb: np.ndarray) -> None:
        if self._process.stdin is None:
            raise RuntimeError(f"ffmpeg stdin is closed for {self._path}")
        self._process.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        returncode = self._process.wait()
        if returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {returncode} for {self._path}")


def _write_camera_videos(
    output_path: Path,
    image_keys: tuple[str, ...],
    images: dict[str, list[np.ndarray]],
    image_masks: dict[str, list[bool]],
    *,
    fps: float,
) -> dict[str, str]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to save RLT replay videos")

    video_files: dict[str, str] = {}
    for key in image_keys:
        valid_frames = [frame for frame, mask in zip(images[key], image_masks[key], strict=False) if mask and frame.size]
        if not valid_frames:
            continue
        first = valid_frames[0]
        height, width = int(first.shape[0]), int(first.shape[1])
        video_path = output_path.with_suffix(f".{key}.mp4")
        tmp_video_path = video_path.with_name(f".{video_path.name}.tmp.mp4")
        writer = _FfmpegRgbMp4Writer(tmp_video_path, fps=fps, width=width, height=height)
        try:
            last_valid = first
            for frame, mask in zip(images[key], image_masks[key], strict=False):
                if mask and frame.size:
                    if frame.shape[:2] != (height, width):
                        raise RuntimeError(
                            f"camera {key} resolution changed from {(height, width)} to {frame.shape[:2]}"
                        )
                    last_valid = frame
                writer.write(last_valid)
            writer.close()
            tmp_video_path.replace(video_path)
        except Exception:
            try:
                writer.close()
            except Exception:
                pass
            tmp_video_path.unlink(missing_ok=True)
            raise
        video_files[key] = video_path.name
    return video_files


def _raw_robot_state(observation: dict[str, Any]) -> np.ndarray:
    for key in ("qpos", "state", "observation.state"):
        if key in observation:
            return np.asarray(observation[key], dtype=np.float32)
    return np.asarray([], dtype=np.float32)


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item") and not isinstance(value, str):
        try:
            value = value.item()
        except ValueError:
            pass
    return str(value)


def _current_rgb_frame(image: Any) -> np.ndarray | None:
    if image is None:
        return None

    frame = np.asarray(image)
    if frame.ndim == 4:
        # Video-memory observations are [history, H, W, C]. Replay x_t should use the current frame.
        frame = frame[-1]
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.moveaxis(frame, 0, -1)
    if frame.ndim != 3:
        return None
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] != 3:
        return None
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating) and frame.max(initial=0.0) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


class RLTOnlineReplayRecorder:
    """Collects online RLT rollout data without re-running the policy.

    The runtime passes single-step actions to subscribers. Chunk-level RLT metadata is
    attached by ChunkedPolicy under action["rlt_replay"] when the policy server
    returns it.
    """

    def __init__(
        self,
        replay_dir: str | Path,
        *,
        terminal_label: str = "unlabeled",
        save_images: bool = False,
        image_keys: tuple[str, ...] = DEFAULT_IMAGE_KEYS,
        policy_metadata: dict[str, Any] | None = None,
    ) -> None:
        self._replay_dir = Path(replay_dir)
        self._replay_dir.mkdir(parents=True, exist_ok=True)
        self._terminal_label = terminal_label
        self._save_images = save_images
        self._image_keys = image_keys
        self._policy_metadata = policy_metadata or {}
        self._episode_index = self._next_episode_index()
        self.on_episode_start()

    def _next_episode_index(self) -> int:
        existing = sorted(self._replay_dir.glob("episode_*.npz"))
        if not existing:
            return 0
        last = existing[-1].stem.rsplit("_", 1)[-1]
        try:
            return int(last) + 1
        except ValueError:
            return len(existing)

    def on_episode_start(self) -> None:
        self._recording_enabled = False
        self._timestamps: list[float] = []
        self._raw_states: list[np.ndarray] = []
        self._executed_actions: list[np.ndarray] = []
        self._tasks: list[str] = []
        self._subtasks: list[str] = []
        self._chunk_indices: list[int] = []

        self._chunk_start_steps: list[int] = []
        self._chunk_ids: list[int] = []
        self._chunk_tasks: list[str] = []
        self._chunk_subtasks: list[str] = []
        self._tokens: list[np.ndarray] = []
        self._embeddings: list[np.ndarray] = []
        self._masks: list[np.ndarray] = []
        self._noises: list[np.ndarray] = []
        self._norm_states: list[np.ndarray] = []
        self._reference_chunks: list[np.ndarray] = []
        self._policy_chunks: list[np.ndarray] = []
        self._actor_enabled: list[bool] = []
        self._chunk_q_min: list[float] = []
        self._chunk_q1: list[float] = []
        self._chunk_q2: list[float] = []
        self._vla_chunk_q_min: list[float] = []
        self._vla_chunk_q1: list[float] = []
        self._vla_chunk_q2: list[float] = []
        self._actor_chunk_q_min: list[float] = []
        self._actor_chunk_q1: list[float] = []
        self._actor_chunk_q2: list[float] = []
        self._state_is_normalized: list[bool] = []
        self._state_normalization: str | None = None
        self._seen_chunk_ids: set[int] = set()
        self._images: dict[str, list[np.ndarray]] = {key: [] for key in self._image_keys}
        self._image_masks: dict[str, list[bool]] = {key: [] for key in self._image_keys}

    def begin_recording(self) -> None:
        logging.info("RLT online replay recording started.")
        self.on_episode_start()
        self._recording_enabled = True

    def set_terminal_label(self, label: str) -> None:
        if label not in {"unlabeled", "success", "failure"}:
            raise ValueError(f"Unsupported RLT terminal label: {label}")
        self._terminal_label = label

    def on_step(self, observation: dict, action: dict) -> None:
        if not self._recording_enabled:
            return

        replay = action.get("rlt_replay") if isinstance(action, dict) else None
        executed_action = np.asarray(action.get("actions", []), dtype=np.float32)
        self._timestamps.append(time.time())
        self._raw_states.append(_raw_robot_state(observation))
        self._executed_actions.append(executed_action)
        task = _string_value(observation.get("task"))
        subtask = _string_value(observation.get("subtask"))
        self._tasks.append(task)
        self._subtasks.append(subtask)
        if self._save_images:
            self._record_images(observation)

        chunk_index = int(replay.get("chunk_index", -1)) if replay else -1
        self._chunk_indices.append(chunk_index)

        if replay is None or chunk_index in self._seen_chunk_ids:
            return

        rlt_token = replay.get("rlt_token")
        embeddings = replay.get("rlt_embeddings")
        mask = replay.get("rlt_mask")
        noise = replay.get("rlt_noise")
        self._seen_chunk_ids.add(chunk_index)
        self._chunk_start_steps.append(len(self._timestamps) - 1)
        self._chunk_ids.append(chunk_index)
        self._chunk_tasks.append(task)
        self._chunk_subtasks.append(subtask)
        if rlt_token is not None:
            self._tokens.append(np.asarray(rlt_token, dtype=np.float32))
        if embeddings is not None:
            self._embeddings.append(np.asarray(embeddings, dtype=np.float32))
        if mask is not None:
            self._masks.append(np.asarray(mask, dtype=np.bool_))
        if noise is not None:
            self._noises.append(np.asarray(noise, dtype=np.float32))
        self._norm_states.append(np.asarray(replay["rlt_state"], dtype=np.float32))
        self._reference_chunks.append(np.asarray(replay["rlt_reference_action_chunk"], dtype=np.float32))
        self._policy_chunks.append(np.asarray(replay["rlt_policy_action_chunk"], dtype=np.float32))
        self._actor_enabled.append(bool(replay.get("rlt_actor_enabled", False)))
        self._chunk_q_min.append(float(replay.get("rlt_chunk_q_min", np.nan)))
        self._chunk_q1.append(float(replay.get("rlt_chunk_q1", np.nan)))
        self._chunk_q2.append(float(replay.get("rlt_chunk_q2", np.nan)))
        self._vla_chunk_q_min.append(float(replay.get("rlt_vla_chunk_q_min", np.nan)))
        self._vla_chunk_q1.append(float(replay.get("rlt_vla_chunk_q1", np.nan)))
        self._vla_chunk_q2.append(float(replay.get("rlt_vla_chunk_q2", np.nan)))
        self._actor_chunk_q_min.append(float(replay.get("rlt_actor_chunk_q_min", np.nan)))
        self._actor_chunk_q1.append(float(replay.get("rlt_actor_chunk_q1", np.nan)))
        self._actor_chunk_q2.append(float(replay.get("rlt_actor_chunk_q2", np.nan)))
        self._state_is_normalized.append(bool(replay.get("rlt_state_is_normalized", True)))
        self._state_normalization = str(replay.get("rlt_state_normalization", "policy_input_transform"))

    def _record_images(self, observation: dict[str, Any]) -> None:
        images = observation.get("images", {}) if isinstance(observation, dict) else {}
        if not isinstance(images, dict):
            images = {}
        for key in self._image_keys:
            frame = _current_rgb_frame(images.get(key))
            if frame is None:
                if self._images[key]:
                    zero_frame = np.zeros_like(self._images[key][-1])
                else:
                    zero_frame = np.zeros((0, 0, 3), dtype=np.uint8)
                self._images[key].append(zero_frame)
                self._image_masks[key].append(False)
                continue
            self._images[key].append(frame)
            self._image_masks[key].append(True)

    def on_episode_end(self, episode_subdir: str | None = None) -> None:
        if not self._timestamps:
            logging.warning("No RLT replay data collected; skipping save.")
            return

        episode_name = f"episode_{self._episode_index:06d}"
        if episode_subdir:
            episode_name = f"{episode_name}_{episode_subdir}"
        output_path = self._replay_dir / f"{episode_name}.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        raw_state_dim = int(max((state.shape[-1] for state in self._raw_states if state.ndim > 0), default=0))
        action_dim = int(max((action.shape[-1] for action in self._executed_actions if action.ndim > 0), default=0))
        token_dim = int(self._tokens[0].shape[-1]) if self._tokens else 0
        embedding_shape = self._embeddings[0].shape if self._embeddings else (0, 0)
        mask_shape = self._masks[0].shape if self._masks else (0,)
        noise_shape = self._noises[0].shape if self._noises else (0, 0)
        norm_state_dim = int(self._norm_states[0].shape[-1]) if self._norm_states else 0
        reference_shape = self._reference_chunks[0].shape if self._reference_chunks else (0, 0)

        metadata = {
            "format": "rlt_online_episode_v1",
            "terminal_label": self._terminal_label,
            "terminal_success": -1 if self._terminal_label == "unlabeled" else int(self._terminal_label == "success"),
            "state_is_normalized": bool(all(self._state_is_normalized)) if self._state_is_normalized else False,
            "state_normalization": self._state_normalization,
            "num_steps": len(self._timestamps),
            "num_chunks": len(self._chunk_ids),
            "created_at": time.time(),
            "policy_metadata": self._policy_metadata,
        }
        fps = 50.0
        try:
            runtime_metadata = self._policy_metadata.get("runtime", {})
            if isinstance(runtime_metadata, dict):
                fps = float(runtime_metadata.get("policy_hz", fps) or fps)
        except Exception:
            fps = 50.0

        payload = {
            "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False)),
            "timestamps": np.asarray(self._timestamps, dtype=np.float64),
            "task": np.asarray(self._tasks),
            "subtask": np.asarray(self._subtasks),
            "raw_state": _array_or_empty(self._raw_states, shape_tail=(raw_state_dim,)),
            "executed_action": _array_or_empty(self._executed_actions, shape_tail=(action_dim,)),
            "step_chunk_index": np.asarray(self._chunk_indices, dtype=np.int32),
            "chunk_id": np.asarray(self._chunk_ids, dtype=np.int32),
            "chunk_start_step": np.asarray(self._chunk_start_steps, dtype=np.int32),
            "chunk_task": np.asarray(self._chunk_tasks),
            "chunk_subtask": np.asarray(self._chunk_subtasks),
            "rlt_token": _array_or_empty(self._tokens, shape_tail=(token_dim,)),
            "rlt_embeddings": _array_or_empty(self._embeddings, shape_tail=embedding_shape),
            "rlt_mask": _array_or_empty(self._masks, shape_tail=mask_shape, dtype=np.bool_),
            "rlt_noise": _array_or_empty(self._noises, shape_tail=noise_shape),
            "norm_state": _array_or_empty(self._norm_states, shape_tail=(norm_state_dim,)),
            "reference_action_chunk": _array_or_empty(self._reference_chunks, shape_tail=reference_shape),
            "policy_action_chunk": _array_or_empty(self._policy_chunks, shape_tail=reference_shape),
            "actor_enabled": np.asarray(self._actor_enabled, dtype=np.bool_),
            "chunk_q_min": np.asarray(self._chunk_q_min, dtype=np.float32),
            "chunk_q1": np.asarray(self._chunk_q1, dtype=np.float32),
            "chunk_q2": np.asarray(self._chunk_q2, dtype=np.float32),
            "vla_chunk_q_min": np.asarray(self._vla_chunk_q_min, dtype=np.float32),
            "vla_chunk_q1": np.asarray(self._vla_chunk_q1, dtype=np.float32),
            "vla_chunk_q2": np.asarray(self._vla_chunk_q2, dtype=np.float32),
            "actor_chunk_q_min": np.asarray(self._actor_chunk_q_min, dtype=np.float32),
            "actor_chunk_q1": np.asarray(self._actor_chunk_q1, dtype=np.float32),
            "actor_chunk_q2": np.asarray(self._actor_chunk_q2, dtype=np.float32),
        }
        if self._save_images:
            video_files = _write_camera_videos(
                output_path,
                self._image_keys,
                self._images,
                self._image_masks,
                fps=fps,
            )
            metadata["image_storage"] = "mp4_sidecar"
            metadata["image_keys"] = list(video_files)
            metadata["video_files"] = video_files
            metadata["video_fps"] = fps
            payload["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))

        tmp_path = output_path.with_name(f".{output_path.name}.tmp")
        try:
            logging.info(
                "Saving RLT online replay: %s steps=%d chunks=%d save_images=%s",
                output_path,
                len(self._timestamps),
                len(self._chunk_ids),
                self._save_images,
            )
            with tmp_path.open("wb") as f:
                np.savez(f, **payload)
            tmp_path.replace(output_path)
            logging.info(
                "Saved RLT online replay: %s steps=%d chunks=%d",
                output_path,
                len(self._timestamps),
                len(self._chunk_ids),
            )
            self._episode_index += 1
            self.on_episode_start()
        except Exception:
            logging.exception("Failed to save RLT online replay: %s", output_path)
            tmp_path.unlink(missing_ok=True)
