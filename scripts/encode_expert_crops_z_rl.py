from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from scripts.reencode_clean_no_actor_z_rl import RLTokenPolicyEncoder


DEFAULT_CHECKPOINT = Path(
    "checkpoints/eii_rinse_11repo_cam4_fullft_rl_token_small_query/"
    "rinse_11repo_rl_token_small_query_512_from_9000_20260615/9999"
)
DEFAULT_CONFIG = "eii_rinse_11repo_cam4_fullft_rl_token_small_query"
CAMERA_PATHS = {
    "cam_high": "observation.images.cam_high",
    "cam_low": "observation.images.cam_low",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
}


@dataclasses.dataclass(frozen=True)
class EncodeExpertArgs:
    dataset_root: Path
    crop_root: Path
    output_root: Path
    checkpoint_path: Path
    config_name: str
    prompt: str = "Twist off the bottle cap."
    convert_bgr_to_rgb: bool = False
    require_camera: tuple[str, ...] = ("cam_low",)
    overwrite: bool = False
    limit_episodes: int | None = None


@dataclasses.dataclass(frozen=True)
class EncodeExpertSummary:
    requested_episodes: int
    encoded_episodes: int
    skipped: dict[str, int]
    output_root: Path


@dataclasses.dataclass(frozen=True)
class _FrameRequest:
    frame_index: int
    global_index: int
    state: np.ndarray


def encode_expert_crop_z(args: EncodeExpertArgs) -> EncodeExpertSummary:
    requests = _collect_frame_requests(args.dataset_root, args.crop_root)
    if args.limit_episodes is not None:
        requests = dict(list(sorted(requests.items()))[: args.limit_episodes])
    skipped: defaultdict[str, int] = defaultdict(int)
    if not requests:
        return EncodeExpertSummary(0, 0, {}, args.output_root)

    encoder = RLTokenPolicyEncoder(
        config_name=args.config_name,
        checkpoint_path=args.checkpoint_path,
        prompt=args.prompt,
        convert_bgr_to_rgb=args.convert_bgr_to_rgb,
        require_camera=args.require_camera,
    )
    encoded = 0
    readers: dict[str, _ExpertVideoFrameReader] = {}
    try:
        for (dataset_id, episode_index), frame_requests in sorted(requests.items()):
            output_path = args.output_root / dataset_id / f"episode_{episode_index:06d}_z_rl.npz"
            if output_path.exists() and not args.overwrite:
                skipped["output_exists"] += 1
                continue
            dataset_dir = args.dataset_root / dataset_id
            reader = readers.get(dataset_id)
            if reader is None:
                reader = _ExpertVideoFrameReader(dataset_dir, convert_bgr_to_rgb=args.convert_bgr_to_rgb)
                readers[dataset_id] = reader
            z_rows = []
            frame_indices = []
            for request in frame_requests:
                obs = {
                    "images": reader.read_all(request.global_index),
                    "state": np.asarray(request.state, dtype=np.float32),
                    "prompt": args.prompt,
                }
                result = encoder._policy.infer(obs, use_rtc=False)  # noqa: SLF001 - reuse policy wrapper for batch script.
                if "z_rl" not in result:
                    raise RuntimeError("policy inference did not return z_rl")
                z_rows.append(np.asarray(result["z_rl"], dtype=np.float32))
                frame_indices.append(int(request.frame_index))
            _write_npz(
                output_path,
                {
                    "frame_index": np.asarray(frame_indices, dtype=np.int64),
                    "z_rl": np.stack(z_rows, axis=0).astype(np.float32),
                    "metadata": np.asarray(
                        json.dumps(
                            {
                                "dataset_id": dataset_id,
                                "episode_index": int(episode_index),
                                "source": "expert_crop_rl_token_encoder",
                                "rl_token_checkpoint_path": str(args.checkpoint_path),
                                "rl_token_config_name": args.config_name,
                                "prompt": args.prompt,
                                "convert_bgr_to_rgb": args.convert_bgr_to_rgb,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    ),
                },
            )
            encoded += 1
            logging.info(
                "encoded expert z cache dataset=%s episode=%s frames=%s -> %s",
                dataset_id,
                episode_index,
                len(frame_indices),
                output_path,
            )
    finally:
        for reader in readers.values():
            reader.close()
    return EncodeExpertSummary(
        requested_episodes=len(requests),
        encoded_episodes=encoded,
        skipped=dict(sorted(skipped.items())),
        output_root=args.output_root,
    )


def _collect_frame_requests(dataset_root: Path, crop_root: Path) -> dict[tuple[str, int], list[_FrameRequest]]:
    by_episode: dict[tuple[str, int], dict[int, _FrameRequest]] = defaultdict(dict)
    episode_cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for crop_path in sorted(crop_root.glob("*/*.json")):
        crop = json.loads(crop_path.read_text(encoding="utf-8"))
        dataset_id = str(crop.get("dataset_id") or crop_path.parent.name)
        episode_index = int(crop["episode_index"])
        key = (dataset_id, episode_index)
        episode = episode_cache.get(key)
        if episode is None:
            episode = _load_episode_rows(dataset_root / dataset_id, episode_index)
            episode_cache[key] = episode
        rel_time = episode["timestamp"] - float(episode["timestamp"][0])
        mask = (rel_time >= float(crop["start_sec"])) & (rel_time < float(crop["end_sec"]))
        for row in np.flatnonzero(mask):
            frame_index = int(episode["frame_index"][row])
            by_episode[key][frame_index] = _FrameRequest(
                frame_index=frame_index,
                global_index=int(episode["index"][row]),
                state=np.asarray(episode["state"][row], dtype=np.float32),
            )
    return {key: [items[idx] for idx in sorted(items)] for key, items in by_episode.items()}


def _load_episode_rows(dataset_dir: Path, episode_index: int) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {key: [] for key in ("frame_index", "index", "timestamp", "state")}
    for parquet_path in sorted((dataset_dir / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(
            parquet_path,
            columns=["episode_index", "frame_index", "index", "timestamp", "observation.state"],
        )
        episodes = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        mask = episodes == int(episode_index)
        if not np.any(mask):
            continue
        parts["frame_index"].append(np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)[mask])
        parts["index"].append(np.asarray(table["index"].to_pylist(), dtype=np.int64)[mask])
        parts["timestamp"].append(np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)[mask])
        parts["state"].append(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[mask])
    if not parts["frame_index"]:
        raise FileNotFoundError(f"{dataset_dir} missing episode {episode_index}")
    frame_index = np.concatenate(parts["frame_index"], axis=0)
    order = np.argsort(frame_index)
    return {
        "frame_index": frame_index[order],
        "index": np.concatenate(parts["index"], axis=0)[order],
        "timestamp": np.concatenate(parts["timestamp"], axis=0)[order],
        "state": np.concatenate(parts["state"], axis=0)[order],
    }


class _ExpertVideoFrameReader:
    def __init__(self, dataset_dir: Path, *, convert_bgr_to_rgb: bool) -> None:
        import av

        self._av = av
        self._convert_bgr_to_rgb = convert_bgr_to_rgb
        self._videos: dict[str, list[tuple[int, int, Path]]] = {}
        self._streams: dict[Path, Any] = {}
        self._stream_indices: dict[Path, int] = {}
        for camera, camera_dir in CAMERA_PATHS.items():
            videos = []
            cursor = 0
            for path in sorted((dataset_dir / "videos" / camera_dir).glob("chunk-*/file-*.mp4")):
                with av.open(str(path)) as container:
                    stream = container.streams.video[0]
                    frame_count = int(stream.frames)
                if frame_count <= 0:
                    continue
                videos.append((cursor, cursor + frame_count, path))
                cursor += frame_count
            if videos:
                self._videos[camera] = videos
        if "cam_high" not in self._videos:
            raise FileNotFoundError(f"{dataset_dir} missing cam_high videos")

    def read_all(self, global_index: int) -> dict[str, np.ndarray]:
        return {camera: self.read(camera, global_index) for camera in self._videos}

    def read(self, camera: str, global_index: int) -> np.ndarray:
        for start, end, path in self._videos[camera]:
            if start <= global_index < end:
                local_index = int(global_index - start)
                container, frames = self._stream(path)
                cursor = self._stream_indices[path]
                if local_index < cursor:
                    container.close()
                    self._streams.pop(path, None)
                    self._stream_indices.pop(path, None)
                    container, frames = self._stream(path)
                    cursor = 0
                frame = None
                while cursor <= local_index:
                    try:
                        frame = next(frames)
                    except StopIteration as exc:
                        raise RuntimeError(f"failed to read global frame {global_index} from {path}") from exc
                    cursor += 1
                self._stream_indices[path] = cursor
                if frame is None:
                    raise RuntimeError(f"failed to read global frame {global_index} from {path}")
                fmt = "rgb24" if self._convert_bgr_to_rgb else "bgr24"
                return np.asarray(frame.to_ndarray(format=fmt), dtype=np.uint8)
        raise IndexError(f"global frame {global_index} not found for camera {camera}")

    def _stream(self, path: Path):
        stream = self._streams.get(path)
        if stream is None:
            container = self._av.open(str(path))
            frames = container.decode(video=0)
            stream = (container, frames)
            self._streams[path] = stream
            self._stream_indices[path] = 0
        return stream

    def close(self) -> None:
        for container, _frames in self._streams.values():
            container.close()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    tmp_path.replace(path)


def _parse_args() -> EncodeExpertArgs:
    parser = argparse.ArgumentParser(description="Encode saved Expert-for-D crops into frame-level z_rl caches.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/eii/.cache/huggingface/lerobot/lyl472324464"))
    parser.add_argument(
        "--crop-root",
        type=Path,
        default=Path("/home/eii/data/openpi0.5-rtc-reward-learning/replay/discriminator_expert_crops"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/eii/data/openpi0.5-rtc-reward-learning/replay/expert_crop_z_rl_cache_20260629"),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--prompt", default="Twist off the bottle cap.")
    parser.add_argument("--convert-bgr-to-rgb", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--require-camera", action="append", default=["cam_low"])
    ns = parser.parse_args()
    return EncodeExpertArgs(
        dataset_root=ns.dataset_root,
        crop_root=ns.crop_root,
        output_root=ns.output_root,
        checkpoint_path=ns.checkpoint_path,
        config_name=ns.config_name,
        prompt=ns.prompt,
        convert_bgr_to_rgb=ns.convert_bgr_to_rgb,
        overwrite=ns.overwrite,
        limit_episodes=ns.limit_episodes,
        require_camera=tuple(ns.require_camera),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = encode_expert_crop_z(_parse_args())
    print(json.dumps(dataclasses.asdict(summary), ensure_ascii=False, default=str, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
