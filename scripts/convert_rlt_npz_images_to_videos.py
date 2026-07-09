from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

import numpy as np


IMAGE_PREFIX = "image_"
IMAGE_MASK_PREFIX = "image_mask_"


def _read_npy_header(member) -> tuple[tuple[int, ...], np.dtype]:
    version = np.lib.format.read_magic(member)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(member)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(member)
    else:
        raise ValueError(f"unsupported npy version: {version}")
    if fortran_order:
        raise ValueError("Fortran-order image arrays are not supported")
    return tuple(int(dim) for dim in shape), np.dtype(dtype)


def _camera_members(npz_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(npz_path) as zf:
        names = set(zf.namelist())
    cameras = {}
    for name in names:
        if not name.startswith(IMAGE_PREFIX) or not name.endswith(".npy"):
            continue
        if name.startswith(IMAGE_MASK_PREFIX):
            continue
        camera = name.removeprefix(IMAGE_PREFIX).removesuffix(".npy")
        cameras[camera] = name
    return dict(sorted(cameras.items()))


def _ffmpeg_command(output_path: Path, *, width: int, height: int, fps: float) -> list[str]:
    return [
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
        str(output_path),
    ]


def _write_video_from_member(npz_path: Path, member_name: str, output_path: Path, *, fps: float) -> None:
    tmp_path = output_path.with_name(f".{output_path.name}.tmp.mp4")
    with zipfile.ZipFile(npz_path) as zf:
        with zf.open(member_name) as member:
            shape, dtype = _read_npy_header(member)
            if len(shape) != 4 or shape[0] <= 0 or shape[-1] not in (1, 3, 4):
                raise ValueError(f"unsupported image array shape for {member_name}: {shape}")
            if dtype != np.dtype("uint8"):
                raise ValueError(f"unsupported image dtype for {member_name}: {dtype}")
            frame_count, height, width, channels = shape
            frame_bytes = int(np.prod(shape[1:]) * dtype.itemsize)
            proc = subprocess.Popen(_ffmpeg_command(tmp_path, width=width, height=height, fps=fps), stdin=subprocess.PIPE)
            try:
                assert proc.stdin is not None
                for _ in range(frame_count):
                    raw = member.read(frame_bytes)
                    if len(raw) != frame_bytes:
                        raise ValueError(f"truncated frame data in {member_name}")
                    frame = np.frombuffer(raw, dtype=dtype).reshape(shape[1:])
                    if channels == 4:
                        frame = frame[..., :3]
                    elif channels == 1:
                        frame = np.repeat(frame, 3, axis=-1)
                    proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                proc.stdin.close()
                returncode = proc.wait()
                if returncode != 0:
                    raise RuntimeError(f"ffmpeg exited with code {returncode} for {output_path}")
                tmp_path.replace(output_path)
            except Exception:
                if proc.stdin is not None:
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass
                proc.wait()
                tmp_path.unlink(missing_ok=True)
                raise


def _metadata_from_npz(data: np.lib.npyio.NpzFile) -> dict:
    if "metadata_json" not in data:
        return {}
    return json.loads(str(data["metadata_json"]))


def _fps_from_metadata(metadata: dict) -> float:
    for key in ("video_fps", "fps"):
        if key in metadata:
            return float(metadata[key] or 50.0)
    policy_metadata = metadata.get("policy_metadata", {})
    if isinstance(policy_metadata, dict):
        runtime_metadata = policy_metadata.get("runtime", {})
        if isinstance(runtime_metadata, dict) and runtime_metadata.get("policy_hz"):
            return float(runtime_metadata["policy_hz"])
    return 50.0


def _rewrite_npz_without_images(npz_path: Path, metadata: dict) -> None:
    tmp_path = npz_path.with_name(f".{npz_path.name}.tmp")
    with np.load(npz_path, allow_pickle=False) as data:
        payload = {}
        for key in data.files:
            if key.startswith(IMAGE_PREFIX) or key.startswith(IMAGE_MASK_PREFIX):
                continue
            if key == "metadata_json":
                continue
            payload[key] = np.asarray(data[key])
        payload["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
        with tmp_path.open("wb") as f:
            np.savez(f, **payload)
    tmp_path.replace(npz_path)


def convert_episode(npz_path: Path, *, force: bool = False) -> bool:
    with np.load(npz_path, allow_pickle=False) as data:
        metadata = _metadata_from_npz(data)
    if metadata.get("image_storage") == "mp4_sidecar" and metadata.get("video_files") and not force:
        return False

    cameras = _camera_members(npz_path)
    if not cameras:
        raise ValueError(f"{npz_path} has no embedded image arrays to convert")
    fps = _fps_from_metadata(metadata)

    video_files = {}
    for camera, member_name in cameras.items():
        video_path = npz_path.with_suffix(f".{camera}.mp4")
        _write_video_from_member(npz_path, member_name, video_path, fps=fps)
        video_files[camera] = video_path.name

    metadata["image_storage"] = "mp4_sidecar"
    metadata["image_keys"] = sorted(video_files)
    metadata["video_files"] = video_files
    metadata["video_fps"] = fps
    _rewrite_npz_without_images(npz_path, metadata)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RLT online replay npz image arrays to sidecar mp4 videos.")
    parser.add_argument("replay_dir", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=0)
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")

    episodes = sorted(path for path in args.replay_dir.glob("episode_*.npz") if not path.name.startswith("."))
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    converted = 0
    skipped = 0
    start = time.monotonic()
    for index, episode in enumerate(episodes, start=1):
        episode_start = time.monotonic()
        try:
            did_convert = convert_episode(episode, force=args.force)
        except Exception as exc:
            print(f"ERROR {episode}: {exc}", flush=True)
            raise
        if did_convert:
            converted += 1
            print(
                f"converted {index}/{len(episodes)} {episode.name} elapsed={time.monotonic() - episode_start:.1f}s",
                flush=True,
            )
        else:
            skipped += 1
            print(f"skipped {index}/{len(episodes)} {episode.name}", flush=True)
    print(
        f"done converted={converted} skipped={skipped} total={len(episodes)} elapsed={time.monotonic() - start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
