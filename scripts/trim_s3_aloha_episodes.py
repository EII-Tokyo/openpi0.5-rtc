#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py


VIDEO_NAMES = [
    "cam_high.mp4",
    "cam_left_wrist.mp4",
    "cam_low.mp4",
    "cam_right_wrist.mp4",
]


def run(cmd: list[str], *, capture_output: bool = False) -> str:
    proc = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=capture_output,
    )
    return proc.stdout if capture_output else ""


def s3_join(prefix: str, suffix: str) -> str:
    return prefix.rstrip("/") + "/" + suffix.lstrip("/")


def list_episode_names(prefix: str) -> list[str]:
    output = run(["aws", "s3", "ls", prefix], capture_output=True)
    names = []
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("PRE "):
            continue
        name = line.split(None, 1)[1].rstrip("/")
        if re.fullmatch(r"episode_\d+", name):
            names.append(name)
    return sorted(names, key=lambda name: int(name.split("_")[1]))


def get_video_frame_count(path: Path) -> int:
    output = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
    )
    data = json.loads(output)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe did not return any video stream for {path}")
    frame_count = streams[0].get("nb_read_frames")
    if frame_count in (None, "N/A"):
        raise RuntimeError(f"ffprobe did not return frame count for {path}")
    return int(frame_count)


def inspect_hdf5_frame_count(path: Path) -> tuple[int, list[str]]:
    with h5py.File(path, "r") as f:
        frame_count = int(f["action"].shape[0])
        mismatched = []

        def visit(name: str, obj: h5py.Dataset) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            if obj.shape and obj.shape[0] == frame_count:
                return
            if obj.shape and obj.name not in {"/compress_len"} and obj.shape[0] != frame_count:
                mismatched.append(f"{name}:{obj.shape}")

        f.visititems(visit)
    return frame_count, mismatched


def trim_hdf5(src: Path, dst: Path, target_frames: int) -> None:
    with h5py.File(src, "r") as fin:
        source_frames = int(fin["action"].shape[0])
        with h5py.File(dst, "w") as fout:
            for key, value in fin.attrs.items():
                fout.attrs[key] = value

            def copy_node(src_group: h5py.Group, dst_group: h5py.Group) -> None:
                for name, item in src_group.items():
                    if isinstance(item, h5py.Group):
                        new_group = dst_group.create_group(name)
                        for key, value in item.attrs.items():
                            new_group.attrs[key] = value
                        copy_node(item, new_group)
                        continue

                    if not isinstance(item, h5py.Dataset):
                        continue

                    should_trim = item.shape and item.shape[0] == source_frames
                    data = item[:target_frames] if should_trim else item[...]
                    chunks = item.chunks
                    if should_trim and chunks is not None:
                        chunks = (min(chunks[0], target_frames), *chunks[1:])
                    ds = dst_group.create_dataset(
                        name,
                        data=data,
                        dtype=item.dtype,
                        chunks=chunks,
                        compression=item.compression,
                        compression_opts=item.compression_opts,
                        shuffle=item.shuffle,
                        fletcher32=item.fletcher32,
                        fillvalue=item.fillvalue,
                    )
                    for key, value in item.attrs.items():
                        ds.attrs[key] = value

            copy_node(fin, fout)


def trim_video(src: Path, dst: Path, target_frames: int) -> None:
    run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-stats",
            "-y",
            "-i",
            str(src),
            "-vf",
            f"trim=end_frame={target_frames},setpts=PTS-STARTPTS",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(dst),
        ]
    )


def process_episode(
    dataset_prefix: str,
    episode_name: str,
    trim_frames: int,
    *,
    dry_run: bool,
) -> tuple[int, int]:
    episode_prefix = s3_join(dataset_prefix, episode_name) + "/"
    with tempfile.TemporaryDirectory(prefix=f"trim_{episode_name}_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        local_hdf5 = tmpdir / "episode.hdf5"
        run(
            [
                "aws",
                "s3",
                "cp",
                "--only-show-errors",
                s3_join(episode_prefix, "episode.hdf5"),
                str(local_hdf5),
            ]
        )

        hdf5_frames, hdf5_mismatches = inspect_hdf5_frame_count(local_hdf5)
        if hdf5_mismatches:
            raise RuntimeError(
                f"{episode_name}: hdf5 contains datasets with unexpected leading dims: "
                + ", ".join(hdf5_mismatches[:10])
            )

        local_videos: list[Path] = []
        video_counts: dict[str, int] = {}
        for video_name in VIDEO_NAMES:
            local_video = tmpdir / video_name
            run(
                [
                    "aws",
                    "s3",
                    "cp",
                    "--only-show-errors",
                    s3_join(episode_prefix, video_name),
                    str(local_video),
                ]
            )
            local_videos.append(local_video)
            video_counts[video_name] = get_video_frame_count(local_video)

        mismatched_videos = {
            name: count for name, count in video_counts.items() if count != hdf5_frames
        }
        if mismatched_videos:
            mismatch_text = ", ".join(f"{name}={count}" for name, count in mismatched_videos.items())
            raise RuntimeError(
                f"{episode_name}: source frame mismatch, hdf5={hdf5_frames}, {mismatch_text}"
            )

        target_frames = hdf5_frames - trim_frames
        if target_frames <= 0:
            raise RuntimeError(
                f"{episode_name}: cannot trim {trim_frames} frames from {hdf5_frames}-frame episode"
            )

        print(f"{episode_name}: {hdf5_frames} -> {target_frames}")
        if dry_run:
            return hdf5_frames, target_frames

        trimmed_hdf5 = tmpdir / "episode.trimmed.hdf5"
        trim_hdf5(local_hdf5, trimmed_hdf5, target_frames)

        trimmed_videos: list[Path] = []
        for video_path in local_videos:
            trimmed_video = tmpdir / f"{video_path.stem}.trimmed.mp4"
            trim_video(video_path, trimmed_video, target_frames)
            new_count = get_video_frame_count(trimmed_video)
            if new_count != target_frames:
                raise RuntimeError(
                    f"{episode_name}: trimmed video {video_path.name} has {new_count} frames, expected {target_frames}"
                )
            trimmed_videos.append(trimmed_video)

        new_hdf5_frames, _ = inspect_hdf5_frame_count(trimmed_hdf5)
        if new_hdf5_frames != target_frames:
            raise RuntimeError(
                f"{episode_name}: trimmed hdf5 has {new_hdf5_frames} frames, expected {target_frames}"
            )

        run(
            [
                "aws",
                "s3",
                "cp",
                "--only-show-errors",
                str(trimmed_hdf5),
                s3_join(episode_prefix, "episode.hdf5"),
            ]
        )
        for trimmed_video in trimmed_videos:
            remote_name = trimmed_video.name.replace(".trimmed", "")
            run(
                [
                    "aws",
                    "s3",
                    "cp",
                    "--only-show-errors",
                    str(trimmed_video),
                    s3_join(episode_prefix, remote_name),
                ]
            )

        return hdf5_frames, target_frames


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Trim the last N frames from each Aloha episode on S3, keeping hdf5 and mp4 lengths matched."
    )
    parser.add_argument("--s3-prefix", required=True, help="S3 prefix containing episode_* directories")
    parser.add_argument("--trim-frames", type=int, default=100)
    parser.add_argument("--episode", action="append", dest="episodes", help="Process only specific episode directory names, e.g. episode_3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.trim_frames <= 0:
        raise SystemExit("--trim-frames must be > 0")

    episodes = args.episodes or list_episode_names(args.s3_prefix)
    if not episodes:
        raise SystemExit("No episode_* directories found")

    processed = 0
    total_before = 0
    total_after = 0
    for episode_name in episodes:
        before, after = process_episode(
            args.s3_prefix,
            episode_name,
            args.trim_frames,
            dry_run=args.dry_run,
        )
        processed += 1
        total_before += before
        total_after += after

    print(
        f"processed={processed} total_before={total_before} total_after={total_after} "
        f"trimmed_each={args.trim_frames} dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
