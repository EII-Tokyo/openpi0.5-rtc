from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil

import pandas as pd
from torchcodec.decoders import VideoDecoder

import lerobot.datasets.aggregate as lerobot_aggregate
import lerobot.datasets.dataset_tools as lerobot_dataset_tools
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import split_dataset
from lerobot.datasets.utils import update_chunk_file_indices
from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s


DEFAULT_CSV = Path("/tmp/eii_excel_extract/sheet2_first_two_columns_raw.csv")
DEFAULT_DEST_REPO_ID = "michios/droid_xxjd_combined"
DEFAULT_LEROBOT_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot"
DEFAULT_STAGING_ROOT = Path("/tmp/droid_xxjd_combined_rebuild")

REPO_ALIASES = {
    # The spreadsheet uses this older name, but the local source dataset is cached under the v2 name.
    "michios/droid_xxjd_8_canonical": "michios/droid_xxjd_8_2_canonical",
}


def _fixed_keep_episodes_from_video_with_av(
    input_path: Path,
    output_path: Path,
    episodes_to_keep: list[tuple[float, float]],
    fps: float,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
) -> None:
    """Keep episode ranges using integer frame indices."""
    from fractions import Fraction

    import av

    if not episodes_to_keep:
        raise ValueError("No episodes to keep")

    frame_ranges = sorted((round(start * fps), round(end * fps)) for start, end in episodes_to_keep)

    in_container = av.open(str(input_path))
    if not in_container.streams.video:
        raise ValueError(f"No video streams found in {input_path}.")

    v_in = in_container.streams.video[0]
    out = av.open(str(output_path), mode="w")
    fps_fraction = Fraction(fps).limit_denominator(1000)
    v_out = out.add_stream(vcodec, rate=fps_fraction)
    v_out.width = v_in.codec_context.width
    v_out.height = v_in.codec_context.height
    v_out.pix_fmt = pix_fmt
    v_out.time_base = Fraction(1, int(fps))
    out.start_encoding()

    output_frame_count = 0
    range_idx = 0
    for packet in in_container.demux(v_in):
        for frame in packet.decode():
            if frame is None:
                continue
            frame_time = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0
            frame_idx = round(frame_time * fps)

            while range_idx < len(frame_ranges) and frame_idx >= frame_ranges[range_idx][1]:
                range_idx += 1
            if range_idx >= len(frame_ranges):
                break

            start_frame, end_frame = frame_ranges[range_idx]
            if frame_idx < start_frame or frame_idx >= end_frame:
                continue

            new_frame = frame.reformat(width=v_out.width, height=v_out.height, format=v_out.pix_fmt)
            new_frame.pts = output_frame_count
            new_frame.time_base = Fraction(1, int(fps))
            for pkt in v_out.encode(new_frame):
                out.mux(pkt)
            output_frame_count += 1

    for pkt in v_out.encode():
        out.mux(pkt)

    out.close()
    in_container.close()


def parse_episode_csv(csv_path: Path, lerobot_root: Path) -> dict[str, list[int]]:
    df = pd.read_csv(csv_path)
    df["dataset"] = df["A"].ffill()
    df["episode"] = pd.to_numeric(df["B"], errors="coerce")
    df = df.dropna(subset=["dataset", "episode"])
    df["episode"] = df["episode"].astype(int)

    episodes_by_repo: dict[str, list[int]] = {}
    for raw_repo_id, group in df.groupby("dataset", sort=False):
        repo_id = str(raw_repo_id)
        if not (lerobot_root / repo_id).exists() and repo_id in REPO_ALIASES:
            repo_id = REPO_ALIASES[repo_id]
        episodes_by_repo[repo_id] = sorted(group["episode"].unique().tolist())

    return episodes_by_repo


def _fixed_aggregate_videos(src_meta, dst_meta, videos_idx, video_files_size_in_mb, chunk_size):
    """LeRobot v3.0 aggregation with per-source-video destination file tracking.

    The installed aggregate_videos records per-source timestamp offsets, but metadata update later writes
    only the final destination file index for a camera. That is wrong when a camera spans multiple mp4s.
    """
    for key in videos_idx:
        videos_idx[key]["episode_duration"] = 0.0
        videos_idx[key]["src_to_dst"] = {}

    for key, video_idx in videos_idx.items():
        unique_chunk_file_pairs = sorted(
            {
                (chunk, file)
                for chunk, file in zip(
                    src_meta.episodes[f"videos/{key}/chunk_index"],
                    src_meta.episodes[f"videos/{key}/file_index"],
                    strict=False,
                )
            }
        )

        chunk_idx = video_idx["chunk"]
        file_idx = video_idx["file"]
        current_file_duration = video_idx.get("current_file_duration", 0.0)

        for src_chunk_idx, src_file_idx in unique_chunk_file_pairs:
            src_path = src_meta.root / lerobot_aggregate.DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=src_chunk_idx,
                file_index=src_file_idx,
            )
            dst_path = dst_meta.root / lerobot_aggregate.DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )
            src_duration = get_video_duration_in_s(src_path)

            if not dst_path.exists():
                offset = 0.0
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src_path), str(dst_path))
                current_file_duration = src_duration
            else:
                src_size = lerobot_aggregate.get_file_size_in_mb(src_path)
                dst_size = lerobot_aggregate.get_file_size_in_mb(dst_path)
                if dst_size + src_size >= video_files_size_in_mb:
                    chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunk_size)
                    dst_path = dst_meta.root / lerobot_aggregate.DEFAULT_VIDEO_PATH.format(
                        video_key=key,
                        chunk_index=chunk_idx,
                        file_index=file_idx,
                    )
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(str(src_path), str(dst_path))
                    offset = 0.0
                    current_file_duration = src_duration
                else:
                    offset = current_file_duration
                    concatenate_video_files([dst_path, src_path], dst_path)
                    current_file_duration += src_duration

            video_idx["src_to_dst"][(src_chunk_idx, src_file_idx)] = {
                "chunk": chunk_idx,
                "file": file_idx,
                "offset": offset,
            }
            video_idx["episode_duration"] += src_duration

        video_idx["chunk"] = chunk_idx
        video_idx["file"] = file_idx
        video_idx["current_file_duration"] = current_file_duration
        video_idx["latest_duration"] = video_idx.get("latest_duration", 0.0) + video_idx["episode_duration"]

    return videos_idx


def _fixed_update_meta_data(df, dst_meta, meta_idx, data_idx, videos_idx):
    df["meta/episodes/chunk_index"] = df["meta/episodes/chunk_index"] + meta_idx["chunk"]
    df["meta/episodes/file_index"] = df["meta/episodes/file_index"] + meta_idx["file"]
    df["data/chunk_index"] = df["data/chunk_index"] + data_idx["chunk"]
    df["data/file_index"] = df["data/file_index"] + data_idx["file"]

    for key, video_idx in videos_idx.items():
        chunk_col = f"videos/{key}/chunk_index"
        file_col = f"videos/{key}/file_index"
        from_col = f"videos/{key}/from_timestamp"
        to_col = f"videos/{key}/to_timestamp"
        src_to_dst = video_idx.get("src_to_dst", {})
        if not src_to_dst:
            raise RuntimeError(f"Missing source-to-destination video mapping for {key}.")

        for idx in df.index:
            src_key = (df.at[idx, chunk_col], df.at[idx, file_col])
            dst = src_to_dst[src_key]
            df.at[idx, chunk_col] = dst["chunk"]
            df.at[idx, file_col] = dst["file"]
            df.at[idx, from_col] += dst["offset"]
            df.at[idx, to_col] += dst["offset"]

    df["dataset_from_index"] = df["dataset_from_index"] + dst_meta.info["total_frames"]
    df["dataset_to_index"] = df["dataset_to_index"] + dst_meta.info["total_frames"]
    df["episode_index"] = df["episode_index"] + dst_meta.info["total_episodes"]
    return df


def merge_with_fixed_video_metadata(subset_datasets: list[LeRobotDataset], dest_repo_id: str, dest_root: Path) -> None:
    original_aggregate_videos = lerobot_aggregate.aggregate_videos
    original_update_meta_data = lerobot_aggregate.update_meta_data
    try:
        lerobot_aggregate.aggregate_videos = _fixed_aggregate_videos
        lerobot_aggregate.update_meta_data = _fixed_update_meta_data
        lerobot_aggregate.aggregate_datasets(
            repo_ids=[dataset.repo_id for dataset in subset_datasets],
            roots=[dataset.root for dataset in subset_datasets],
            aggr_repo_id=dest_repo_id,
            aggr_root=dest_root,
        )
    finally:
        lerobot_aggregate.aggregate_videos = original_aggregate_videos
        lerobot_aggregate.update_meta_data = original_update_meta_data


def validate_video_metadata(dest_root: Path, repo_id: str) -> None:
    dataset = LeRobotDataset(repo_id, root=dest_root, download_videos=False)
    fps = dataset.meta.fps
    failures: list[str] = []
    frame_counts: dict[Path, int] = {}

    for video_key in dataset.meta.video_keys:
        for episode_index in range(dataset.meta.total_episodes):
            episode = dataset.meta.episodes[episode_index]
            rel_path = dataset.meta.get_video_file_path(int(episode["episode_index"]), video_key)
            video_path = dest_root / rel_path
            if video_path not in frame_counts:
                frame_counts[video_path] = len(VideoDecoder(str(video_path)))
            frame_count = frame_counts[video_path]
            max_ts = episode[f"videos/{video_key}/from_timestamp"] + (episode["length"] - 1) / fps
            max_frame = round(max_ts * fps)
            if max_frame >= frame_count:
                failures.append(
                    f"episode={int(episode['episode_index'])} key={video_key} path={rel_path} "
                    f"max_frame={max_frame} frame_count={frame_count}"
                )

    if failures:
        joined = "\n".join(failures[:20])
        raise RuntimeError(f"Video metadata points beyond decoded video frames:\n{joined}")

    logging.info(
        "Validated %s episodes, %s frames, %s video files.",
        dataset.meta.total_episodes,
        dataset.meta.total_frames,
        len(frame_counts),
    )


def load_or_create_subset(source: LeRobotDataset, episodes: list[int], split_root: Path) -> LeRobotDataset:
    selected_root = split_root / "selected"
    if selected_root.exists():
        try:
            existing = LeRobotDataset(f"{source.repo_id}_selected", root=selected_root, download_videos=False)
            if existing.meta.total_episodes == len(episodes):
                logging.info(
                    "Reusing staged subset %s at %s with %s episodes and %s frames.",
                    existing.repo_id,
                    existing.root,
                    existing.meta.total_episodes,
                    existing.meta.total_frames,
                )
                return existing
            logging.warning(
                "Staged subset at %s has %s episodes, expected %s; recreating it.",
                selected_root,
                existing.meta.total_episodes,
                len(episodes),
            )
        except Exception as exc:
            logging.warning("Could not load staged subset at %s (%s); recreating it.", selected_root, exc)
        shutil.rmtree(split_root)

    split = split_dataset(source, {"selected": episodes}, output_dir=split_root)["selected"]
    logging.info(
        "Created subset %s at %s with %s episodes and %s frames.",
        split.repo_id,
        split.root,
        split.meta.total_episodes,
        split.meta.total_frames,
    )
    return split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--dest-repo-id", default=DEFAULT_DEST_REPO_ID)
    parser.add_argument("--lerobot-root", type=Path, default=DEFAULT_LEROBOT_ROOT)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    parser.add_argument("--clean-staging", action="store_true")
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dest_root = args.lerobot_root / args.dest_repo_id
    if args.validate_only:
        validate_video_metadata(dest_root, args.dest_repo_id)
        return

    episodes_by_repo = parse_episode_csv(args.csv, args.lerobot_root)
    for repo_id, episodes in episodes_by_repo.items():
        source_root = args.lerobot_root / repo_id
        if not source_root.exists():
            raise FileNotFoundError(f"Source dataset is not cached: {source_root}")
        logging.info(
            "Selected %s unique episodes from %s (min=%s max=%s).",
            len(episodes),
            repo_id,
            min(episodes),
            max(episodes),
        )

    if args.clean_staging and args.staging_root.exists():
        shutil.rmtree(args.staging_root)
    args.staging_root.mkdir(parents=True, exist_ok=True)

    subset_datasets: list[LeRobotDataset] = []
    original_keep_episodes = lerobot_dataset_tools._keep_episodes_from_video_with_av
    lerobot_dataset_tools._keep_episodes_from_video_with_av = _fixed_keep_episodes_from_video_with_av
    try:
        for repo_id, episodes in episodes_by_repo.items():
            source = LeRobotDataset(repo_id, root=args.lerobot_root / repo_id, download_videos=False)
            split_root = args.staging_root / repo_id.replace("/", "__")
            split = load_or_create_subset(source, episodes, split_root)
            subset_datasets.append(split)
    finally:
        lerobot_dataset_tools._keep_episodes_from_video_with_av = original_keep_episodes

    if dest_root.exists():
        raise FileExistsError(f"Destination already exists; remove it first: {dest_root}")

    merge_with_fixed_video_metadata(subset_datasets, args.dest_repo_id, dest_root)
    validate_video_metadata(dest_root, args.dest_repo_id)

    if not args.keep_staging:
        shutil.rmtree(args.staging_root)


if __name__ == "__main__":
    main()
