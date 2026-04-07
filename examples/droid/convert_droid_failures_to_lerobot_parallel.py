"""
Parallel version of convert_droid_failures_to_lerobot.py.

Uses a ThreadPoolExecutor for GCS downloads + MP4 decoding, with the main
thread acting as the serial dataset writer.  Typical speedup: ~num_workers x.

Memory sizing: each in-flight episode holds full-resolution camera frames
(~8-9 MB/frame) during load_trajectory(), before the resize step.  A 500-frame
episode peaks at ~4.5 GB.  Set num_workers so that num_workers × peak_episode_gb
fits comfortably in RAM.  Example: 64 GB RAM → num_workers=4 works safely.

Usage:
  python convert_droid_failures_to_lerobot_parallel.py \\
    --repo_id michios/droid_failure_v2 \\
    --num_workers 4 \\
    --max_episodes 50   # for testing

Resume after a crash:
  python convert_droid_failures_to_lerobot_parallel.py \\
    --repo_id michios/droid_failure_v2 \\
    --num_workers 4 \\
    --resume

  Progress is tracked in <episode_list_stem>.progress.jsonl next to the episode
  list file.  Resuming skips already-processed GCS paths and appends new
  episodes to the existing dataset on disk.
"""

import gc
import json
import shutil
import tempfile
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import h5py
import numpy as np
import tyro
from tqdm import tqdm

from convert_droid_failures_to_lerobot import (
    FPS,
    GCS_BASE,
    MIN_FRAMES,
    STATIC_THRESHOLD,
    build_annotation_index,
    compute_idle_flags,
    find_next_annotation,
    gsutil_cp,
    gsutil_cp_dir,
    gsutil_ls,
    load_trajectory,
    resize_image,
)

DATASET_FEATURES = {
    "exterior_image_1_left": {"dtype": "image", "shape": (180, 320, 3), "names": ["height", "width", "channel"]},
    "exterior_image_2_left": {"dtype": "image", "shape": (180, 320, 3), "names": ["height", "width", "channel"]},
    "wrist_image_left":      {"dtype": "image", "shape": (180, 320, 3), "names": ["height", "width", "channel"]},
    "joint_position":        {"dtype": "float32", "shape": (7,),        "names": ["joint_position"]},
    "gripper_position":      {"dtype": "float32", "shape": (1,),        "names": ["gripper_position"]},
    "actions":               {"dtype": "float32", "shape": (8,),        "names": ["actions"]},
    "is_idle":               {"dtype": "float32", "shape": (1,),        "names": ["is_idle"]},
}


# ---------------------------------------------------------------------------
# Per-episode worker (runs in thread pool)
# ---------------------------------------------------------------------------

def process_episode(episode_gcs_path: str, annotation_index: dict) -> dict:
    """
    Downloads, filters, and decodes one episode.

    Returns one of:
      {"skip": "too_short" | "static"}
      {"error": <message>}
      {"frames": list[dict], "task": str, "uuid": str}
    """
    episode_gcs_path = episode_gcs_path.rstrip("/") + "/"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        try:
            # Step 1: metadata
            meta_files = gsutil_ls(f"{episode_gcs_path}metadata_*.json")
            if not meta_files:
                return {"error": f"no metadata in {episode_gcs_path}"}
            meta_local = tmpdir / "metadata.json"
            gsutil_cp(meta_files[0], str(meta_local))
            with open(meta_local) as f:
                meta = json.load(f)
            uuid = meta["uuid"]
            traj_length = meta.get("trajectory_length", 0)

            # Step 2: filter too-short
            if traj_length < MIN_FRAMES:
                return {"skip": "too_short"}

            # Step 3: download trajectory.h5
            h5_local = tmpdir / "trajectory.h5"
            gsutil_cp(f"{episode_gcs_path}trajectory.h5", str(h5_local))

            # Step 4: filter static
            with h5py.File(h5_local, "r") as f:
                joint_velocities = f["action"]["joint_velocity"][:]
            if np.all(np.std(joint_velocities, axis=0) < STATIC_THRESHOLD):
                return {"skip": "static"}

            # Step 5: annotation
            task = find_next_annotation(uuid, annotation_index)
            print(f"  [TASK] {uuid} -> {task!r}")

            # Step 6: download MP4s
            mp4_local_dir = tmpdir / "MP4"
            mp4_local_dir.mkdir()
            gsutil_cp_dir(f"{episode_gcs_path}recordings/MP4/", str(mp4_local_dir) + "/")

            # Step 7: decode trajectory into plain numpy dicts
            idle_flags = compute_idle_flags(joint_velocities)
            trajectory = load_trajectory(str(h5_local), str(mp4_local_dir))

            if not trajectory:
                return {"error": f"empty trajectory for {uuid}"}

            frames = []
            for i, step in enumerate(trajectory):
                cam_type_dict = step["observation"]["camera_type"]
                wrist_ids    = [k for k, v in cam_type_dict.items() if v == 0]
                exterior_ids = [k for k, v in cam_type_dict.items() if v != 0]
                if len(wrist_ids) == 0 or len(exterior_ids) < 2:
                    return {"error": f"degenerate cameras for {uuid}"}
                frames.append({
                    "exterior_image_1_left": resize_image(
                        step["observation"]["image"][exterior_ids[0]][..., ::-1], (320, 180)
                    ),
                    "exterior_image_2_left": resize_image(
                        step["observation"]["image"][exterior_ids[1]][..., ::-1], (320, 180)
                    ),
                    "wrist_image_left": resize_image(
                        step["observation"]["image"][wrist_ids[0]][..., ::-1], (320, 180)
                    ),
                    "joint_position": np.asarray(
                        step["observation"]["robot_state"]["joint_positions"], dtype=np.float32
                    ),
                    "gripper_position": np.asarray(
                        step["observation"]["robot_state"]["gripper_position"][None], dtype=np.float32
                    ),
                    "actions": np.concatenate([
                        step["action"]["joint_velocity"],
                        step["action"]["gripper_position"][None],
                    ], dtype=np.float32),
                    "is_idle": np.array([float(idle_flags[i])], dtype=np.float32),
                    "task": task,
                })

            return {"frames": frames, "task": task, "uuid": uuid}

        except Exception as e:
            return {"error": f"{episode_gcs_path}: {e}"}


# ---------------------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------------------

class _LastEpisodeProxy:
    """Lightweight proxy for meta.episodes that only keeps the last row in memory.

    LeRobotDataset only ever accesses ``episodes[-1]`` and ``len(episodes)``
    when writing.  Loading the full episodes HF Dataset for thousands of
    episodes consumes 10+ GB of RAM, so this proxy avoids that by reading
    only the tail of the last parquet file.
    """

    def __init__(self, last_row: dict, total: int):
        self._last = last_row
        self._total = total

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        if idx == -1 or idx == self._total - 1:
            return self._last
        raise IndexError(f"Only last episode is loaded (requested index {idx})")


def _build_lightweight_meta(repo_id: str, root=None, metadata_buffer_size: int = 10):
    """Build a LeRobotDatasetMetadata without the costly load_episodes().

    This replaces the full HF Dataset of episodes with a tiny proxy that
    only stores the last episode row – sufficient for the write path.
    """
    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import (
        CODEBASE_VERSION,
        HF_LEROBOT_HOME,
        LeRobotDatasetMetadata,
    )
    from lerobot.datasets.utils import load_info, load_stats, load_tasks

    meta = LeRobotDatasetMetadata.__new__(LeRobotDatasetMetadata)
    meta.repo_id = repo_id
    meta.revision = CODEBASE_VERSION
    meta.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
    meta.writer = None
    meta.latest_episode = None
    meta.metadata_buffer = []
    meta.metadata_buffer_size = metadata_buffer_size

    # Lightweight loads (all < 1 MB)
    meta.info = load_info(meta.root)
    meta.tasks = load_tasks(meta.root)
    meta.stats = load_stats(meta.root)

    # Instead of loading ALL episodes, read only the last row of the last
    # episode parquet file via pyarrow (no HF Dataset / arrow cache).
    ep_dir = meta.root / "meta" / "episodes"
    ep_files = sorted(ep_dir.glob("*/*.parquet"))
    if ep_files:
        last_table = pq.read_table(
            str(ep_files[-1]),
            columns=[
                "episode_index",
                "meta/episodes/chunk_index",
                "meta/episodes/file_index",
                "dataset_from_index",
                "dataset_to_index",
                "data/chunk_index",
                "data/file_index",
            ],
        )
        last_row = {col: last_table[col][-1].as_py() for col in last_table.column_names}
        meta.episodes = _LastEpisodeProxy(last_row, meta.info["total_episodes"])
    else:
        meta.episodes = None

    return meta


def _open_dataset_for_resume(
    repo_id: str,
    root=None,
    image_writer_threads: int = 0,
    image_writer_processes: int = 0,
):
    """
    Reconstruct a LeRobotDataset in write mode from an existing on-disk dataset.

    Uses a lightweight metadata loader that avoids reading all episode rows
    into memory (which can consume 10+ GB for large datasets).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import get_safe_default_codec

    obj = LeRobotDataset.__new__(LeRobotDataset)
    obj.meta = _build_lightweight_meta(repo_id, root, metadata_buffer_size=10)
    obj.repo_id = obj.meta.repo_id
    obj.root = obj.meta.root
    obj.revision = None
    obj.tolerance_s = 1e-4
    obj.image_writer = None
    obj.batch_encoding_size = 1
    obj.episodes_since_last_encoding = 0

    if image_writer_processes or image_writer_threads:
        obj.start_image_writer(image_writer_processes, image_writer_threads)

    # create_episode_buffer uses meta.total_episodes as the episode index,
    # so new episodes will be numbered correctly after existing ones.
    obj.episode_buffer = obj.create_episode_buffer()
    obj.episodes = None
    obj.hf_dataset = obj.create_hf_dataset()
    obj.image_transforms = None
    obj.delta_timestamps = None
    obj.delta_indices = None
    obj.video_backend = get_safe_default_codec()
    obj.writer = None
    obj.latest_episode = None
    obj._current_file_start_frame = None
    obj._lazy_loading = False
    # Count existing frames so frame indices continue from the right offset
    obj._recorded_frames = obj.meta.total_frames
    obj._writer_closed_for_reading = False
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _open_or_create_dataset(repo_id: str, output_path, resume: bool):
    """Create a new dataset or reopen an existing one for resume."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if resume and output_path.exists():
        dataset = _open_dataset_for_resume(
            repo_id=repo_id,
            image_writer_threads=4,
            image_writer_processes=2,
        )
        print(f"  Opened dataset: {dataset.meta.total_episodes} episodes, "
              f"{dataset.meta.total_frames} frames on disk.")
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type="panda",
            fps=FPS,
            features=DATASET_FEATURES,
            image_writer_threads=4,
            image_writer_processes=2,
        )
    return dataset


def main(
    repo_id: str,
    annotations_path: str = "droid_language_annotations.json",
    max_episodes: int = -1,
    push_to_hub: bool = False,
    resume: bool = False,
    episode_list_path: str = "failure_episode_paths.txt",
    # Each in-flight worker holds a full decoded episode in RAM.
    # Peak memory ≈ num_workers × episode_size (see module docstring).
    num_workers: int = 2,
    # Periodically close and reopen the dataset to release accumulated memory.
    # Set to 0 to disable.  200 keeps peak RSS well under 16 GB on 32 GB machines.
    restart_every: int = 200,
):
    try:
        from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME

        output_path = HF_LEROBOT_HOME / repo_id

        # Progress file lives next to the episode list: <stem>.progress.jsonl
        ep_list_path = Path(episode_list_path)
        progress_path = ep_list_path.parent / (ep_list_path.stem + ".progress.jsonl")

        if not resume:
            if output_path.exists():
                print(f"Removing existing dataset at {output_path}")
                shutil.rmtree(output_path)
            # Always clear stale progress file on a fresh (non-resume) run
            if progress_path.exists():
                progress_path.unlink()

        print("Building annotation index...")
        annotation_index = build_annotation_index(annotations_path)
        print(f"  {sum(len(v) for v in annotation_index.values())} annotated episodes indexed "
              f"across {len(annotation_index)} (lab, user_id) pairs")

        # -----------------------------------------------------------------
        # Resume: load which GCS paths were already processed
        # -----------------------------------------------------------------
        already_done: dict[str, str] = {}  # normalized_path -> outcome
        if resume and progress_path.exists():
            print(f"Loading resume progress from {progress_path}...")
            with open(progress_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        # Normalize path the same way process_episode does
                        key = record["path"].rstrip("/") + "/"
                        already_done[key] = record["outcome"]
            print(f"  {len(already_done)} episodes already processed, will skip them.")

        # -----------------------------------------------------------------
        # Create or resume dataset
        # -----------------------------------------------------------------
        dataset = _open_or_create_dataset(repo_id, output_path, resume)

        # Load or enumerate episode list
        if Path(episode_list_path).exists():
            print(f"Loading episode list from {episode_list_path}...")
            all_episode_gcs_paths = Path(episode_list_path).read_text().splitlines()
        else:
            print("Enumerating failure episodes from GCS...")
            date_dirs = gsutil_ls(f"{GCS_BASE}/*/failure/")
            all_episode_gcs_paths = []
            for date_dir in tqdm(date_dirs, desc="Enumerating dates"):
                episode_dirs = [p for p in gsutil_ls(date_dir) if p.endswith("/")]
                all_episode_gcs_paths.extend(episode_dirs)
            Path(episode_list_path).write_text("\n".join(all_episode_gcs_paths))
            print(f"Saved episode list to {episode_list_path}")

        # Normalize all paths (consistent trailing slash) and drop blanks
        all_episode_gcs_paths = [
            p.rstrip("/") + "/" for p in all_episode_gcs_paths if p.strip()
        ]
        print(f"Found {len(all_episode_gcs_paths)} total failure episodes")

        # Filter out already-processed paths
        if already_done:
            all_episode_gcs_paths = [p for p in all_episode_gcs_paths if p not in already_done]
            print(f"  {len(all_episode_gcs_paths)} episodes remaining after resume filter")

        if max_episodes > 0:
            all_episode_gcs_paths = all_episode_gcs_paths[:max_episodes]
            print(f"Limiting to {max_episodes} episodes for testing")

        stats = {"total": len(all_episode_gcs_paths), "too_short": 0, "static": 0, "converted": 0, "errors": 0}
        oom_abort = False
        episodes_since_restart = 0

        # Keep num_workers*2 futures queued so workers stay busy while the main
        # thread writes frames to the dataset.
        max_inflight = num_workers * 2

        # Open progress file in append mode for crash-safe incremental writes.
        # Written only from the main thread, so no locking needed.
        progress_file = open(progress_path, "a")
        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                paths_iter = iter(all_episode_gcs_paths)
                in_flight: dict = {}  # future -> gcs_path

                def safe_submit(path):
                    nonlocal oom_abort
                    try:
                        return executor.submit(process_episode, path, annotation_index)
                    except MemoryError:
                        print("  [OOM] Unable to start worker thread. Aborting remaining episodes.")
                        stats["errors"] += 1
                        oom_abort = True
                        return None

                for _ in range(max_inflight):
                    path = next(paths_iter, None)
                    if path is None:
                        break
                    future = safe_submit(path)
                    if future is not None:
                        in_flight[future] = path
                    if oom_abort:
                        break

                with tqdm(total=len(all_episode_gcs_paths), desc="Processing episodes") as pbar:
                    while in_flight:
                        done, _ = wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)
                        for future in done:
                            current_path = in_flight.pop(future)
                            outcome = "error"
                            try:
                                result = future.result()
                            except MemoryError:
                                print("  [OOM] Worker ran out of memory. Aborting remaining episodes.")
                                stats["errors"] += 1
                                oom_abort = True
                                progress_file.write(
                                    json.dumps({"path": current_path, "outcome": "error"}) + "\n"
                                )
                                progress_file.flush()
                                break
                            except Exception as e:
                                print(f"  [ERROR] Worker exception: {e}")
                                stats["errors"] += 1
                                progress_file.write(
                                    json.dumps({"path": current_path, "outcome": "error"}) + "\n"
                                )
                                progress_file.flush()
                                pbar.update(1)
                                next_path = next(paths_iter, None)
                                if next_path is not None:
                                    f2 = safe_submit(next_path)
                                    if f2 is not None:
                                        in_flight[f2] = next_path
                                continue

                            if "skip" in result:
                                stats[result["skip"]] += 1
                                outcome = f"skip:{result['skip']}"
                            elif "error" in result:
                                print(f"  [ERROR] {result['error']}")
                                stats["errors"] += 1
                                outcome = "error"
                            else:
                                try:
                                    for frame in result["frames"]:
                                        dataset.add_frame(frame)
                                    dataset.save_episode()
                                    stats["converted"] += 1
                                    outcome = "converted"
                                    episodes_since_restart += 1
                                except MemoryError:
                                    print("  [OOM] Dataset writer ran out of memory. Aborting remaining episodes.")
                                    stats["errors"] += 1
                                    outcome = "error"
                                    oom_abort = True
                                    dataset.clear_episode_buffer(delete_images=False)
                                except Exception as e:
                                    print(f"  [ERROR] Dataset write failed: {e}")
                                    print(traceback.format_exc())
                                    stats["errors"] += 1
                                    outcome = "error"
                                    # Reset the episode buffer so the next episode starts clean.
                                    # save_episode() pops 'size' and 'task' from the buffer before
                                    # it may fail, leaving it in a corrupted state that causes
                                    # KeyError('size') for every subsequent add_frame() call.
                                    dataset.clear_episode_buffer(delete_images=False)

                            progress_file.write(
                                json.dumps({"path": current_path, "outcome": outcome}) + "\n"
                            )
                            progress_file.flush()
                            pbar.update(1)

                            # --- Periodic restart to prevent OOM ---
                            if restart_every > 0 and episodes_since_restart >= restart_every:
                                print(f"\n  [RESTART] {episodes_since_restart} episodes since last restart. "
                                      f"Closing dataset to free memory...")
                                dataset.stop_image_writer()
                                dataset.finalize()
                                del dataset
                                gc.collect()
                                # Release pyarrow memory back to OS
                                try:
                                    import pyarrow as pa
                                    pa.default_memory_pool().release_unused()
                                except Exception:
                                    pass
                                dataset = _open_or_create_dataset(repo_id, output_path, resume=True)
                                episodes_since_restart = 0
                                print(f"  [RESTART] Reopened: {dataset.meta.total_episodes} episodes, "
                                      f"{dataset.meta.total_frames} frames on disk.\n")

                            next_path = next(paths_iter, None)
                            if next_path is not None:
                                f2 = safe_submit(next_path)
                                if f2 is not None:
                                    in_flight[f2] = next_path

                        if oom_abort:
                            for f in in_flight:
                                f.cancel()
                            in_flight.clear()
                            break
        finally:
            progress_file.close()
            dataset.stop_image_writer()
            dataset.finalize()

        print("\n" + "=" * 60)
        print("Conversion Summary:")
        print(f"  Total episodes:   {stats['total']}")
        print(f"  Too short (<5s):  {stats['too_short']}")
        print(f"  Static joints:    {stats['static']}")
        print(f"  Errors:           {stats['errors']}")
        print(f"  Converted:        {stats['converted']}")
        print("=" * 60)

        if push_to_hub:
            print(f"\nPushing to HuggingFace Hub: {repo_id}")
            dataset.push_to_hub(
                tags=["droid", "panda", "failure"],
                private=False,
                push_videos=True,
                license="cc-by-4.0",
            )
    except MemoryError:
        print("  [OOM] Out of memory in main thread. Aborting.")
        return


if __name__ == "__main__":
    tyro.cli(main)
