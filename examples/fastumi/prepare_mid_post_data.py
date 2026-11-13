"""Download, audit, and index pinned single-arm episodes without duplicating videos."""

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time

import av
import numpy as np
import pyarrow.parquet as pq
import requests
import tyro

from examples.fastumi.prepare_two_phase import validate_task


class Downloader:
    def __init__(self, root, repo, revision):
        self.root, self.repo, self.revision = root, repo, revision
        self.lock = threading.Lock()
        self.next_request = 0.0
        self.headers = {"Authorization": "Bearer " + os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else {}

    def get(self, relative):
        path = self.root / relative
        marker = path.with_suffix(path.suffix + ".download.json")
        if path.exists() and marker.exists():
            record = json.loads(marker.read_text())
            if path.stat().st_size == record["size"] and record["revision"] == self.revision:
                return path, record["sha256"]
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".partial")
        for attempt in range(12):
            with self.lock:
                wait = max(0, self.next_request - time.monotonic())
                self.next_request = max(self.next_request, time.monotonic()) + 0.125
            time.sleep(wait)
            try:
                url = f"https://huggingface.co/datasets/{self.repo}/resolve/{self.revision}/{relative}"
                with requests.get(url, headers=self.headers, stream=True, timeout=(30, 180)) as response:
                    if response.status_code in (429, 500, 502, 503, 504):
                        pause = min(600, max(30, int(response.headers.get("Retry-After", "60"))))
                        print(f"Retrying HTTP {response.status_code}: {relative} in {pause}s", flush=True)
                        time.sleep(pause)
                        continue
                    if response.status_code == 404:
                        raise FileNotFoundError(f"Pinned source file is absent: {relative}")
                    response.raise_for_status()
                    digest = hashlib.sha256()
                    size = 0
                    with partial.open("wb") as out:
                        for chunk in response.iter_content(1024 * 1024):
                            out.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                    if response.headers.get("Content-Length") and size != int(response.headers["Content-Length"]):
                        raise RuntimeError(f"Incomplete transfer: {relative}")
                partial.replace(path)
                marker.write_text(json.dumps({"revision": self.revision, "size": size, "sha256": digest.hexdigest()}))
                return path, digest.hexdigest()
            except (requests.RequestException, RuntimeError):
                if attempt == 11:
                    raise
                time.sleep(min(120, 5 * (attempt + 1)))
        raise RuntimeError(f"Download retries exhausted: {relative}")


def audit_arrays(table, length, offset):
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    timestamp = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    tasks = np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
    if state.shape != (length, 7) or action.shape != state.shape:
        raise ValueError("Episode lengths or pose dimensions disagree")
    if not np.isfinite(state).all() or not np.isfinite(action).all() or not np.isfinite(timestamp).all():
        raise ValueError("Nonfinite episode values")
    expected = state if offset == 1 else np.concatenate([state[1:], state[-1:]])
    if not np.array_equal(action, expected):
        raise ValueError(f"Unexpected action timing for offset {offset}")
    if np.any(state[:, 6] < 0) or np.any(state[:, 6] > 1):
        raise ValueError("Gripper is not an opening fraction in [0,1]")
    if not np.allclose(timestamp, np.arange(length) / 20, atol=1e-3, rtol=0):
        raise ValueError("Video timestamps must start at zero and advance at 20 Hz")
    return state, action, timestamp, tasks


def main(
    root: Path,
    fan_cache: Path,
    workers: int = 12,
    split: Path = Path("examples/fastumi/mid_post_public_split.proposed.json"),
    fan_manifest: Path = Path("examples/fastumi/run3_datasets.json"),
):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    spec = json.loads(split.read_text())
    download = Downloader(root / "public", spec["repo_id"], spec["revision"])
    assignments = {}
    for phase, sources in spec["public_phases"].items():
        for source in sources:
            for episode in source["episode_indices"]:
                key = (source["path"], episode)
                if key in assignments:
                    raise ValueError("Public episode appears in both phases")
                assignments[key] = phase
    excluded = {(e["path"], e["episode_index"]) for e in spec.get("excluded_episodes", [])}
    if excluded & set(assignments):
        raise ValueError("An excluded episode appears in a training phase")
    jobs = []
    for task_path in sorted({p for p, _ in assignments}):
        meta = {}
        for name in ("info.json", "episodes.jsonl", "tasks.jsonl"):
            path, _ = download.get(f"{task_path}/meta/{name}")
            meta[name] = (
                json.loads(path.read_text())
                if name.endswith(".json")
                else [json.loads(line) for line in path.read_text().splitlines() if line]
            )
        info, episodes, tasks = meta["info.json"], meta["episodes.jsonl"], meta["tasks.jsonl"]
        validate_task({"path": task_path, "info": info, "episodes": episodes, "tasks": tasks})
        selected_ids = {e for p, e in assignments if p == task_path}
        excluded_ids = {e for p, e in excluded if p == task_path}
        if {e["episode_index"] for e in episodes} != selected_ids | excluded_ids:
            raise ValueError("Split does not cover the pinned source")
        for episode in episodes:
            if episode["episode_index"] in excluded_ids:
                continue
            jobs.append(
                {
                    "source": task_path,
                    "episode": episode["episode_index"],
                    "length": episode["length"],
                    "group": "public",
                    "phase": assignments[(task_path, episode["episode_index"])],
                    "info": info,
                    "tasks": {str(t["task_index"]): t["task"] for t in tasks},
                    "camera": "observation.images.camera_rgb_image",
                    "action_offset": 0,
                }
            )
    for source in json.loads(fan_manifest.read_text())["datasets"]:
        record = fan_cache / ".fastumi-sources" / (source["repo_id"].replace("/", "__") + ".json")
        if json.loads(record.read_text()) != source:
            raise ValueError("Fan cache revision differs from the approved run 3 source")
        raw = fan_cache / (source["repo_id"] + "_old")
        info = json.loads((raw / "meta/info.json").read_text())
        for line in (raw / "meta/episodes.jsonl").read_text().splitlines():
            episode = json.loads(line)
            jobs.append(
                {
                    "source": source["repo_id"],
                    "episode": episode["episode_index"],
                    "length": episode["length"],
                    "group": "fan",
                    "phase": "both",
                    "raw": str(raw),
                    "info": info,
                    "tasks": None,
                    "camera": "observation.images.front",
                    "action_offset": 1,
                }
            )
    print(f"Preparing {len(jobs)} episodes; public partitions remain disjoint", flush=True)
    (root / "numeric").mkdir(exist_ok=True)

    def prepare(job):
        info = job["info"]
        args = {
            "episode_index": job["episode"],
            "episode_chunk": job["episode"] // info["chunks_size"],
            "video_key": job["camera"],
        }
        data_file, video_file = info["data_path"].format(**args), info["video_path"].format(**args)
        if job["group"] == "public":
            data, _ = download.get(f"{job['source']}/{data_file}")
            video, video_hash = download.get(f"{job['source']}/{video_file}")
        else:
            data, video = Path(job["raw"]) / data_file, Path(job["raw"]) / video_file
            with video.open("rb") as stream:
                video_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        table = pq.read_table(data)
        state, action, timestamp, task_indices = audit_arrays(table, job["length"], job["action_offset"])
        tasks = job["tasks"] or {str(int(t)): "put the fan in the box" for t in np.unique(task_indices)}
        if not set(map(str, np.unique(task_indices))).issubset(tasks):
            raise ValueError("Frame task indices do not resolve to instructions")
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            if (stream.width, stream.height) != (1280, 720):
                raise ValueError("Unexpected video resolution")
            if stream.frames and stream.frames != job["length"]:
                raise ValueError(f"Video/pose frame count mismatch: {video}")
        numeric = f"numeric/{job['source'].replace('/', '__')}__{job['episode']:06d}.npz"
        np.savez(root / numeric, state=state, action=action, timestamp=timestamp, task_index=task_indices)
        return {k: job[k] for k in ("source", "episode", "length", "group", "phase", "action_offset")} | {
            "tasks": tasks,
            "numeric": numeric,
            "video": str(video),
            "video_sha256": video_hash,
            "trajectory_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
        }

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(prepare, job) for job in jobs}
        try:
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                if len(results) % 100 == 0 or len(results) == len(jobs):
                    free = shutil.disk_usage(root).free / 1e9
                    print(f"Audited {len(results)}/{len(jobs)} episodes; free disk {free:.1f} GB", flush=True)
                    if free < 450:
                        raise RuntimeError("Less than 450 GB remains for training checkpoints; stopping preparation")
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    seen = {}
    for episode in results:
        if episode["group"] != "public":
            continue
        for kind in ("trajectory_sha256", "video_sha256"):
            key = (kind, episode[kind])
            if key in seen and seen[key] != episode["phase"]:
                raise ValueError(
                    "Duplicate public trajectory/video crosses phase boundary; regroup duplicates before training"
                )
            seen[key] = episode["phase"]
    results.sort(key=lambda e: (e["group"], e["source"], e["episode"]))
    for group in ("fan", "midtrain", "posttrain"):
        selected = [
            e
            for e in results
            if (e["group"] == "fan" if group == "fan" else e["group"] == "public" and e["phase"] == group)
        ]
        count = sum(e["length"] for e in selected)
        directory = root / "arrays" / group
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {
            name: np.lib.format.open_memmap(
                directory / f"{name}.npy",
                mode="w+",
                dtype=dtype,
                shape=(count, 7) if name in ("state", "action") else (count,),
            )
            for name, dtype in (
                ("state", np.float32),
                ("action", np.float32),
                ("timestamp", np.float64),
                ("task_index", np.int64),
            )
        }
        start = 0
        for episode in selected:
            with np.load(root / episode["numeric"], allow_pickle=False) as numeric:
                for name, array in arrays.items():
                    array[start : start + episode["length"]] = numeric[name]
            episode["numeric"] = f"arrays/{group}"
            episode["array_start"] = start
            start += episode["length"]
        for array in arrays.values():
            array.flush()
    index = root / "episodes.json"
    index.write_text(json.dumps(results, indent=2) + "\n")
    report = {
        "ready": True,
        "episodes": len(results),
        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "public_revision": spec["revision"],
        "fan_manifest_sha256": hashlib.sha256(fan_manifest.read_bytes()).hexdigest(),
    }
    (root / "ready.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "DATA READY: all selected poses, action timing, task labels, video counts, and cross-phase duplicates audited",
        flush=True,
    )


if __name__ == "__main__":
    tyro.cli(main)
