"""Prepare pinned FastUMI datasets in an isolated cache; never launches training."""

import json
from pathlib import Path
import re

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset
from lerobot.utils.constants import HF_LEROBOT_HOME
import numpy as np
import pyarrow.parquet as pq
import tyro

POSE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def validate_source(info: dict, filenames: set[str]) -> None:
    if info["codebase_version"] != "v2.1":
        raise ValueError("This preparation tool expects LeRobot v2.1 source datasets")
    if info["fps"] != 20 or info["total_episodes"] <= 0 or info["total_frames"] <= 0:
        raise ValueError("Expected a nonempty 20 Hz dataset")
    for key in ("observation.state", "action"):
        feature = info["features"][key]
        if feature["shape"] != [7] or feature.get("names") != POSE_NAMES:
            raise ValueError(f"Unexpected XYZ/RPY/gripper schema: {key}")
    camera = "observation.images.front"
    feature = info["features"][camera]
    if feature["dtype"] != "video" or feature["shape"] != [720, 1280, 3]:
        raise ValueError("Expected the 720x1280 wrist RGB video")
    for episode in range(info["total_episodes"]):
        args = {"episode_index": episode, "episode_chunk": episode // info["chunks_size"], "video_key": camera}
        for key in ("data_path", "video_path"):
            required = info[key].format(**args)
            if required not in filenames:
                raise ValueError(f"Upload incomplete: missing {required}")
    for required in ("meta/episodes.jsonl", "meta/episodes_stats.jsonl", "meta/tasks.jsonl"):
        if required not in filenames:
            raise ValueError(f"Upload incomplete: missing {required}")


def audit_poses(root: Path, info: dict) -> dict:
    frames = 0
    largest_step = 0.0
    for episode in range(info["total_episodes"]):
        path = root / info["data_path"].format(episode_index=episode, episode_chunk=episode // info["chunks_size"])
        table = pq.read_table(path)
        state = np.asarray(table["observation.state"].to_pylist())
        action = np.asarray(table["action"].to_pylist())
        if state.shape != (len(table), 7) or not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"Invalid pose arrays: {path}")
        # This recipe takes t+1..t+16 because source actions equal measured poses.
        if not np.array_equal(state, action):
            raise ValueError(f"Action/state alignment differs; inspect before applying the +1 offset: {path}")
        if np.any(state[:, 6] < 0) or np.any(state[:, 6] > 1):
            raise ValueError(f"Expected gripper opening fractions in [0,1]: {path}")
        if len(state) > 1:
            largest_step = max(largest_step, float(np.linalg.norm(np.diff(state[:, :3], axis=0), axis=1).max()))
        frames += len(state)
    if frames != info["total_frames"]:
        raise ValueError(f"Frame count mismatch: {frames} != {info['total_frames']}")
    return {"episodes": info["total_episodes"], "frames": frames, "max_translation_step_m": largest_step}


def main(manifest: Path = Path("examples/fastumi/run3_datasets.json"), *, allow_pending: bool = False):
    spec = json.loads(manifest.read_text())
    sources = spec["datasets"]
    ids = [item["repo_id"] for item in sources]
    if not sources or len(ids) != len(set(ids)):
        raise ValueError("Manifest must list distinct datasets")
    pending = [item["repo_id"] for item in sources if item["revision"] is None]
    if pending and not allow_pending:
        raise ValueError(f"Wait for uploads to finish and pin revisions before preparing the full mixture: {pending}")
    api = HfApi()
    results = []
    for item in sources:
        repo_id, revision = item["repo_id"], item["revision"]
        if revision is None:
            print(f"PENDING: {repo_id}")
            continue
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"Pin an immutable commit SHA for {repo_id}")
        remote = api.dataset_info(repo_id, revision=revision)
        info = json.loads(
            Path(hf_hub_download(repo_id, "meta/info.json", repo_type="dataset", revision=revision)).read_text()
        )
        validate_source(info, {f.rfilename for f in remote.siblings})
        root = HF_LEROBOT_HOME / repo_id
        record = HF_LEROBOT_HOME / ".fastumi-sources" / f"{repo_id.replace('/', '__')}.json"
        if record.exists():
            if json.loads(record.read_text()) != item:
                raise ValueError(f"Cache revision differs; use a new HF_LEROBOT_HOME: {root}")
        elif root.exists():
            raise ValueError(f"Untracked existing cache; use an isolated HF_LEROBOT_HOME: {root}")
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(item, indent=2) + "\n")
        local_info = root / "meta/info.json"
        converted = local_info.exists() and json.loads(local_info.read_text())["codebase_version"] == "v3.0"
        if converted:
            # The converter preserves v2.1 source trajectories for audit.
            audit = audit_poses(root.with_name(root.name + "_old"), info)
        else:
            for suffix in ("_old", "_v30"):
                if root.with_name(root.name + suffix).exists():
                    raise FileExistsError(f"Inspect interrupted conversion before retrying: {root}{suffix}")
            snapshot_download(repo_id, repo_type="dataset", revision=revision, local_dir=root)
            audit = audit_poses(root, info)
            convert_dataset(repo_id, root=HF_LEROBOT_HOME, push_to_hub=False)
        converted_info = json.loads(local_info.read_text())
        if (converted_info["total_frames"], converted_info["total_episodes"]) != (
            info["total_frames"],
            info["total_episodes"],
        ):
            raise ValueError(f"Conversion changed counts: {repo_id}")
        results.append({**item, **audit})
        print(f"READY: {repo_id}: {audit}")
    output = {"config_name": spec["config_name"], "datasets": results, "pending": pending, "ready": not pending}
    target = HF_LEROBOT_HOME / f"{spec['run_name']}.preparation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Preparation report: {target}; complete={not pending}. No training launched.")


if __name__ == "__main__":
    tyro.cli(main)
