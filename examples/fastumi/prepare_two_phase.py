"""Pin and partition FastUMI episode metadata; does not download videos or train."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
import tyro

PUBLIC_REPO = "IPEC-COMMUNITY/FastUMI_100k_lerobot"
POSE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def validate_task(task: dict) -> None:
    info, episodes = task["info"], task["episodes"]
    if info["codebase_version"] != "v2.1" or info["fps"] != 20:
        raise ValueError(f"Expected LeRobot v2.1 at 20 Hz: {task['path']}")
    for key in ("observation.state", "action"):
        feature = info["features"][key]
        names = feature.get("names")
        if isinstance(names, dict):
            names = names.get("motors")
        if feature["shape"] != [7] or names != POSE_NAMES:
            raise ValueError(f"Expected single-arm XYZ/RPY/gripper: {task['path']} {key}")
    camera = info["features"].get("observation.images.camera_rgb_image", {})
    if camera.get("dtype") != "video" or camera.get("shape") != [720, 1280, 3]:
        raise ValueError(f"Unexpected public camera schema: {task['path']}")
    ids = [e["episode_index"] for e in episodes]
    if len(ids) != len(set(ids)) or set(ids) != set(range(info["total_episodes"])):
        raise ValueError(f"Missing or duplicate episode IDs: {task['path']}")
    if any(e["length"] <= 0 for e in episodes) or sum(e["length"] for e in episodes) != info["total_frames"]:
        raise ValueError(f"Episode frame counts disagree: {task['path']}")
    labels = {t["task"] for t in task["tasks"]}
    if not labels or any(not isinstance(t, str) or not t.strip() for t in labels):
        raise ValueError(f"Missing task instructions: {task['path']}")
    if any(not e.get("tasks") or not set(e["tasks"]).issubset(labels) for e in episodes):
        raise ValueError(f"Episode instructions absent from task metadata: {task['path']}")


def partition(
    inventory: dict,
    posttrain_public_episodes: int,
    seed: int = 42,
    *,
    excluded_episodes: set[tuple[str, int]] | None = None,
) -> dict:
    """Sample complete episodes without replacement, independently of file ordering."""
    revision = inventory["revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("An immutable 40-character source revision is required")
    tasks = sorted(inventory["tasks"], key=lambda task: task["path"])
    if not tasks or len({t["path"] for t in tasks}) != len(tasks):
        raise ValueError("Expected distinct, nonempty tasks")
    for task in tasks:
        if not task["path"].startswith("single_arm/"):
            raise ValueError("This adapter supports single-arm tasks only")
        validate_task(task)
    identities = [(task["path"], e["episode_index"]) for task in tasks for e in task["episodes"]]
    excluded = excluded_episodes or set()
    if not excluded.issubset(set(identities)):
        raise ValueError("Exclusions contain unknown public episode IDs")
    identities = [identity for identity in identities if identity not in excluded]
    if not 0 < posttrain_public_episodes < len(identities):
        raise ValueError("Reserve a positive count smaller than the public episode total")

    def rank(identity):
        key = json.dumps([seed, inventory["repo_id"], revision, *identity], separators=(",", ":"))
        return hashlib.sha256(key.encode()).digest(), identity

    reserved = set(sorted(identities, key=rank)[:posttrain_public_episodes])
    phases = {"midtrain": [], "posttrain": []}
    for task in tasks:
        for phase, phase_sources in phases.items():
            chosen = sorted(
                (
                    e
                    for e in task["episodes"]
                    if (task["path"], e["episode_index"]) not in excluded
                    and ((task["path"], e["episode_index"]) in reserved) == (phase == "posttrain")
                ),
                key=lambda e: e["episode_index"],
            )
            if not chosen:
                continue
            phase_sources.append(
                {
                    "path": task["path"],
                    "episode_indices": [e["episode_index"] for e in chosen],
                    "episodes": len(chosen),
                    "frames": sum(e["length"] for e in chosen),
                    "camera_key": "observation.images.camera_rgb_image",
                    "action_sequence_offset": 0,
                    "prompt_source": "dataset_task",
                }
            )
    mid = {(s["path"], e) for s in phases["midtrain"] for e in s["episode_indices"]}
    post = {(s["path"], e) for s in phases["posttrain"] for e in s["episode_indices"]}
    if mid & post or mid | post != set(identities):
        raise AssertionError("Public partition must be disjoint and exhaustive")
    return {
        "repo_id": inventory["repo_id"],
        "revision": revision,
        "seed": seed,
        "selection": "uniform episode sampling without replacement via seeded SHA256 ranking",
        "scope": "single_arm",
        "public_phases": phases,
        "summary": {
            phase: {"episodes": sum(s["episodes"] for s in sources), "frames": sum(s["frames"] for s in sources)}
            for phase, sources in phases.items()
        },
        "excluded_episodes": [{"path": path, "episode_index": episode} for path, episode in sorted(excluded)],
        "public_episode_overlap": 0,
        "full_pose_audit_required": True,
    }


def fetch_inventory(revision: str, cache_dir: Path | None = None) -> dict:
    api = HfApi()
    pinned = api.dataset_info(PUBLIC_REPO, revision=revision).sha
    paths = sorted(
        x.path
        for x in api.list_repo_tree(PUBLIC_REPO, path_in_repo="single_arm", revision=pinned, repo_type="dataset")
        if x.path.count("/") == 1
    )
    tasks = []
    for path in paths:
        files = {}
        for filename in ("info.json", "episodes.jsonl", "tasks.jsonl"):
            local = hf_hub_download(
                PUBLIC_REPO, f"{path}/meta/{filename}", repo_type="dataset", revision=pinned, cache_dir=cache_dir
            )
            text = Path(local).read_text()
            files[filename] = (
                json.loads(text)
                if filename.endswith(".json")
                else [json.loads(line) for line in text.splitlines() if line]
            )
        tasks.append(
            {
                "path": path,
                "info": files["info.json"],
                "episodes": files["episodes.jsonl"],
                "tasks": files["tasks.jsonl"],
            }
        )
    return {"repo_id": PUBLIC_REPO, "revision": pinned, "tasks": tasks}


def main(
    *,
    posttrain_public_episodes: int,
    output: Path,
    revision: str = "4cf7cca8fdc15d6f71ee0055553ce8a5e78aa9a0",
    seed: int = 42,
    inventory_path: Path | None = None,
    availability: Path | None = None,
):
    inventory = json.loads(inventory_path.read_text()) if inventory_path else fetch_inventory(revision)
    if inventory["repo_id"] != PUBLIC_REPO or inventory["revision"] != revision:
        raise ValueError("Inventory does not match requested pinned source")
    excluded = set()
    if availability is not None:
        report = json.loads(availability.read_text())
        if (report["repo_id"], report["revision"]) != (PUBLIC_REPO, revision):
            raise ValueError("Availability report belongs to another source revision")
        excluded = {(item["path"], item["episode_index"]) for item in report["excluded_episodes"]}
    result = partition(inventory, posttrain_public_episodes, seed, excluded_episodes=excluded)
    serialized = json.dumps(result, indent=2) + "\n"
    if output.exists() and output.read_text() != serialized:
        raise FileExistsError("Refusing to overwrite a different split; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized)
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved {output}; no videos downloaded and no training launched.")


if __name__ == "__main__":
    tyro.cli(main)
