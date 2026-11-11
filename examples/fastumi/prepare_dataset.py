"""Download FastUMI and convert its local copy to this repo's LeRobot v3 format."""

import json

from huggingface_hub import snapshot_download
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset
from lerobot.utils.constants import HF_LEROBOT_HOME

REPO_ID = "michios/fastumi_pick_fan"


def main():
    root = HF_LEROBOT_HOME / REPO_ID
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    if info.get("codebase_version") != "v3.0":
        # The upstream converter can delete previous conversion directories.
        # Refuse that behavior so an interrupted conversion is recoverable.
        for suffix in ("_old", "_v30"):
            sibling = root.with_name(root.name + suffix)
            if sibling.exists():
                raise FileExistsError(f"Inspect the previous conversion at {sibling} before retrying.")
        # Use main: the v2.1 tag can predate episode curation on the Hub.
        snapshot_download(REPO_ID, repo_type="dataset", revision="main", local_dir=root)
        info = json.loads(info_path.read_text())
        if info["codebase_version"] == "v2.1":
            convert_dataset(REPO_ID, root=HF_LEROBOT_HOME, push_to_hub=False)
        elif info["codebase_version"] != "v3.0":
            raise ValueError(f"Unsupported dataset version: {info['codebase_version']}")

    info = json.loads(info_path.read_text())
    features = info["features"]
    for key in ("observation.state", "action"):
        if features[key]["shape"] != [7]:
            raise ValueError(f"Expected seven-dimensional {key}, got {features[key]}")
    if "observation.images.front" not in features:
        raise ValueError("Missing observation.images.front")
    print(f"Ready: {root} ({info['total_episodes']} episodes, {info['total_frames']} frames)")


if __name__ == "__main__":
    main()
