"""Watch finalized local Orbax checkpoints and back them up to a private HF repo."""

import json
import logging
from pathlib import Path
import time

from huggingface_hub import HfApi
import tyro


def pending_checkpoints(root: Path, uploaded: set[int], interval: int, final_step: int) -> list[Path]:
    if not root.exists():
        return []
    # Orbax writes to *.orbax-checkpoint-tmp-* and atomically renames the
    # completed directory to its numeric step on this local filesystem.
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir()
            and path.name.isdecimal()
            and int(path.name) not in uploaded
            and (int(path.name) == final_step or int(path.name) > 0 and int(path.name) % interval == 0)
            and all((path / name).is_dir() for name in ("params", "assets", "train_state"))
        ),
        key=lambda path: int(path.name),
    )


def main(
    checkpoint_dir: Path,
    repo_id: str = "michios/pi05-fastumi-pick-fan-full",
    interval: int = 10_000,
    final_step: int = 29_999,
    poll_seconds: float = 60,
):
    if interval <= 0 or poll_seconds <= 0:
        raise ValueError("interval and poll_seconds must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True)
    if not api.repo_info(repo_id).private:
        raise ValueError(f"Expected a private checkpoint repository: {repo_id}")
    # Include the run name so repeated experiments do not overwrite each other.
    prefix = checkpoint_dir.name
    ledger = checkpoint_dir.parent / f"{prefix}.hf-uploads.json"
    record = json.loads(ledger.read_text()) if ledger.exists() else {"repo_id": repo_id, "steps": []}
    if record["repo_id"] != repo_id:
        raise ValueError(f"Upload ledger belongs to another repository: {ledger}")
    uploaded = set(record["steps"])
    logging.info("Watching %s -> https://huggingface.co/%s/tree/main/%s", checkpoint_dir, repo_id, prefix)
    while final_step not in uploaded:
        for checkpoint in pending_checkpoints(checkpoint_dir, uploaded, interval, final_step):
            step = int(checkpoint.name)
            try:
                logging.info("Uploading checkpoint %s (full training state: %s)", step, step == final_step)
                result = api.upload_folder(
                    repo_id=repo_id,
                    folder_path=checkpoint,
                    path_in_repo=f"{prefix}/{step}",
                    # Intermediate checkpoints support inference. The final one
                    # also includes optimizer, RNG, and unaveraged parameters.
                    allow_patterns=None if step == final_step else ["params/**", "assets/**"],
                    commit_message=f"Back up {prefix} checkpoint {step}",
                )
                uploaded.add(step)
                ledger.parent.mkdir(parents=True, exist_ok=True)
                temporary = ledger.with_suffix(".tmp")
                temporary.write_text(json.dumps({"repo_id": repo_id, "steps": sorted(uploaded)}, indent=2) + "\n")
                temporary.replace(ledger)
                logging.info("Uploaded checkpoint %s: %s", step, result.commit_url)
            except Exception:
                logging.exception("Upload failed for %s; will retry after the polling interval", checkpoint)
                break
        if final_step not in uploaded:
            time.sleep(poll_seconds)
    logging.info("Final checkpoint backed up successfully")


if __name__ == "__main__":
    tyro.cli(main)
