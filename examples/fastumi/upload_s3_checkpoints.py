"""Upload selected finalized checkpoints to S3; requires AWS CLI v2 credentials."""

import hashlib
import json
import logging
from pathlib import Path
import subprocess
import tempfile
import time
from urllib.parse import urlparse

import tyro


def pending_checkpoints(root: Path, uploaded: set[int], interval: int, final_step: int) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        (
            p
            for p in root.iterdir()
            if p.is_dir()
            and p.name.isdecimal()
            and int(p.name) not in uploaded
            and int(p.name) <= final_step
            and (int(p.name) == final_step or (int(p.name) > 0 and int(p.name) % interval == 0))
            and all((p / name).is_dir() for name in ("params", "assets", "train_state"))
        ),
        key=lambda p: int(p.name),
    )


def aws(*args: str) -> str:
    return subprocess.check_output(["aws", *args, "--no-cli-pager"], text=True)


def upload_checkpoint(checkpoint: Path, destination: str, *, full: bool) -> None:
    parsed = urlparse(destination)
    bucket, prefix = parsed.netloc, parsed.path.strip("/")
    manifest = {"step": int(checkpoint.name), "full_training_state": full, "files": []}
    files = sorted(
        p
        for p in checkpoint.rglob("*")
        if p.is_file() and (full or p.relative_to(checkpoint).parts[0] in ("params", "assets"))
    )
    if not files:
        raise ValueError(f"Empty checkpoint: {checkpoint}")
    for path in files:
        relative = path.relative_to(checkpoint).as_posix()
        key = f"{prefix}/{checkpoint.name}/{relative}"
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        size = path.stat().st_size
        # AWS validates the checksum in transit. Store the full-file hash as
        # metadata too, since multipart S3 checksums use a different convention.
        aws(
            "s3",
            "cp",
            str(path),
            f"s3://{bucket}/{key}",
            "--only-show-errors",
            "--checksum-algorithm",
            "SHA256",
            "--metadata",
            f"sha256={digest}",
        )
        head = json.loads(aws("s3api", "head-object", "--bucket", bucket, "--key", key))
        if head["ContentLength"] != size or head.get("Metadata", {}).get("sha256") != digest:
            raise RuntimeError(f"S3 size/hash metadata mismatch: {key}")
        manifest["files"].append({"path": relative, "size": size, "sha256": digest})
    # A manifest is a completion marker; write it only after every file succeeds.
    with tempfile.TemporaryDirectory() as directory:
        marker = Path(directory) / "checkpoint_manifest.json"
        marker.write_text(json.dumps(manifest, indent=2) + "\n")
        aws(
            "s3",
            "cp",
            str(marker),
            f"{destination.rstrip('/')}/{checkpoint.name}/{marker.name}",
            "--only-show-errors",
            "--checksum-algorithm",
            "SHA256",
        )
    wandb_id = checkpoint.parent / "wandb_id.txt"
    if wandb_id.exists():
        aws("s3", "cp", str(wandb_id), f"{destination.rstrip('/')}/wandb_id.txt", "--only-show-errors")


def main(
    checkpoint_dir: Path,
    destination: str = "s3://openpi-tokyo/checkpoints/umi_pi05_run2_relative_pose",
    interval: int = 10_000,
    final_step: int = 29_999,
    poll_seconds: float = 60,
):
    parsed = urlparse(destination)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("Supply an S3 bucket and a dedicated run prefix")
    if interval <= 0 or poll_seconds <= 0:
        raise ValueError("interval and poll_seconds must be positive")
    destination = destination.rstrip("/")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ledger = checkpoint_dir.parent / f"{checkpoint_dir.name}.s3-uploads.json"
    record = json.loads(ledger.read_text()) if ledger.exists() else {"destination": destination, "steps": []}
    if record["destination"] != destination:
        raise ValueError(f"Upload ledger belongs to another destination: {ledger}")
    uploaded = set(record["steps"])
    while final_step not in uploaded:
        for checkpoint in pending_checkpoints(checkpoint_dir, uploaded, interval, final_step):
            step = int(checkpoint.name)
            try:
                logging.info("Uploading %s -> %s", checkpoint, destination)
                upload_checkpoint(checkpoint, destination, full=step == final_step)
                uploaded.add(step)
                ledger.parent.mkdir(parents=True, exist_ok=True)
                temporary = ledger.with_suffix(".tmp")
                temporary.write_text(json.dumps({"destination": destination, "steps": sorted(uploaded)}, indent=2))
                temporary.replace(ledger)
                logging.info("Verified backup for step %s", step)
            except Exception:
                logging.exception("Backup failed; retrying after the polling interval")
                break
        if final_step not in uploaded:
            time.sleep(poll_seconds)
    logging.info("Final checkpoint backup complete")


if __name__ == "__main__":
    tyro.cli(main)
