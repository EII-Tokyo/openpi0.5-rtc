"""Pre-download every unique Hugging Face dataset used by a training config."""

import argparse
import concurrent.futures
import os
from pathlib import Path
import time

from huggingface_hub import snapshot_download

import openpi.training.config as _config


def _download(repo_id: str, file_workers: int) -> tuple[str, str, float]:
    start = time.monotonic()
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    local_dir = hf_home / "lerobot" / repo_id
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        max_workers=file_workers,
    )
    return repo_id, path, time.monotonic() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name")
    parser.add_argument("--repo-workers", type=int, default=8)
    parser.add_argument("--file-workers", type=int, default=8)
    args = parser.parse_args()

    repo_ids = list(dict.fromkeys(_config.get_config(args.config_name).data.repo_ids))
    print(
        f"Downloading {len(repo_ids)} unique datasets with "
        f"{args.repo_workers} concurrent repos and {args.file_workers} workers per repo",
        flush=True,
    )

    failures: list[tuple[str, BaseException]] = []
    # Each snapshot download manages its own thread pool and tqdm state. Separate
    # processes keep those globals isolated while still downloading repos in parallel.
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.repo_workers) as executor:
        futures = {executor.submit(_download, repo_id, args.file_workers): repo_id for repo_id in repo_ids}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            repo_id = futures[future]
            try:
                _, path, elapsed = future.result()
                print(f"[{completed}/{len(repo_ids)}] {repo_id} -> {path} ({elapsed:.1f}s)", flush=True)
            except BaseException as exc:
                failures.append((repo_id, exc))
                print(f"[{completed}/{len(repo_ids)}] FAILED {repo_id}: {exc!r}", flush=True)

    if failures:
        details = "\n".join(f"- {repo_id}: {exc!r}" for repo_id, exc in failures)
        raise RuntimeError(f"Failed to download {len(failures)} dataset(s):\n{details}")


if __name__ == "__main__":
    main()
