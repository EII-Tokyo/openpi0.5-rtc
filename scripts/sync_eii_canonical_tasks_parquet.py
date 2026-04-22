#!/usr/bin/env python3
"""Download, audit, and optionally fix LeRobot meta/tasks.parquet for EII datasets.

Repo IDs are parsed from _EII_DATA_SYSTEM_HUB_NO_TEAR_REPO_IDS in src/openpi/training/config.py.
Checks Hugging Face dataset revisions main and v3.0.

  .venv/bin/python scripts/sync_eii_canonical_tasks_parquet.py --dry-run
  HF_TOKEN=... .venv/bin/python scripts/sync_eii_canonical_tasks_parquet.py --apply
"""

from __future__ import annotations

import argparse
import io
import pathlib
import re
import sys
import urllib.error
import urllib.request

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_eii_repo_ids() -> list[str]:
    cfg = _REPO_ROOT / "src" / "openpi" / "training" / "config.py"
    raw = cfg.read_text(encoding="utf-8")
    m = re.search(r"_EII_DATA_SYSTEM_HUB_NO_TEAR_REPO_IDS = \[([\s\S]*?)\n\]", raw)
    if not m:
        raise RuntimeError("Could not find _EII_DATA_SYSTEM_HUB_NO_TEAR_REPO_IDS in config.py")
    return re.findall(r'"([^"]+)"', m.group(1))


import pandas as pd  # noqa: E402

CANONICAL_TASK = (
    "Do the followings: 1. If the bottle cap is facing left, rotate the bottle 180 degrees. "
    "2. Pick up the bottle. 3. Twist off the bottle cap. 4. Put the bottle into the box on the left. "
    "5. Put the cap into the box on the right. If the bottle cap falls onto the table, pick it up. "
    "6. Return to home position."
)


def _tasks_url(repo_id: str, revision: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/meta/tasks.parquet"


def download_tasks_parquet(repo_id: str, revision: str):
    url = _tasks_url(repo_id, revision)
    try:
        data = urllib.request.urlopen(url, timeout=60).read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    return pd.read_parquet(io.BytesIO(data))


def canonical_tasks_dataframe():
    return pd.DataFrame({"task_index": [0]}, index=[CANONICAL_TASK])


def tasks_need_update(df):
    if df.shape[0] != 1:
        return True, f"expected exactly 1 task row, got {df.shape[0]}"
    if "task_index" not in df.columns:
        return True, "missing column task_index"
    cur = list(df.index)[0]
    if isinstance(cur, bytes):
        cur = cur.decode("utf-8")
    if not isinstance(cur, str):
        return True, f"task index is not str (type={type(cur)})"
    if cur == CANONICAL_TASK:
        return False, "ok"
    return True, "task text differs from canonical"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-dir",
        type=pathlib.Path,
        default=_REPO_ROOT / "tmp" / "eii_tasks_parquet_audit",
        help="Where to save downloaded tasks.parquet copies.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report and save files; do not upload.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Upload fixed meta/tasks.parquet (needs HF_TOKEN with write access).",
    )
    parser.add_argument("--revisions", nargs="+", default=["main", "v3.0"])
    args = parser.parse_args()

    if args.apply and args.dry_run:
        parser.error("Use only one of --apply and --dry-run")

    repo_ids = _load_eii_repo_ids()

    args.download_dir.mkdir(parents=True, exist_ok=True)

    hf_api = None
    if args.apply:
        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            raise SystemExit("huggingface_hub is required for --apply") from e
        hf_api = HfApi()

    summary_ok = []
    summary_fix = []
    summary_skip = []

    for repo_id in repo_ids:
        safe = repo_id.replace("/", "__")
        for rev in args.revisions:
            try:
                df = download_tasks_parquet(repo_id, rev)
            except Exception as e:
                summary_skip.append((repo_id, rev, f"download error: {e}"))
                print(f"[SKIP] {repo_id} @{rev}: {e}")
                continue
            if df is None:
                summary_skip.append((repo_id, rev, "404 meta/tasks.parquet"))
                print(f"[SKIP] {repo_id} @{rev}: no tasks.parquet")
                continue

            out_path = args.download_dir / f"{safe}__{rev.replace('/', '_')}.parquet"
            df.to_parquet(out_path, index=True)
            need, reason = tasks_need_update(df)
            if not need:
                summary_ok.append((repo_id, rev))
                print(f"[OK]   {repo_id} @{rev}")
                continue

            print(f"[FIX]  {repo_id} @{rev}: {reason}")
            if df.shape[0] == 1:
                cur = list(df.index)[0]
                preview = cur if len(str(cur)) < 200 else str(cur)[:200] + "..."
                print(f"        current: {preview!r}")

            new_df = canonical_tasks_dataframe()
            new_path = args.download_dir / f"{safe}__{rev}.canonical.parquet"
            new_df.to_parquet(new_path, index=True)
            summary_fix.append((repo_id, rev, reason))

            if args.apply:
                hf_api.upload_file(
                    path_or_fileobj=str(new_path),
                    path_in_repo="meta/tasks.parquet",
                    repo_id=repo_id,
                    repo_type="dataset",
                    revision=rev,
                    commit_message="chore: set canonical twist+bottle task in tasks.parquet",
                )
                print(f"        uploaded -> {repo_id} (revision={rev})")

    print("\n=== Summary ===")
    print(f"ok (already canonical): {len(summary_ok)}")
    print(f"would fix / fixed:      {len(summary_fix)}")
    print(f"skipped:                {len(summary_skip)}")
    if args.dry_run and summary_fix:
        print("\nRe-run with HF_TOKEN and --apply to upload to Hub (main and v3.0).")


if __name__ == "__main__":
    main()
