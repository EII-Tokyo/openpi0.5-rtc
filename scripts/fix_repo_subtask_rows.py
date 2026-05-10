#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import tempfile

import pandas as pd
from huggingface_hub import HfApi, snapshot_download


def _v3_is_tag_only(hf_api: HfApi, repo_id: str) -> bool:
    refs = hf_api.list_repo_refs(repo_id, repo_type="dataset")
    tags = {t.name for t in refs.tags}
    branches = {b.name for b in refs.branches}
    return "v3.0" in tags and "v3.0" not in branches


def _upload_replace_v3_tag(
    hf_api: HfApi,
    repo_id: str,
    *,
    local_file: pathlib.Path,
    path_in_repo: str,
    commit_message: str,
) -> None:
    tag = "v3.0"
    tmp = "__eii_tmp_v3_subtask_fix"
    base_sha = hf_api.repo_info(repo_id, repo_type="dataset", revision=tag).sha
    try:
        hf_api.delete_branch(repo_id, branch=tmp, repo_type="dataset")
    except Exception:
        pass
    hf_api.create_branch(repo_id, branch=tmp, revision=base_sha, repo_type="dataset")
    try:
        hf_api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            revision=tmp,
            commit_message=commit_message,
        )
        tip = hf_api.repo_info(repo_id, repo_type="dataset", revision=tmp).sha
        hf_api.delete_tag(repo_id, tag=tag, repo_type="dataset")
        hf_api.create_tag(repo_id, tag=tag, revision=tip, repo_type="dataset", exist_ok=False)
    finally:
        try:
            hf_api.delete_branch(repo_id, branch=tmp, repo_type="dataset")
        except Exception:
            pass


def _patch_parquet_file(parquet_path: pathlib.Path) -> tuple[pathlib.Path, int]:
    df = pd.read_parquet(parquet_path)
    if "subtask" not in df.columns:
        return parquet_path, 0

    changed = 0
    new_values = []
    for raw in df["subtask"].tolist():
        obj = json.loads(raw) if isinstance(raw, str) else raw
        if (
            isinstance(obj, dict)
            and obj.get("bottle_state") == "Bottle label is still attached"
            and obj.get("subtask") == "Tear off label"
        ):
            obj = dict(obj)
            obj["bottle_state"] = "Bottle position is incorrect"
            obj["subtask"] = "Adjust bottle position"
            changed += 1
        new_values.append(json.dumps(obj, ensure_ascii=False) if isinstance(raw, str) else obj)

    if changed == 0:
        return parquet_path, 0

    patched = parquet_path.with_suffix(".patched.parquet")
    df = df.copy()
    df["subtask"] = new_values
    df.to_parquet(patched, index=False)
    return patched, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--revisions", nargs="+", default=["main", "v3.0"])
    args = parser.parse_args()

    hf_api = HfApi()
    commit_message = (
        "fix: replace erroneous Tear off label rows with Adjust bottle position"
    )

    for revision in args.revisions:
        local_dir = pathlib.Path(
            snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                revision=revision,
                allow_patterns=["data/**/*.parquet"],
            )
        )
        print(f"[{revision}] downloaded: {local_dir}")
        total_changed = 0
        patched_files: list[tuple[pathlib.Path, pathlib.Path]] = []
        for parquet_path in sorted(local_dir.glob("data/**/*.parquet")):
            patched, changed = _patch_parquet_file(parquet_path)
            if changed:
                total_changed += changed
                patched_files.append((parquet_path, patched))
                print(f"[{revision}] changed {changed} rows in {parquet_path.relative_to(local_dir)}")
        print(f"[{revision}] total_changed={total_changed}")

        if not args.apply or total_changed == 0:
            continue

        for original, patched in patched_files:
            rel = original.relative_to(local_dir).as_posix()
            if revision == "v3.0" and _v3_is_tag_only(hf_api, args.repo_id):
                _upload_replace_v3_tag(
                    hf_api,
                    args.repo_id,
                    local_file=patched,
                    path_in_repo=rel,
                    commit_message=commit_message,
                )
            else:
                hf_api.upload_file(
                    path_or_fileobj=str(patched),
                    path_in_repo=rel,
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    revision=revision,
                    commit_message=commit_message,
                )
            print(f"[{revision}] uploaded {rel}")


if __name__ == "__main__":
    main()
