import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


def build_readme(info: dict) -> str:
    info_json = json.dumps(info, ensure_ascii=False, indent=4)
    return f"""---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*/*.parquet
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{info_json}
```
"""


def upload_episode_metadata(api: HfApi, repo_id: str, repo_path: Path) -> None:
    episodes_root = repo_path / "meta" / "episodes"
    if not episodes_root.exists():
        return

    for file_path in sorted(episodes_root.rglob("*.parquet")):
        rel_path = file_path.relative_to(repo_path).as_posix()
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=rel_path,
            path_or_fileobj=str(file_path),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--local-root", default=str(Path.home() / ".cache/huggingface/lerobot"))
    args = parser.parse_args()

    repo_path = Path(args.local_root) / args.repo_id
    info_path = repo_path / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    readme = build_readme(info)
    api = HfApi()
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_in_repo="README.md",
        path_or_fileobj=readme.encode("utf-8"),
    )
    upload_episode_metadata(api, args.repo_id, repo_path)
    print(f"Updated README for {args.repo_id}")


if __name__ == "__main__":
    main()
