import argparse
from pathlib import Path

import pandas as pd


def rebuild(repo_id: str, lerobot_root: Path) -> None:
    root = lerobot_root / repo_id
    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data parquet files found under {root / 'data'}")

    tasks_df = pd.read_parquet(root / "meta" / "tasks.parquet")
    task_names_by_index = {int(v): str(k) for k, v in tasks_df["task_index"].to_dict().items()}

    episode_rows: list[dict] = []
    for data_file in data_files:
        rel = data_file.relative_to(root).as_posix()
        parts = rel.split("/")
        chunk_index = int(parts[1].split("-")[1])
        file_index = int(parts[2].split("-")[1].split(".")[0])

        df = pd.read_parquet(data_file, columns=["episode_index", "task_index", "index"])
        for episode_index, group in df.groupby("episode_index", sort=True):
            task_indices = pd.unique(group["task_index"]).tolist()
            episode_rows.append(
                {
                    "episode_index": int(episode_index),
                    "tasks": [task_names_by_index[int(idx)] for idx in task_indices if int(idx) in task_names_by_index],
                    "length": int(len(group)),
                    "data/chunk_index": chunk_index,
                    "data/file_index": file_index,
                    "dataset_from_index": int(group["index"].min()),
                    "dataset_to_index": int(group["index"].max()) + 1,
                    "meta/episodes/chunk_index": 0,
                    "meta/episodes/file_index": 0,
                }
            )

    episode_rows.sort(key=lambda row: row["episode_index"])
    out_dir = root / "meta" / "episodes" / "chunk-000"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "file-000.parquet"
    pd.DataFrame(episode_rows).to_parquet(out_path, index=False)
    print(f"Rebuilt {len(episode_rows)} episodes at {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--lerobot-root", default=str(Path.home() / ".cache" / "huggingface" / "lerobot"))
    args = parser.parse_args()
    rebuild(args.repo_id, Path(args.lerobot_root))


if __name__ == "__main__":
    main()
