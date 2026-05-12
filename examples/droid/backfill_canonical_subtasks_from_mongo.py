from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from huggingface_hub import HfApi
import numpy as np
import pandas as pd
import tyro

from examples.droid.canonical_lerobot import HF_LEROBOT_HOME
from examples.droid.canonical_lerobot import SUBTASKS_PATH
from examples.droid.canonical_lerobot import SubtaskRegistry
from examples.droid.canonical_lerobot import normalize_datetime
from examples.droid.canonical_lerobot import resolve_subtask_for_frame
from examples.droid.droid_metadata import HOUSEKEEPING_SLICE_LABELS
from examples.droid.droid_mongo import ReadOnlyDroidMongo


EPISODE_STATS_PATH = Path("meta/episodes")
DATA_PATH = Path("data")
MIGRATION_REPORT_PATH = Path("meta/episode_migration.parquet")
STATS_JSON_PATH = Path("meta/stats.json")
CONVERSION_REPORT_PATH = Path("meta/conversion_report.json")


@dataclass
class BackfillSubtasksConfig:
    repo_id: str
    root: Path | None = None
    mongo_url: str = "mongodb://localhost:27017"
    mongo_db_name: str = "eii_data_system"
    mongo_project_path_filters: list[str] | None = None
    mongo_project_date_filters: list[str] | None = None
    dry_run: bool = False
    push_to_hub: bool = False


def _resolve_root(repo_id: str, root: Path | None) -> Path:
    return root if root is not None else HF_LEROBOT_HOME / repo_id


def _numeric_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Cannot compute stats for an empty array.")
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "count": int(arr.size),
        "q01": float(np.quantile(arr, 0.01)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "q99": float(np.quantile(arr, 0.99)),
    }


def _build_subtasks_from_slices(slices: list[Any]) -> list[dict[str, Any]]:
    subtasks: list[dict[str, Any]] = []
    for slice_item in slices:
        label = str(slice_item.label or "").strip()
        if not label or label in HOUSEKEEPING_SLICE_LABELS:
            continue
        subtasks.append(
            {
                "start_frame": int(slice_item.start_index),
                "end_frame": int(slice_item.end_index) + 1,
                "subtask": label,
            }
        )
    return subtasks


def _load_episode_lengths(root: Path) -> dict[int, int]:
    lengths: dict[int, int] = {}
    for parquet_path in sorted((root / EPISODE_STATS_PATH).rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            lengths[int(row["episode_index"])] = int(row["length"])
    return lengths


def _load_migration(root: Path) -> pd.DataFrame:
    path = root / MIGRATION_REPORT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Migration report not found: {path}")
    return pd.read_parquet(path)


def _resolve_candidate_datetime(row: pd.Series) -> str | None:
    for value in (row.get("datetime"), row.get("mongo_source_folder_name")):
        if pd.isna(value) or value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        return normalize_datetime(text)
    return None


def _resolve_date_filters(
    migration_df: pd.DataFrame,
    explicit_filters: list[str] | None,
) -> list[str] | None:
    if explicit_filters:
        return explicit_filters
    if "datetime" not in migration_df.columns:
        return None
    values = migration_df["datetime"].dropna().astype(str).tolist()
    dates = sorted({value[:10] for value in values if len(value) >= 10})
    return dates or None


def _pick_episode_id_from_migration_row(row: pd.Series, episodes_by_datetime: dict[str, list[Any]]) -> str | None:
    mongo_episode_id = row.get("mongo_episode_id")
    if not pd.isna(mongo_episode_id) and str(mongo_episode_id).strip():
        return str(mongo_episode_id)

    candidate_datetime = _resolve_candidate_datetime(row)
    if candidate_datetime is None:
        return None
    matches = episodes_by_datetime.get(candidate_datetime, [])
    if len(matches) == 1:
        return matches[0].id
    return None


def _build_episode_subtask_indices(
    root: Path,
    *,
    mongo_url: str,
    mongo_db_name: str,
    mongo_project_path_filters: list[str] | None,
    mongo_project_date_filters: list[str] | None,
) -> tuple[dict[int, np.ndarray], SubtaskRegistry]:
    episode_lengths = _load_episode_lengths(root)
    migration_df = _load_migration(root)
    registry = SubtaskRegistry(root)

    mongo = ReadOnlyDroidMongo.connect(mongo_url, db_name=mongo_db_name)
    resolved_date_filters = _resolve_date_filters(migration_df, mongo_project_date_filters)
    metadata_index = mongo.build_export_index(
        path_substrings=mongo_project_path_filters,
        date_substrings=resolved_date_filters,
    )
    unique_episode_ids = {
        episode_id
        for episode_id in (
            _pick_episode_id_from_migration_row(row, metadata_index.episodes_by_datetime)
            for _, row in migration_df.iterrows()
        )
        if episode_id
    }
    slices_by_mongo_id = {episode_id: mongo.get_slices_for_episode(episode_id) for episode_id in unique_episode_ids}

    episode_to_indices: dict[int, np.ndarray] = {}
    for _, row in migration_df.iterrows():
        destination_episode_index = row.get("destination_episode_index")
        if pd.isna(destination_episode_index):
            continue
        episode_index = int(destination_episode_index)
        episode_length = episode_lengths.get(episode_index)
        if episode_length is None:
            raise KeyError(f"Missing episode length for episode_index={episode_index}")

        mongo_episode_id = _pick_episode_id_from_migration_row(row, metadata_index.episodes_by_datetime)
        if mongo_episode_id is None:
            label = "unknown"
            label_index = registry.get_or_add(label)
            episode_to_indices[episode_index] = np.full((episode_length,), label_index, dtype=np.int64)
            continue

        subtasks = _build_subtasks_from_slices(slices_by_mongo_id[str(mongo_episode_id)])
        annotation = SimpleNamespace(subtasks=subtasks)
        frame_indices = np.empty((episode_length,), dtype=np.int64)
        for frame_index in range(episode_length):
            label = resolve_subtask_for_frame(annotation, frame_index=frame_index, episode_length=episode_length)
            frame_indices[frame_index] = registry.get_or_add(label)
        episode_to_indices[episode_index] = frame_indices

    return episode_to_indices, registry


def _update_data_parquets(
    root: Path,
    episode_to_indices: dict[int, np.ndarray],
    *,
    dry_run: bool,
) -> np.ndarray:
    all_subtask_values: list[np.ndarray] = []
    for parquet_path in sorted((root / DATA_PATH).rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        updated = False
        for episode_index, group_index in df.groupby("episode_index").groups.items():
            episode_index_int = int(episode_index)
            indices = episode_to_indices.get(episode_index_int)
            if indices is None:
                continue
            frame_ids = df.loc[group_index, "frame_index"].to_numpy(dtype=np.int64)
            new_values = indices[frame_ids]
            old_values = df.loc[group_index, "subtask_index"].to_numpy(dtype=np.int64)
            if not np.array_equal(old_values, new_values):
                df.loc[group_index, "subtask_index"] = new_values
                updated = True
            all_subtask_values.append(new_values)
        if updated and not dry_run:
            df.to_parquet(parquet_path, index=False)
    if not all_subtask_values:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(all_subtask_values)


def _update_episode_stats_parquets(
    root: Path,
    episode_to_indices: dict[int, np.ndarray],
    *,
    dry_run: bool,
) -> None:
    stats_fields = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
    for parquet_path in sorted((root / EPISODE_STATS_PATH).rglob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        updated = False
        for row_index, row in df.iterrows():
            episode_index = int(row["episode_index"])
            indices = episode_to_indices.get(episode_index)
            if indices is None:
                continue
            stats = _numeric_stats(indices)
            for field in stats_fields:
                column = f"stats/subtask_index/{field}"
                if column in df.columns and df.at[row_index, column] != stats[field]:
                    df.at[row_index, column] = stats[field]
                    updated = True
        if updated and not dry_run:
            df.to_parquet(parquet_path, index=False)


def _update_stats_json(root: Path, all_subtask_values: np.ndarray, *, dry_run: bool) -> None:
    if all_subtask_values.size == 0:
        return
    path = root / STATS_JSON_PATH
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    stats = _numeric_stats(all_subtask_values)
    payload["subtask_index"] = {key: [value] for key, value in stats.items()}
    if not dry_run:
        path.write_text(json.dumps(payload, indent=2))


def _update_conversion_report(root: Path, registry: SubtaskRegistry, all_subtask_values: np.ndarray, *, dry_run: bool) -> None:
    path = root / CONVERSION_REPORT_PATH
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    unknown_index = registry.label_to_index.get("unknown", 0)
    payload["subtask_vocab_size"] = len(registry.index_to_label)
    payload["frames_unknown_subtask"] = int(np.sum(all_subtask_values == unknown_index)) if all_subtask_values.size else 0
    if not dry_run:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main(config: BackfillSubtasksConfig) -> None:
    root = _resolve_root(config.repo_id, config.root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    episode_to_indices, registry = _build_episode_subtask_indices(
        root,
        mongo_url=config.mongo_url,
        mongo_db_name=config.mongo_db_name,
        mongo_project_path_filters=config.mongo_project_path_filters,
        mongo_project_date_filters=config.mongo_project_date_filters,
    )
    logging.info("Loaded subtask mappings for %s episodes", len(episode_to_indices))

    all_subtask_values = _update_data_parquets(root, episode_to_indices, dry_run=config.dry_run)
    _update_episode_stats_parquets(root, episode_to_indices, dry_run=config.dry_run)
    _update_stats_json(root, all_subtask_values, dry_run=config.dry_run)
    _update_conversion_report(root, registry, all_subtask_values, dry_run=config.dry_run)
    if not config.dry_run:
        registry.write()

    logging.info(
        "Backfilled subtasks for %s frames, vocabulary size=%s%s",
        int(all_subtask_values.size),
        len(registry.index_to_label),
        " (dry run)" if config.dry_run else "",
    )

    if config.push_to_hub:
        if config.dry_run:
            raise ValueError("Cannot push to hub in dry-run mode.")
        HfApi().upload_folder(
            folder_path=str(root),
            repo_id=config.repo_id,
            repo_type="dataset",
            commit_message="feat: backfill canonical subtask annotations from mongo",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main(tyro.cli(BackfillSubtasksConfig))
