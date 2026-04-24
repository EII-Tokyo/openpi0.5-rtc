from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import importlib
from typing import Any

from examples.droid.canonical_lerobot import normalize_datetime


def _normalize_doc_id(doc: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(doc)
    doc_id = normalized.pop("_id", None)
    if doc_id is not None:
        normalized["id"] = str(doc_id)
    return normalized


def _collection_find(collection: Any, query: dict[str, Any], projection: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [_normalize_doc_id(doc) for doc in collection.find(query, projection)]


def _matches_path_filters(s3_address: str, path_substrings: Iterable[str] | None, date_substrings: Iterable[str] | None) -> bool:
    if path_substrings:
        path_terms = [term for term in path_substrings if term]
        if path_terms and not any(term in s3_address for term in path_terms):
            return False
    if date_substrings:
        date_terms = [term for term in date_substrings if term]
        if date_terms and not any(term in s3_address for term in date_terms):
            return False
    return True


@dataclass(frozen=True)
class DroidMongoSlice:
    id: str
    episode_id: str
    start_index: int
    end_index: int
    label: str
    sub_label: str | None = None
    created_at: str | None = None
    is_success: bool | None = None
    status: str | None = None
    bottle_segment_index: int | None = None
    tags: list[str] | None = None
    s3_key: str | None = None
    s3_bucket: str | None = None
    s3_path: str | None = None
    completed_at: str | None = None
    skip_reason: str | None = None


@dataclass(frozen=True)
class DroidMongoEpisode:
    id: str
    project_id: str
    episode_index: int
    folder_name: str | None
    prompts: dict[str, Any] | None
    s3_address: str | None
    trajectory_s3_path: str | None
    metadata: dict[str, Any] | None
    data_count: int
    datetime_value: str | None


@dataclass(frozen=True)
class DroidMongoProject:
    id: str
    name: str | None
    s3_address: str | None
    created_at: str | None


@dataclass(frozen=True)
class DroidMongoExportIndex:
    projects: list[DroidMongoProject]
    episodes: list[DroidMongoEpisode]
    slices_by_episode_id: dict[str, list[DroidMongoSlice]]
    episodes_by_datetime: dict[str, list[DroidMongoEpisode]]
    episodes_by_task_length: dict[tuple[str, int], list[DroidMongoEpisode]]


class ReadOnlyDroidMongo:
    def __init__(self, db: Any):
        self._db = db

    @classmethod
    def connect(cls, mongo_url: str, db_name: str | None = None) -> ReadOnlyDroidMongo:
        try:
            pymongo = importlib.import_module("pymongo")
        except ImportError as exc:
            raise ImportError("pymongo is required to read DROID metadata from MongoDB.") from exc

        client = pymongo.MongoClient(mongo_url)
        if db_name is not None:
            db = client[db_name]
        else:
            default_db = client.get_default_database()
            db = default_db if default_db is not None else client["eii_data_system"]
        return cls(db)

    def find_droid_projects(
        self,
        *,
        path_substrings: Iterable[str] | None = None,
        date_substrings: Iterable[str] | None = None,
    ) -> list[DroidMongoProject]:
        docs = _collection_find(
            self._db["projects"],
            {"type": "droid", "is_deleted": {"$ne": True}},
            {"name": 1, "s3_address": 1, "created_at": 1},
        )
        projects: list[DroidMongoProject] = []
        for payload in docs:
            s3_address = str(payload.get("s3_address") or "")
            if not _matches_path_filters(s3_address, path_substrings, date_substrings):
                continue
            projects.append(
                DroidMongoProject(
                    id=str(payload["id"]),
                    name=payload.get("name"),
                    s3_address=payload.get("s3_address"),
                    created_at=payload.get("created_at"),
                )
            )
        return projects

    def get_droid_episodes(self, project_id: str) -> list[DroidMongoEpisode]:
        docs = _collection_find(
            self._db["episodes"],
            {"type": "droid_episode", "project_id": project_id},
            {
                "project_id": 1,
                "episode_index": 1,
                "folder_name": 1,
                "prompts": 1,
                "s3_address": 1,
                "trajectory_s3_path": 1,
                "metadata": 1,
                "data_count": 1,
            },
        )
        episodes: list[DroidMongoEpisode] = []
        for payload in docs:
            folder_name = payload.get("folder_name")
            datetime_value = normalize_datetime(folder_name) if folder_name else None
            episodes.append(
                DroidMongoEpisode(
                    id=str(payload["id"]),
                    project_id=str(payload["project_id"]),
                    episode_index=int(payload.get("episode_index") or 0),
                    folder_name=folder_name,
                    prompts=payload.get("prompts") if isinstance(payload.get("prompts"), dict) else None,
                    s3_address=payload.get("s3_address"),
                    trajectory_s3_path=payload.get("trajectory_s3_path"),
                    metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
                    data_count=int(payload.get("data_count") or 0),
                    datetime_value=datetime_value,
                )
            )
        return episodes

    def get_slices_for_episode(self, episode_id: str) -> list[DroidMongoSlice]:
        docs = _collection_find(
            self._db["slices"],
            {"type": "episode_slice", "episode_id": episode_id},
            {
                "episode_id": 1,
                "start_index": 1,
                "end_index": 1,
                "label": 1,
                "sub_label": 1,
                "created_at": 1,
                "is_success": 1,
                "status": 1,
                "bottle_segment_index": 1,
                "tags": 1,
                "s3_key": 1,
                "s3_bucket": 1,
                "s3_path": 1,
                "completed_at": 1,
                "skip_reason": 1,
            },
        )
        slices: list[DroidMongoSlice] = []
        for payload in docs:
            start_index = payload.get("start_index")
            end_index = payload.get("end_index")
            label = payload.get("label")
            if start_index is None or end_index is None or label is None:
                continue
            tags = payload.get("tags")
            slices.append(
                DroidMongoSlice(
                    id=str(payload["id"]),
                    episode_id=str(payload["episode_id"]),
                    start_index=int(start_index),
                    end_index=int(end_index),
                    label=str(label),
                    sub_label=payload.get("sub_label"),
                    created_at=payload.get("created_at"),
                    is_success=payload.get("is_success"),
                    status=payload.get("status"),
                    bottle_segment_index=payload.get("bottle_segment_index"),
                    tags=list(tags) if isinstance(tags, list) else None,
                    s3_key=payload.get("s3_key"),
                    s3_bucket=payload.get("s3_bucket"),
                    s3_path=payload.get("s3_path"),
                    completed_at=payload.get("completed_at"),
                    skip_reason=payload.get("skip_reason"),
                )
            )
        return sorted(slices, key=lambda item: (item.start_index, item.end_index, item.created_at or "", item.id))

    def build_export_index(
        self,
        *,
        path_substrings: Iterable[str] | None = None,
        date_substrings: Iterable[str] | None = None,
    ) -> DroidMongoExportIndex:
        projects = self.find_droid_projects(path_substrings=path_substrings, date_substrings=date_substrings)
        episodes: list[DroidMongoEpisode] = []
        slices_by_episode_id: dict[str, list[DroidMongoSlice]] = {}
        episodes_by_datetime: dict[str, list[DroidMongoEpisode]] = {}
        episodes_by_task_length: dict[tuple[str, int], list[DroidMongoEpisode]] = {}

        for project in projects:
            project_episodes = self.get_droid_episodes(project.id)
            episodes.extend(project_episodes)
            for episode in project_episodes:
                slices_by_episode_id[episode.id] = self.get_slices_for_episode(episode.id)
                if episode.datetime_value:
                    episodes_by_datetime.setdefault(episode.datetime_value, []).append(episode)
                task = ""
                if isinstance(episode.prompts, dict):
                    task = str(episode.prompts.get("default") or "")
                episodes_by_task_length.setdefault((task, int(episode.data_count)), []).append(episode)

        return DroidMongoExportIndex(
            projects=projects,
            episodes=episodes,
            slices_by_episode_id=slices_by_episode_id,
            episodes_by_datetime=episodes_by_datetime,
            episodes_by_task_length=episodes_by_task_length,
        )
