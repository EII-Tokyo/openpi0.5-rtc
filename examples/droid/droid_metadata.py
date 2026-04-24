from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any

from examples.droid.canonical_lerobot import normalize_datetime
from examples.droid.canonical_lerobot import normalize_episode_identity
from examples.droid.canonical_lerobot import parse_uuid
from examples.droid.droid_mongo import DroidMongoEpisode
from examples.droid.droid_mongo import DroidMongoExportIndex

HOUSEKEEPING_SLICE_LABELS = {
    "need to be truncated",
    "bad action",
    "good action",
}


def parse_speed_value(prompts: dict[str, Any] | None) -> float | None:
    if not isinstance(prompts, dict):
        return None
    raw = prompts.get("speed")
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        match = re.search(r"speed\s*:\s*([-+]?\d*\.?\d+)", text, flags=re.IGNORECASE)
        if match is None:
            match = re.search(r"([-+]?\d*\.?\d+)\s*(?:m/s|mps)\b", text, flags=re.IGNORECASE)
        if match is None:
            match = re.fullmatch(r"([-+]?\d*\.?\d+)", text)
        if match is None:
            return None
        return float(match.group(1))


@dataclass(frozen=True)
class MongoBackedEpisodeMetadata:
    uuid: str | None
    task: str | None
    subtasks: list[dict[str, Any]]
    conveyor_speed: float | None
    building: str
    collector_id: str
    datetime_value: str
    mongo_episode_id: str | None
    mongo_project_id: str | None
    match_status: str
    match_candidates: tuple[str, ...] = ()
    source_folder_name: str | None = None

    @property
    def identity(self) -> tuple[str | None, str | None, str | None]:
        return parse_uuid(self.uuid)


def _build_subtasks(index: DroidMongoExportIndex, mongo_episode: DroidMongoEpisode | None) -> list[dict[str, Any]]:
    if mongo_episode is None:
        return []
    slices = index.slices_by_episode_id.get(mongo_episode.id, [])
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


def _match_mongo_episode(index: DroidMongoExportIndex, candidate_datetimes: list[str]) -> tuple[DroidMongoEpisode | None, str, tuple[str, ...]]:
    matches: list[DroidMongoEpisode] = []
    seen_ids: set[str] = set()
    for datetime_value in candidate_datetimes:
        for episode in index.episodes_by_datetime.get(datetime_value, []):
            if episode.id in seen_ids:
                continue
            seen_ids.add(episode.id)
            matches.append(episode)
    if not matches:
        return None, "unmatched", ()
    if len(matches) > 1:
        return None, "ambiguous", tuple(episode.id for episode in matches)
    return matches[0], "matched", ()


def _match_mongo_episode_by_task_length(
    index: DroidMongoExportIndex,
    *,
    task: str,
    episode_length: int,
) -> tuple[DroidMongoEpisode | None, str, tuple[str, ...]]:
    matches = index.episodes_by_task_length.get((task, int(episode_length)), [])
    if not matches:
        return None, "unmatched", ()
    if len(matches) > 1:
        return None, "ambiguous_task_length", tuple(episode.id for episode in matches)
    return matches[0], "matched_task_length", ()


def _build_candidate_datetimes(
    *,
    folder_name: str | None = None,
    source_datetime: str | None = None,
    source_date: str | None = None,
    annotation_uuid: str | None = None,
) -> list[str]:
    candidates: list[str] = []
    for raw_value in (folder_name, source_datetime, source_date):
        if raw_value:
            normalized = normalize_datetime(raw_value)
            if normalized not in candidates:
                candidates.append(normalized)
    _, _, uuid_datetime = parse_uuid(annotation_uuid)
    if uuid_datetime and uuid_datetime not in candidates:
        candidates.append(uuid_datetime)
    return candidates


def _base_identity(
    *,
    building: str | None,
    collector_id: str | None,
    datetime_value: str | None,
    date_value: str | None,
    uuid: str | None,
    fallback_prefix: str,
) -> tuple[str, str, str]:
    return normalize_episode_identity(
        building=building,
        collector_id=collector_id,
        datetime_value=datetime_value,
        date_value=date_value,
        uuid=uuid,
        fallback_prefix=fallback_prefix,
    )


def resolve_raw_episode_metadata(
    *,
    index: DroidMongoExportIndex | None,
    folder_name: str,
    annotation: Any = None,
    fallback_task: str = "Do something",
) -> MongoBackedEpisodeMetadata:
    annotation_uuid = getattr(annotation, "uuid", None)
    building, collector_id, datetime_value = _base_identity(
        building=None,
        collector_id=None,
        datetime_value=folder_name,
        date_value=None,
        uuid=annotation_uuid,
        fallback_prefix=folder_name,
    )
    candidate_datetimes = _build_candidate_datetimes(folder_name=folder_name, annotation_uuid=annotation_uuid)
    mongo_episode, match_status, match_candidates = _match_mongo_episode(index, candidate_datetimes) if index else (None, "unmatched", ())
    if match_status == "ambiguous":
        logging.warning("Ambiguous Mongo match for raw episode %s: %s", folder_name, ", ".join(match_candidates))
    task = fallback_task
    if mongo_episode is not None and isinstance(mongo_episode.prompts, dict):
        task = str(mongo_episode.prompts.get("default") or fallback_task)
    elif annotation is not None and getattr(annotation, "task", None):
        task = str(annotation.task)
    speed = parse_speed_value(mongo_episode.prompts if mongo_episode is not None else None)
    return MongoBackedEpisodeMetadata(
        uuid=annotation_uuid,
        task=task,
        subtasks=_build_subtasks(index, mongo_episode) if index else [],
        conveyor_speed=speed,
        building=building,
        collector_id=collector_id,
        datetime_value=datetime_value,
        mongo_episode_id=None if mongo_episode is None else mongo_episode.id,
        mongo_project_id=None if mongo_episode is None else mongo_episode.project_id,
        match_status=match_status,
        match_candidates=match_candidates,
        source_folder_name=folder_name,
    )


def resolve_legacy_episode_metadata(
    *,
    index: DroidMongoExportIndex | None,
    source_frame: dict[str, Any],
    source_repo_id: str,
    source_episode_index: int,
    source_task: str,
    source_length: int | None = None,
    annotation: Any = None,
    allow_task_length_fallback: bool = False,
) -> MongoBackedEpisodeMetadata:
    annotation_uuid = getattr(annotation, "uuid", None)
    source_datetime = source_frame.get("datetime")
    source_date = source_frame.get("date")
    building, collector_id, datetime_value = _base_identity(
        building=source_frame.get("building"),
        collector_id=source_frame.get("collector_id"),
        datetime_value=source_datetime,
        date_value=source_date,
        uuid=annotation_uuid,
        fallback_prefix=f"{source_repo_id}-episode-{source_episode_index:06d}",
    )
    candidate_datetimes = _build_candidate_datetimes(
        source_datetime=source_datetime,
        source_date=source_date,
        annotation_uuid=annotation_uuid,
    )
    mongo_episode, match_status, match_candidates = _match_mongo_episode(index, candidate_datetimes) if index else (None, "unmatched", ())
    if (
        index is not None
        and mongo_episode is None
        and match_status == "unmatched"
        and allow_task_length_fallback
        and source_length is not None
    ):
        mongo_episode, match_status, match_candidates = _match_mongo_episode_by_task_length(
            index,
            task=source_task,
            episode_length=source_length,
        )
    if match_status == "ambiguous":
        logging.warning(
            "Ambiguous Mongo match for legacy %s episode %s: %s",
            source_repo_id,
            source_episode_index,
            ", ".join(match_candidates),
        )
    if match_status == "ambiguous_task_length":
        logging.warning(
            "Ambiguous task/length fallback match for legacy %s episode %s: %s",
            source_repo_id,
            source_episode_index,
            ", ".join(match_candidates),
        )
    if mongo_episode is not None and mongo_episode.datetime_value:
        datetime_value = mongo_episode.datetime_value
    task = source_task
    if mongo_episode is not None and isinstance(mongo_episode.prompts, dict):
        task = str(mongo_episode.prompts.get("default") or source_task)
    elif annotation is not None and getattr(annotation, "task", None):
        task = str(annotation.task)
    speed = parse_speed_value(mongo_episode.prompts if mongo_episode is not None else None)
    return MongoBackedEpisodeMetadata(
        uuid=annotation_uuid,
        task=task,
        subtasks=_build_subtasks(index, mongo_episode) if index else [],
        conveyor_speed=speed,
        building=building,
        collector_id=collector_id,
        datetime_value=datetime_value,
        mongo_episode_id=None if mongo_episode is None else mongo_episode.id,
        mongo_project_id=None if mongo_episode is None else mongo_episode.project_id,
        match_status=match_status,
        match_candidates=match_candidates,
        source_folder_name=None if mongo_episode is None else mongo_episode.folder_name,
    )
