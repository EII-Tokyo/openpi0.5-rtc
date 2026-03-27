from __future__ import annotations

from typing import Any

import numpy as np

from examples.droid.canonical_lerobot import compute_env_stats_for_frame
from examples.droid.canonical_lerobot import resolve_subtask_for_frame
from examples.droid.convert_legacy_lerobot_to_canonical import _map_legacy_frame
from examples.droid.convert_raw_droid_to_canonical_lerobot import _build_raw_frame
from examples.droid.droid_metadata import HOUSEKEEPING_SLICE_LABELS
from examples.droid.droid_metadata import parse_speed_value
from examples.droid.droid_metadata import resolve_legacy_episode_metadata
from examples.droid.droid_metadata import resolve_raw_episode_metadata
from examples.droid.droid_mongo import ReadOnlyDroidMongo


class _FakeCollection:
    def __init__(self, docs: list[dict[str, Any]]):
        self._docs = docs

    def find(self, query: dict[str, Any], projection: dict[str, Any] | None = None):
        def matches(doc: dict[str, Any]) -> bool:
            for key, value in query.items():
                if isinstance(value, dict) and "$ne" in value:
                    if doc.get(key) == value["$ne"]:
                        return False
                    continue
                if doc.get(key) != value:
                    return False
            return True

        rows = [doc for doc in self._docs if matches(doc)]
        if projection is None:
            return rows
        projected = []
        for doc in rows:
            payload = {"_id": doc["_id"]}
            for key, enabled in projection.items():
                if enabled and key in doc:
                    payload[key] = doc[key]
            projected.append(payload)
        return projected


class _FakeDb:
    def __init__(self, collections: dict[str, _FakeCollection]):
        self._collections = collections

    def __getitem__(self, key: str):
        return self._collections[key]


def _make_index():
    db = _FakeDb(
        {
            "projects": _FakeCollection(
                [
                    {
                        "_id": "project-a",
                        "type": "droid",
                        "is_deleted": False,
                        "name": "Droid Project",
                        "s3_address": "s3://bucket/droid_xxjd_data/success/2026-02-02",
                        "created_at": "2026-02-03T00:00:00",
                    }
                ]
            ),
            "episodes": _FakeCollection(
                [
                    {
                        "_id": "episode-a",
                        "type": "droid_episode",
                        "project_id": "project-a",
                        "episode_index": 7,
                        "folder_name": "Sun_Feb_2_12:34:56_2026",
                        "prompts": {"default": "Put the bottle in the bin", "speed": "0.05"},
                        "s3_address": "s3://bucket/droid_xxjd_data/success/2026-02-02/Sun_Feb_2_12:34:56_2026",
                        "trajectory_s3_path": "s3://bucket/traj.npz",
                        "metadata": {},
                        "data_count": 7,
                    }
                ]
            ),
            "slices": _FakeCollection(
                [
                    {
                        "_id": "slice-1",
                        "type": "episode_slice",
                        "episode_id": "episode-a",
                        "start_index": 0,
                        "end_index": 2,
                        "label": "reach",
                        "status": "pending",
                    },
                    {
                        "_id": "slice-2",
                        "type": "episode_slice",
                        "episode_id": "episode-a",
                        "start_index": 2,
                        "end_index": 4,
                        "label": "place",
                        "status": "failed",
                    },
                    {
                        "_id": "slice-3",
                        "type": "episode_slice",
                        "episode_id": "episode-a",
                        "start_index": 5,
                        "end_index": 6,
                        "label": "bad action",
                        "status": "completed",
                    },
                ]
            ),
        }
    )
    return ReadOnlyDroidMongo(db).build_export_index(date_substrings=["2026-02-02"])


def test_read_only_mongo_builds_episode_and_slice_index():
    index = _make_index()
    assert len(index.projects) == 1
    assert len(index.episodes) == 1
    assert index.episodes[0].datetime_value == "2026-02-02-12h-34m-56s"
    assert len(index.slices_by_episode_id["episode-a"]) == 3


def test_parse_speed_value_handles_valid_missing_and_invalid_values():
    assert np.isclose(parse_speed_value({"speed": "0.1"}), 0.1)
    assert np.isclose(parse_speed_value({"speed": "Put the fan in the cardboard box (Speed: 0m/s)"}), 0.0)
    assert np.isclose(parse_speed_value({"speed": "Put the battery in the box (speed: 0.04 m/s)"}), 0.04)
    assert parse_speed_value({"speed": " fast "}) is None
    assert parse_speed_value({}) is None


def test_resolve_raw_episode_metadata_matches_by_folder_datetime():
    index = _make_index()
    resolved = resolve_raw_episode_metadata(index=index, folder_name="Sun_Feb_2_12:34:56_2026", annotation=None)
    assert resolved.match_status == "matched"
    assert resolved.mongo_episode_id == "episode-a"
    assert resolved.task == "Put the bottle in the bin"
    assert np.isclose(resolved.conveyor_speed, 0.05)
    assert resolve_subtask_for_frame(resolved, frame_index=0, episode_length=5) == "reach"
    assert resolve_subtask_for_frame(resolved, frame_index=3, episode_length=5) == "place"


def test_resolve_legacy_episode_metadata_matches_by_annotation_uuid_datetime():
    class Annotation:
        uuid = "EII+collector+2026-02-02-12h-34m-56s"
        task = "Ignore me"

    index = _make_index()
    source_frame = {
        "task": "Legacy task",
        "building": "EII",
        "collector_id": "collector",
        "datetime": "",
    }
    resolved = resolve_legacy_episode_metadata(
        index=index,
        source_frame=source_frame,
        source_repo_id="repo",
        source_episode_index=0,
        source_task="Legacy task",
        annotation=Annotation(),
    )
    assert resolved.match_status == "matched"
    assert resolved.mongo_episode_id == "episode-a"
    assert resolved.building == "EII"
    assert resolved.collector_id == "collector"
    assert resolved.datetime_value == "2026-02-02-12h-34m-56s"


def test_resolve_legacy_episode_metadata_supports_task_length_fallback():
    index = _make_index()
    source_frame = {
        "task": "Put the bottle in the bin",
        "building": None,
        "collector_id": None,
        "datetime": None,
    }
    resolved = resolve_legacy_episode_metadata(
        index=index,
        source_frame=source_frame,
        source_repo_id="repo",
        source_episode_index=0,
        source_task="Put the bottle in the bin",
        source_length=7,
        annotation=None,
        allow_task_length_fallback=True,
    )
    assert resolved.match_status == "matched_task_length"
    assert resolved.mongo_episode_id == "episode-a"
    assert resolved.datetime_value == "2026-02-02-12h-34m-56s"


def test_resolve_metadata_filters_housekeeping_labels():
    index = _make_index()
    resolved = resolve_raw_episode_metadata(index=index, folder_name="Sun_Feb_2_12:34:56_2026", annotation=None)
    labels = {segment["subtask"] for segment in resolved.subtasks}
    assert "reach" in labels
    assert "place" in labels
    assert not labels.intersection(HOUSEKEEPING_SLICE_LABELS)


def test_unmatched_episode_uses_nan_speed_and_unknown_subtask():
    index = _make_index()
    resolved = resolve_raw_episode_metadata(index=index, folder_name="Sun_Feb_3_12:34:56_2026", annotation=None)
    stats = compute_env_stats_for_frame({}, frame_index=0, episode_length=2, annotation=resolved)
    assert resolved.match_status == "unmatched"
    assert np.isnan(stats["environment.conveyor_speed"][0])
    assert resolve_subtask_for_frame(resolved, frame_index=0, episode_length=2) == "unknown"


def test_legacy_frame_mapping_includes_resolved_speed():
    class Annotation:
        uuid = "EII+collector+2026-02-02-12h-34m-56s"
        task = "Ignore me"

    index = _make_index()
    source_frame = {
        "task": "Legacy task",
        "building": "EII",
        "collector_id": "collector",
        "datetime": "2026-02-02-12h-34m-56s",
        "observation.images.wrist_left": np.zeros((180, 320, 3), dtype=np.uint8),
        "observation.images.exterior_1_left": np.zeros((180, 320, 3), dtype=np.uint8),
        "observation.images.exterior_2_left": np.zeros((180, 320, 3), dtype=np.uint8),
        "observation.state.joint_position": np.zeros(7, dtype=np.float32),
        "observation.state.gripper_position": np.zeros(1, dtype=np.float32),
        "action.source_joint_velocity_gripper": np.zeros(8, dtype=np.float32),
    }
    resolved = resolve_legacy_episode_metadata(
        index=index,
        source_frame=source_frame,
        source_repo_id="repo",
        source_episode_index=0,
        source_task="Legacy task",
        annotation=Annotation(),
    )
    frame, subtask_label = _map_legacy_frame(
        source_frame,
        frame_index=0,
        episode_length=5,
        task=resolved.task or "Legacy task",
        building=resolved.building,
        collector_id=resolved.collector_id,
        datetime_value=resolved.datetime_value,
        is_episode_successful=True,
        annotation=resolved,
    )
    assert subtask_label == "reach"
    assert np.isclose(frame["environment.conveyor_speed"][0], 0.05)


def test_raw_frame_mapping_includes_resolved_speed():
    index = _make_index()
    resolved = resolve_raw_episode_metadata(index=index, folder_name="Sun_Feb_2_12:34:56_2026", annotation=None)
    step = {
        "observation": {
            "camera_type": {"wrist": 0, "ext1": 1, "ext2": 1},
            "timestamp": {"cameras": {"wrist_frame_received": 0.0, "ext1_frame_received": 0.0, "ext2_frame_received": 0.0}},
            "image": {
                "wrist": np.zeros((180, 320, 3), dtype=np.uint8),
                "ext1": np.zeros((180, 320, 3), dtype=np.uint8),
                "ext2": np.zeros((180, 320, 3), dtype=np.uint8),
            },
            "robot_state": {
                "joint_positions": np.zeros(7, dtype=np.float32),
                "gripper_position": np.zeros(1, dtype=np.float32),
                "cartesian_position": np.zeros(6, dtype=np.float32),
            },
        },
        "action": {
            "joint_position": np.zeros(7, dtype=np.float32),
            "joint_velocity": np.zeros(7, dtype=np.float32),
            "gripper_position": np.zeros(1, dtype=np.float32),
            "gripper_velocity": np.zeros(1, dtype=np.float32),
            "cartesian_position": np.zeros(6, dtype=np.float32),
            "cartesian_velocity": np.zeros(6, dtype=np.float32),
        },
    }
    frame, subtask_label = _build_raw_frame(
        step,
        frame_index=0,
        episode_length=5,
        task=resolved.task or "Do something",
        building=resolved.building,
        collector_id=resolved.collector_id,
        datetime_value=resolved.datetime_value,
        is_episode_successful=True,
        annotation=resolved,
    )
    assert subtask_label == "reach"
    assert np.isclose(frame["environment.conveyor_speed"][0], 0.05)
