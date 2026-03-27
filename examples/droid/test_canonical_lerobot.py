from __future__ import annotations

import json

import numpy as np

from examples.droid.canonical_lerobot import (
    AnnotationIndex,
    SubtaskRegistry,
    build_canonical_features,
    compute_env_stats_for_frame,
    normalize_datetime,
    normalize_episode_identity,
    resolve_subtask_for_frame,
    validate_episode_frames_sync,
    validate_raw_camera_timestamps,
)
from examples.droid.convert_legacy_lerobot_to_canonical import _normalize_raw_episode_datetime


def test_build_canonical_features_uses_datetime_and_videos():
    features = build_canonical_features()
    assert "datetime" in features
    assert "date" not in features
    assert features["observation.images.wrist_left"]["dtype"] == "video"
    assert features["subtask_index"]["dtype"] == "int64"


def test_normalize_datetime_accepts_multiple_source_formats():
    assert normalize_datetime("2025-11-14-17h-05m-00s") == "2025-11-14-17h-05m-00s"
    assert normalize_datetime("2025_11_14_17:05:00") == "2025-11-14-17h-05m-00s"
    assert normalize_datetime("Fri_Nov_14_17:05:00_2025") == "2025-11-14-17h-05m-00s"


def test_normalize_episode_identity_prefers_uuid_and_datetime():
    building, collector_id, dt = normalize_episode_identity(
        uuid="EII+4b1a56cc+2025-11-14-17h-05m-00s",
        datetime_value="Fri_Nov_14_17:05:00_2025",
    )
    assert building == "EII"
    assert collector_id == "4b1a56cc"
    assert dt == "2025-11-14-17h-05m-00s"


def test_annotation_index_supports_uuid_and_ordered_matching(tmp_path):
    path = tmp_path / "annotations.jsonl"
    rows = [
        {"uuid": "EII+abc+2025-11-14-17h-05m-00s", "prompts": {"default": "task-a", "speed": 0.05}},
        {"uuid": "EII+abc+2025-11-14-17h-06m-00s", "prompts": {"default": "task-b"}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    annotations = AnnotationIndex.load(path)

    matched = annotations.match_raw_episode("Fri_Nov_14_17:05:00_2025")
    assert matched is not None
    assert matched.task == "task-a"
    assert matched.conveyor_speed == 0.05

    ordered = annotations.match_legacy_episode(source_repo_id="repo", source_episode_index=0, task="task-a")
    assert ordered is not None
    assert ordered.task == "task-a"


def test_legacy_annotation_mismatch_refuses_assignment(tmp_path):
    path = tmp_path / "annotations.jsonl"
    rows = [
        {"uuid": "EII+abc+2025-11-14-17h-05m-00s", "prompts": {"default": "task-a"}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    annotations = AnnotationIndex.load(path)

    matched = annotations.match_legacy_episode(source_repo_id="repo", source_episode_index=0, task="different-task")
    assert matched is None


def test_resolve_subtask_for_frame_supports_segment_specs():
    class Annotation:
        subtasks = [
            {"start_frame": 0, "end_frame": 2, "subtask": "reach"},
            {"start_frame": 2, "end_frame": 4, "subtask": "place"},
        ]

    assert resolve_subtask_for_frame(Annotation(), frame_index=0, episode_length=4) == "reach"
    assert resolve_subtask_for_frame(Annotation(), frame_index=3, episode_length=4) == "place"


def test_compute_env_stats_for_frame_only_emits_conveyor_speed():
    class Annotation:
        conveyor_speed = 0.04

    stats = compute_env_stats_for_frame({}, frame_index=1, episode_length=3, annotation=Annotation())
    assert set(stats) == {"environment.conveyor_speed"}
    assert np.isclose(stats["environment.conveyor_speed"][0], 0.04)


def test_compute_env_stats_for_frame_marks_missing_speed_as_nan():
    stats = compute_env_stats_for_frame({}, frame_index=1, episode_length=3, annotation=None)
    assert np.isnan(stats["environment.conveyor_speed"][0])


def test_subtask_registry_round_trip(tmp_path):
    registry = SubtaskRegistry(tmp_path)
    assert registry.get_or_add("unknown") == 0
    assert registry.get_or_add("reach") == 1
    registry.write()

    reloaded = SubtaskRegistry(tmp_path)
    assert reloaded.get_or_add("unknown") == 0
    assert reloaded.get_or_add("reach") == 1


def test_validate_raw_camera_timestamps_rejects_desync():
    timestamps = {
        "wrist_frame_received": 0.0,
        "ext1_frame_received": 0.0,
        "ext2_frame_received": 0.2,
    }
    try:
        validate_raw_camera_timestamps(timestamps, ["wrist", "ext1", "ext2"], tolerance_s=0.05)
    except ValueError as exc:
        assert "out of sync" in str(exc)
    else:
        raise AssertionError("Expected raw camera timestamp validation to fail.")


def test_validate_episode_frames_sync_rejects_missing_camera():
    frame = {
        "observation.images.wrist_left": np.zeros((180, 320, 3), dtype=np.uint8),
        "observation.images.exterior_1_left": np.zeros((180, 320, 3), dtype=np.uint8),
    }
    try:
        validate_episode_frames_sync([frame])
    except ValueError as exc:
        assert "Missing camera frame" in str(exc)
    else:
        raise AssertionError("Expected episode camera validation to fail.")


def test_normalize_raw_episode_datetime_handles_double_underscore_day():
    assert _normalize_raw_episode_datetime("Wed_Oct__8_14:41:50_2025") == "2025-10-08-14h-41m-50s"
