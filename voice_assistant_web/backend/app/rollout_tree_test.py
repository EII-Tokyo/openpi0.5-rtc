import json
import os

import numpy as np
import pytest

from voice_assistant_web.backend.app import main
from voice_assistant_web.backend.app.main import _scan_rollout_tree


def _write_key_region(path, *, key_region_id, start_time, end_time):
    path.mkdir(parents=True)
    (path / "cam_high.mp4").write_bytes(b"mp4")
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "key_region_id": key_region_id,
                "phase": "warmup",
                "reward": 0,
                "start_time": start_time,
                "end_time": end_time,
                "num_replay_transitions": 1,
            }
        )
    )
    os.utime(path, (end_time, end_time))


def test_key_region_directories_are_sorted_newest_first(tmp_path):
    old = tmp_path / "key_region_zz_old"
    new = tmp_path / "key_region_aa_new"
    _write_key_region(old, key_region_id="zz_old", start_time=10.0, end_time=20.0)
    _write_key_region(new, key_region_id="aa_new", start_time=30.0, end_time=40.0)

    tree = _scan_rollout_tree(tmp_path, "key_regions/task/day/warmup")

    names = [child["name"] for child in tree["children"] if child["type"] == "directory"]
    assert names[:2] == ["key_region_aa_new", "key_region_zz_old"]


class _FakeRLTControl:
    def __init__(self, segments):
        self.segments = {segment["key_region_id"]: dict(segment) for segment in segments}
        self.committed = []

    def list_segments(self, *, limit=500):
        return list(self.segments.values())[:limit]

    def commit_key_region_from_files(self, *, key_region_id, phase, reward, shard_path, num_replay_transitions):
        self.committed.append(
            {
                "key_region_id": key_region_id,
                "phase": phase,
                "reward": reward,
                "shard_path": shard_path,
                "num_replay_transitions": num_replay_transitions,
            }
        )
        self.segments[key_region_id] = {
            "key_region_id": key_region_id,
            "status": "committed",
            "phase": phase,
            "reward": reward,
            "shard_path": shard_path,
            "num_replay_transitions": num_replay_transitions,
            "updated_at": 100.0,
        }


def test_key_region_review_reconciles_saved_accepted_sample_as_trainable(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    key_region_id = "ready"
    rollout_dir = rollout_root / "key_regions/task/2026-06-01/warmup" / f"key_region_{key_region_id}"
    shard_path = replay_root / "rlt_key_regions/task/2026-06-01/shards" / f"key_region_{key_region_id}.npz"
    rollout_dir.mkdir(parents=True)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "cam_right_wrist.mp4").write_bytes(b"mp4")
    (rollout_dir / "manifest.json").write_text(
        json.dumps(
            {
                "key_region_id": key_region_id,
                "phase": "warmup",
                "reward": 1,
                "start_time": 10.0,
                "end_time": 12.0,
                "score_time": 13.0,
                "num_replay_transitions": 3,
                "segment_status": "committed",
                "train_eligible": True,
            }
        )
    )
    np.savez(shard_path, done=np.asarray([False, False, True]), reward_seq=np.ones((3, 10)))
    fake_control = _FakeRLTControl(
        [
            {
                "key_region_id": key_region_id,
                "status": "accepted",
                "phase": "warmup",
                "reward": 1,
                "shard_path": None,
                "num_replay_transitions": 0,
                "updated_at": 9.0,
            }
        ]
    )

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", fake_control)

    records = main._key_region_review_records()

    assert fake_control.committed == [
        {
            "key_region_id": key_region_id,
            "phase": "warmup",
            "reward": 1,
            "shard_path": str(shard_path),
            "num_replay_transitions": 3,
        }
    ]
    assert records[0]["status"] == "committed"
    assert records[0]["trainable"] is True
    assert records[0].get("incomplete_reason") is None


def test_key_region_review_hides_deleted_tombstones(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    fake_control = _FakeRLTControl(
        [
            {
                "key_region_id": "deleted",
                "status": "deleted",
                "phase": "warmup",
                "reward": 1,
                "shard_path": "/app/replay/rlt_key_regions/task/day/shards/key_region_deleted.npz",
                "num_replay_transitions": 3,
                "updated_at": 9.0,
            }
        ]
    )

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", fake_control)

    assert main._key_region_review_records() == []


def test_key_region_review_reports_missing_npz_before_train_eligibility(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    fake_control = _FakeRLTControl(
        [
            {
                "key_region_id": "missing",
                "status": "committed",
                "phase": "warmup",
                "reward": 1,
                "shard_path": "/app/replay/rlt_key_regions/task/day/shards/key_region_missing.npz",
                "num_replay_transitions": 3,
                "updated_at": 9.0,
            }
        ]
    )

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", fake_control)

    records = main._key_region_review_records()

    assert records[0]["trainable"] is False
    assert records[0]["incomplete_reason"] == "missing_npz"


def test_key_region_review_counts_committed_container_clean_manual_shard_as_trainable(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    shard_path = replay_root / "rlt_key_regions_clean" / "manual" / "key_region_manual.crop_1.npz"
    shard_path.parent.mkdir(parents=True)
    np.savez(shard_path, done=np.asarray([True]), reward_seq=np.ones((1, 10)))
    fake_control = _FakeRLTControl(
        [
            {
                "key_region_id": "manual",
                "status": "committed",
                "phase": "warmup",
                "reward": 1,
                "shard_path": "/app/replay/rlt_key_regions_clean/manual/key_region_manual.crop_1.npz",
                "num_replay_transitions": 1,
                "updated_at": 9.0,
            }
        ]
    )

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", fake_control)

    records = main._key_region_review_records()
    summary = main._key_region_review_summary(records)

    assert records[0]["npz_exists"] is True
    assert records[0]["local_shard_path"] == str(shard_path.resolve())
    assert records[0]["batch"] == "manual"
    assert records[0]["trainable"] is True
    assert records[0].get("incomplete_reason") is None
    assert summary.trainable == 1


def test_key_region_review_requires_current_segment_shard_even_when_raw_orphan_exists(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    key_region_id = "missing_clean"
    rollout_dir = rollout_root / "key_regions/task/2026-06-01/warmup" / f"key_region_{key_region_id}"
    raw_shard_path = replay_root / "rlt_key_regions/task/2026-06-01/shards" / f"key_region_{key_region_id}.npz"
    missing_clean_path = replay_root / "rlt_key_regions_clean/task/2026-06-01/shards" / f"key_region_{key_region_id}.crop_1.npz"
    rollout_dir.mkdir(parents=True)
    raw_shard_path.parent.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "cam_right_wrist.mp4").write_bytes(b"mp4")
    (rollout_dir / "manifest.json").write_text(
        json.dumps(
            {
                "key_region_id": key_region_id,
                "phase": "warmup",
                "reward": 1,
                "start_time": 10.0,
                "end_time": 12.0,
                "score_time": 13.0,
                "num_replay_transitions": 3,
                "segment_status": "committed",
                "train_eligible": True,
                "shard_path": str(missing_clean_path),
            }
        )
    )
    np.savez(raw_shard_path, done=np.asarray([False, False, True]), reward_seq=np.ones((3, 10)))
    fake_control = _FakeRLTControl(
        [
            {
                "key_region_id": key_region_id,
                "status": "committed",
                "phase": "warmup",
                "reward": 1,
                "shard_path": str(missing_clean_path),
                "num_replay_transitions": 3,
                "updated_at": 9.0,
            }
        ]
    )

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", fake_control)

    records = main._key_region_review_records()

    assert records[0]["npz_exists"] is False
    assert records[0]["trainable"] is False
    assert records[0]["incomplete_reason"] == "missing_npz"


def test_key_region_review_reports_video_duration_and_region_offsets(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    key_region_id = "timed"
    rollout_dir = rollout_root / "key_regions/task/2026-06-01/warmup" / f"key_region_{key_region_id}"
    shard_path = replay_root / "rlt_key_regions/task/2026-06-01/shards" / f"key_region_{key_region_id}.npz"
    rollout_dir.mkdir(parents=True)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "cam_right_wrist.mp4").write_bytes(b"mp4")
    (rollout_dir / "manifest.json").write_text(
        json.dumps(
            {
                "key_region_id": key_region_id,
                "phase": "warmup",
                "reward": 1,
                "start_time": 100.0,
                "end_time": 101.214,
                "score_time": 102.0,
                "num_frames": 175,
                "fps": 50.0,
                "num_replay_transitions": 3,
                "segment_status": "committed",
                "train_eligible": True,
            }
        )
    )
    np.savez(shard_path, done=np.asarray([False, False, True]), reward_seq=np.ones((3, 10)))

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", _FakeRLTControl([]))

    records = main._key_region_review_records()

    assert records[0]["duration_seconds"] == pytest.approx(3.5)
    assert records[0]["key_region_duration_seconds"] == pytest.approx(1.214)
    assert records[0]["key_region_start_sec"] == pytest.approx(2.0)
    assert records[0]["key_region_end_sec"] == pytest.approx(3.214)


def _write_review_record(rollout_root, replay_root, *, key_region_id, reward, score_time):
    rollout_dir = rollout_root / "key_regions/task/2026-06-01/warmup" / f"key_region_{key_region_id}"
    shard_path = replay_root / "rlt_key_regions/task/2026-06-01/shards" / f"key_region_{key_region_id}.npz"
    rollout_dir.mkdir(parents=True)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "cam_right_wrist.mp4").write_bytes(b"mp4")
    (rollout_dir / "manifest.json").write_text(
        json.dumps(
            {
                "key_region_id": key_region_id,
                "phase": "warmup",
                "reward": reward,
                "start_time": score_time - 2.0,
                "end_time": score_time - 1.0,
                "score_time": score_time,
                "num_replay_transitions": 3,
                "segment_status": "committed",
                "train_eligible": True,
            }
        )
    )
    np.savez(shard_path, done=np.asarray([False, False, True]), reward_seq=np.ones((3, 10)))
    return shard_path


def _write_cropped_review_record(rollout_root, replay_root, *, key_region_id, reward, score_time):
    rollout_dir = rollout_root / "key_regions/task/2026-06-01/warmup" / f"key_region_{key_region_id}"
    shard_path = replay_root / "rlt_key_regions_clean/task/2026-06-01/shards" / f"key_region_{key_region_id}.crop_1.npz"
    rollout_dir.mkdir(parents=True)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "cam_right_wrist.mp4").write_bytes(b"mp4")
    (rollout_dir / "manifest.json").write_text(
        json.dumps(
            {
                "key_region_id": key_region_id,
                "phase": "warmup",
                "reward": reward,
                "start_time": score_time - 2.0,
                "end_time": score_time - 1.0,
                "score_time": score_time,
                "num_replay_transitions": 3,
                "segment_status": "committed",
                "train_eligible": True,
                "crop_start_sec": 0.25,
                "crop_end_sec": 1.25,
                "shard_path": str(shard_path),
            }
        )
    )
    np.savez(shard_path, done=np.asarray([False, False, True]), reward_seq=np.ones((3, 10)))
    return shard_path


def test_key_region_review_page_paginates_and_summarizes_records(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    _write_review_record(rollout_root, replay_root, key_region_id="old_failure", reward=0, score_time=10.0)
    _write_review_record(rollout_root, replay_root, key_region_id="middle_success", reward=1, score_time=20.0)
    _write_review_record(rollout_root, replay_root, key_region_id="new_failure", reward=0, score_time=30.0)

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", _FakeRLTControl([]))

    page = main.rlt_key_region_review(limit=1, offset=1, reward="failure")

    assert page.total == 2
    assert page.next_offset is None
    assert page.summary.success == 1
    assert page.summary.failure == 2
    assert page.summary.trainable == 3
    assert [record.key_region_id for record in page.items] == ["old_failure"]


def test_key_region_review_page_can_focus_target_record(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    _write_review_record(rollout_root, replay_root, key_region_id="old_failure", reward=0, score_time=10.0)
    _write_review_record(rollout_root, replay_root, key_region_id="middle_success", reward=1, score_time=20.0)
    _write_review_record(rollout_root, replay_root, key_region_id="new_failure", reward=0, score_time=30.0)

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", _FakeRLTControl([]))

    page = main.rlt_key_region_review(limit=1, offset=0, focus_key_region_id="old_failure")

    assert page.offset == 2
    assert [record.key_region_id for record in page.items] == ["old_failure"]


def test_key_region_review_needs_crop_lists_uncropped_trainable_candidates(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    _write_review_record(rollout_root, replay_root, key_region_id="raw_candidate", reward=0, score_time=10.0)

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", _FakeRLTControl([]))

    page = main.rlt_key_region_review(limit=20, status="needsCrop")

    assert [record.key_region_id for record in page.items] == ["raw_candidate"]
    assert page.summary.needs_crop == 1


def test_preference_candidates_only_use_clean_cropped_records(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    _write_review_record(rollout_root, replay_root, key_region_id="raw", reward=1, score_time=10.0)
    _write_cropped_review_record(rollout_root, replay_root, key_region_id="clean", reward=1, score_time=20.0)

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", _FakeRLTControl([]))

    records = main._preference_candidate_records(batch="all", reward="all")

    assert [record["key_region_id"] for record in records] == ["clean"]


def test_preference_pair_sampling_is_budgeted_and_skip_is_deferred(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    _write_cropped_review_record(rollout_root, replay_root, key_region_id="success_a", reward=1, score_time=30.0)
    _write_cropped_review_record(rollout_root, replay_root, key_region_id="success_b", reward=1, score_time=20.0)
    _write_cropped_review_record(rollout_root, replay_root, key_region_id="failure_a", reward=0, score_time=10.0)

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", _FakeRLTControl([]))
    monkeypatch.setattr(main.settings, "rlt_segment_db_path", str(tmp_path / "segments.sqlite3"))

    first = main.rlt_preference_next_pair(pair_type="success_success")
    assert first.pair_type == "success_success"
    assert {first.left.key_region_id, first.right.key_region_id} == {"success_a", "success_b"}

    main.rlt_preference_record(
        main.RLTPreferenceRequest(
            left_key_region_id=first.left.key_region_id,
            right_key_region_id=first.right.key_region_id,
            preference="skip",
        )
    )

    skipped = main.rlt_preference_next_pair(pair_type="success_success")
    assert skipped.left is not None
    assert {skipped.left.key_region_id, skipped.right.key_region_id} == {"success_a", "success_b"}

    main.rlt_preference_record(
        main.RLTPreferenceRequest(
            left_key_region_id=first.left.key_region_id,
            right_key_region_id=first.right.key_region_id,
            preference="left",
        )
    )

    exhausted = main.rlt_preference_next_pair(pair_type="success_success")
    assert exhausted.left is None
    assert exhausted.remaining_unseen_pairs == 0


def test_key_region_review_reports_batch_and_local_paths(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    first_shard = _write_review_record(
        rollout_root,
        replay_root,
        key_region_id="first_batch",
        reward=1,
        score_time=10.0,
    )
    second_dir = rollout_root / "key_regions/task/2026-06-02/warmup/key_region_second_batch"
    second_shard = replay_root / "rlt_key_regions/task/2026-06-02/shards/key_region_second_batch.npz"
    second_dir.mkdir(parents=True)
    second_shard.parent.mkdir(parents=True, exist_ok=True)
    (second_dir / "cam_right_wrist.mp4").write_bytes(b"mp4")
    (second_dir / "manifest.json").write_text(
        json.dumps(
            {
                "key_region_id": "second_batch",
                "phase": "warmup",
                "reward": 0,
                "start_time": 18.0,
                "end_time": 19.0,
                "score_time": 20.0,
                "num_replay_transitions": 3,
                "segment_status": "committed",
                "train_eligible": True,
            }
        )
    )
    np.savez(second_shard, done=np.asarray([False, False, True]), reward_seq=np.ones((3, 10)))

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", _FakeRLTControl([]))

    page = main.rlt_key_region_review(limit=20, batch="2026-06-01")

    assert page.batches == ["2026-06-02", "2026-06-01"]
    assert [record.key_region_id for record in page.items] == ["first_batch"]
    record = page.items[0]
    assert record.batch == "2026-06-01"
    assert record.local_rollout_path == str(
        rollout_root / "key_regions/task/2026-06-01/warmup/key_region_first_batch"
    )
    assert record.local_shard_path == str(first_shard)
    assert all(item.key_region_id != "second_batch" for item in page.items)


def test_key_region_detail_returns_single_record(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    _write_review_record(rollout_root, replay_root, key_region_id="target", reward=1, score_time=10.0)
    _write_review_record(rollout_root, replay_root, key_region_id="other", reward=0, score_time=20.0)

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", _FakeRLTControl([]))

    record = main.rlt_key_region_detail("target")

    assert record.key_region_id == "target"
    assert record.reward == 1
    assert record.video_paths == [
        "key_regions/task/2026-06-01/warmup/key_region_target/cam_right_wrist.mp4"
    ]
