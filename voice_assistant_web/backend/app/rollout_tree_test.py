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
                "replay_state_grain": "paper_subsampled_anchor",
                "formal_replay_ready": True,
                "z_rl_dim": 2048,
            }
        )
    )
    np.savez(
        shard_path,
        done=np.asarray([False, False, True]),
        reward_seq=np.ones((3, 10)),
        manifest=json.dumps(
            {
                "train_eligible": True,
                "replay_state_grain": "paper_subsampled_anchor",
                "formal_replay_ready": True,
                "z_rl_dim": 2048,
            }
        ),
    )
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


def test_key_region_review_hides_archived_tombstones(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    fake_control = _FakeRLTControl(
        [
            {
                "key_region_id": "archived",
                "status": "archived",
                "phase": "rl",
                "reward": 1,
                "shard_path": "/app/replay/rlt_key_regions/task/2026-07-07/shards/key_region_archived.npz",
                "num_replay_transitions": 3,
                "updated_at": 1783400000.0,
            }
        ]
    )

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", fake_control)

    assert main._key_region_review_records() == []
    assert main._key_region_review_batches_from_files() == []


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
    np.savez(
        shard_path,
        done=np.asarray([True]),
        reward_seq=np.ones((1, 10)),
        manifest=json.dumps(
            {
                "train_eligible": True,
                "replay_state_grain": "paper_subsampled_anchor",
                "formal_replay_ready": True,
                "z_rl_dim": 2048,
            }
        ),
    )
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


def test_key_region_review_uses_date_from_absolute_clean_shard_path(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    external_root = tmp_path / "external_data"
    shard_path = external_root / "replay" / "rlt_key_regions_clean" / "task" / "2026-06-19" / "shards" / "key_region_abs.crop_1.npz"
    shard_path.parent.mkdir(parents=True)
    np.savez(shard_path, done=np.asarray([True]), reward_seq=np.ones((1, 10)))
    fake_control = _FakeRLTControl(
        [
            {
                "key_region_id": "abs",
                "status": "committed",
                "phase": "warmup",
                "reward": 1,
                "shard_path": str(shard_path),
                "num_replay_transitions": 1,
                "updated_at": 1781942010.0,
            }
        ]
    )

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", fake_control)

    records = main._key_region_review_records()

    assert records[0]["batch"] == "2026-06-19"


def test_key_region_review_batches_skip_deleted_file_records(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    _write_review_record(rollout_root, replay_root, key_region_id="visible", reward=1, score_time=20.0)
    stale_shard = replay_root / "rlt_key_regions/task/2026-06-22/shards/key_region_stale.npz"
    stale_shard.parent.mkdir(parents=True)
    np.savez(stale_shard, done=np.asarray([True]), reward_seq=np.ones((1, 10)))

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(
        main,
        "rlt_control",
        _FakeRLTControl(
            [
                {
                    "key_region_id": "stale",
                    "status": "deleted",
                    "phase": "warmup",
                    "reward": 0,
                    "shard_path": str(stale_shard),
                    "num_replay_transitions": 1,
                    "updated_at": 9.0,
                }
            ]
        ),
    )

    page = main.rlt_key_region_review(limit=20)

    assert page.batches == ["2026-06-01"]


def test_key_region_review_batches_use_segment_batch_over_stale_raw_shard(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    key_region_id = "same_key"
    stale_shard = replay_root / "rlt_key_regions/task/2026-06-22/shards" / f"key_region_{key_region_id}.npz"
    clean_shard = replay_root / "rlt_key_regions_clean/manual" / f"key_region_{key_region_id}.crop_1.npz"
    stale_shard.parent.mkdir(parents=True)
    clean_shard.parent.mkdir(parents=True)
    np.savez(stale_shard, done=np.asarray([True]), reward_seq=np.ones((1, 10)))
    np.savez(clean_shard, done=np.asarray([True]), reward_seq=np.ones((1, 10)))

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(
        main,
        "rlt_control",
        _FakeRLTControl(
            [
                {
                    "key_region_id": key_region_id,
                    "status": "committed",
                    "phase": "warmup",
                    "reward": 1,
                    "shard_path": str(clean_shard),
                    "num_replay_transitions": 1,
                    "updated_at": 9.0,
                }
            ]
        ),
    )

    assert main._key_region_review_batches_from_files() == ["manual"]


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


def _write_review_record(rollout_root, replay_root, *, key_region_id, reward, score_time, formal=True):
    rollout_dir = rollout_root / "key_regions/task/2026-06-01/warmup" / f"key_region_{key_region_id}"
    shard_path = replay_root / "rlt_key_regions/task/2026-06-01/shards" / f"key_region_{key_region_id}.npz"
    rollout_dir.mkdir(parents=True)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "cam_right_wrist.mp4").write_bytes(b"mp4")
    manifest = {
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
    if formal:
        manifest.update(
            {
                "replay_state_grain": "paper_subsampled_anchor",
                "formal_replay_ready": True,
                "z_rl_dim": 2048,
            }
        )
    (rollout_dir / "manifest.json").write_text(json.dumps(manifest))
    np.savez(shard_path, done=np.asarray([False, False, True]), reward_seq=np.ones((3, 10)), manifest=json.dumps(manifest))
    return shard_path


def _write_cropped_review_record(rollout_root, replay_root, *, key_region_id, reward, score_time):
    rollout_dir = rollout_root / "key_regions/task/2026-06-01/warmup" / f"key_region_{key_region_id}"
    shard_path = replay_root / "rlt_key_regions_clean/task/2026-06-01/shards" / f"key_region_{key_region_id}.crop_1.npz"
    rollout_dir.mkdir(parents=True)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "cam_right_wrist.mp4").write_bytes(b"mp4")
    manifest = {
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
        "replay_state_grain": "paper_subsampled_anchor",
        "formal_replay_ready": True,
        "z_rl_dim": 2048,
    }
    (rollout_dir / "manifest.json").write_text(json.dumps(manifest))
    np.savez(shard_path, done=np.asarray([False, False, True]), reward_seq=np.ones((3, 10)), manifest=json.dumps(manifest))
    return shard_path


def _write_action_review_record(
    rollout_root,
    replay_root,
    *,
    key_region_id,
    reward,
    score_time,
    action_offset,
):
    shard_path = _write_cropped_review_record(
        rollout_root,
        replay_root,
        key_region_id=key_region_id,
        reward=reward,
        score_time=score_time,
    )
    reference_action = np.zeros((3, 10, 14), dtype=np.float32)
    action = reference_action + np.float32(action_offset)
    np.savez(
        shard_path,
        done=np.asarray([False, False, True]),
        reward_seq=np.ones((3, 10), dtype=np.float32),
        action=action,
        reference_action=reference_action,
    )
    return shard_path


def test_key_region_review_filters_no_actor_records(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    _write_action_review_record(
        rollout_root,
        replay_root,
        key_region_id="vla_only",
        reward=1,
        score_time=20.0,
        action_offset=0.0,
    )
    _write_action_review_record(
        rollout_root,
        replay_root,
        key_region_id="actor_changed",
        reward=0,
        score_time=10.0,
        action_offset=0.01,
    )

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", _FakeRLTControl([]))

    page = main.rlt_key_region_review(limit=20, status="noActor")

    assert [record.key_region_id for record in page.items] == ["vla_only"]
    assert page.items[0].actor_inference_kind == "no_actor"
    assert page.items[0].actor_delta_p95 == pytest.approx(0.0)


def test_key_region_review_filters_actor_modified_records(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    _write_action_review_record(
        rollout_root,
        replay_root,
        key_region_id="vla_only",
        reward=1,
        score_time=20.0,
        action_offset=0.0,
    )
    _write_action_review_record(
        rollout_root,
        replay_root,
        key_region_id="actor_changed",
        reward=0,
        score_time=10.0,
        action_offset=0.01,
    )

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", _FakeRLTControl([]))

    page = main.rlt_key_region_review(limit=20, status="actorModified")

    assert [record.key_region_id for record in page.items] == ["actor_changed"]
    assert page.items[0].actor_inference_kind == "actor_or_modified"
    assert page.items[0].actor_delta_p95 == pytest.approx(0.01)


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


def test_key_region_review_default_page_only_reads_action_metrics_for_visible_records(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    _write_review_record(rollout_root, replay_root, key_region_id="old_failure", reward=0, score_time=10.0)
    _write_review_record(rollout_root, replay_root, key_region_id="middle_success", reward=1, score_time=20.0)
    _write_review_record(rollout_root, replay_root, key_region_id="new_failure", reward=0, score_time=30.0)
    calls = []

    def fake_action_metrics(shard_path):
        calls.append(shard_path)
        return {"actor_inference_kind": "unknown"}

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", _FakeRLTControl([]))
    monkeypatch.setattr(main, "_rlt_action_delta_metrics", fake_action_metrics)

    page = main.rlt_key_region_review(limit=1, offset=0)

    assert [record.key_region_id for record in page.items] == ["new_failure"]
    assert len(calls) == 1


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


def test_key_region_review_page_searches_date_time_and_key_region_id(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    _write_review_record(rollout_root, replay_root, key_region_id="early_target", reward=0, score_time=10.0)
    _write_review_record(rollout_root, replay_root, key_region_id="late_target", reward=1, score_time=20.0)

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", _FakeRLTControl([]))

    by_id = main.rlt_key_region_review(limit=20, search="late_target")
    by_date = main.rlt_key_region_review(limit=20, search="1970-01-01")
    by_time = main.rlt_key_region_review(limit=20, search="1970-01-01 09:00:20")

    assert [record.key_region_id for record in by_id.items] == ["late_target"]
    assert [record.key_region_id for record in by_date.items] == ["late_target", "early_target"]
    assert [record.key_region_id for record in by_time.items] == ["late_target"]
    assert by_time.items[0].review_datetime == "1970-01-01 09:00:20"


def test_key_region_review_page_searches_clean_shard_crop_datetime(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    rollout_dir = rollout_root / "key_regions/task/2026-06-01/warmup/key_region_crop_time"
    shard_path = replay_root / "rlt_key_regions_clean/task/2026-06-01/shards/key_region_crop_time.crop_1781942010812.npz"
    rollout_dir.mkdir(parents=True)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "cam_right_wrist.mp4").write_bytes(b"mp4")
    (rollout_dir / "manifest.json").write_text(
        json.dumps(
            {
                "key_region_id": "crop_time",
                "phase": "warmup",
                "reward": 0,
                "start_time": 1000.0,
                "end_time": 1001.0,
                "score_time": 1002.0,
                "num_replay_transitions": 3,
                "segment_status": "committed",
                "train_eligible": True,
                "shard_path": str(shard_path),
            }
        )
    )
    np.savez(shard_path, done=np.asarray([False, False, True]), reward_seq=np.ones((3, 10)))

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", _FakeRLTControl([]))

    page = main.rlt_key_region_review(limit=20, search="2026-06-20 16:53:30")

    assert [record.key_region_id for record in page.items] == ["crop_time"]
    assert page.items[0].crop_datetime == "2026-06-20 16:53:30"


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


def test_key_region_review_marks_legacy_replay_as_not_formal_trainable(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    _write_review_record(rollout_root, replay_root, key_region_id="legacy", reward=1, score_time=10.0, formal=False)

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", _FakeRLTControl([]))

    page = main.rlt_key_region_review(limit=20)

    assert page.items[0].conversion_status == "legacy_unmarked_requires_audit"
    assert page.items[0].trainable is False
    assert page.items[0].incomplete_reason == "legacy_unmarked_requires_audit"


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


def test_key_region_review_batch_filter_does_not_scan_other_batch_manifests(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    _write_review_record(rollout_root, replay_root, key_region_id="wanted", reward=1, score_time=10.0)
    other_dir = rollout_root / "key_regions/task/2026-06-02/warmup/key_region_other"
    other_dir.mkdir(parents=True)
    (other_dir / "cam_right_wrist.mp4").write_bytes(b"mp4")
    (other_dir / "manifest.json").write_text("{not-json")

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", _FakeRLTControl([]))
    scanned_batches = []
    original_manifest_summary = main._manifest_summary

    def tracking_manifest_summary(rollout_dir):
        scanned_batches.append(main._batch_from_rollout_path(rollout_dir))
        return original_manifest_summary(rollout_dir)

    monkeypatch.setattr(main, "_manifest_summary", tracking_manifest_summary)

    page = main.rlt_key_region_review(limit=20, batch="2026-06-01")

    assert [record.key_region_id for record in page.items] == ["wanted"]
    assert page.total == 1
    assert page.batches == ["2026-06-02", "2026-06-01"]
    assert scanned_batches == ["2026-06-01"]


def test_key_region_review_defaults_to_latest_batch(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    _write_review_record(rollout_root, replay_root, key_region_id="old", reward=1, score_time=10.0)
    latest_dir = rollout_root / "key_regions/task/2026-06-02/warmup/key_region_latest"
    latest_shard = replay_root / "rlt_key_regions/task/2026-06-02/shards/key_region_latest.npz"
    latest_dir.mkdir(parents=True)
    latest_shard.parent.mkdir(parents=True, exist_ok=True)
    (latest_dir / "cam_right_wrist.mp4").write_bytes(b"mp4")
    (latest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "key_region_id": "latest",
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
    np.savez(latest_shard, done=np.asarray([False, False, True]), reward_seq=np.ones((3, 10)))

    monkeypatch.setattr(main, "ROLLOUTS_ROOT", rollout_root)
    monkeypatch.setattr(main, "REPLAY_ROOT", replay_root)
    monkeypatch.setattr(main, "rlt_control", _FakeRLTControl([]))

    page = main.rlt_key_region_review(limit=20)

    assert page.batches == ["2026-06-02", "2026-06-01"]
    assert [record.key_region_id for record in page.items] == ["latest"]
    assert page.summary.total == 1


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
