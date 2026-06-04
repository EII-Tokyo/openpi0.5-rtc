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
    shard_path.parent.mkdir(parents=True)
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


def test_key_region_review_reports_video_duration_and_region_offsets(tmp_path, monkeypatch):
    rollout_root = tmp_path / "rollouts"
    replay_root = tmp_path / "replay"
    key_region_id = "timed"
    rollout_dir = rollout_root / "key_regions/task/2026-06-01/warmup" / f"key_region_{key_region_id}"
    shard_path = replay_root / "rlt_key_regions/task/2026-06-01/shards" / f"key_region_{key_region_id}.npz"
    rollout_dir.mkdir(parents=True)
    shard_path.parent.mkdir(parents=True)
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
