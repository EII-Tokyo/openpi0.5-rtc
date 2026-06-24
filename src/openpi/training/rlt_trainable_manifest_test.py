import json
import sqlite3

import numpy as np

from openpi.training import rlt_trainable_manifest


def _arrays(num_transitions: int, *, reward: float = 1.0) -> dict[str, np.ndarray]:
    action = np.ones((num_transitions, 10, 14), dtype=np.float32)
    reward_seq = np.zeros((num_transitions, 10), dtype=np.float32)
    done = np.zeros((num_transitions,), dtype=np.bool_)
    done[-1] = True
    reward_seq[-1, -1] = reward
    return {
        "z_rl": np.ones((num_transitions, 8), dtype=np.float32),
        "proprio": np.ones((num_transitions, 14), dtype=np.float32),
        "action": action,
        "reference_action": action * 0.5,
        "reward_seq": reward_seq,
        "next_z_rl": np.ones((num_transitions, 8), dtype=np.float32) * 2,
        "next_proprio": np.ones((num_transitions, 14), dtype=np.float32) * 3,
        "next_reference_action": action * 0.25,
        "done": done,
    }


def _write_segment_db(db_path, rows):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE segments (
                key_region_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                phase TEXT NOT NULL,
                reward INTEGER,
                shard_path TEXT,
                num_replay_transitions INTEGER NOT NULL DEFAULT 0,
                invalid_reason TEXT,
                created_at REAL NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL DEFAULT 0
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO segments
            (key_region_id, status, phase, reward, shard_path, num_replay_transitions, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def test_build_manifest_maps_container_clean_paths_and_summarizes_committed_shards(tmp_path):
    data_root = tmp_path / "data"
    clean_root = data_root / "replay" / "rlt_key_regions_clean"
    manual_shard = clean_root / "manual" / "key_region_manual.crop_1.npz"
    dated_shard = clean_root / "task" / "2026-06-22" / "shards" / "key_region_dated.crop_1.npz"
    deleted_shard = clean_root / "task" / "2026-06-22" / "shards" / "key_region_deleted.crop_1.npz"
    invalid_shard = clean_root / "manual" / "key_region_invalid.crop_1.npz"
    for path in (manual_shard, dated_shard, deleted_shard, invalid_shard):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(manual_shard, **_arrays(5, reward=1.0))
    np.savez(dated_shard, **_arrays(7, reward=0.0))
    np.savez(deleted_shard, **_arrays(11, reward=1.0))
    np.savez(invalid_shard, z_rl=np.ones((3, 8), dtype=np.float32))

    db_path = tmp_path / "segments.sqlite3"
    _write_segment_db(
        db_path,
        [
            (
                "manual",
                "committed",
                "warmup",
                1,
                "/app/replay/rlt_key_regions_clean/manual/key_region_manual.crop_1.npz",
                5,
                100.0,
            ),
            ("dated", "committed", "warmup", 0, str(dated_shard), 7, 90.0),
            ("deleted", "deleted", "warmup", 1, str(deleted_shard), 11, 80.0),
            (
                "invalid",
                "committed",
                "warmup",
                0,
                "/app/replay/rlt_key_regions_clean/manual/key_region_invalid.crop_1.npz",
                3,
                70.0,
            ),
            (
                "missing",
                "committed",
                "warmup",
                0,
                "/app/replay/rlt_key_regions_clean/manual/key_region_missing.crop_1.npz",
                3,
                60.0,
            ),
        ],
    )

    manifest = tmp_path / "trainable.jsonl"
    result = rlt_trainable_manifest.build_manifest_from_segment_db(
        db_path,
        output_path=manifest,
        clean_root=clean_root,
    )

    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    by_id = {row["key_region_id"]: row for row in rows}
    assert set(by_id) == {"manual", "dated"}
    assert by_id["manual"]["shard_path"] == str(manual_shard.resolve())
    assert by_id["manual"]["source_shard_path"].startswith("/app/replay/")
    assert by_id["manual"]["batch"] == "manual"
    assert by_id["dated"]["batch"] == "2026-06-22"
    assert result.summary.num_shards == 2
    assert result.summary.num_transitions == 12
    assert result.summary.success_episodes == 1
    assert result.summary.failure_episodes == 1
    assert result.skipped_by_reason == {"invalid_shard": 1, "missing_file": 1}


def test_read_manifest_paths_returns_ordered_paths(tmp_path):
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"shard_path": str(first)}),
                json.dumps({"shard_path": str(second)}),
            ]
        )
        + "\n"
    )

    assert rlt_trainable_manifest.read_manifest_paths(manifest) == [first.resolve(), second.resolve()]
