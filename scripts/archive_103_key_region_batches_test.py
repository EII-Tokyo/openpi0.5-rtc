import sqlite3

from scripts.archive_103_key_region_batches import archive_key_region_batches


def test_archive_key_region_batches_moves_active_files_and_removes_active_ledger_rows(tmp_path):
    data_root = tmp_path / "data"
    batch = "2026-07-07"
    task = "twist_off_the_bottle_cap"
    replay_dir = data_root / "replay/rlt_key_regions" / task / batch
    rollout_dir = data_root / "rollouts/key_regions" / task / batch
    pending_dir = data_root / "replay/rlt_anchor_token_jobs/pending"
    replay_dir.mkdir(parents=True)
    rollout_dir.mkdir(parents=True)
    pending_dir.mkdir(parents=True)
    (replay_dir / "manifest.jsonl").write_text("{}\n")
    shard_dir = replay_dir / "shards"
    shard_dir.mkdir()
    (shard_dir / "key_region_keep.npz").write_bytes(b"npz")
    (rollout_dir / "rl/key_region_keep").mkdir(parents=True)
    (rollout_dir / "rl/key_region_keep/manifest.json").write_text("{}")
    (pending_dir / "key_region_keep.json").write_text("{}")

    db_path = data_root / "segment_db/segments.sqlite3"
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE segments (
            key_region_id TEXT PRIMARY KEY,
            status TEXT,
            phase TEXT,
            reward INTEGER,
            shard_path TEXT,
            num_replay_transitions INTEGER,
            invalid_reason TEXT,
            created_at REAL,
            updated_at REAL
        );
        CREATE TABLE segment_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_region_id TEXT,
            event TEXT
        );
        """
    )
    conn.execute(
        """
        INSERT INTO segments (
            key_region_id, status, phase, reward, shard_path,
            num_replay_transitions, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "keep",
            "committed",
            "rl",
            1,
            f"/app/replay/rlt_key_regions/{task}/{batch}/shards/key_region_keep.npz",
            3,
            1783400000.0,
            1783400001.0,
        ),
    )
    conn.execute("INSERT INTO segment_events (key_region_id, event) VALUES (?, ?)", ("keep", "committed"))
    conn.commit()
    conn.close()

    archive_root = tmp_path / "archive"

    result = archive_key_region_batches(
        data_root=data_root,
        archive_root=archive_root,
        task=task,
        batches=[batch],
        execute=True,
    )

    assert result.executed is True
    assert result.archived_key_region_ids == ["keep"]
    assert not replay_dir.exists()
    assert not rollout_dir.exists()
    assert not (pending_dir / "key_region_keep.json").exists()
    assert (archive_root / "replay/rlt_key_regions" / task / batch / "manifest.jsonl").exists()
    assert (archive_root / "rollouts/key_regions" / task / batch / "rl/key_region_keep/manifest.json").exists()
    assert (archive_root / "replay/rlt_anchor_token_jobs/pending/key_region_keep.json").exists()
    assert (archive_root / "segment_db/segments.sqlite3.before_archive").exists()

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT count(*) FROM segments").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM segment_events").fetchone()[0] == 0
    conn.close()
