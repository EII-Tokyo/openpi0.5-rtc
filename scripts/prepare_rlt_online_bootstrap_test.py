import json

import pytest

from scripts import prepare_rlt_online_bootstrap


def _write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def test_prepare_online_bootstrap_deduplicates_no_actor_rows(tmp_path):
    shard_a = tmp_path / "clean" / "a.npz"
    shard_b = tmp_path / "clean" / "b.npz"
    shard_a.parent.mkdir()
    shard_a.write_bytes(b"a")
    shard_b.write_bytes(b"b")
    source = tmp_path / "source.jsonl"
    _write_manifest(
        source,
        [
            {
                "key_region_id": "a",
                "shard_path": str(shard_a),
                "source_shard_path": str(shard_a),
                "batch": "2026-06-17",
                "num_replay_transitions": 5,
                "success_episodes": 1,
                "failure_episodes": 0,
                "selection": {"selected_for_rtc_only_no_actor": True},
            },
            {
                "key_region_id": "a",
                "shard_path": str(shard_a),
                "source_shard_path": str(shard_a),
                "batch": "2026-06-17",
                "num_replay_transitions": 5,
                "success_episodes": 1,
                "failure_episodes": 0,
                "selection": {"selected_for_rtc_only_no_actor": True},
            },
            {
                "key_region_id": "b",
                "shard_path": str(shard_b),
                "source_shard_path": str(shard_b),
                "batch": "2026-06-22",
                "num_replay_transitions": 7,
                "success_episodes": 0,
                "failure_episodes": 1,
                "selection": {"selected_for_rtc_only_no_actor": True},
            },
        ],
    )

    result = prepare_rlt_online_bootstrap.prepare_bootstrap(
        prepare_rlt_online_bootstrap.Args(
            source_manifest=source,
            output_dir=tmp_path / "out",
            expected_count=None,
        )
    )

    assert result.summary["num_shards"] == 2
    assert result.summary["num_transitions"] == 12
    assert result.summary["success_episodes"] == 1
    assert result.summary["failure_episodes"] == 1
    assert result.skipped_by_reason == {"duplicate_key_region_id": 1}
    rows = [json.loads(line) for line in result.manifest_path.read_text().splitlines()]
    assert [row["key_region_id"] for row in rows] == ["a", "b"]
    assert rows[0]["bootstrap_source"] == "no_actor_clean"
    assert json.loads(result.summary_path.read_text())["num_shards"] == 2


def test_prepare_online_bootstrap_writes_remote_manifest(tmp_path):
    shard = tmp_path / "clean" / "a.npz"
    shard.parent.mkdir()
    shard.write_bytes(b"a")
    source = tmp_path / "source.jsonl"
    _write_manifest(
        source,
        [
            {
                "key_region_id": "a",
                "shard_path": str(shard),
                "source_shard_path": str(shard),
                "selection": {"selected_for_rtc_only_no_actor": True},
            }
        ],
    )

    result = prepare_rlt_online_bootstrap.prepare_bootstrap(
        prepare_rlt_online_bootstrap.Args(
            source_manifest=source,
            output_dir=tmp_path / "out",
            remote_shard_root="/app/replay/rlt_online_bootstrap/no_actor_clean/shards",
        )
    )

    remote_manifest = result.remote_manifest_path
    assert remote_manifest is not None
    rows = [json.loads(line) for line in remote_manifest.read_text().splitlines()]
    assert rows[0]["shard_path"] == "/app/replay/rlt_online_bootstrap/no_actor_clean/shards/a.npz"
    assert rows[0]["local_shard_path"] == str(shard.resolve())
    assert result.summary["remote_manifest_path"] == str(remote_manifest.resolve())


def test_prepare_online_bootstrap_rejects_wrong_expected_count(tmp_path):
    shard = tmp_path / "clean" / "a.npz"
    shard.parent.mkdir()
    shard.write_bytes(b"a")
    source = tmp_path / "source.jsonl"
    _write_manifest(
        source,
        [
            {
                "key_region_id": "a",
                "shard_path": str(shard),
                "source_shard_path": str(shard),
                "selection": {"selected_for_rtc_only_no_actor": True},
            }
        ],
    )

    with pytest.raises(ValueError, match="Expected 2 bootstrap shards, got 1"):
        prepare_rlt_online_bootstrap.prepare_bootstrap(
            prepare_rlt_online_bootstrap.Args(
                source_manifest=source,
                output_dir=tmp_path / "out",
                expected_count=2,
            )
        )
