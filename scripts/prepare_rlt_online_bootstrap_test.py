import json

import numpy as np
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
    shard.parent.mkdir(parents=True)
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


def test_prepare_online_bootstrap_can_label_holdout_eval_rows(tmp_path):
    shard = tmp_path / "clean" / "twist_off_the_bottle_cap" / "2026-06-17" / "shards" / "holdout.npz"
    shard.parent.mkdir(parents=True)
    np.savez(
        shard,
        action=np.zeros((2, 10, 14), dtype=np.float32),
        done=np.asarray([False, True]),
        reward_seq=np.asarray([[0.0] * 10, [0.0] * 9 + [1.0]], dtype=np.float32),
    )
    source = tmp_path / "holdout_manifest.jsonl"
    _write_manifest(source, [{"shard_path": str(shard)}])

    result = prepare_rlt_online_bootstrap.prepare_bootstrap(
        prepare_rlt_online_bootstrap.Args(
            source_manifest=source,
            output_dir=tmp_path / "out",
            output_name="holdout",
            bootstrap_source="online_holdout_eval_only",
        )
    )

    rows = [json.loads(line) for line in result.manifest_path.read_text().splitlines()]
    assert rows[0]["bootstrap_source"] == "online_holdout_eval_only"
    assert rows[0]["batch"] == "2026-06-17"
    assert rows[0]["num_replay_transitions"] == 2
    assert rows[0]["success_episodes"] == 1
    assert result.summary["bootstrap_source"] == "online_holdout_eval_only"
    assert "2026-06-17" in result.summary["by_batch"]
    assert result.summary["num_transitions"] == 2
    assert result.summary["success_episodes"] == 1


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


def test_prepare_online_bootstrap_from_training_summary_loaded_shards(tmp_path):
    shard_a = tmp_path / "clean" / "a.npz"
    shard_b = tmp_path / "clean" / "b.npz"
    shard_a.parent.mkdir()
    shard_a.write_bytes(b"a")
    shard_b.write_bytes(b"b")
    summary = tmp_path / "training_summary.json"
    summary.write_text(
        json.dumps(
            {
                "loaded_shards": [str(shard_a), str(shard_b)],
                "replay_stats": {"num_shards": 2, "replay_size": 12, "success_episodes": 1, "failure_episodes": 1},
            }
        )
    )

    result = prepare_rlt_online_bootstrap.prepare_bootstrap(
        prepare_rlt_online_bootstrap.Args(
            source_manifest=None,
            training_summary=summary,
            output_dir=tmp_path / "out",
        )
    )

    rows = [json.loads(line) for line in result.manifest_path.read_text().splitlines()]
    assert result.summary["num_shards"] == 2
    assert [row["shard_path"] for row in rows] == [str(shard_a.resolve()), str(shard_b.resolve())]
    assert rows[0]["bootstrap_source"] == "training_summary_loaded_shards"
