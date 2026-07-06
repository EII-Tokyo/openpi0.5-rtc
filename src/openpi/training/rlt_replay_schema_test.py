import json

import numpy as np

from openpi.training import rlt_replay_schema


def test_classify_formal_paper_anchor_replay_as_trainable():
    manifest = {
        "train_eligible": True,
        "voided": False,
        "replay_state_grain": "paper_subsampled_anchor",
        "formal_replay_ready": True,
        "z_rl_dim": 2048,
    }

    status = rlt_replay_schema.classify_replay_manifest(manifest, z_dim=2048)

    assert status.status == "formal_replay_ready"
    assert status.trainable
    assert status.reason == "paper_subsampled_anchor"


def test_classify_runtime_cache_block_as_requiring_offline_reencode():
    manifest = {
        "train_eligible": False,
        "replay_state_grain": "runtime_action_cache_block",
        "requires_offline_reencode": True,
        "formal_replay_ready": False,
    }

    status = rlt_replay_schema.classify_replay_manifest(manifest, z_dim=2048)

    assert status.status == "requires_offline_reencode"
    assert not status.trainable
    assert status.reason == "runtime_action_cache_block"


def test_classify_legacy_unmarked_replay_as_needing_audit_even_if_train_eligible():
    manifest = {
        "train_eligible": True,
        "replay_status": "written",
        "z_rl_dim": 2048,
    }

    status = rlt_replay_schema.classify_replay_manifest(manifest, z_dim=2048)

    assert status.status == "legacy_unmarked_requires_audit"
    assert not status.trainable
    assert "missing replay_state_grain" in status.reason


def test_load_manifest_from_npz_handles_scalar_json_manifest(tmp_path):
    shard = tmp_path / "shard.npz"
    manifest = {"replay_state_grain": "paper_subsampled_anchor"}
    np.savez(shard, manifest=json.dumps(manifest))

    with np.load(shard, allow_pickle=False) as data:
        assert rlt_replay_schema.load_manifest_from_npz(data) == manifest
