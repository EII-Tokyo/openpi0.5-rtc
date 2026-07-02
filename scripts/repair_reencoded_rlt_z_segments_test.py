from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.repair_reencoded_rlt_z_segments import SegmentRepairArgs
from scripts.repair_reencoded_rlt_z_segments import align_to_reference_segments
from scripts.repair_reencoded_rlt_z_segments import change_rows
from scripts.repair_reencoded_rlt_z_segments import repair_reencoded_z_segments


def test_align_to_reference_segments_repeats_first_value_per_segment() -> None:
    values = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
    reference = np.asarray([[0], [0], [1], [1], [1], [2]], dtype=np.float32)

    aligned = align_to_reference_segments(values, reference)

    np.testing.assert_array_equal(aligned[0], values[0])
    np.testing.assert_array_equal(aligned[1], values[0])
    np.testing.assert_array_equal(aligned[2], values[2])
    np.testing.assert_array_equal(aligned[3], values[2])
    np.testing.assert_array_equal(aligned[4], values[2])
    np.testing.assert_array_equal(aligned[5], values[5])
    np.testing.assert_array_equal(change_rows(aligned), change_rows(reference))


def test_repair_reencoded_z_segments_writes_aligned_shard(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    shard = input_root / "shards" / "a.npz"
    shard.parent.mkdir(parents=True, exist_ok=True)
    z = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)
    proprio = np.asarray([[0], [0], [1], [1], [1], [2]], dtype=np.float32)
    manifest = {"key_region_id": "abc"}
    np.savez_compressed(
        shard,
        z_rl=z,
        proprio=proprio,
        action=np.zeros((6, 10, 14), dtype=np.float32),
        reference_action=np.zeros((6, 10, 14), dtype=np.float32),
        reward_seq=np.zeros((6, 10), dtype=np.float32),
        next_z_rl=z + 100,
        next_proprio=proprio,
        next_reference_action=np.zeros((6, 10, 14), dtype=np.float32),
        done=np.zeros((6,), dtype=bool),
        manifest=np.asarray(json.dumps(manifest)),
    )

    summary = repair_reencoded_z_segments(
        SegmentRepairArgs(
            input_root=input_root,
            output_root=tmp_path / "output",
            execute=True,
        )
    )

    assert summary.converted == 1
    with np.load(tmp_path / "output" / "shards" / "a.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(change_rows(data["z_rl"]), change_rows(data["proprio"]))
        np.testing.assert_array_equal(change_rows(data["next_z_rl"]), change_rows(data["next_proprio"]))
        repaired_manifest = json.loads(str(data["manifest"].item()))
    assert repaired_manifest["z_rl_segment_alignment"] == "proprio_change_rows"
