import json

import h5py
import numpy as np

from scripts import build_rlt_timeline_replay


def test_cli_writes_trainable_shard_and_manifest(tmp_path):
    h5_path = tmp_path / "episode.hdf5"
    out = tmp_path / "shards" / "key_region_demo.npz"
    manifest_path = tmp_path / "manifest.jsonl"
    with h5py.File(h5_path, "w") as root:
        root.attrs["key_region_id"] = "demo"
        root.attrs["reward"] = 0
        root.create_dataset("action", data=np.arange(8 * 2, dtype=np.float32).reshape(8, 2))
        root.create_dataset("reference_action", data=np.arange(8 * 2, dtype=np.float32).reshape(8, 2))
        timeline = root.create_group("rlt_timeline")
        timeline.attrs["z_rl_source"] = "vla_same_forward_runtime_output"
        timeline.create_dataset("z_rl", data=np.arange(8 * 3, dtype=np.float32).reshape(8, 3))
        timeline.create_dataset("proprio", data=np.arange(8 * 4, dtype=np.float32).reshape(8, 4))
        timeline.create_dataset("valid", data=np.ones((8,), dtype=np.bool_))

    build_rlt_timeline_replay.main(
        [
            "--hdf5",
            str(h5_path),
            "--output",
            str(out),
            "--manifest",
            str(manifest_path),
            "--train-horizon",
            "2",
            "--chunk-stride",
            "2",
        ]
    )

    with np.load(out, allow_pickle=False) as data:
        assert data["action"].shape == (3, 2, 2)
        shard_manifest = json.loads(str(data["manifest"].item()))
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "key_region_id": "demo",
            "num_transitions": 3,
            "replay_state_grain": "paper_subsampled_anchor",
            "shard_path": str(out.resolve()),
            "source_format": "rlt_timeline_hdf5",
            "z_dim": 3,
        }
    ]
    assert shard_manifest["source_format"] == "rlt_timeline_hdf5"
