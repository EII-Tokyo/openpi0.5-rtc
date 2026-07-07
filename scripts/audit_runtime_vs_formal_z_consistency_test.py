import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.audit_runtime_vs_formal_z_consistency import audit_one_key_region, summarize_rows


def _write_npz(path: Path, z_rl: np.ndarray, *, key_region_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        z_rl=z_rl.astype(np.float32),
        next_z_rl=z_rl.astype(np.float32),
        manifest=np.array(json.dumps({"key_region_id": key_region_id})),
    )


def _write_h5(path: Path, cached_z_rl: np.ndarray, *, key_region_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as root:
        root.attrs["key_region_id"] = key_region_id
        rlt = root.create_group("rlt")
        rlt.create_dataset("cached_z_rl", data=cached_z_rl.astype(np.float32))


def test_audit_one_key_region_reports_best_and_rowwise_similarity(tmp_path):
    key_region_id = "abc"
    runtime_z = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    formal_z = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype=np.float32)
    cached_z = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    runtime_npz = tmp_path / "runtime" / f"key_region_{key_region_id}.npz"
    formal_npz = tmp_path / "formal" / f"key_region_{key_region_id}.npz"
    h5_path = tmp_path / "rollouts" / f"key_region_{key_region_id}" / "episode.hdf5"
    _write_npz(runtime_npz, runtime_z, key_region_id=key_region_id)
    _write_npz(formal_npz, formal_z, key_region_id=key_region_id)
    _write_h5(h5_path, cached_z, key_region_id=key_region_id)

    row = audit_one_key_region(
        key_region_id=key_region_id,
        runtime_npz=runtime_npz,
        formal_npz=formal_npz,
        h5_path=h5_path,
    )

    assert row["key_region_id"] == key_region_id
    assert row["runtime_rows"] == 3
    assert row["runtime_unique_rows_rounded6"] == 2
    assert row["formal_unique_rows_rounded6"] == 3
    assert row["formal_vs_runtime_rowwise_cos_mean"] > 0.9
    assert row["formal_vs_cached_best_cos_min"] == pytest.approx(0.8)
    assert row["formal_vs_cached_best_unique_indices"] == 2


def test_summarize_rows_flags_low_cosine_as_not_equivalent():
    summary = summarize_rows(
        [
            {
                "formal_vs_cached_best_cos_mean": 0.88,
                "formal_vs_cached_best_cos_min": 0.80,
                "formal_vs_runtime_rowwise_cos_mean": 0.87,
                "runtime_unique_rows_rounded6": 4,
                "runtime_rows": 49,
                "formal_unique_rows_rounded6": 49,
            }
        ]
    )

    assert summary["sample_count"] == 1
    assert summary["is_runtime_cached_equivalent_to_formal"] is False
    assert summary["severity"] == "high"
