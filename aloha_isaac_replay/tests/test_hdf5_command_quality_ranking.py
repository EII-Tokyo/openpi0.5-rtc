from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from aloha_isaac_replay.scripts.rank_hdf5_command_quality_candidates import (
    rank_hdf5_command_quality_candidates,
)


def _write_hdf5(path: Path, qpos: np.ndarray) -> None:
    with h5py.File(path, "w") as h5:
        h5.create_dataset("observations/qpos", data=qpos.astype(np.float32))


def _write_mapping(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "dof_mapping:",
                "  - canonical_name: left_shoulder",
                "    dataset_index: 1",
                "    sign: 1.0",
                "    offset: 0.0",
                "    scale: 1.0",
                "    unit: rad",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_rank_hdf5_command_quality_candidates_prefers_smooth_window(tmp_path: Path) -> None:
    smooth = np.zeros((80, 14), dtype=np.float64)
    smooth[:, 1] = np.linspace(0.0, 0.1, smooth.shape[0])
    spiky = np.zeros((80, 14), dtype=np.float64)
    spiky[:, 1] = (np.arange(spiky.shape[0]) % 2) * 0.1
    _write_hdf5(tmp_path / "smooth.hdf5", smooth)
    _write_hdf5(tmp_path / "spiky.hdf5", spiky)
    _write_mapping(tmp_path / "mapping.yaml")

    report = rank_hdf5_command_quality_candidates(
        hdf5_paths=[tmp_path / "spiky.hdf5", tmp_path / "smooth.hdf5"],
        mapping_path=tmp_path / "mapping.yaml",
        output_dir=tmp_path / "out",
        hdf5_rate_hz=50.0,
        window_size_frames=40,
        window_stride_frames=20,
        spike_threshold_rad_s=2.0,
        top_per_episode=1,
    )

    assert report["scanned_episode_count"] == 2
    assert report["best_candidate_windows"][0]["episode_path"].endswith("smooth.hdf5")
    assert report["best_candidate_windows"][0]["classification"] == "COMMAND_SMOOTHNESS_PASS"
    assert Path(report["json"]).exists()
    assert Path(report["csv"]).exists()
    assert Path(report["episodes_csv"]).exists()
    assert Path(report["markdown"]).exists()
