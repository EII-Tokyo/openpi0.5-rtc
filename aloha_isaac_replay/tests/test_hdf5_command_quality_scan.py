from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from aloha_isaac_replay.scripts.scan_hdf5_command_quality import scan_hdf5_command_quality


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
                "  - canonical_name: left_elbow",
                "    dataset_index: 2",
                "    sign: 1.0",
                "    offset: 0.0",
                "    scale: 1.0",
                "    unit: rad",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_hdf5_command_quality_passes_smooth_window(tmp_path: Path) -> None:
    qpos = np.zeros((16, 14), dtype=np.float64)
    qpos[:, 1] = np.linspace(0.0, 0.15, qpos.shape[0])
    _write_hdf5(tmp_path / "episode.hdf5", qpos)
    _write_mapping(tmp_path / "mapping.yaml")

    report = scan_hdf5_command_quality(
        hdf5_path=tmp_path / "episode.hdf5",
        mapping_path=tmp_path / "mapping.yaml",
        output_dir=tmp_path / "out",
        hdf5_rate_hz=50.0,
        spike_threshold_rad_s=2.0,
    )

    assert report["overall_classification"] == "COMMAND_SMOOTHNESS_PASS"
    assert report["overall_recommendation"] == "ALLOW_ISAAC_REPLAY_GATE"
    assert report["formal_replay_targets_modified"] is False
    assert Path(report["json"]).exists()
    assert Path(report["csv"]).exists()
    assert Path(report["markdown"]).exists()


def test_hdf5_command_quality_reports_repeated_spike_cluster(tmp_path: Path) -> None:
    qpos = np.zeros((50, 14), dtype=np.float64)
    qpos[10, 1] = 0.00
    qpos[11, 1] = 0.08
    qpos[12, 1] = 0.16
    qpos[13:, 1] = 0.16
    _write_hdf5(tmp_path / "episode.hdf5", qpos)
    _write_mapping(tmp_path / "mapping.yaml")

    report = scan_hdf5_command_quality(
        hdf5_path=tmp_path / "episode.hdf5",
        mapping_path=tmp_path / "mapping.yaml",
        output_dir=tmp_path / "out",
        hdf5_rate_hz=50.0,
        spike_threshold_rad_s=2.0,
        cluster_gap_steps=3,
    )

    shoulder = report["per_joint"]["left_shoulder"]
    assert report["overall_classification"] == "REPEATED_SPIKE_CLUSTER"
    assert report["overall_recommendation"] == "BLOCK_CCD_FIX_COMMAND_CONTINUITY_FIRST"
    assert shoulder["classification"] == "REPEATED_SPIKE_CLUSTER"
    assert shoulder["spike_count"] == 2
    assert shoulder["spike_clusters"][0]["cluster_start_step"] == 10
    assert shoulder["spike_clusters"][0]["cluster_end_step"] == 11
    assert shoulder["max_abs_target_velocity_hdf5_frame"] == 11
    assert shoulder["max_abs_target_velocity"] == pytest.approx(4.0)


def test_hdf5_command_quality_rejects_invalid_window(tmp_path: Path) -> None:
    _write_hdf5(tmp_path / "episode.hdf5", np.zeros((8, 14), dtype=np.float64))
    _write_mapping(tmp_path / "mapping.yaml")

    with pytest.raises(ValueError, match="invalid frame window"):
        scan_hdf5_command_quality(
            hdf5_path=tmp_path / "episode.hdf5",
            mapping_path=tmp_path / "mapping.yaml",
            output_dir=tmp_path / "out",
            start_frame=7,
            end_frame=7,
        )
