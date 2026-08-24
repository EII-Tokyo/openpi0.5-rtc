#!/usr/bin/env python3
"""Build the self-contained episode-0 replay payload from the source HDF5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


EXPECTED_SOURCE_SHA256 = "7f2c7fc2010e64982116e8d4417710840672b365563cd8322967f42a7c98628e"
EXPECTED_FRAMES = 918
EXPECTED_FPS = 50.0
LABELS = [
    (0, 173, "PICK_UP", "Bottle on table, opening faces right: Pick up with left hand"),
    (174, 649, "UNSCREW_CAP", "Bottle in left hand and capped: Unscrew cap"),
    (650, 799, "DISPOSE", "Bottle in left hand, cap removed, and cap in right hand: Bottle to left trash bin, cap to right trash bin"),
    (800, 917, "RETURN", "No bottle on table: Return to initial pose"),
]
LEFT_CLOSED_RUNS = [(0, 8), (113, 762)]
RIGHT_CLOSED_RUNS = [(0, 5), (217, 290), (391, 469), (583, 755), (904, 917)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[5]
    default_source = Path("/home/eii/project/bottles_data/2026-05-11_twist/episode_0/episode.hdf5")
    default_output = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output_dir.resolve()
    source_hash = sha256(source)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"episode-0 source SHA mismatch: {source_hash}")

    with h5py.File(source, "r") as episode:
        action = np.asarray(episode["action"], dtype=np.float32)
        qpos = np.asarray(episode["observations/qpos"], dtype=np.float32)
        fps = float(episode.attrs["video_fps"])
        frames = int(episode.attrs["video_frame_count"])
        if action.shape != (EXPECTED_FRAMES, 14) or qpos.shape != (EXPECTED_FRAMES, 14):
            raise ValueError(f"unexpected episode shapes: action={action.shape}, qpos={qpos.shape}")
        if frames != EXPECTED_FRAMES or not np.isclose(fps, EXPECTED_FPS):
            raise ValueError(f"unexpected timing: frames={frames}, fps={fps}")
        if not np.all(np.isfinite(action)) or not np.all(np.isfinite(qpos)):
            raise ValueError("episode contains non-finite joint values")

    state = np.empty(EXPECTED_FRAMES, dtype="U16")
    label = np.empty(EXPECTED_FRAMES, dtype="U160")
    for start, end, state_name, text in LABELS:
        state[start : end + 1] = state_name
        label[start : end + 1] = text

    output.mkdir(parents=True, exist_ok=True)
    payload_path = output / "episode_0_replay_data.npz"
    np.savez_compressed(
        payload_path,
        action=action,
        qpos=qpos,
        state=state,
        label=label,
        frequency_hz=np.asarray(EXPECTED_FPS, dtype=np.float64),
    )
    payload_hash = sha256(payload_path)
    manifest = {
        "schema": "aloha_episode_replay/v1",
        "episode": 0,
        "classification": "KINEMATIC_VISUAL_REPLAY_NOT_PHYSICS_ACCEPTANCE",
        "source": {
            "relative_dataset_path": "2026-05-11_twist/episode_0/episode.hdf5",
            "sha256": source_hash,
            "command_dataset": "/action",
            "readback_dataset": "/observations/qpos",
        },
        "payload": {"path": payload_path.name, "sha256": payload_hash},
        "timing": {"frame_count": EXPECTED_FRAMES, "frequency_hz": EXPECTED_FPS},
        "joint_layout": {
            "left": ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper_normalized"],
            "right": ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper_normalized"],
            "active_isaac_dofs": ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "left_finger"],
        },
        "gripper_mapping": {"closed_m": 0.0440, "open_m": 0.0579, "normalized_closed_threshold": 0.35, "normalized_open_threshold": 0.8},
        "authoritative_manual_labels": [
            {"start_frame": start, "end_frame": end, "state": state_name, "label": text}
            for start, end, state_name, text in LABELS
        ],
        "derived_closed_runs": {"left": LEFT_CLOSED_RUNS, "right": RIGHT_CLOSED_RUNS},
        "object_replay": {
            "bottle_attach_frame": 174,
            "bottle_release_frame": 768,
            "cap_right_grasp_runs": [[217, 290], [391, 469], [583, 755]],
            "cap_release_frame": 768,
            "initial_pose_source": "existing remote_stream_cap_stage.usda",
            "continuity_rule": "capture relative transform on each attach transition",
        },
        "safety": {"uses_ros": False, "touches_real_robot": False, "saves_stage": False, "authors_session_layer_only": True},
    }
    manifest_path = output / "episode_0_replay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "PASS", "payload": str(payload_path), "payload_sha256": payload_hash}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
