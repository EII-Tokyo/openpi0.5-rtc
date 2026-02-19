#!/usr/bin/env python3
"""Run PI05 high-level autoregressive subtask generation from one HDF5 frame."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageFile

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


ImageFile.LOAD_TRUNCATED_IMAGES = True


def _decode_jpeg_bytes(buf: np.ndarray) -> np.ndarray:
    img = Image.open(io.BytesIO(np.asarray(buf, dtype=np.uint8).tobytes())).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def load_aloha_obs(hdf5_path: Path, frame_idx: int, prompt: str) -> dict:
    with h5py.File(hdf5_path, "r") as f:
        state = np.asarray(f["observations/qpos"][frame_idx], dtype=np.float32)
        images = {cam: _decode_jpeg_bytes(f["observations/images"][cam][frame_idx]) for cam in f["observations/images"].keys()}
    return {"state": state, "images": images, "prompt": prompt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, default=Path("episode_0.hdf5"))
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--config", type=str, default="pi05_aloha")
    parser.add_argument("--checkpoint", type=str, default="gs://openpi-assets/checkpoints/pi05_base")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-text-token-id", type=int, default=240000)
    args = parser.parse_args()

    obs = load_aloha_obs(args.hdf5, args.frame_idx, args.prompt)
    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config),
        args.checkpoint,
    )
    result = policy.infer_subtask(
        obs,
        temperature=args.temperature,
        max_text_token_id=args.max_text_token_id,
    )
    print(
        json.dumps(
            {
                "hdf5": str(args.hdf5),
                "frame_idx": args.frame_idx,
                "prompt": args.prompt,
                "subtask_text": result["subtask_text"],
                "subtask_tokens": result["subtask_tokens"].tolist(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
