from __future__ import annotations

import argparse
import io
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def _to_hwc(image: np.ndarray) -> np.ndarray:
    # Compressed-image dataset style: 1D uint8 bytes (e.g. jpeg/png payload).
    if image.ndim == 1 and image.dtype == np.uint8:
        with Image.open(io.BytesIO(image.tobytes())) as im:
            return np.asarray(im.convert("RGB"))
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        return np.transpose(image, (1, 2, 0))
    return image


def _load_hdf5_example(hdf5_path: Path, frame_index: int, prompt: str, subtask: str | None):
    with h5py.File(hdf5_path, "r") as f:
        obs = {
            "images": {
                "cam_high": _to_hwc(f["observations/images/cam_high"][frame_index]),
                "cam_left_wrist": _to_hwc(f["observations/images/cam_left_wrist"][frame_index]),
                "cam_right_wrist": _to_hwc(f["observations/images/cam_right_wrist"][frame_index]),
            },
            "state": f["observations/qpos"][frame_index].astype(np.float32),
            "prompt": prompt,
            "subtask": subtask if subtask is not None else prompt,
        }
        actions = f["action"][frame_index : frame_index + 16].astype(np.float32)
    return obs, actions


def _add_batch_dim(tree):
    out = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            out[k] = {kk: jnp.asarray(vv)[None, ...] for kk, vv in v.items()}
        else:
            out[k] = jnp.asarray(v)[None, ...]
    return out


def _fit_action_horizon(actions: np.ndarray, target_horizon: int) -> np.ndarray:
    cur = actions.shape[0]
    if cur == target_horizon:
        return actions
    if cur > target_horizon:
        return actions[:target_horizon]
    pad = np.zeros((target_horizon - cur, actions.shape[1]), dtype=actions.dtype)
    return np.concatenate([actions, pad], axis=0)


def main():
    parser = argparse.ArgumentParser(description="Compute PI0/PI05 loss from one HDF5 trajectory slice.")
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--config", type=str, default="pi05_aloha_pen_uncap")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="twist off the bottle cap")
    parser.add_argument("--subtask", type=str, default=None, help="If empty, fallback to --prompt.")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    policy = _policy_config.create_trained_policy(_config.get_config(args.config), args.checkpoint)
    model = policy._model

    obs, actions = _load_hdf5_example(args.hdf5, args.frame_index, args.prompt, args.subtask)
    sample = {**obs, "actions": actions}

    # Reuse training/inference raw transforms (tokenization, repacking, etc.).
    for transform in policy._input_transforms_raw:
        # print(f"transform={transform}")
        sample = transform(sample)

    actions = sample.pop("actions")
    actions = _fit_action_horizon(np.asarray(actions, dtype=np.float32), int(model.action_horizon))
    obs = _add_batch_dim(sample)
    actions = jnp.asarray(actions)[None, ...]

    obs_struct = _model.Observation.from_dict(obs)
    rng = jax.random.key(args.seed)
    total_loss, flow_loss, subtask_ar_loss = model.compute_loss_with_metrics(rng, obs_struct, actions, train=False)

    print(f"config={args.config}")
    print(f"checkpoint={args.checkpoint}")
    print(f"hdf5={args.hdf5}")
    print(f"frame_index={args.frame_index}")
    print(f"total_loss={float(np.array(total_loss).mean()):.8f}")
    print(f"flow_loss={float(np.array(flow_loss).mean()):.8f}")
    print(f"subtask_ar_loss={float(np.array(subtask_ar_loss).mean()):.8f}")


if __name__ == "__main__":
    main()
