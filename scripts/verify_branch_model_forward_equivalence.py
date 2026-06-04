#!/usr/bin/env python3
"""Compare main vs minimal PI05 model forward outputs on fixed observations."""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


BRANCH_CODE = r'''
import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_base
from openpi.models import pi0_config

try:
    from openpi.data import transforms
except ImportError:
    transforms = None


def make_config():
    try:
        return pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy")
    except TypeError:
        return pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy")


def make_observation(cfg):
    batch_size = 1
    image = jnp.linspace(-1, 1, batch_size * 224 * 224 * 3, dtype=jnp.float32).reshape(batch_size, 224, 224, 3)
    new_images = {
        "cam_high": image,
        "cam_low": image * 0.5,
        "cam_left_wrist": image * 0.25,
        "cam_right_wrist": image * -0.25,
    }
    old_images = {
        "base_0_rgb": new_images["cam_high"],
        "base_1_rgb": new_images["cam_low"],
        "left_wrist_0_rgb": new_images["cam_left_wrist"],
        "right_wrist_0_rgb": new_images["cam_right_wrist"],
    }
    image_keys = tuple(cfg.inputs_spec(batch_size=batch_size)[0].images.keys())
    images = new_images if "cam_high" in image_keys else old_images
    return model_base.Observation(
        images={key: images[key] for key in image_keys},
        image_masks={key: jnp.ones((batch_size,), dtype=jnp.bool_) for key in image_keys},
        state=jnp.zeros((batch_size, cfg.action_dim), dtype=jnp.float32),
        tokenized_prompt=jnp.ones((batch_size, cfg.max_token_len), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((batch_size, cfg.max_token_len), dtype=jnp.bool_),
    )


def preprocess_if_external(observation, cfg, rng=None, train=False):
    if transforms is None or not hasattr(transforms, "AlohaTransformPipeline"):
        return observation
    return transforms.AlohaTransformPipeline.preprocess_observation(
        rng,
        observation,
        train=train,
        image_resolution=cfg.image_resolution,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = make_config()
    net = cfg.create(jax.random.key(0))
    observation = make_observation(cfg)
    actions = jnp.zeros((1, cfg.action_horizon, cfg.action_dim), dtype=jnp.float32)
    prev_action = jnp.zeros((1, cfg.action_horizon, cfg.action_dim), dtype=jnp.float32)

    loss_obs = preprocess_if_external(observation, cfg, train=False)
    sample_obs = preprocess_if_external(observation, cfg, train=False)

    result = {
        "loss": np.asarray(net.compute_loss(jax.random.key(1), loss_obs, actions, train=False)),
        "sample_action_chunk": np.asarray(net.sample_action_chunk(jax.random.key(2), sample_obs, denoising_steps=2)),
        "sample_action_chunk_with_inference_time_rtc": np.asarray(net.sample_action_chunk_with_inference_time_rtc(jax.random.key(3), prev_action, sample_obs, denoising_steps=2)),
    }
    with Path(args.out).open("wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
'''


def run_branch(tree: Path, out: Path) -> None:
    env = os.environ.copy()
    paths = [str(tree / "src")]
    old_client = tree / "packages" / "openpi-client" / "src"
    if old_client.exists():
        paths.append(str(old_client))
    env["PYTHONPATH"] = os.pathsep.join(paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    subprocess.run([sys.executable, "-c", BRANCH_CODE, "--out", str(out)], cwd=tree, env=env, check=True)


def max_abs(lhs, rhs) -> float:
    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    if lhs.shape != rhs.shape:
        raise AssertionError(f"shape mismatch: {lhs.shape} vs {rhs.shape}")
    return float(np.max(np.abs(lhs - rhs))) if lhs.size else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-tree", required=True, type=Path)
    parser.add_argument("--candidate-tree", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as td:
        main_out = Path(td) / "main.pkl"
        cand_out = Path(td) / "candidate.pkl"
        run_branch(args.main_tree, main_out)
        run_branch(args.candidate_tree, cand_out)
        main_data = pickle.loads(main_out.read_bytes())
        cand_data = pickle.loads(cand_out.read_bytes())

    failed = False
    for key in ("loss", "sample_action_chunk", "sample_action_chunk_with_inference_time_rtc"):
        diff = max_abs(main_data[key], cand_data[key])
        print(f"{key}: max_abs_diff={diff:.10g}")
        failed = failed or diff > args.atol
    if failed:
        raise SystemExit(f"Model forward equivalence failed with atol={args.atol}")
    print("Model forward equivalence passed.")


if __name__ == "__main__":
    main()
