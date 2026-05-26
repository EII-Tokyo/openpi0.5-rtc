#!/usr/bin/env python3
"""Compare one train_step between main and minimal branches on a fixed batch."""

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
import dataclasses
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
from flax import traverse_util

from openpi.models import model as model_base
from openpi.models import pi0_config
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as train_config
from openpi.training import sharding
from scripts import train as train_script


OLD_TO_NEW_IMAGE_KEYS = {
    "base_0_rgb": "cam_high",
    "base_1_rgb": "cam_low",
    "left_wrist_0_rgb": "cam_left_wrist",
    "right_wrist_0_rgb": "cam_right_wrist",
}


def replace_if_field(obj, **kwargs):
    fields = {field.name for field in dataclasses.fields(obj)}
    return dataclasses.replace(obj, **{key: value for key, value in kwargs.items() if key in fields})


def make_dummy_model_config():
    kwargs = {
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
        "image_resolution": (64, 64),
        "action_dim": 14,
        "action_horizon": 4,
        "max_token_len": 32,
        "pi05": True,
    }
    fields = {field.name for field in dataclasses.fields(pi0_config.Pi0Config)}
    return pi0_config.Pi0Config(**{key: value for key, value in kwargs.items() if key in fields})


def image_index(name):
    canonical = OLD_TO_NEW_IMAGE_KEYS.get(name, name)
    order = {
        "cam_high": 0,
        "cam_low": 1,
        "cam_left_wrist": 2,
        "cam_right_wrist": 3,
    }
    return order.get(canonical, 9)


def make_observation_and_actions(model_cfg, batch_size):
    obs_spec, actions_spec = model_cfg.inputs_spec(batch_size=batch_size)
    image_keys = sorted(obs_spec.images.keys(), key=image_index)
    images = {}
    image_masks = {}
    for idx, key in enumerate(image_keys):
        shape = obs_spec.images[key].shape
        total = int(np.prod(shape))
        value = np.linspace(-1.0, 1.0, total, dtype=np.float32).reshape(shape)
        images[key] = jnp.asarray(value + idx * 0.01)
        image_masks[key] = jnp.ones((batch_size,), dtype=jnp.bool_)

    state_shape = obs_spec.state.shape
    state = jnp.asarray(np.linspace(-0.5, 0.5, int(np.prod(state_shape)), dtype=np.float32).reshape(state_shape))

    token_shape = obs_spec.tokenized_prompt.shape
    tokenized_prompt = jnp.arange(int(np.prod(token_shape)), dtype=jnp.int32).reshape(token_shape) % 32
    tokenized_prompt_mask = jnp.ones(token_shape, dtype=jnp.bool_)

    action_shape = actions_spec.shape
    actions = jnp.asarray(np.linspace(-0.25, 0.25, int(np.prod(action_shape)), dtype=np.float32).reshape(action_shape))

    observation = model_base.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )
    return observation, actions, image_keys


def flatten_params(params):
    if hasattr(params, "to_pure_dict"):
        params = params.to_pure_dict()
    flat = traverse_util.flatten_dict(params, sep="/")
    return {key: np.asarray(value) for key, value in flat.items()}


def tree_leaves_as_arrays(tree):
    return [np.asarray(value) for value in jax.tree_util.tree_leaves(tree)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = train_config.get_config("debug")
    dummy_model = make_dummy_model_config()
    cfg = replace_if_field(
        cfg,
        batch_size=1,
        num_workers=0,
        gradient_accumulation_steps=1,
        freeze_filter=nnx.Not(nnx_utils.PathRegex("action_out_proj/.*")),
        ema_decay=None,
        log_interval=1,
        save_interval=1000,
        overwrite=True,
    )
    object.__setattr__(cfg, "model", dummy_model)

    mesh = sharding.make_mesh(1)
    rng = jax.random.key(0)
    step_rng = jax.random.key(20260527)

    with sharding.set_mesh(mesh):
        state, _ = train_script.init_train_state(cfg, rng, mesh, resume=False)
        observation, actions, image_keys = make_observation_and_actions(cfg.model, cfg.batch_size)
        new_state, metrics = train_script.train_step(cfg, step_rng, state, (observation, actions))

    payload = {
        "image_keys": image_keys,
        "metrics": {key: np.asarray(value) for key, value in metrics.items()},
        "params": flatten_params(new_state.params),
        "opt_state": tree_leaves_as_arrays(new_state.opt_state),
        "step": np.asarray(new_state.step),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(payload, f)


if __name__ == "__main__":
    main()
'''


def run_branch(tree: Path, label: str, tmpdir: Path) -> dict:
    script = tmpdir / f"{label}_runner.py"
    out = tmpdir / f"{label}.pkl"
    script.write_text(BRANCH_CODE)

    env = os.environ.copy()
    pythonpath = [str(tree), str(tree / "src")]
    client_src = tree / "packages" / "openpi-client" / "src"
    if client_src.exists():
        pythonpath.append(str(client_src))
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    subprocess.run(
        [sys.executable, str(script), "--out", str(out)],
        cwd=str(tree),
        env=env,
        check=True,
    )
    with open(out, "rb") as f:
        return pickle.load(f)


def max_abs_diff(a, b) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def compare_named_arrays(title: str, main: dict, candidate: dict, atol: float) -> bool:
    ok = True
    main_keys = set(main)
    candidate_keys = set(candidate)
    missing = sorted(main_keys - candidate_keys)
    extra = sorted(candidate_keys - main_keys)
    if missing or extra:
        print(f"{title}: key mismatch")
        if missing:
            print(f"  missing in candidate: {missing[:20]}")
        if extra:
            print(f"  extra in candidate: {extra[:20]}")
        ok = False

    for key in sorted(main_keys & candidate_keys):
        diff = max_abs_diff(main[key], candidate[key])
        status = "OK" if diff <= atol else "FAIL"
        print(f"{title}.{key}: max_abs_diff={diff:.12g} [{status}]")
        ok = ok and diff <= atol
    return ok


def compare_leaf_lists(title: str, main: list, candidate: list, atol: float) -> bool:
    ok = True
    if len(main) != len(candidate):
        print(f"{title}: leaf count mismatch main={len(main)} candidate={len(candidate)}")
        ok = False

    for idx, (left, right) in enumerate(zip(main, candidate)):
        diff = max_abs_diff(left, right)
        status = "OK" if diff <= atol else "FAIL"
        print(f"{title}[{idx}]: max_abs_diff={diff:.12g} [{status}]")
        ok = ok and diff <= atol
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-tree", required=True, type=Path, help="Path to main branch worktree")
    parser.add_argument("--candidate-tree", required=True, type=Path, help="Path to minimal/current branch worktree")
    parser.add_argument("--atol", type=float, default=0.0, help="Allowed max_abs_diff")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="openpi_train_step_equiv_") as d:
        tmpdir = Path(d)
        print(f"Running main branch from {args.main_tree}")
        main_payload = run_branch(args.main_tree.resolve(), "main", tmpdir)
        print(f"Running candidate branch from {args.candidate_tree}")
        candidate_payload = run_branch(args.candidate_tree.resolve(), "candidate", tmpdir)

    print(f"main image_keys: {main_payload['image_keys']}")
    print(f"candidate image_keys: {candidate_payload['image_keys']}")

    ok = True
    ok = compare_named_arrays("metrics", main_payload["metrics"], candidate_payload["metrics"], args.atol) and ok
    ok = compare_named_arrays("params", main_payload["params"], candidate_payload["params"], args.atol) and ok
    ok = compare_leaf_lists("opt_state", main_payload["opt_state"], candidate_payload["opt_state"], args.atol) and ok

    step_diff = max_abs_diff(main_payload["step"], candidate_payload["step"])
    print(f"step: max_abs_diff={step_diff:.12g} [{'OK' if step_diff <= args.atol else 'FAIL'}]")
    ok = ok and step_diff <= args.atol
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
