#!/usr/bin/env python3
"""Compare one policy.infer call between main and minimal branches."""

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
import inspect
import pickle
from pathlib import Path

import numpy as np
import torch

from openpi.policies import policy as policy_lib
from openpi.training import config as train_config

try:
    from openpi.data import transforms
except ImportError:
    import openpi.transforms as transforms


def make_runtime_obs(include_low, include_task):
    rng = np.random.default_rng(20260529)
    images = {
        "cam_high": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "cam_left_wrist": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "cam_right_wrist": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
    }
    if include_low:
        images["cam_low"] = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    obs = {
        "images": images,
        "state": rng.normal(size=(14,)).astype(np.float32),
        "prompt": "open bottle",
    }
    if include_task:
        obs["task"] = "open bottle"
    return obs


def image_keys_from_repack(data_config):
    for item in data_config.repack_transforms.inputs:
        if isinstance(item, transforms.RepackTransform):
            structure = item.structure
            images = structure.get("images") if isinstance(structure, dict) else None
            if isinstance(images, dict):
                return tuple(images)
    return None


def create_old_branch_transforms(cfg, assets_root):
    if assets_root is not None and hasattr(cfg, "assets_base_dir"):
        cfg = dataclasses.replace(cfg, assets_base_dir=str(Path(assets_root) / "assets"))
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    image_keys = image_keys_from_repack(data_config)
    input_transforms = [
        *([transforms.FilterImages(image_keys)] if image_keys is not None else []),
        transforms.InjectDefaultPrompt(None),
        *data_config.data_transforms.inputs,
        transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    output_transforms = [
        *data_config.model_transforms.outputs,
        transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ]
    include_low = image_keys is not None and "cam_low" in image_keys
    return input_transforms, output_transforms, include_low, False


def create_current_branch_transforms(cfg):
    pipe = cfg.data.transform_pipeline
    return pipe.policy_input_transforms(), pipe.policy_output_transforms(), "cam_low" in pipe.raw_image_keys, True


class FakeTorchPolicyModel:
    def __init__(self, action_horizon, action_dim):
        self.action_horizon = action_horizon
        self.action_dim = action_dim

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def sample_actions(self, device, observation, **kwargs):
        batch_size = observation.state.shape[0]
        total = batch_size * self.action_horizon * self.action_dim
        return torch.linspace(
            -0.25,
            0.25,
            total,
            dtype=observation.state.dtype,
            device=observation.state.device,
        ).reshape(batch_size, self.action_horizon, self.action_dim)

    def guided_inference(self, device, prev_action, observation, **kwargs):
        return self.sample_actions(device, observation, **kwargs)


def make_policy(model, input_transforms, output_transforms):
    kwargs = {
        "transforms": input_transforms,
        "output_transforms": output_transforms,
        "pytorch_device": "cpu",
        "is_pytorch": True,
    }
    params = inspect.signature(policy_lib.Policy).parameters
    return policy_lib.Policy(model, **{key: value for key, value in kwargs.items() if key in params})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--assets-root", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = train_config.get_config(args.config)
    if getattr(cfg.data, "transform_pipeline", None) is not None:
        input_transforms, output_transforms, include_low, include_task = create_current_branch_transforms(cfg)
    else:
        input_transforms, output_transforms, include_low, include_task = create_old_branch_transforms(cfg, args.assets_root)

    model = FakeTorchPolicyModel(
        action_horizon=getattr(cfg.model, "action_horizon", 50),
        action_dim=getattr(cfg.model, "action_dim", 32),
    )
    policy = make_policy(model, input_transforms, output_transforms)
    output = policy.infer(make_runtime_obs(include_low, include_task))
    output = {key: value for key, value in output.items() if key != "policy_timing"}

    with Path(args.out).open("wb") as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    main()
'''


def run_branch(tree: Path, config: str, out: Path, assets_root: Path) -> None:
    env = os.environ.copy()
    paths = [str(tree), str(tree / "src")]
    old_client = tree / "packages" / "openpi-client" / "src"
    if old_client.exists():
        paths.append(str(old_client))
    env["PYTHONPATH"] = os.pathsep.join(paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    subprocess.run(
        [
            sys.executable,
            "-c",
            BRANCH_CODE,
            "--config",
            config,
            "--assets-root",
            str(assets_root),
            "--out",
            str(out),
        ],
        cwd=assets_root,
        env=env,
        check=True,
    )


def max_abs(lhs, rhs) -> float:
    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    if lhs.shape != rhs.shape:
        raise AssertionError(f"shape mismatch: {lhs.shape} vs {rhs.shape}")
    return float(np.max(np.abs(lhs - rhs))) if lhs.size else 0.0


def compare_dict(prefix: str, main, candidate, atol: float) -> bool:
    ok = True
    if isinstance(main, dict) and isinstance(candidate, dict):
        main_keys = set(main)
        candidate_keys = set(candidate)
        if main_keys != candidate_keys:
            print(f"{prefix}: key mismatch main={sorted(main_keys)} candidate={sorted(candidate_keys)}")
            ok = False
        for key in sorted(main_keys & candidate_keys):
            ok = compare_dict(f"{prefix}.{key}" if prefix else key, main[key], candidate[key], atol) and ok
        return ok

    diff = max_abs(main, candidate)
    status = "OK" if diff <= atol else "FAIL"
    print(f"{prefix}: max_abs_diff={diff:.12g} [{status}]")
    return diff <= atol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-tree", required=True, type=Path)
    parser.add_argument("--candidate-tree", required=True, type=Path)
    parser.add_argument("--config", default="eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo")
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=None,
        help="Working directory used for relative assets paths. Defaults to --candidate-tree.",
    )
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()
    assets_root = args.assets_root or args.candidate_tree

    with tempfile.TemporaryDirectory(prefix="openpi_policy_equiv_") as td:
        main_out = Path(td) / "main.pkl"
        candidate_out = Path(td) / "candidate.pkl"
        run_branch(args.main_tree, args.config, main_out, assets_root)
        run_branch(args.candidate_tree, args.config, candidate_out, assets_root)
        main_output = pickle.loads(main_out.read_bytes())
        candidate_output = pickle.loads(candidate_out.read_bytes())

    ok = compare_dict("", main_output, candidate_output, args.atol)
    if not ok:
        raise SystemExit(f"Policy infer equivalence failed with atol={args.atol}")
    print("Policy infer equivalence passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
