#!/usr/bin/env python3
"""Compare main vs minimal ALOHA input transforms on fixed synthetic samples."""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


OLD_TO_NEW_IMAGE_KEYS = {
    "base_0_rgb": "cam_high",
    "base_1_rgb": "cam_low",
    "left_wrist_0_rgb": "cam_left_wrist",
    "right_wrist_0_rgb": "cam_right_wrist",
}


BRANCH_CODE = r'''
import argparse
import pickle
from pathlib import Path

import numpy as np

from openpi.training import config as train_config

try:
    from openpi.data import transforms as transforms
except ImportError:
    import openpi.transforms as transforms


def get_data_config(cfg):
    if hasattr(cfg.data, "resolve"):
        return cfg.data.resolve(cfg.assets_dirs, cfg.model)
    return cfg.data


def transform_names(items):
    return [type(item).__name__ for item in items]


def make_lerobot_sample(include_low=True):
    rng = np.random.default_rng(20260527)
    sample = {
        "observation.images.cam_high": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation.images.cam_left_wrist": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation.images.cam_right_wrist": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation.state": rng.normal(size=(14,)).astype(np.float32),
        "action": rng.normal(size=(50, 14)).astype(np.float32),
        "task": "open bottle",
        "prompt": "open bottle",
        "subtask": "unscrew",
    }
    if include_low:
        sample["observation.images.cam_low"] = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    return sample


def make_runtime_obs(include_low=True):
    rng = np.random.default_rng(20260528)
    images = {
        "cam_high": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "cam_left_wrist": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "cam_right_wrist": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
    }
    if include_low:
        images["cam_low"] = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    return {
        "images": images,
        "state": rng.normal(size=(14,)).astype(np.float32),
        "task": "open bottle",
        "prompt": "open bottle",
        "subtask": "unscrew",
    }


def apply_all(data, items):
    data = dict(data)
    for item in items:
        data = item(data)
    return data


def training_transforms(data_config):
    if getattr(data_config, "transform_pipeline", None) is not None:
        return data_config.transform_pipeline.training_input_transforms()
    prompt_from_task = getattr(data_config, "prompt_from_task", True)
    return [
        *([transforms.PromptFromLeRobotTask()] if prompt_from_task else []),
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        transforms.Normalize(None, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]


def policy_transforms(data_config):
    if getattr(data_config, "transform_pipeline", None) is not None:
        return data_config.transform_pipeline.policy_input_transforms()

    image_keys = None
    for item in data_config.repack_transforms.inputs:
        if isinstance(item, transforms.RepackTransform):
            structure = item.structure
            images = structure.get("images") if isinstance(structure, dict) else None
            if isinstance(images, dict):
                image_keys = tuple(images)
                break
    return [
        *([transforms.FilterImages(image_keys)] if image_keys is not None else []),
        transforms.InjectDefaultPrompt(None),
        *data_config.data_transforms.inputs,
        transforms.Normalize(None, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]


def pack_output(data):
    return {
        "image_order": tuple(data["image"].keys()),
        "image": {key: np.asarray(value) for key, value in data["image"].items()},
        "image_mask": {key: np.asarray(value) for key, value in data["image_mask"].items()},
        "state": np.asarray(data["state"]),
        "actions": np.asarray(data["actions"]) if "actions" in data else None,
        "tokenized_prompt": np.asarray(data["tokenized_prompt"]),
        "tokenized_prompt_mask": np.asarray(data["tokenized_prompt_mask"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = train_config.get_config(args.config)
    data_config = get_data_config(cfg)
    include_low = "cam_low" in getattr(data_config, "transform_pipeline", data_config).raw_image_keys if getattr(data_config, "transform_pipeline", None) else any(
        isinstance(item, transforms.RepackTransform) and isinstance(item.structure, dict) and "cam_low" in item.structure.get("images", {})
        for item in data_config.repack_transforms.inputs
    )

    train_items = training_transforms(data_config)
    policy_items = policy_transforms(data_config)
    result = {
        "training_transform_names": transform_names(train_items),
        "policy_transform_names": transform_names(policy_items),
        "training": pack_output(apply_all(make_lerobot_sample(include_low=include_low), train_items)),
        "policy": pack_output(apply_all(make_runtime_obs(include_low=include_low), policy_items)),
    }
    with Path(args.out).open("wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
'''


def run_branch(tree: Path, config: str, out: Path, cwd: Path) -> None:
    env = os.environ.copy()
    paths = [str(tree / "src")]
    old_client = tree / "packages" / "openpi-client" / "src"
    if old_client.exists():
        paths.append(str(old_client))
    env["PYTHONPATH"] = os.pathsep.join(paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    subprocess.run(
        [sys.executable, "-c", BRANCH_CODE, "--config", config, "--out", str(out)],
        cwd=cwd,
        env=env,
        check=True,
    )


def canonical_image_key(key: str) -> str:
    return OLD_TO_NEW_IMAGE_KEYS.get(key, key)


def max_abs(lhs, rhs) -> float:
    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    if lhs.shape != rhs.shape:
        raise AssertionError(f"shape mismatch: {lhs.shape} vs {rhs.shape}")
    if lhs.dtype.kind in "bOUS" or rhs.dtype.kind in "bOUS":
        return 0.0 if np.array_equal(lhs, rhs) else float("inf")
    return float(np.max(np.abs(lhs - rhs))) if lhs.size else 0.0


def compare_section(name: str, main: dict, candidate: dict) -> dict[str, float]:
    diffs: dict[str, float] = {}
    main_order = [canonical_image_key(key) for key in main["image_order"]]
    cand_order = [canonical_image_key(key) for key in candidate["image_order"]]
    if main_order != cand_order:
        raise AssertionError(f"{name} image order mismatch: {main_order} vs {cand_order}")
    for old_key, new_key in zip(main["image_order"], cand_order, strict=True):
        diffs[f"{name}.image.{new_key}"] = max_abs(main["image"][old_key], candidate["image"][new_key])
        diffs[f"{name}.image_mask.{new_key}"] = max_abs(main["image_mask"][old_key], candidate["image_mask"][new_key])
    for key in ("state", "actions", "tokenized_prompt", "tokenized_prompt_mask"):
        if main[key] is None and candidate[key] is None:
            continue
        diffs[f"{name}.{key}"] = max_abs(main[key], candidate[key])
    return diffs


def main() -> None:
    parser = argparse.ArgumentParser()
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

    with tempfile.TemporaryDirectory() as td:
        main_out = Path(td) / "main.pkl"
        cand_out = Path(td) / "candidate.pkl"
        run_branch(args.main_tree, args.config, main_out, assets_root)
        run_branch(args.candidate_tree, args.config, cand_out, assets_root)
        main_data = pickle.loads(main_out.read_bytes())
        cand_data = pickle.loads(cand_out.read_bytes())

    print("main training transforms:", main_data["training_transform_names"])
    print("candidate training transforms:", cand_data["training_transform_names"])
    print("main policy transforms:", main_data["policy_transform_names"])
    print("candidate policy transforms:", cand_data["policy_transform_names"])

    diffs = {}
    diffs.update(compare_section("training", main_data["training"], cand_data["training"]))
    diffs.update(compare_section("policy", main_data["policy"], cand_data["policy"]))
    failed = False
    for key in sorted(diffs):
        print(f"{key}: max_abs_diff={diffs[key]:.10g}")
        failed = failed or diffs[key] > args.atol
    if failed:
        raise SystemExit(f"Transform equivalence failed with atol={args.atol}")
    print("Transform equivalence passed.")


if __name__ == "__main__":
    main()
