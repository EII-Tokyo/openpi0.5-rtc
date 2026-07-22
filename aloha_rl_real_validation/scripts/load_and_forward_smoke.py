#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch


def _build_model(root: Path, *, add_value_head: bool):
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    cfg_json = json.loads((root / "config.json").read_text(encoding="utf-8"))
    train_config = get_openpi_config("pi05_aloha_robotwin", model_path=str(root))
    model_cfg = OpenPi0Config(**train_config.model.__dict__)
    model_cfg.__dict__["config_name"] = "pi05_aloha_robotwin"
    model_cfg.__dict__["action_dim"] = int(cfg_json["action_dim"])
    model_cfg.__dict__["action_horizon"] = int(cfg_json["action_horizon"])
    model_cfg.__dict__["action_chunk"] = int(cfg_json["action_horizon"])
    model_cfg.__dict__["action_env_dim"] = 14
    model_cfg.__dict__["num_images_in_input"] = 3
    model_cfg.__dict__["num_steps"] = 5
    model_cfg.__dict__["noise_level"] = 0.3
    model_cfg.__dict__["train_expert_only"] = False
    model_cfg.__dict__["add_value_head"] = add_value_head
    model_cfg.__dict__["detach_critic_input"] = True
    with init_empty_weights():
        model = OpenPi0ForRLActionPrediction(model_cfg)
    data_config = train_config.data.create(train_config.assets_dirs, model_cfg)
    from openpi.training import checkpoints as _checkpoints
    import openpi.transforms as transforms

    norm_stats = _checkpoints.load_norm_stats(root, data_config.asset_id)
    model.setup_wrappers(
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--add-value-head", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    root = Path(args.checkpoint)

    started = time.time()
    model = _build_model(root, add_value_head=args.add_value_head)
    # Accelerate understands model.safetensors.index.json and avoids the
    # all_state_dict merge used by RLinf's default helper.
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(root),
        device_map={"": args.device},
        no_split_module_classes=["GemmaDecoderLayer"],
    )
    model.eval()
    load_s = time.time() - started

    b = 1
    env_obs = {
        "main_images": torch.zeros((b, 224, 224, 3), dtype=torch.uint8, device=args.device),
        "wrist_images": torch.zeros((b, 2, 224, 224, 3), dtype=torch.uint8, device=args.device),
        "extra_view_images": None,
        "states": torch.zeros((b, 14), dtype=torch.float32, device=args.device),
        "task_descriptions": ["adjust bottle"] * b,
    }
    latencies = []
    out_shape = None
    finite = True
    with torch.no_grad():
        for _ in range(args.iterations):
            t0 = time.time()
            actions, _info = model.predict_action_batch(env_obs, mode="eval", compute_values=False)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            latencies.append(time.time() - t0)
            out_shape = list(actions.shape)
            finite = finite and bool(torch.isfinite(actions).all().item())
    peak = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
    payload = {
        "checkpoint": str(root),
        "add_value_head": args.add_value_head,
        "load_s": load_s,
        "iterations": args.iterations,
        "latency_s": latencies,
        "output_shape": out_shape,
        "finite": finite,
        "cuda_peak_allocated_bytes": int(peak),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if finite else 2)


if __name__ == "__main__":
    main()
