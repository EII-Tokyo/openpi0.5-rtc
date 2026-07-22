#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from accelerate import init_empty_weights
from safetensors import safe_open


def _load_checkpoint_index(root: Path) -> tuple[dict[str, str], dict[str, list[int]]]:
    index_path = root / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    shapes: dict[str, list[int]] = {}
    for filename in sorted(set(weight_map.values())):
        path = root / filename
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                shapes[key] = list(handle.get_tensor(key).shape)
    return weight_map, shapes


def _make_empty_model(root: Path, *, add_value_head: bool, backend: str):
    cfg_json = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if backend == "openpi":
        from openpi.models.pi0_config import Pi0Config
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

        model_cfg = Pi0Config(
            pi05=True,
            action_dim=int(cfg_json["action_dim"]),
            action_horizon=int(cfg_json["action_horizon"]),
            paligemma_variant=cfg_json.get("paligemma_variant", "gemma_2b"),
            action_expert_variant=cfg_json.get("action_expert_variant", "gemma_300m"),
            dtype=cfg_json.get("precision", "bfloat16"),
        )
        with init_empty_weights():
            model = PI0Pytorch(model_cfg)
        return model
    if backend == "rlinf":
        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
        from rlinf.models.embodiment.openpi.openpi_action_model import (
            OpenPi0Config,
            OpenPi0ForRLActionPrediction,
        )

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
        return model
    raise ValueError(f"Unsupported backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--add-value-head", action="store_true")
    parser.add_argument("--backend", choices=["openpi", "rlinf"], default="openpi")
    args = parser.parse_args()
    root = Path(args.checkpoint)
    weight_map, ckpt_shapes = _load_checkpoint_index(root)
    model = _make_empty_model(root, add_value_head=args.add_value_head, backend=args.backend)
    model_shapes = {k: list(v.shape) for k, v in model.state_dict().items()}

    missing = sorted(set(model_shapes) - set(ckpt_shapes))
    unexpected = sorted(set(ckpt_shapes) - set(model_shapes))
    shape_mismatch = {
        key: {"model": model_shapes[key], "checkpoint": ckpt_shapes[key]}
        for key in sorted(set(model_shapes) & set(ckpt_shapes))
        if model_shapes[key] != ckpt_shapes[key]
    }
    payload = {
        "checkpoint": str(root),
        "add_value_head": args.add_value_head,
        "backend": args.backend,
        "model_key_count": len(model_shapes),
        "checkpoint_key_count": len(ckpt_shapes),
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "shape_mismatch_count": len(shape_mismatch),
        "missing_keys_sample": missing[:50],
        "unexpected_keys_sample": unexpected[:50],
        "shape_mismatch_sample": dict(list(shape_mismatch.items())[:50]),
        "strict_key_shape_pass": not missing and not unexpected and not shape_mismatch,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["strict_key_shape_pass"] else 2)


if __name__ == "__main__":
    main()
