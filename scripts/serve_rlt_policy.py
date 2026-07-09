from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
import subprocess
import time

import numpy as np
import orbax.checkpoint as ocp

from openpi.data import transforms as _transforms
from openpi.rlt import actor_critic
from openpi.rlt import policy as rlt_policy
from openpi.rlt import token_model
from openpi.serving import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

DEFAULT_BASE_CONFIG = "eii_rinse_11repo_cam4_fullft"
DEFAULT_BASE_CHECKPOINT = "/home/eii/openpi0.5-rtc/checkpoints/eii_rinse_11repo_cam4_fullft/rinse_11repo_insertx5_fullft_bs256_nw64_fsdp8_20260513/9000"


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _load_token(path: Path):
    if (path / "rlt_token_config.json").exists() and (path / "params").exists():
        config = token_model.RLTTokenConfig(**json.loads((path / "rlt_token_config.json").read_text()))
        params = ocp.StandardCheckpointer().restore(path / "params")
        if isinstance(params, dict) and set(params) == {"params"}:
            params = params["params"]
        return params, config
    config = token_model.RLTTokenConfig(**json.loads((path / "config.json").read_text()))
    return _load_pickle(path / "params.pkl"), config


def _load_actor(path: Path):
    data = json.loads((path / "config.json").read_text())
    allowed = actor_critic.RLTActorCriticConfig.__dataclass_fields__.keys()
    config = actor_critic.RLTActorCriticConfig(**{key: data[key] for key in allowed if key in data})
    return _load_pickle(path / "params.pkl"), config


def _make_dummy_obs() -> dict:
    return _transforms.make_aloha_example()


def _warmup_policy(policy: rlt_policy.RLTPolicy, *, warmup_steps: int) -> None:
    if warmup_steps <= 0:
        logging.info("Skipping RLT policy warmup by request.")
        return
    dummy_obs = _make_dummy_obs()
    timings = []
    actor_modes = [False, True] if policy.actor_available else [False]
    for actor_enabled in actor_modes:
        prev_action = None
        for i in range(warmup_steps):
            start = time.monotonic()
            outputs = policy.infer(
                dummy_obs,
                chunking_mode="inference_time",
                prev_action=prev_action,
                rlt_actor_enabled=actor_enabled,
            )
            elapsed_ms = (time.monotonic() - start) * 1000
            timings.append(elapsed_ms)
            prev_action = outputs.get("model_actions")
            rlt_token = outputs.get("rlt_token")
            token_shape = np.asarray(rlt_token).shape if rlt_token is not None else None
            logging.info(
                "RLT warmup actor=%s step %d/%d finished in %.1f ms actions=%s rlt_token=%s",
                actor_enabled,
                i + 1,
                warmup_steps,
                elapsed_ms,
                np.asarray(outputs["actions"]).shape,
                token_shape,
            )
    if len(timings) > 1:
        logging.info("RLT warmup post-compile latency %.1f ms", timings[-1])


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--rlt-dir", default="")
    parser.add_argument("--rlt-token-checkpoint", default="")
    parser.add_argument("--rlt-actor-checkpoint", default="")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--denoising-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--rlt-actor-inference-reference-clip", type=float, default=None)
    args = parser.parse_args()

    train_config = _config.get_config(args.base_config)
    base_policy = _policy_config.create_trained_policy(
        train_config,
        args.base_checkpoint,
        sample_kwargs={"denoising_steps": args.denoising_steps},
    )
    token_params = actor_params = token_config = actor_config = None
    if args.rlt_token_checkpoint:
        token_params, token_config = _load_token(Path(args.rlt_token_checkpoint))
        logging.info("Loaded RLT token checkpoint from %s", args.rlt_token_checkpoint)
    if args.rlt_actor_checkpoint:
        actor_params, actor_config = _load_actor(Path(args.rlt_actor_checkpoint))
        logging.info("Loaded RLT actor checkpoint from %s", args.rlt_actor_checkpoint)
    if args.rlt_dir and not args.rlt_token_checkpoint and not args.rlt_actor_checkpoint:
        rlt_dir = Path(args.rlt_dir)
        token_path = rlt_dir / "rlt_token"
        actor_path = rlt_dir / "rlt_actor_critic"
        if not token_path.exists():
            raise ValueError(f"RLT token checkpoint missing under {token_path}")
        token_params, token_config = _load_token(token_path)
        if actor_path.exists():
            actor_params, actor_config = _load_actor(actor_path)
        logging.info("Loaded RLT checkpoints from %s", rlt_dir)
    if token_params is None or token_config is None:
        raise ValueError("RLT serving requires --rlt-token-checkpoint or rlt_dir/rlt_token.")
    policy = rlt_policy.RLTPolicy(
        base_policy,
        token_params=token_params,
        actor_params=actor_params,
        token_config=token_config,
        actor_config=actor_config,
        actor_inference_reference_clip=args.rlt_actor_inference_reference_clip,
        metadata={
            "rlt": {
                "base_config": args.base_config,
                "base_checkpoint": args.base_checkpoint,
                "rlt_dir": args.rlt_dir,
                "rlt_token_checkpoint": args.rlt_token_checkpoint,
                "rlt_actor_checkpoint": args.rlt_actor_checkpoint,
                "actor_enabled": actor_params is not None and actor_config is not None,
                "actor_inference_reference_clip": args.rlt_actor_inference_reference_clip,
                "denoising_steps": args.denoising_steps,
            },
            "code": {
                "git_revision": _git_revision(),
            },
        },
    )
    _warmup_policy(policy, warmup_steps=args.warmup_steps)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
