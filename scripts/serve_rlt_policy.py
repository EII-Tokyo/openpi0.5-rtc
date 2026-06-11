from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

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
    config = token_model.RLTTokenConfig(**json.loads((path / "config.json").read_text()))
    return _load_pickle(path / "params.pkl"), config


def _load_actor(path: Path):
    config = actor_critic.RLTActorCriticConfig(**json.loads((path / "config.json").read_text()))
    return _load_pickle(path / "params.pkl"), config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--rlt-dir", default="")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--denoising-steps", type=int, default=10)
    args = parser.parse_args()

    train_config = _config.get_config(args.base_config)
    base_policy = _policy_config.create_trained_policy(
        train_config,
        args.base_checkpoint,
        sample_kwargs={"denoising_steps": args.denoising_steps},
    )
    token_params = actor_params = token_config = actor_config = None
    if args.rlt_dir:
        rlt_dir = Path(args.rlt_dir)
        token_path = rlt_dir / "rlt_token"
        actor_path = rlt_dir / "rlt_actor_critic"
        if token_path.exists() and actor_path.exists():
            token_params, token_config = _load_token(token_path)
            actor_params, actor_config = _load_actor(actor_path)
            logging.info("Loaded RLT checkpoints from %s", rlt_dir)
        else:
            logging.warning("RLT checkpoint missing under %s; serving base VLA fallback", rlt_dir)
    policy = rlt_policy.RLTPolicy(
        base_policy,
        token_params=token_params,
        actor_params=actor_params,
        token_config=token_config,
        actor_config=actor_config,
    )
    websocket_policy_server.WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
