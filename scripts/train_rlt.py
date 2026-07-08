import dataclasses
import json
import logging
import pathlib
import platform
import time

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import tyro
import wandb

from openpi.models import rlt
from openpi.training import rlt_training


@dataclasses.dataclass
class Args:
    replay_npz: pathlib.Path
    output_dir: pathlib.Path = pathlib.Path("./checkpoints/rlt_actor_critic/debug")
    num_train_steps: int = 10_000
    batch_size: int = 64
    seed: int = 0
    log_interval: int = 50
    save_interval: int = 1_000
    policy_delay: int = 2
    actor_publish_interval: int = 500
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    reference_dropout: float = 0.3
    actor_output_mode: str = rlt.ACTOR_OUTPUT_MODE_RESIDUAL_CLIPPED
    train_action_horizon: int | None = 10
    expected_replay_action_horizon: int | None = 10
    wandb_enabled: bool = True
    wandb_project: str = "openpi"
    wandb_run_name: str = "rlt_actor_critic"
    overwrite: bool = False


def _init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d [%(levelname).1s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _require_arrays(data: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"Replay NPZ is missing required arrays: {missing}")
    return {key: data[key] for key in keys}


def _resolve_train_action_horizon(replay_action_horizon: int, train_action_horizon: int | None) -> int:
    horizon = replay_action_horizon if train_action_horizon is None else train_action_horizon
    if horizon <= 0:
        raise ValueError("train_action_horizon must be positive")
    if horizon > replay_action_horizon:
        raise ValueError(
            f"train_action_horizon={horizon} exceeds replay action_horizon={replay_action_horizon}"
        )
    if horizon != 10:
        logging.warning("RLT paper default train action horizon is C=10; got %d", horizon)
    return horizon


def _sample_batch(
    arrays: dict[str, np.ndarray],
    rng: np.random.Generator,
    batch_size: int,
    *,
    train_action_horizon: int | None = None,
) -> rlt_training.RLTReplayBatch:
    replay_size = len(arrays["z_rl"])
    indices = rng.integers(0, replay_size, size=batch_size)
    horizon = _resolve_train_action_horizon(arrays["action"].shape[-2], train_action_horizon)
    return rlt_training.make_replay_batch(
        z_rl=jnp.asarray(arrays["z_rl"][indices]),
        proprio=jnp.asarray(arrays["proprio"][indices]),
        action=jnp.asarray(arrays["action"][indices, :horizon]),
        reference_action=jnp.asarray(arrays["reference_action"][indices, :horizon]),
        reward_seq=jnp.asarray(arrays["reward_seq"][indices, :horizon]),
        next_z_rl=jnp.asarray(arrays["next_z_rl"][indices]),
        next_proprio=jnp.asarray(arrays["next_proprio"][indices]),
        next_reference_action=jnp.asarray(arrays["next_reference_action"][indices, :horizon]),
        done=jnp.asarray(arrays["done"][indices].astype(np.bool_)),
    )


def _save_actor_for_inference(
    state: rlt_training.RLTTrainState,
    output_dir: pathlib.Path,
    step: int,
    *,
    action_horizon: int,
) -> None:
    actor_dir = output_dir / "inference_actor" / f"{step:08d}"
    actor_dir.mkdir(parents=True, exist_ok=True)
    actor_params = rlt_training.actor_params_for_inference(state).to_pure_dict()
    (actor_dir / "actor.msgpack").write_bytes(serialization.to_bytes(actor_params))
    (actor_dir / "metadata.json").write_text(
        json.dumps(
            {
                "step": step,
                "type": "rlt_inference_actor",
                "note": "Stable actor export. Runtime should switch only at chunk/idle boundary.",
                "action_horizon": action_horizon,
            },
            indent=2,
        )
    )
    latest_path = output_dir / "inference_actor" / "LATEST"
    latest_path.write_text(str(actor_dir))


def main(args: Args) -> None:
    _init_logging()
    logging.info("Running on %s", platform.node())
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists. Pass --overwrite to replace it.")
        import shutil

        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    required_keys = (
        "z_rl",
        "proprio",
        "action",
        "reference_action",
        "reward_seq",
        "next_z_rl",
        "next_proprio",
        "next_reference_action",
        "done",
    )
    arrays = _require_arrays(np.load(args.replay_npz), required_keys)
    replay_size = len(arrays["z_rl"])
    if replay_size < args.batch_size:
        raise ValueError(f"Replay size {replay_size} is smaller than batch_size {args.batch_size}.")
    logging.info("Loaded replay NPZ with %d transitions", replay_size)

    z_dim = int(arrays["z_rl"].shape[-1])
    proprio_dim = int(arrays["proprio"].shape[-1])
    replay_action_horizon = int(arrays["action"].shape[-2])
    if args.expected_replay_action_horizon is not None and replay_action_horizon != args.expected_replay_action_horizon:
        raise ValueError(
            f"Expected replay action horizon {args.expected_replay_action_horizon}, got {replay_action_horizon}"
        )
    action_horizon = _resolve_train_action_horizon(replay_action_horizon, args.train_action_horizon)
    action_dim = int(arrays["action"].shape[-1])
    config = rlt_training.RLTTrainingConfig(
        model=rlt.RLTConfig(
            z_dim=z_dim,
            proprio_dim=proprio_dim,
            action_horizon=action_horizon,
            action_dim=action_dim,
            reference_dropout=args.reference_dropout,
            actor_output_mode=args.actor_output_mode,
        ),
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        policy_delay=args.policy_delay,
        actor_publish_interval=args.actor_publish_interval,
    )
    state = rlt_training.init_train_state(config, jax.random.key(args.seed))
    replay_rng = np.random.default_rng(args.seed)

    if args.wandb_enabled:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                **{key: str(value) if isinstance(value, pathlib.Path) else value for key, value in dataclasses.asdict(args).items()},
                "replay_size": replay_size,
                "replay_action_horizon": replay_action_horizon,
                "train_action_horizon": action_horizon,
            },
        )
    else:
        wandb.init(mode="disabled")

    _save_actor_for_inference(state, args.output_dir, 0, action_horizon=action_horizon)
    start = time.perf_counter()
    infos = []
    for step in tqdm.tqdm(range(args.num_train_steps), total=args.num_train_steps, dynamic_ncols=True):
        batch = _sample_batch(arrays, replay_rng, args.batch_size, train_action_horizon=action_horizon)
        train_rng = jax.random.fold_in(jax.random.key(args.seed), step)
        state, info = rlt_training.train_step(state, batch, train_rng)
        infos.append(jax.device_get(info))

        next_step = int(state.step)
        if bool(info["publish_actor"]):
            _save_actor_for_inference(state, args.output_dir, next_step, action_horizon=action_horizon)
        if next_step % args.save_interval == 0:
            _save_actor_for_inference(state, args.output_dir / "snapshots", next_step, action_horizon=action_horizon)
        if next_step % args.log_interval == 0:
            reduced = {
                key: float(np.mean([np.asarray(item[key]) for item in infos]))
                for key in infos[0]
                if np.asarray(infos[0][key]).ndim == 0
            }
            reduced["replay_size"] = replay_size
            reduced["steps_per_sec"] = args.log_interval / max(time.perf_counter() - start, 1e-6)
            wandb.log({f"rlt/{key}": value for key, value in reduced.items()}, step=next_step)
            logging.info("step=%d %s", next_step, " ".join(f"{k}={v:.4f}" for k, v in reduced.items()))
            infos = []
            start = time.perf_counter()


if __name__ == "__main__":
    main(tyro.cli(Args))
