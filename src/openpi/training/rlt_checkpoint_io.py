import dataclasses
import datetime
import hashlib
import json
import pathlib
import platform
import time

from flax import nnx
from flax import serialization
import jax
import numpy as np

from openpi.models import rlt
from openpi.training import rlt_replay_store
from openpi.training import rlt_training


def atomic_write_bytes(path: pathlib.Path, data: bytes) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


def atomic_write_text(path: pathlib.Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def with_runtime_beta(state: rlt_training.RLTTrainState, beta: float) -> rlt_training.RLTTrainState:
    model = nnx.merge(state.model_def, state.params)
    beta = float(beta)
    if float(model.config.beta) == beta:
        return state
    config = dataclasses.replace(model.config, beta=beta)
    model.config = config
    model.actor.config = config
    model.critic.q1.config = config
    model.critic.q2.config = config
    model.target_actor.config = config
    model.target_critic.q1.config = config
    model.target_critic.q2.config = config
    return dataclasses.replace(state, model_def=nnx.graphdef(model), params=nnx.state(model))


def load_inference_actor_checkpoint(
    state: rlt_training.RLTTrainState,
    checkpoint_dir: pathlib.Path,
) -> tuple[rlt_training.RLTTrainState, dict]:
    checkpoint_dir = resolve_inference_checkpoint_dir(checkpoint_dir)
    metadata_path = checkpoint_dir / "metadata.json"
    actor_path = checkpoint_dir / "actor.msgpack"
    critic_path = checkpoint_dir / "critic.msgpack"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {checkpoint_dir}")
    if not actor_path.exists():
        raise FileNotFoundError(f"actor.msgpack not found in {checkpoint_dir}")
    if not critic_path.exists():
        raise FileNotFoundError(f"critic.msgpack not found in {checkpoint_dir}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("type") != "rlt_inference_actor":
        raise ValueError(f"{checkpoint_dir} is not an RLT inference actor checkpoint")
    loaded_config = rlt.RLTConfig(**metadata["rlt_config"])
    model = nnx.merge(state.model_def, state.params)
    assert_compatible_rlt_config(model.config, loaded_config, checkpoint_dir)

    actor_state = nnx.state(model.actor)
    actor_pure = serialization.from_bytes(actor_state.to_pure_dict(), actor_path.read_bytes())
    actor_state.replace_by_pure_dict(actor_pure)
    nnx.update(model.actor, actor_state)
    target_actor_state = nnx.state(model.target_actor)
    target_actor_state.replace_by_pure_dict(actor_pure)
    nnx.update(model.target_actor, target_actor_state)

    critic_state = nnx.state(model.critic)
    critic_pure = serialization.from_bytes(critic_state.to_pure_dict(), critic_path.read_bytes())
    critic_state.replace_by_pure_dict(critic_pure)
    nnx.update(model.critic, critic_state)
    target_critic_state = nnx.state(model.target_critic)
    target_critic_state.replace_by_pure_dict(critic_pure)
    nnx.update(model.target_critic, target_critic_state)
    return dataclasses.replace(
        state,
        step=jax.numpy.asarray(0, dtype=jax.numpy.int32),
        params=nnx.state(model),
        model_def=nnx.graphdef(model),
    ), metadata


def resolve_inference_checkpoint_dir(path: pathlib.Path) -> pathlib.Path:
    path = path.expanduser()
    if path.name == "LATEST":
        return pathlib.Path(path.read_text().strip()).expanduser()
    if path.is_dir() and (path / "LATEST").exists():
        return pathlib.Path((path / "LATEST").read_text().strip()).expanduser()
    return path


def assert_compatible_rlt_config(current: rlt.RLTConfig, loaded: rlt.RLTConfig, checkpoint_dir: pathlib.Path) -> None:
    fields = ("z_dim", "proprio_dim", "action_horizon", "action_dim", "hidden_dim", "num_layers")
    mismatches = {
        field: (getattr(current, field), getattr(loaded, field))
        for field in fields
        if getattr(current, field) != getattr(loaded, field)
    }
    if mismatches:
        raise ValueError(f"Incompatible RLT checkpoint {checkpoint_dir}: {mismatches}")


def format_log_metric(key: str, value) -> str:
    if isinstance(value, bool):
        return f"{key}={int(value)}"
    if isinstance(value, int | float | np.number):
        return f"{key}={float(value):.4f}"
    return f"{key}={value}"


def reduce_numeric_infos(infos: list[dict[str, object]]) -> dict[str, float]:
    reduced: dict[str, float] = {}
    if not infos:
        return reduced
    for key in infos[0]:
        values = [item.get(key) for item in infos]
        if any(value is None for value in values):
            continue
        first = np.asarray(values[0])
        if first.ndim != 0 or first.dtype.kind not in "biuf":
            continue
        reduced[key] = float(np.mean([np.asarray(value) for value in values]))
    return reduced


def save_actor_for_inference(
    state: rlt_training.RLTTrainState,
    output_dir: pathlib.Path,
    step: int,
    *,
    action_horizon: int,
    replay_shape: rlt_replay_store.ReplayShape | None = None,
    train_shape: rlt_replay_store.ReplayShape | None = None,
    replay_stats: rlt_replay_store.ReplayStats | None = None,
    z_rl_normalization: rlt_replay_store.ZRLNormalization | None = None,
    source_script: str = "scripts/train_rlt_online.py",
) -> pathlib.Path:
    actor_dir = output_dir / "inference_actor" / f"{step:08d}"
    actor_dir.mkdir(parents=True, exist_ok=True)
    actor_params = rlt_training.actor_params_for_inference(state).to_pure_dict()
    critic_params = rlt_training.critic_params_for_inference(state).to_pure_dict()
    actor_bytes = serialization.to_bytes(actor_params)
    critic_bytes = serialization.to_bytes(critic_params)
    atomic_write_bytes(actor_dir / "actor.msgpack", actor_bytes)
    atomic_write_bytes(actor_dir / "critic.msgpack", critic_bytes)
    model = nnx.merge(state.model_def, state.params)
    actor_loss_config = {
        "actor_loss_mode": rlt_training.actor_loss_mode_name(int(state.actor_loss_mode)),
        "awbc_temperature": float(state.awbc_temperature),
        "awbc_max_weight": float(state.awbc_max_weight),
        "awbc_min_advantage": float(state.awbc_min_advantage),
        "awbc_max_action_delta_norm": float(state.awbc_max_action_delta_norm),
    }
    atomic_write_text(
        actor_dir / "metadata.json",
        json.dumps(
            {
                "format_version": 1,
                "created_at_unix": time.time(),
                "created_at_iso": datetime.datetime.now(datetime.UTC).isoformat(),
                "source_script": source_script,
                "host": platform.node(),
                "step": int(step),
                "type": "rlt_inference_actor",
                "note": "Stable actor export. Runtime should switch only at chunk/idle boundary.",
                "actor_file": "actor.msgpack",
                "actor_sha256": sha256_bytes(actor_bytes),
                "critic_file": "critic.msgpack",
                "critic_sha256": sha256_bytes(critic_bytes),
                "action_horizon": int(action_horizon),
                "rlt_config": dataclasses.asdict(model.config),
                "actor_loss_config": actor_loss_config,
                "replay_shape": _shape_metadata(replay_shape),
                "train_shape": _shape_metadata(train_shape),
                "replay_stats": _stats_metadata(replay_stats),
                "z_rl_normalization": _z_rl_normalization_metadata(z_rl_normalization),
            },
            indent=2,
            sort_keys=True,
        ),
    )
    latest_path = output_dir / "inference_actor" / "LATEST"
    atomic_write_text(latest_path, str(actor_dir))
    return actor_dir


def save_training_checkpoint(
    state: rlt_training.RLTTrainState,
    output_dir: pathlib.Path,
    step: int,
    store: rlt_replay_store.RLTReplayStore,
) -> pathlib.Path:
    checkpoint_dir = output_dir / "checkpoints" / f"{step:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(state.step),
        "params": _to_msgpackable_state(state.params),
        "actor_opt_state": _to_msgpackable_state(state.actor_opt_state),
        "critic_opt_state": _to_msgpackable_state(state.critic_opt_state),
    }
    atomic_write_bytes(checkpoint_dir / "train_state.msgpack", serialization.to_bytes(payload))
    atomic_write_text(
        checkpoint_dir / "metadata.json",
        json.dumps(
            {
                "step": int(state.step),
                "replay_stats": dataclasses.asdict(store.stats),
                "replay_shape": None if store.shape is None else dataclasses.asdict(store.shape),
                "train_shape": None if store.sample_shape is None else dataclasses.asdict(store.sample_shape),
                "loaded_shards": [str(path) for path in store.loaded_paths],
            },
            indent=2,
        ),
    )
    atomic_write_text(output_dir / "checkpoints" / "LATEST", str(checkpoint_dir))
    return checkpoint_dir


def state_for_actor_gate(
    state: rlt_training.RLTTrainState,
    *,
    actor_enabled: bool,
    policy_delay: int,
    actor_publish_interval: int,
) -> rlt_training.RLTTrainState:
    if actor_enabled:
        return dataclasses.replace(
            state,
            policy_delay=policy_delay,
            actor_publish_interval=actor_publish_interval,
        )
    return dataclasses.replace(
        state,
        policy_delay=1_000_000_000,
        actor_publish_interval=0,
    )


def _shape_metadata(shape: rlt_replay_store.ReplayShape | None) -> dict[str, int] | None:
    return None if shape is None else dataclasses.asdict(shape)


def _stats_metadata(stats: rlt_replay_store.ReplayStats | None) -> dict[str, int] | None:
    return None if stats is None else dataclasses.asdict(stats)


def _z_rl_normalization_metadata(stats: rlt_replay_store.ZRLNormalization | None) -> dict[str, list[float]] | None:
    if stats is None:
        return None
    return {
        "mean": np.asarray(stats.mean, dtype=np.float32).tolist(),
        "std": np.asarray(stats.std, dtype=np.float32).tolist(),
    }


def _to_msgpackable_state(value):
    state_dict = serialization.to_state_dict(value)
    return _convert_nnx_state_objects(state_dict)


def _convert_nnx_state_objects(value):
    if isinstance(value, nnx.State):
        return _convert_nnx_state_objects(value.to_pure_dict())
    if isinstance(value, dict):
        return {str(key): _convert_nnx_state_objects(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_convert_nnx_state_objects(item) for item in value]
    return value
