from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
import time

import numpy as np
import orbax.checkpoint as ocp

from openpi.data import transforms as _transforms
from openpi.rlt import policy as rlt_policy
from openpi.rlt import token_model
from openpi.serving import policy_config as _policy_config
from openpi.training import config as _config


DEFAULT_BASE_CONFIG = "eii_rinse_11repo_cam4_fullft"
DEFAULT_BASE_CHECKPOINT = (
    "/app/checkpoints/eii_rinse_11repo_cam4_fullft/"
    "rinse_11repo_insertx5_fullft_bs256_nw64_fsdp8_20260513/9000"
)
DEFAULT_RLT_TOKEN_CHECKPOINT = "/app/rlt_runs/rlt_token_rinse_9000_bs64_nw4_warmup2000_10000_abs/9999"
DEFAULT_REPLAY_DIR = "/app/data/rlt_online_replay/rinse_smoke"


def _load_token(path: Path):
    config_path = path / "rlt_token_config.json"
    if not config_path.exists():
        config_path = path / "config.json"
    config = token_model.RLTTokenConfig(**json.loads(config_path.read_text()))
    params = ocp.StandardCheckpointer().restore(path / "params")
    if isinstance(params, dict) and set(params) == {"params"}:
        params = params["params"]
    return params, config


def _metadata(data: np.lib.npyio.NpzFile) -> dict:
    if "metadata_json" not in data:
        return {}
    return json.loads(str(data["metadata_json"]))


def _metadata_with_sidecar(path: Path, data: np.lib.npyio.NpzFile) -> dict:
    meta = _metadata(data)
    for suffix in (".label.json", ".trim.json"):
        sidecar = path.with_suffix(path.suffix + suffix)
        if sidecar.exists():
            meta.update(json.loads(sidecar.read_text()))
    return meta


def _image_keys(data: np.lib.npyio.NpzFile) -> list[str]:
    keys = []
    for key in data.files:
        if key.startswith("image_") and not key.startswith("image_mask_"):
            keys.append(key.removeprefix("image_"))
    return sorted(keys)


def _observation_at(
    data: np.lib.npyio.NpzFile,
    image_arrays: dict[str, np.ndarray],
    step: int,
    *,
    task: str,
    subtask: str,
) -> dict:
    images = {}
    for key, value in image_arrays.items():
        frame = value[step]
        if frame.size:
            images[key] = frame
    return {
        "state": np.asarray(data["raw_state"][step], dtype=np.float32),
        "images": images,
        "task": task,
        "subtask": subtask,
    }


def _action_chunk(actions: np.ndarray, start: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    action_dim = actions.shape[-1]
    chunk = np.zeros((horizon, action_dim), dtype=np.float32)
    mask = np.zeros((horizon,), dtype=np.bool_)
    end = min(start + horizon, actions.shape[0])
    valid = max(end - start, 0)
    if valid:
        chunk[:valid] = actions[start:end]
        mask[:valid] = True
    return chunk, mask


def _pad_action_dim(chunk: np.ndarray, action_dim: int) -> np.ndarray:
    if chunk.shape[-1] == action_dim:
        return chunk
    if chunk.shape[-1] > action_dim:
        return chunk[..., :action_dim]
    padded = np.zeros((*chunk.shape[:-1], action_dim), dtype=chunk.dtype)
    padded[..., : chunk.shape[-1]] = chunk
    return padded


def _terminal_label(meta: dict) -> tuple[str, int]:
    label = str(meta.get("terminal_label", "unlabeled"))
    if label == "success":
        return label, 1
    if label == "failure":
        return label, 0
    return label, -1


def _output_path(path: Path, output_dir: Path | None, stride: int, next_offset_steps: int) -> Path:
    name = f"{path.stem}.replay_stride{stride}_next{next_offset_steps}.npz"
    if output_dir is None:
        return path.with_name(name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / name


def _discounted_terminal_return(terminal_success: int, horizon: int, gamma: float) -> float:
    if terminal_success < 0:
        return 0.0
    return float(terminal_success) * float(gamma ** max(horizon - 1, 0))


def _make_action_transform(train_config: _config.TrainConfig, checkpoint_dir: str | Path):
    checkpoint_dir = Path(checkpoint_dir)
    transform_pipeline = train_config.data.transform_pipeline
    if transform_pipeline is None:
        raise ValueError("A transform pipeline is required for RLT replay generation.")
    checkpoint_assets = _transforms.AssetsConfig(
        assets_dir=str(checkpoint_dir / "assets"),
        asset_id=transform_pipeline.assets.asset_id,
    )
    transform_pipeline = dataclasses.replace(transform_pipeline, assets=checkpoint_assets)
    return _transforms.compose(
        [
            *transform_pipeline.raw_state_action_transforms(),
            _transforms.Normalize(transform_pipeline._require_norm_stats(), use_quantiles=transform_pipeline.use_quantile_norm),
        ]
    )


def _normalized_action_chunk(
    action_transform,
    *,
    data: np.lib.npyio.NpzFile,
    image_arrays: dict[str, np.ndarray],
    actions: np.ndarray,
    local_step: int,
    source_step: int,
    horizon: int,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw_chunk, mask = _action_chunk(actions, local_step, horizon)
    task = str(data["task"][source_step])
    subtask = str(data["subtask"][source_step])
    sample = {
        "observation": {
            "state": np.asarray(data["raw_state"][source_step], dtype=np.float32),
            "images": {key: value[source_step] for key, value in image_arrays.items()},
        },
        "action": raw_chunk,
        "task": task,
        "subtask": subtask,
    }
    transformed = action_transform(sample)
    return np.asarray(transformed["actions"], dtype=np.float32)[..., :action_dim], mask


def _process_episode(
    policy: rlt_policy.RLTPolicy,
    episode_path: Path,
    *,
    output_dir: Path | None,
    stride: int,
    action_horizon: int,
    action_dim: int,
    next_offset_steps: int,
    gamma: float,
    action_transform,
    max_transitions: int | None,
    force: bool,
    save_embeddings: bool,
) -> Path | None:
    output_path = _output_path(episode_path, output_dir, stride, next_offset_steps)
    if output_path.exists() and not force:
        logging.info("Skipping existing output: %s", output_path)
        return output_path

    start_time = time.monotonic()
    logging.info("Loading episode: %s", episode_path)
    with np.load(episode_path, allow_pickle=False) as data:
        meta = _metadata_with_sidecar(episode_path, data)
        label, terminal_success = _terminal_label(meta)
        if "raw_state" not in data or "executed_action" not in data:
            raise KeyError(f"{episode_path} missing raw_state or executed_action")
        if "task" not in data or "subtask" not in data:
            raise KeyError(f"{episode_path} missing task or subtask; collect replay with task/subtask fields")
        image_keys = _image_keys(data)
        if not image_keys:
            raise KeyError(f"{episode_path} has no image_* arrays")
        image_arrays = {key: data[f"image_{key}"] for key in image_keys}
        all_actions = np.asarray(data["executed_action"], dtype=np.float32)
        raw_num_steps = int(all_actions.shape[0])
        trim_start_step = int(meta.get("trim_start_step", 0) or 0)
        trim_end_step = int(meta.get("trim_end_step", raw_num_steps) or raw_num_steps)
        trim_start_step = max(0, min(trim_start_step, raw_num_steps - 1))
        trim_end_step = max(trim_start_step + 1, min(trim_end_step, raw_num_steps))
        actions = all_actions[trim_start_step:trim_end_step]
        cropped_num_steps = int(actions.shape[0])
        num_steps = cropped_num_steps - (cropped_num_steps % 2)
        if num_steps <= 0:
            raise ValueError(f"{episode_path} has no usable even-length crop: {trim_start_step}:{trim_end_step}")
        actions = actions[:num_steps]
        source_steps = np.arange(trim_start_step, trim_start_step + num_steps, dtype=np.int32)
        terminal_local_step = num_steps - 1
        terminal_source_step = int(source_steps[terminal_local_step])
        steps = np.arange(0, num_steps, stride, dtype=np.int32)
        if max_transitions is not None:
            steps = steps[:max_transitions]
        next_steps = np.minimum(steps + next_offset_steps, terminal_local_step).astype(np.int32)
        source_training_steps = source_steps[steps]
        source_next_steps = source_steps[next_steps]
        required_source_steps = np.unique(np.concatenate([source_training_steps, source_next_steps])).astype(np.int32)

        tokens_by_step = {}
        embeddings_by_step = {}
        masks_by_step = {}
        states_by_step = {}
        reference_by_step = {}
        task_by_step = {}
        subtask_by_step = {}
        previous_model_action: np.ndarray | None = None

        for idx, step in enumerate(required_source_steps):
            step_task = str(data["task"][int(step)])
            step_subtask = str(data["subtask"][int(step)])
            obs = _observation_at(data, image_arrays, int(step), task=step_task, subtask=step_subtask)
            out = policy.infer(obs, chunking_mode="inference_time", prev_action=previous_model_action)
            token = np.asarray(out["rlt_token"], dtype=np.float32)
            state = np.asarray(out["rlt_state"], dtype=np.float32)
            reference = np.asarray(out["rlt_reference_action_chunk"], dtype=np.float32)[..., :action_dim]
            previous_model_action = np.asarray(out["model_actions"], dtype=np.float32)

            step_key = int(step)
            tokens_by_step[step_key] = token
            if save_embeddings:
                embeddings_by_step[step_key] = np.asarray(out["rlt_embeddings"], dtype=np.float32)
                masks_by_step[step_key] = np.asarray(out["rlt_mask"], dtype=np.bool_)
            states_by_step[step_key] = state
            reference_by_step[step_key] = reference[:action_horizon]
            task_by_step[step_key] = step_task
            subtask_by_step[step_key] = step_subtask
            if idx == 0 or (idx + 1) % 30 == 0:
                logging.info(
                    "%s generated %d/%d policy contexts last_step=%d",
                    episode_path.name,
                    idx + 1,
                    len(required_source_steps),
                    int(step),
                )

        token_arr = np.asarray([tokens_by_step[int(step)] for step in source_training_steps], dtype=np.float32)
        next_token_arr = np.asarray([tokens_by_step[int(step)] for step in source_next_steps], dtype=np.float32)
        state_arr = np.asarray([states_by_step[int(step)] for step in source_training_steps], dtype=np.float32)
        next_state_arr = np.asarray([states_by_step[int(step)] for step in source_next_steps], dtype=np.float32)
        ref_arr = np.asarray([reference_by_step[int(step)] for step in source_training_steps], dtype=np.float32)
        next_ref_arr = np.asarray([reference_by_step[int(step)] for step in source_next_steps], dtype=np.float32)

        executed_chunks = []
        executed_masks = []
        for local_step, source_step in zip(steps, source_training_steps, strict=True):
            executed, executed_mask = _normalized_action_chunk(
                action_transform,
                data=data,
                image_arrays=image_arrays,
                actions=actions,
                local_step=int(local_step),
                source_step=int(source_step),
                horizon=action_horizon,
                action_dim=action_dim,
            )
            executed_chunks.append(executed[:action_horizon])
            executed_masks.append(executed_mask[:action_horizon])

        task_arr = np.asarray([task_by_step[int(step)] for step in source_training_steps])
        subtask_arr = np.asarray([subtask_by_step[int(step)] for step in source_training_steps])

    executed_arr = np.asarray(executed_chunks, dtype=np.float32)
    executed_mask_arr = np.asarray(executed_masks, dtype=np.bool_)
    done = np.zeros((len(steps),), dtype=np.float32)
    reward = np.zeros((len(steps),), dtype=np.float32)
    if len(steps):
        done[-1] = 1.0
        reward[-1] = _discounted_terminal_return(terminal_success, action_horizon, gamma)

    source_episode = np.asarray(str(episode_path))
    output_meta = {
        "format": "rlt_replay_buffer_v1",
        "source_episode": str(episode_path),
        "stride": stride,
        "next_offset_steps": next_offset_steps,
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "raw_num_steps": raw_num_steps,
        "trim_start_step": trim_start_step,
        "trim_end_step": trim_end_step,
        "cropped_num_steps": num_steps,
        "terminal_step": terminal_source_step,
        "context_sampling_mode": "inference_time_rtc",
        "gamma": gamma,
        "save_embeddings": save_embeddings,
        "terminal_label": label,
        "terminal_success": terminal_success,
        "created_at": time.time(),
    }
    payload = {
        "metadata_json": np.asarray(json.dumps(output_meta, ensure_ascii=False)),
        "task": task_arr,
        "subtask": subtask_arr,
        "rlt_token": token_arr,
        "next_rlt_token": next_token_arr,
        "state": state_arr,
        "next_state": next_state_arr,
        "reference_action_chunk": ref_arr,
        "next_reference_action_chunk": next_ref_arr,
        "executed_action_chunk": executed_arr,
        "executed_action_mask": executed_mask_arr,
        "reward": reward,
        "done": done,
        "episode_id": np.zeros((len(steps),), dtype=np.int32),
        "step_index": source_training_steps,
        "local_step_index": steps,
        "next_step_index": source_next_steps,
        "is_intervention": np.zeros((len(steps),), dtype=np.float32),
        "source_episode": source_episode,
    }
    if save_embeddings:
        embeddings_arr = np.asarray([embeddings_by_step[int(step)] for step in source_training_steps], dtype=np.float32)
        next_embeddings_arr = np.asarray([embeddings_by_step[int(step)] for step in source_next_steps], dtype=np.float32)
        mask_arr = np.asarray([masks_by_step[int(step)] for step in source_training_steps], dtype=np.bool_)
        next_mask_arr = np.asarray([masks_by_step[int(step)] for step in source_next_steps], dtype=np.bool_)
        payload.update(
            {
                "rlt_embeddings": embeddings_arr,
                "rlt_mask": mask_arr,
                "next_rlt_embeddings": next_embeddings_arr,
                "next_rlt_mask": next_mask_arr,
            }
        )
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    with tmp_path.open("wb") as f:
        np.savez(f, **payload)
    tmp_path.replace(output_path)
    logging.info(
        "Saved %s transitions=%d size=%.1fMB elapsed=%.1fs",
        output_path,
        len(steps),
        output_path.stat().st_size / 1e6,
        time.monotonic() - start_time,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--rlt-token-checkpoint", default=DEFAULT_RLT_TOKEN_CHECKPOINT)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--denoising-steps", type=int, default=10)
    parser.add_argument("--rtc-replan-start-step", type=int, default=25)
    parser.add_argument("--rtc-handoff-delay-steps", type=int, default=10)
    parser.add_argument("--rtc-guidance-scale", type=float, default=8.0)
    parser.add_argument("--episode-name", default="")
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-transitions", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError(f"--chunk-size must be positive; got {args.chunk_size}")

    replay_dir = Path(args.replay_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    episodes = sorted(
        path
        for path in replay_dir.glob("episode_*.npz")
        if ".rlt_tokens_stride" not in path.name
        and ".replay_stride" not in path.name
        and not path.name.startswith(".")
    )
    if args.episode_name:
        episodes = [path for path in episodes if path.name == args.episode_name]
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        raise FileNotFoundError(f"No episode_*.npz files found under {replay_dir}")

    train_config = _config.get_config(args.base_config)
    action_transform = _make_action_transform(train_config, args.base_checkpoint)
    base_policy = _policy_config.create_trained_policy(
        train_config,
        args.base_checkpoint,
        sample_kwargs={
            "denoising_steps": args.denoising_steps,
            "replan_start_step": args.rtc_replan_start_step,
            "handoff_delay_steps": args.rtc_handoff_delay_steps,
            "guidance_scale": args.rtc_guidance_scale,
        },
    )
    token_params, token_config = _load_token(Path(args.rlt_token_checkpoint))
    policy = rlt_policy.RLTPolicy(base_policy, token_params=token_params, token_config=token_config)

    for episode in episodes:
        _process_episode(
            policy,
            episode,
            output_dir=output_dir,
            stride=args.stride,
            action_horizon=args.chunk_size,
            action_dim=args.action_dim,
            next_offset_steps=args.chunk_size,
            gamma=args.gamma,
            action_transform=action_transform,
            max_transitions=args.max_transitions if args.max_transitions > 0 else None,
            force=args.force,
            save_embeddings=args.save_embeddings,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
