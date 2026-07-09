from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
import time

import cv2
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



def _metadata(data: np.lib.npyio.NpzFile, path: Path | None = None) -> dict:
    if "metadata_json" not in data:
        metadata = {}
    else:
        metadata = json.loads(str(data["metadata_json"]))
    if path is not None:
        for suffix in (".label.json", ".trim.json"):
            sidecar = path.with_suffix(path.suffix + suffix)
            if sidecar.exists():
                metadata.update(json.loads(sidecar.read_text()))
    return metadata


def _image_keys(data: np.lib.npyio.NpzFile, metadata: dict | None = None) -> list[str]:
    metadata = metadata or {}
    video_files = metadata.get("video_files", {})
    if isinstance(video_files, dict) and video_files:
        return sorted(str(key) for key in video_files)
    return []


def _decode_video_rgb(path: Path, *, expected_frames: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open replay video: {path}")
    frames = []
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"replay video has no frames: {path}")
    if len(frames) < expected_frames:
        frames.extend([frames[-1]] * (expected_frames - len(frames)))
    elif len(frames) > expected_frames:
        frames = frames[:expected_frames]
    return np.asarray(frames, dtype=np.uint8)


def _episode_image_arrays(
    data: np.lib.npyio.NpzFile,
    metadata: dict,
    episode_path: Path,
    *,
    expected_frames: int,
) -> dict[str, np.ndarray]:
    video_files = metadata.get("video_files", {})
    if not isinstance(video_files, dict) or not video_files:
        raise KeyError(f"{episode_path} has no video_files metadata")

    image_arrays = {}
    episode_dir = episode_path.parent.resolve()
    for key, video_name in sorted(video_files.items()):
        video_path = (episode_path.parent / str(video_name)).resolve()
        if episode_dir not in (video_path, *video_path.parents):
            raise ValueError(f"video path escapes episode directory: {video_path}")
        image_arrays[str(key)] = _decode_video_rgb(video_path, expected_frames=expected_frames)
    return image_arrays


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


def _frame_returns(meta: dict, raw_num_steps: int) -> dict[int, float]:
    raw = meta.get("frame_rewards", {})
    if not isinstance(raw, dict):
        return {}
    returns = {}
    for step, value in raw.items():
        step_index = max(0, min(int(step), max(raw_num_steps - 1, 0)))
        returns[step_index] = float(value)
    return returns


def _total_frame_reward(frame_returns: dict[int, float]) -> float:
    return float(sum(float(value) for value in frame_returns.values()))


def _discounted_transition_return(
    frame_returns: dict[int, float], source_step: int, next_offset_steps: int, gamma: float
) -> float:
    total = 0.0
    for offset in range(next_offset_steps):
        frame_return = frame_returns.get(source_step + offset)
        if frame_return is not None:
            total += float(gamma**offset) * float(frame_return)
    return total


def _output_path(
    path: Path,
    output_dir: Path | None,
    *,
    stride: int,
    train_action_horizon: int,
    next_offset_steps: int,
    samples_per_step: int = 1,
) -> Path:
    sample_suffix = f"_s{samples_per_step}" if samples_per_step > 1 else ""
    name = f"{path.stem}.replay_stride{stride}_h{train_action_horizon}_next{next_offset_steps}{sample_suffix}.npz"
    if output_dir is None:
        return path.with_name(name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / name


def _discounted_terminal_td_reward(terminal_success: int, next_offset_steps: int, gamma: float) -> float:
    if terminal_success < 0:
        return 0.0
    return float(terminal_success) * float(gamma ** max(next_offset_steps - 1, 0))


def _sampling_offset_from_recorded_chunks(
    data: np.lib.npyio.NpzFile,
    *,
    trim_start_step: int,
    num_steps: int,
    stride: int,
    handoff_delay_steps: int,
) -> int:
    if stride <= 1 or "chunk_start_step" not in data:
        return 0
    for source_chunk_start in np.asarray(data["chunk_start_step"], dtype=np.int32):
        local_generation_step = int(source_chunk_start) - trim_start_step - handoff_delay_steps
        if 0 <= local_generation_step < num_steps:
            return local_generation_step % stride
    return 0


def _recorded_noise_by_generation_step(
    data: np.lib.npyio.NpzFile,
    *,
    trim_start_step: int,
    num_steps: int,
    handoff_delay_steps: int,
) -> dict[int, np.ndarray]:
    if "chunk_start_step" not in data or "rlt_noise" not in data:
        return {}
    noise_by_step = {}
    for source_chunk_start, noise in zip(
        np.asarray(data["chunk_start_step"], dtype=np.int32),
        np.asarray(data["rlt_noise"], dtype=np.float32),
        strict=False,
    ):
        local_generation_step = int(source_chunk_start) - trim_start_step - handoff_delay_steps
        if 0 <= local_generation_step < num_steps:
            noise_by_step[int(source_chunk_start) - handoff_delay_steps] = noise
    return noise_by_step


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


def _normalized_action_chunk_for_policy(
    action_transform,
    *,
    data: np.lib.npyio.NpzFile,
    image_arrays: dict[str, np.ndarray],
    actions: np.ndarray,
    local_start: int,
    source_step: int,
    horizon: int,
    model_action_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    action_dim = actions.shape[-1]
    chunk = np.zeros((horizon, action_dim), dtype=np.float32)
    mask = np.zeros((horizon,), dtype=np.bool_)
    src_start = max(local_start, 0)
    src_end = min(local_start + horizon, actions.shape[0])
    if src_end > src_start:
        dst_start = src_start - local_start
        dst_end = dst_start + (src_end - src_start)
        chunk[dst_start:dst_end] = actions[src_start:src_end]
        mask[dst_start:dst_end] = True

    task = str(data["task"][source_step])
    subtask = str(data["subtask"][source_step])
    sample = {
        "observation": {
            "state": np.asarray(data["raw_state"][source_step], dtype=np.float32),
            "images": {key: value[source_step] for key, value in image_arrays.items()},
        },
        "action": chunk,
        "task": task,
        "subtask": subtask,
    }
    transformed = action_transform(sample)
    return _pad_action_dim(np.asarray(transformed["actions"], dtype=np.float32), model_action_dim), mask


def _process_episode(
    policy: rlt_policy.RLTPolicy,
    episode_path: Path,
    *,
    output_dir: Path | None,
    stride: int,
    train_action_horizon: int,
    model_action_horizon: int,
    model_action_dim: int,
    rtc_replan_start_step: int,
    handoff_delay_steps: int,
    action_dim: int,
    next_offset_steps: int,
    gamma: float,
    action_transform,
    max_transitions: int | None,
    samples_per_step: int,
    force: bool,
    save_embeddings: bool,
) -> Path | None:
    output_path = _output_path(
        episode_path,
        output_dir,
        stride=stride,
        train_action_horizon=train_action_horizon,
        next_offset_steps=next_offset_steps,
        samples_per_step=samples_per_step,
    )
    if output_path.exists() and not force:
        logging.info("Skipping existing output: %s", output_path)
        return output_path

    start_time = time.monotonic()
    logging.info("Loading episode: %s", episode_path)
    with np.load(episode_path, allow_pickle=False) as data:
        meta = _metadata(data, episode_path)
        label, terminal_success = _terminal_label(meta)
        if "raw_state" not in data or "executed_action" not in data:
            raise KeyError(f"{episode_path} missing raw_state or executed_action")
        if "task" not in data or "subtask" not in data:
            raise KeyError(f"{episode_path} missing task or subtask; collect replay with task/subtask fields")
        image_keys = _image_keys(data, meta)
        if not image_keys:
            raise KeyError(f"{episode_path} has no video_files metadata")
        all_actions = np.asarray(data["executed_action"], dtype=np.float32)
        raw_num_steps = int(all_actions.shape[0])
        image_arrays = _episode_image_arrays(data, meta, episode_path, expected_frames=raw_num_steps)
        frame_returns = _frame_returns(meta, raw_num_steps)
        trim_start_step = int(meta.get("trim_start_step", 0) or 0)
        trim_end_step = int(meta.get("trim_end_step", raw_num_steps) or raw_num_steps)
        trim_start_step = max(0, min(trim_start_step, raw_num_steps - 1))
        trim_end_step = max(trim_start_step + 1, min(trim_end_step, raw_num_steps))
        crop_num_steps = trim_end_step - trim_start_step
        if crop_num_steps <= 0:
            raise ValueError(f"{episode_path} has no usable crop: {trim_start_step}:{trim_end_step}")

        max_next_source_step = raw_num_steps - train_action_horizon
        max_training_source_step = max_next_source_step - next_offset_steps
        max_training_source_step = min(max_training_source_step, trim_end_step - 1)
        if max_training_source_step < trim_start_step:
            raise ValueError(
                f"{episode_path} crop {trim_start_step}:{trim_end_step} has no valid training step; "
                f"need source step <= {raw_num_steps - train_action_horizon - next_offset_steps} "
                f"for next_offset_steps={next_offset_steps} and action_horizon={train_action_horizon}"
            )

        terminal_source_step = max_training_source_step
        sampling_offset = _sampling_offset_from_recorded_chunks(
            data,
            trim_start_step=trim_start_step,
            num_steps=crop_num_steps,
            stride=stride,
            handoff_delay_steps=handoff_delay_steps,
        )
        source_training_steps = np.arange(
            trim_start_step + sampling_offset,
            max_training_source_step + 1,
            stride,
            dtype=np.int32,
        )
        if max_transitions is not None:
            source_training_steps = source_training_steps[:max_transitions]
        if len(source_training_steps) == 0:
            raise ValueError(f"{episode_path} has no valid source_training_steps after applying stride={stride}")
        source_next_steps = (source_training_steps + next_offset_steps).astype(np.int32)
        if np.any(source_next_steps > max_next_source_step):
            raise AssertionError(
                f"source_next_steps must be <= {max_next_source_step}, got max {int(source_next_steps.max())}"
            )
        steps = (source_training_steps - trim_start_step).astype(np.int32)
        transition_sample_indices = np.tile(np.arange(samples_per_step, dtype=np.int32), len(source_training_steps))
        transition_source_steps = np.repeat(source_training_steps, samples_per_step).astype(np.int32)
        transition_next_steps = np.repeat(source_next_steps, samples_per_step).astype(np.int32)
        transition_steps = np.repeat(steps, samples_per_step).astype(np.int32)
        required_source_steps = np.unique(np.concatenate([source_training_steps, source_next_steps])).astype(np.int32)
        recorded_noise_by_step = _recorded_noise_by_generation_step(
            data,
            trim_start_step=trim_start_step,
            num_steps=raw_num_steps - trim_start_step,
            handoff_delay_steps=handoff_delay_steps,
        )

        tokens_by_step = {}
        embeddings_by_step = {}
        masks_by_step = {}
        normalized_states_by_step = {}
        normalized_reference_by_step = {}
        normalized_model_reference_by_step = {}
        diffusion_noise_by_step = {}
        normalized_prev_action_by_step = {}
        normalized_prev_action_mask_by_step = {}
        task_by_step = {}
        subtask_by_step = {}
        total_policy_contexts = len(required_source_steps) * samples_per_step
        generated_contexts = 0
        for idx, step in enumerate(required_source_steps):
            step_task = str(data["task"][int(step)])
            step_subtask = str(data["subtask"][int(step)])
            obs = _observation_at(data, image_arrays, int(step), task=step_task, subtask=step_subtask)
            local_start = int(step) - rtc_replan_start_step
            normalized_prev_action, normalized_prev_action_mask = _normalized_action_chunk_for_policy(
                action_transform,
                data=data,
                image_arrays=image_arrays,
                actions=all_actions,
                local_start=local_start,
                source_step=max(int(step) - 1, 0),
                horizon=model_action_horizon,
                model_action_dim=model_action_dim,
            )
            for sample_idx in range(samples_per_step):
                infer_kwargs = {
                    "chunking_mode": "inference_time",
                    "prev_action": normalized_prev_action,
                }
                recorded_noise = recorded_noise_by_step.get(int(step))
                if sample_idx == 0 and recorded_noise is not None:
                    infer_kwargs["noise"] = recorded_noise

                out = policy.infer(obs, **infer_kwargs)
                token = np.asarray(out["rlt_token"], dtype=np.float32)
                normalized_state = np.asarray(out["rlt_state"], dtype=np.float32)
                normalized_model_reference = np.asarray(out["rlt_reference_action_chunk"], dtype=np.float32)
                normalized_reference = normalized_model_reference[..., :action_dim]

                step_key = (int(step), int(sample_idx))
                tokens_by_step[step_key] = token
                if save_embeddings:
                    embeddings_by_step[step_key] = np.asarray(out["rlt_embeddings"], dtype=np.float32)
                    masks_by_step[step_key] = np.asarray(out["rlt_mask"], dtype=np.bool_)
                normalized_states_by_step[step_key] = normalized_state
                action_window_start = handoff_delay_steps
                action_window_end = action_window_start + train_action_horizon
                normalized_reference_by_step[step_key] = normalized_reference[action_window_start:action_window_end]
                normalized_model_reference_by_step[step_key] = normalized_model_reference[:model_action_horizon]
                if "model_noise" in out:
                    diffusion_noise_by_step[step_key] = np.asarray(out["model_noise"], dtype=np.float32)
                normalized_prev_action_by_step[step_key] = normalized_prev_action
                normalized_prev_action_mask_by_step[step_key] = normalized_prev_action_mask
                task_by_step[step_key] = step_task
                subtask_by_step[step_key] = step_subtask
                generated_contexts += 1
                if generated_contexts == 1 or generated_contexts % 30 == 0:
                    logging.info(
                        "%s generated %d/%d policy contexts last_step=%d sample=%d",
                        episode_path.name,
                        generated_contexts,
                        total_policy_contexts,
                        int(step),
                        int(sample_idx),
                    )

        transition_keys = [(int(step), int(sample_idx)) for step, sample_idx in zip(transition_source_steps, transition_sample_indices, strict=False)]
        next_transition_keys = [(int(step), int(sample_idx)) for step, sample_idx in zip(transition_next_steps, transition_sample_indices, strict=False)]
        token_arr = np.asarray([tokens_by_step[key] for key in transition_keys], dtype=np.float32)
        next_token_arr = np.asarray([tokens_by_step[key] for key in next_transition_keys], dtype=np.float32)
        normalized_state_arr = np.asarray([normalized_states_by_step[key] for key in transition_keys], dtype=np.float32)
        normalized_next_state_arr = np.asarray([normalized_states_by_step[key] for key in next_transition_keys], dtype=np.float32)
        normalized_ref_arr = np.asarray([normalized_reference_by_step[key] for key in transition_keys], dtype=np.float32)
        normalized_next_ref_arr = np.asarray([normalized_reference_by_step[key] for key in next_transition_keys], dtype=np.float32)
        normalized_model_ref_arr = np.asarray(
            [normalized_model_reference_by_step[key] for key in transition_keys], dtype=np.float32
        )
        normalized_next_model_ref_arr = np.asarray(
            [normalized_model_reference_by_step[key] for key in next_transition_keys], dtype=np.float32
        )
        diffusion_noise_arr = np.asarray([diffusion_noise_by_step[key] for key in transition_keys], dtype=np.float32)
        next_diffusion_noise_arr = np.asarray([diffusion_noise_by_step[key] for key in next_transition_keys], dtype=np.float32)
        normalized_prev_action_arr = np.asarray(
            [normalized_prev_action_by_step[key] for key in transition_keys], dtype=np.float32
        )
        normalized_prev_action_mask_arr = np.asarray(
            [normalized_prev_action_mask_by_step[key] for key in transition_keys], dtype=np.bool_
        )

        normalized_executed_chunks = []
        normalized_executed_masks = []
        for source_step in transition_source_steps:
            normalized_executed, normalized_executed_mask = _normalized_action_chunk(
                action_transform,
                data=data,
                image_arrays=image_arrays,
                actions=all_actions,
                local_step=int(source_step) + handoff_delay_steps,
                source_step=int(source_step),
                horizon=train_action_horizon,
                action_dim=action_dim,
            )
            normalized_executed_chunks.append(normalized_executed[:train_action_horizon])
            normalized_executed_masks.append(normalized_executed_mask[:train_action_horizon])
        task_arr = np.asarray([task_by_step[key] for key in transition_keys])
        subtask_arr = np.asarray([subtask_by_step[key] for key in transition_keys])

    normalized_executed_arr = np.asarray(normalized_executed_chunks, dtype=np.float32)
    normalized_executed_mask_arr = np.asarray(normalized_executed_masks, dtype=np.bool_)
    done = np.zeros((len(transition_steps),), dtype=np.float32)
    td_reward = np.zeros((len(transition_steps),), dtype=np.float32)
    if frame_returns:
        for idx, source_step in enumerate(transition_source_steps):
            td_reward[idx] = _discounted_transition_return(frame_returns, int(source_step), next_offset_steps, gamma)
        td_reward_source = "frame_returns"
    else:
        td_reward_source = "terminal_label"
    if len(steps):
        # Do not bootstrap once the next state is beyond the last source step
        # that can itself form a valid next transition.
        done = (transition_next_steps > max_training_source_step).astype(np.float32)
        if not frame_returns and bool(np.any(done)):
            td_reward[done.astype(bool)] = _discounted_terminal_td_reward(terminal_success, next_offset_steps, gamma)
    source_episode = np.asarray(str(episode_path))
    output_meta = {
        "format": "rlt_replay_buffer_v1",
        "source_episode": str(episode_path),
        "stride": stride,
        "sampling_offset": sampling_offset,
        "next_offset_steps": next_offset_steps,
        "action_horizon": train_action_horizon,
        "train_action_horizon": train_action_horizon,
        "model_action_horizon": model_action_horizon,
        "model_action_dim": model_action_dim,
        "action_dim": action_dim,
        "raw_num_steps": raw_num_steps,
        "trim_start_step": trim_start_step,
        "trim_end_step": trim_end_step,
        "cropped_num_steps": crop_num_steps,
        "max_next_source_step": max_next_source_step,
        "max_training_source_step": max_training_source_step,
        "terminal_step": terminal_source_step,
        "context_sampling_mode": "inference_time_rtc",
        "context_prev_action_source": "current_executing_chunk_at_rtc_replan_step",
        "context_prev_action_start_offset": -rtc_replan_start_step,
        "rtc_replan_start_step": rtc_replan_start_step,
        "rtc_handoff_delay_steps": handoff_delay_steps,
        "actor_action_window_start": handoff_delay_steps,
        "actor_action_window_end": handoff_delay_steps + train_action_horizon,
        "recorded_noise_matches": len(recorded_noise_by_step),
        "samples_per_step": samples_per_step,
        "recorded_noise_sample_index": 0,
        "extra_samples_noise_source": "policy_rng",
        "executed_action_source_space": "aloha_output_space_absolute_joint_targets_plus_normalized_grippers",
        "normalized_executed_action_chunk_space": "model_normalized_space",
        "normalized_prev_action_chunk_space": "model_normalized_space",
        "normalized_prev_action_chunk_source_dim": action_dim,
        "normalized_prev_action_chunk_dim": model_action_dim,
        "normalized_reference_action_chunk_space": "model_normalized_space",
        "normalized_model_reference_action_chunk_space": "model_normalized_space",
        "normalized_state_space": "model_input_transform_space",
        "diffusion_noise_space": "standard_normal_initial_flow_noise",
        "gamma": gamma,
        "save_embeddings": save_embeddings,
        "terminal_label": label,
        "terminal_success": terminal_success,
        "total_reward": _total_frame_reward(frame_returns),
        "frame_rewards": {str(step): value for step, value in sorted(frame_returns.items())},
        "td_reward_source": td_reward_source,
        "td_reward_semantics": "discounted_immediate_reward_for_td_target",
        "td_reward_horizon_steps": next_offset_steps,
        "created_at": time.time(),
    }
    payload = {
        "metadata_json": np.asarray(json.dumps(output_meta, ensure_ascii=False)),
        "task": task_arr,
        "subtask": subtask_arr,
        "rlt_token": token_arr,
        "next_rlt_token": next_token_arr,
        "normalized_state": normalized_state_arr,
        "normalized_next_state": normalized_next_state_arr,
        "normalized_reference_action_chunk": normalized_ref_arr,
        "normalized_next_reference_action_chunk": normalized_next_ref_arr,
        "normalized_model_reference_action_chunk": normalized_model_ref_arr,
        "normalized_next_model_reference_action_chunk": normalized_next_model_ref_arr,
        "diffusion_noise": diffusion_noise_arr,
        "next_diffusion_noise": next_diffusion_noise_arr,
        "normalized_prev_action_chunk": normalized_prev_action_arr,
        "normalized_prev_action_mask": normalized_prev_action_mask_arr,
        "normalized_executed_action_chunk": normalized_executed_arr,
        "normalized_executed_action_mask": normalized_executed_mask_arr,
        "executed_action_mask": normalized_executed_mask_arr,
        "td_reward": td_reward,
        "done": done,
        "episode_id": np.full((len(transition_steps),), int(episode_path.stem.split("_")[-1]), dtype=np.int32),
        "step_index": transition_source_steps,
        "local_step_index": transition_steps,
        "next_step_index": transition_next_steps,
        "sample_index": transition_sample_indices,
        "is_intervention": np.zeros((len(transition_steps),), dtype=np.float32),
        "source_episode": source_episode,
    }
    if save_embeddings:
        embeddings_arr = np.asarray([embeddings_by_step[key] for key in transition_keys], dtype=np.float32)
        next_embeddings_arr = np.asarray([embeddings_by_step[key] for key in next_transition_keys], dtype=np.float32)
        mask_arr = np.asarray([masks_by_step[key] for key in transition_keys], dtype=np.bool_)
        next_mask_arr = np.asarray([masks_by_step[key] for key in next_transition_keys], dtype=np.bool_)
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
        len(transition_steps),
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
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=None, help="Deprecated alias for --train-action-horizon.")
    parser.add_argument("--train-action-horizon", type=int, default=25)
    parser.add_argument("--model-action-horizon", type=int, default=50)
    parser.add_argument("--model-action-dim", type=int, default=32)
    parser.add_argument("--next-offset-steps", type=int, default=25)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--denoising-steps", type=int, default=10)
    parser.add_argument("--rtc-replan-start-step", type=int, default=25)
    parser.add_argument("--rtc-handoff-delay-steps", type=int, default=10)
    parser.add_argument("--rtc-guidance-scale", type=float, default=8.0)
    parser.add_argument("--episode-name", default="")
    parser.add_argument("--episode-ids", default="", help="Comma-separated episode ids to process, e.g. 329,330,331.")
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-transitions", type=int, default=0)
    parser.add_argument("--samples-per-step", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.chunk_size is not None:
        logging.warning("--chunk-size is deprecated; use --train-action-horizon instead.")
        args.train_action_horizon = args.chunk_size
    if args.train_action_horizon <= 0:
        raise ValueError(f"--train-action-horizon must be positive; got {args.train_action_horizon}")
    if args.model_action_horizon <= 0:
        raise ValueError(f"--model-action-horizon must be positive; got {args.model_action_horizon}")
    if args.model_action_dim <= 0:
        raise ValueError(f"--model-action-dim must be positive; got {args.model_action_dim}")
    if args.next_offset_steps <= 0:
        raise ValueError(f"--next-offset-steps must be positive; got {args.next_offset_steps}")
    if args.samples_per_step <= 0:
        raise ValueError(f"--samples-per-step must be positive; got {args.samples_per_step}")
    if args.rtc_replan_start_step >= args.model_action_horizon:
        raise ValueError(
            f"--rtc-replan-start-step must be smaller than --model-action-horizon; "
            f"got {args.rtc_replan_start_step} >= {args.model_action_horizon}"
        )

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
    if args.episode_ids:
        requested = {int(item.strip()) for item in args.episode_ids.split(",") if item.strip()}
        episodes = [path for path in episodes if int(path.stem.split("_")[-1]) in requested]
        present = {int(path.stem.split("_")[-1]) for path in episodes}
        missing = sorted(requested - present)
        if missing:
            raise FileNotFoundError(f"Requested --episode-ids not found under {replay_dir}: {missing}")
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
            train_action_horizon=args.train_action_horizon,
            model_action_horizon=args.model_action_horizon,
            model_action_dim=args.model_action_dim,
            rtc_replan_start_step=args.rtc_replan_start_step,
            handoff_delay_steps=args.rtc_handoff_delay_steps,
            action_dim=args.action_dim,
            next_offset_steps=args.next_offset_steps,
            gamma=args.gamma,
            action_transform=action_transform,
            max_transitions=args.max_transitions if args.max_transitions > 0 else None,
            samples_per_step=args.samples_per_step,
            force=args.force,
            save_embeddings=args.save_embeddings,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
