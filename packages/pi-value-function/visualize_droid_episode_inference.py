"""Run PiValue inference on a DROID episode and visualize results.

This script:
1) Loads a trained PiValue checkpoint.
2) Runs inference on all (or a subset of) frames in a single episode.
3) Plots:
   - expected value over time
   - value distribution at the selected current timestep
   - the 3 camera views at that timestep

Example:
  uv run python packages/pi-value-function/visualize_droid_episode_inference.py \
    --checkpoint packages/pi-value-function/checkpoints/pi_value/my_exp/30000 \
    --episode-index 0
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

import jax
import numpy as np
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict
import matplotlib

# Use a non-interactive backend when running headless.
if "--no-show" in sys.argv or "--mp4-path" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import orbax.checkpoint as ocp
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _add_local_paths() -> None:
    """Allow running this script directly from repo root without install step."""
    this_file = pathlib.Path(__file__).resolve()
    pkg_src = this_file.parent / "src"
    repo_src = this_file.parents[1] / "src"
    for path in (pkg_src, repo_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_add_local_paths()

from pi_value_function.config import PiValueConfig  # noqa: E402
from pi_value_function.serving.value_policy import ValuePolicy  # noqa: E402
from pi_value_function.training.checkpoint_downloader import download_gemma_from_kaggle  # noqa: E402
from pi_value_function.training.data_loader import parse_observation  # noqa: E402
from openpi.models.tokenizer import Gemma3Tokenizer  # noqa: E402
from openpi.shared.image_tools import resize_with_pad  # noqa: E402


def _normalize_key(key_tuple: tuple[Any, ...]) -> tuple[Any, ...]:
    """Normalize string indices like '0' into int indices for nnx.Sequential keys."""
    return tuple(int(k) if isinstance(k, str) and k.isdigit() else k for k in key_tuple)


def _resolve_checkpoint(base_or_step_path: str) -> tuple[pathlib.Path, int]:
    """Resolve checkpoint argument into (base_dir, step_number).

    Supports:
    - /.../exp/30000
    - /.../exp/30000/state
    - /.../exp  (uses latest numeric step child)
    """
    path = pathlib.Path(base_or_step_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    # /exp/30000/state
    if path.name == "state" and path.parent.name.isdigit():
        return path.parent.parent, int(path.parent.name)

    # /exp/30000
    if path.name.isdigit() and (path / "state").exists():
        return path.parent, int(path.name)

    # /exp -> choose latest numeric child that has /state
    if path.is_dir():
        candidates = []
        for child in path.iterdir():
            if child.is_dir() and child.name.isdigit() and (child / "state").exists():
                candidates.append(int(child.name))
        if candidates:
            step = max(candidates)
            return path, step

    raise ValueError(
        f"Could not resolve checkpoint step from: {path}\n"
        "Expected a step dir (.../30000), state dir (.../30000/state), "
        "or an experiment dir containing numeric step subdirs."
    )


def _load_policy_from_checkpoint(
    checkpoint_path: str,
    max_token_len: int = 48,
    return_distribution: bool = True,
) -> ValuePolicy:
    """Load PiValue model and wrap it in ValuePolicy."""
    checkpoint_base_dir, step_number = _resolve_checkpoint(checkpoint_path)
    ckpt_step_dir = checkpoint_base_dir / str(step_number) / "state"
    print(f"Loading checkpoint from: {ckpt_step_dir}")

    mesh = jax.sharding.Mesh(jax.devices(), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(ckpt_step_dir)
        item = {k: metadata[k] for k in metadata}
        restored = ckptr.restore(
            ckpt_step_dir,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=jax.Array),
                    item,
                ),
            ),
        )

    model_state_data = restored["model_state"]

    # NNX state may include trailing "value" wrappers.
    flat_state = flatten_dict(model_state_data)
    if flat_state and all(kp[-1] == "value" for kp in flat_state):
        flat_state = {kp[:-1]: v for kp, v in flat_state.items()}
        model_state_data = unflatten_dict(flat_state)

    flat_ckpt = flatten_dict(model_state_data)
    has_value_proj = any("value_proj" in str(k) for k in flat_ckpt)
    has_value_mlp = any("value_mlp" in str(k) for k in flat_ckpt)
    value_head_layers = 1 if (has_value_proj and not has_value_mlp) else 2

    config = PiValueConfig(
        value_dims=201,
        value_min=-1.0,
        value_max=0.0,
        gemma_variant="gemma-3-270m",
        siglip_variant="siglip2-so400m-patch16-384",
        value_head_layers=value_head_layers,
        max_token_len=max_token_len,
    )

    model = config.create(jax.random.PRNGKey(0))
    _, model_state = nnx.split(model)
    flat_model = flatten_dict(model_state.to_pure_dict())
    ckpt_by_normalized = {_normalize_key(k): v for k, v in flat_ckpt.items()}

    for key in flat_model:
        norm_key = _normalize_key(key)
        if norm_key in ckpt_by_normalized:
            flat_model[key] = ckpt_by_normalized[norm_key]

    nnx.update(model, nnx.State(unflatten_dict(flat_model)))

    _, tokenizer_path = download_gemma_from_kaggle()
    tokenizer = Gemma3Tokenizer(path=tokenizer_path, max_len=config.max_token_len)
    policy = ValuePolicy(model, tokenizer, return_distribution=return_distribution)
    print(f"Loaded model step={int(restored['step'])}, value_head_layers={value_head_layers}")
    return policy


def _to_np(x: Any) -> np.ndarray:
    """Convert torch/jax/numpy scalar/tensor to numpy array."""
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _build_policy_obs_from_sample(sample: dict, prompt_override: str | None) -> dict:
    """Map LeRobot sample to ValuePolicy input dict."""
    parsed = parse_observation(sample)
    state = parsed["state"].astype(np.float32)
    base = np.asarray(resize_with_pad(parsed["image"]["base_0_rgb"], 224, 224))
    right = np.asarray(resize_with_pad(parsed["image"]["right_wrist_0_rgb"], 224, 224))
    left = np.asarray(resize_with_pad(parsed["image"]["left_wrist_0_rgb"], 224, 224))

    prompt = prompt_override if prompt_override is not None else parsed["prompt"]
    prompt = prompt if prompt is not None else ""

    return {
        "observation/exterior_image_1_left": base,
        "observation/exterior_image_2_left": right,
        "observation/wrist_image_left": left,
        "observation/joint_position": state[:7],
        "observation/gripper_position": np.array([state[7]], dtype=np.float32),
        "prompt": prompt,
    }


def _get_episode_bounds(ds: LeRobotDataset, episode_index: int) -> tuple[int, int]:
    """Return [start, end) global frame bounds for an episode."""
    ep = ds.meta.episodes[episode_index]
    if "dataset_from_index" in ep and "dataset_to_index" in ep:
        return int(ep["dataset_from_index"]), int(ep["dataset_to_index"])

    # Fallback for older metadata format.
    start = 0
    for i in range(episode_index):
        start += int(ds.meta.episodes[i]["length"])
    end = start + int(ep["length"])
    return start, end


def _visualize(
    values: np.ndarray,
    distributions: np.ndarray,
    current_idx: int,
    camera_images: dict[str, np.ndarray],
    repo_id: str,
    episode_index: int,
    prompt: str,
    save_path: str | None,
    show: bool,
) -> None:
    fig = plt.figure(figsize=(18, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0])

    ax_value = fig.add_subplot(gs[0, :2])
    ax_dist = fig.add_subplot(gs[0, 2:])
    ax_base = fig.add_subplot(gs[1, 0])
    ax_left = fig.add_subplot(gs[1, 1])
    ax_right = fig.add_subplot(gs[1, 2])
    ax_info = fig.add_subplot(gs[1, 3])

    t = np.arange(len(values))
    ax_value.plot(t, values, color="tab:blue", linewidth=2)
    ax_value.scatter([current_idx], [values[current_idx]], color="tab:red", s=60, zorder=3)
    ax_value.axvline(current_idx, color="tab:red", linestyle="--", linewidth=1)
    ax_value.set_title("Expected Value Over Time")
    ax_value.set_xlabel("Timestep")
    ax_value.set_ylabel("Expected Value")
    ax_value.grid(alpha=0.3)

    support = np.linspace(-1.0, 0.0, distributions.shape[1], dtype=np.float32)
    current_dist = distributions[current_idx]
    ax_dist.plot(support, current_dist, color="tab:green", linewidth=2)
    ax_dist.fill_between(support, 0.0, current_dist, color="tab:green", alpha=0.25)
    ax_dist.axvline(values[current_idx], color="tab:red", linestyle="--", linewidth=1, label="expected value")
    ax_dist.set_title(f"Value Distribution @ t={current_idx}")
    ax_dist.set_xlabel("Value bin")
    ax_dist.set_ylabel("Probability")
    ax_dist.grid(alpha=0.3)
    ax_dist.legend(loc="upper left")

    ax_base.imshow(camera_images["base_0_rgb"])
    ax_base.set_title("base_0_rgb")
    ax_base.axis("off")

    ax_left.imshow(camera_images["left_wrist_0_rgb"])
    ax_left.set_title("left_wrist_0_rgb")
    ax_left.axis("off")

    ax_right.imshow(camera_images["right_wrist_0_rgb"])
    ax_right.set_title("right_wrist_0_rgb")
    ax_right.axis("off")

    ax_info.axis("off")
    info_text = (
        f"repo: {repo_id}\n"
        f"episode: {episode_index}\n"
        f"steps: {len(values)}\n"
        f"current t: {current_idx}\n"
        f"value(t): {values[current_idx]:.4f}\n\n"
        f"prompt:\n{prompt}"
    )
    ax_info.text(0.0, 1.0, info_text, va="top", ha="left", fontsize=10)

    fig.suptitle("PiValue Inference Visualization", fontsize=14)

    if save_path:
        out = pathlib.Path(save_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Saved figure to: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _visualize_loop(
    values: np.ndarray,
    distributions: np.ndarray,
    ds: LeRobotDataset,
    global_indices: list[int],
    start_idx: int,
    repo_id: str,
    episode_index: int,
    prompt: str,
    fps: int,
    mp4_path: str | None,
    show: bool,
) -> None:
    """Animate a looping timestep cursor over one episode."""
    fig = plt.figure(figsize=(18, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0])

    ax_value = fig.add_subplot(gs[0, :2])
    ax_dist = fig.add_subplot(gs[0, 2:])
    ax_base = fig.add_subplot(gs[1, 0])
    ax_left = fig.add_subplot(gs[1, 1])
    ax_right = fig.add_subplot(gs[1, 2])
    ax_info = fig.add_subplot(gs[1, 3])

    t = np.arange(len(values))
    ax_value.plot(t, values, color="tab:blue", linewidth=2)
    cursor = ax_value.scatter([start_idx], [values[start_idx]], color="tab:red", s=60, zorder=3)
    vline = ax_value.axvline(start_idx, color="tab:red", linestyle="--", linewidth=1)
    ax_value.set_title("Expected Value Over Time (Loop)")
    ax_value.set_xlabel("Timestep")
    ax_value.set_ylabel("Expected Value")
    ax_value.grid(alpha=0.3)

    support = np.linspace(-1.0, 0.0, distributions.shape[1], dtype=np.float32)
    current_dist = distributions[start_idx]
    dist_line, = ax_dist.plot(support, current_dist, color="tab:green", linewidth=2)
    dist_fill = ax_dist.fill_between(support, 0.0, current_dist, color="tab:green", alpha=0.25)
    exp_line = ax_dist.axvline(values[start_idx], color="tab:red", linestyle="--", linewidth=1, label="expected value")
    ax_dist.set_title(f"Value Distribution @ t={start_idx}")
    ax_dist.set_xlabel("Value bin")
    ax_dist.set_ylabel("Probability")
    ax_dist.grid(alpha=0.3)
    ax_dist.legend(loc="upper left")

    sample = ds[global_indices[start_idx]]
    sample_np = {k: (_to_np(v) if k != "task" else v) for k, v in sample.items()}
    parsed = parse_observation(sample_np)

    im_base = ax_base.imshow(parsed["image"]["base_0_rgb"])
    ax_base.set_title("base_0_rgb")
    ax_base.axis("off")

    im_left = ax_left.imshow(parsed["image"]["left_wrist_0_rgb"])
    ax_left.set_title("left_wrist_0_rgb")
    ax_left.axis("off")

    im_right = ax_right.imshow(parsed["image"]["right_wrist_0_rgb"])
    ax_right.set_title("right_wrist_0_rgb")
    ax_right.axis("off")

    ax_info.axis("off")
    info_text = ax_info.text(
        0.0,
        1.0,
        (
            f"repo: {repo_id}\n"
            f"episode: {episode_index}\n"
            f"steps: {len(values)}\n"
            f"current t: {start_idx}\n"
            f"value(t): {values[start_idx]:.4f}\n\n"
            f"prompt:\n{prompt}"
        ),
        va="top",
        ha="left",
        fontsize=10,
    )

    frame_order = list(range(start_idx, len(values))) + list(range(0, start_idx))
    interval_ms = int(1000 / max(fps, 1))

    def _update(local_t: int):
        nonlocal dist_fill
        cursor.set_offsets(np.array([[local_t, values[local_t]]], dtype=np.float32))
        vline.set_xdata([local_t, local_t])

        dist_line.set_ydata(distributions[local_t])
        if dist_fill is not None:
            dist_fill.remove()
        dist_fill = ax_dist.fill_between(support, 0.0, distributions[local_t], color="tab:green", alpha=0.25)
        exp_line.set_xdata([values[local_t], values[local_t]])
        ax_dist.set_title(f"Value Distribution @ t={local_t}")

        sample_t = ds[global_indices[local_t]]
        sample_t_np = {k: (_to_np(v) if k != "task" else v) for k, v in sample_t.items()}
        parsed_t = parse_observation(sample_t_np)
        im_base.set_data(parsed_t["image"]["base_0_rgb"])
        im_left.set_data(parsed_t["image"]["left_wrist_0_rgb"])
        im_right.set_data(parsed_t["image"]["right_wrist_0_rgb"])

        info_text.set_text(
            (
                f"repo: {repo_id}\n"
                f"episode: {episode_index}\n"
                f"steps: {len(values)}\n"
                f"current t: {local_t}\n"
                f"value(t): {values[local_t]:.4f}\n\n"
                f"prompt:\n{prompt}"
            )
        )
        return (cursor, vline, dist_line, exp_line, im_base, im_left, im_right, info_text)

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=frame_order,
        interval=interval_ms,
        repeat=True,
        blit=False,
    )

    fig.suptitle("PiValue Inference Loop", fontsize=14)

    if mp4_path:
        out = pathlib.Path(mp4_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = animation.FFMpegWriter(fps=max(fps, 1))
        anim.save(str(out), writer=writer, dpi=120)
        print(f"Saved loop MP4 to: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize PiValue inference on a DROID episode.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path (step dir, state dir, or exp dir).")
    parser.add_argument("--repo-id", type=str, default="michios/droid_xxjd_7")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--current-timestep", type=int, default=-1, help="Current timestep for distribution/camera view.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on number of timesteps to evaluate.")
    parser.add_argument("--prompt-override", type=str, default=None, help="Override prompt for all frames.")
    parser.add_argument("--max-token-len", type=int, default=48)
    parser.add_argument("--save-path", type=str, default=None, help="Path to save figure PNG.")
    parser.add_argument("--no-show", action="store_true", help="Do not open matplotlib window.")
    parser.add_argument("--loop", action="store_true", help="Continuously loop timesteps in this episode.")
    parser.add_argument("--fps", type=int, default=2, help="Loop playback FPS for --loop mode.")
    parser.add_argument("--mp4-path", type=str, default=None, help="Path to save loop animation MP4 (loop mode only).")
    args = parser.parse_args()

    policy = _load_policy_from_checkpoint(
        checkpoint_path=args.checkpoint,
        max_token_len=args.max_token_len,
        return_distribution=True,
    )

    ds = LeRobotDataset(args.repo_id)
    total_episodes = len(ds.meta.episodes)
    if args.episode_index < 0 or args.episode_index >= total_episodes:
        raise IndexError(f"episode_index={args.episode_index} out of range [0, {total_episodes - 1}]")

    start, end = _get_episode_bounds(ds, args.episode_index)
    length = end - start
    if args.max_steps is not None:
        length = min(length, args.max_steps)
    if length <= 0:
        raise ValueError("Selected episode has zero length after max_steps cap.")

    print(f"Running inference: repo={args.repo_id}, episode={args.episode_index}, steps={length}")

    values: list[float] = []
    dists: list[np.ndarray] = []
    global_indices: list[int] = []
    last_prompt = ""

    for local_t in range(length):
        global_idx = start + local_t
        sample = ds[global_idx]
        sample_np = {k: (_to_np(v) if k != "task" else v) for k, v in sample.items()}
        obs = _build_policy_obs_from_sample(sample_np, args.prompt_override)
        last_prompt = obs["prompt"]

        out = policy.infer(obs, return_distribution=True)
        values.append(float(out["value"]))
        dists.append(np.asarray(out["distribution"], dtype=np.float32))
        global_indices.append(global_idx)

        if (local_t + 1) % 100 == 0 or (local_t + 1) == length:
            print(f"  processed {local_t + 1}/{length} steps")

    values_np = np.asarray(values, dtype=np.float32)
    dists_np = np.stack(dists, axis=0)

    current = args.current_timestep
    if current < 0:
        current = length + current
    if current < 0 or current >= length:
        raise IndexError(f"current_timestep={args.current_timestep} resolves to {current}, out of range [0, {length - 1}]")

    if args.mp4_path and not args.loop:
        raise ValueError("--mp4-path is only supported with --loop.")

    if args.loop:
        if args.no_show and not args.mp4_path:
            raise ValueError("--loop with --no-show requires --mp4-path.")
        _visualize_loop(
            values=values_np,
            distributions=dists_np,
            ds=ds,
            global_indices=global_indices,
            start_idx=current,
            repo_id=args.repo_id,
            episode_index=args.episode_index,
            prompt=last_prompt,
            fps=args.fps,
            mp4_path=args.mp4_path,
            show=not args.no_show,
        )
        return

    current_sample = ds[start + current]
    current_sample_np = {k: (_to_np(v) if k != "task" else v) for k, v in current_sample.items()}
    current_parsed = parse_observation(current_sample_np)

    _visualize(
        values=values_np,
        distributions=dists_np,
        current_idx=current,
        camera_images=current_parsed["image"],
        repo_id=args.repo_id,
        episode_index=args.episode_index,
        prompt=last_prompt,
        save_path=args.save_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
