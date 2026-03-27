from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from safetensors.torch import save_file
import torch
from tqdm import tqdm


def _add_local_paths() -> None:
    this_file = Path(__file__).resolve()
    pkg_src = this_file.parents[1] / "src"
    if str(pkg_src) not in sys.path:
        sys.path.insert(0, str(pkg_src))


_add_local_paths()

from topreward.advantage import AdvantageConfig  # noqa: E402
from topreward.advantage import compute_episode_advantages  # noqa: E402
from topreward.data.camera import select_camera  # noqa: E402
from topreward.data.lerobot_loader import load_episode_frames  # noqa: E402
from topreward.model import TOPRewardModel  # noqa: E402


TIER_TO_ID = {
    "bad": 0,
    "neutral": 1,
    "good": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute AWR weights from TOPReward progress")
    parser.add_argument("--dataset_repo", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--camera_key", type=str, default=None)
    parser.add_argument("--camera_strategy", type=str, default="first")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--subsample_fps", type=int, default=None)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--delta_max", type=float, default=2.0)
    parser.add_argument("--good_threshold", type=float, default=0.6)
    parser.add_argument("--bad_threshold", type=float, default=-0.2)
    parser.add_argument("--good_weight", type=float, default=2.0)
    parser.add_argument("--neutral_weight", type=float, default=1.0)
    parser.add_argument("--bad_weight", type=float, default=0.1)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = LeRobotDataset(args.dataset_repo)
    total_episodes = len(dataset.meta.episodes)
    num_episodes = min(args.num_episodes, total_episodes)

    model = TOPRewardModel(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    config = AdvantageConfig(
        tau=args.tau,
        delta_max=args.delta_max,
        good_threshold=args.good_threshold,
        bad_threshold=args.bad_threshold,
        good_weight=args.good_weight,
        neutral_weight=args.neutral_weight,
        bad_weight=args.bad_weight,
    )

    all_episode_idx: list[int] = []
    all_step_idx: list[int] = []
    all_weight: list[float] = []
    all_delta: list[float] = []
    all_increment: list[float] = []
    all_tier_id: list[int] = []

    episode_summaries: list[dict] = []

    for episode_idx in tqdm(range(num_episodes), desc="Computing AWR weights"):
        camera_key = args.camera_key or select_camera(dataset, episode_idx, strategy=args.camera_strategy)
        frames, instruction, metadata = load_episode_frames(
            dataset,
            episode_idx,
            camera_key=camera_key,
            max_frames=args.max_frames,
            subsample_fps=args.subsample_fps,
        )
        if "dataset_fps" in metadata:
            model.video_fps = float(metadata["dataset_fps"])

        effective_instruction = args.instruction if args.instruction is not None else instruction
        if not effective_instruction:
            raise ValueError(
                f"Episode {episode_idx} has no instruction. Provide --instruction override or ensure dataset task text exists."
            )

        action_timestamps = list(range(1, len(frames) + 1))
        adv = compute_episode_advantages(model, frames, effective_instruction, action_timestamps, config)

        increments = adv["increments"]
        deltas = adv["deltas"]
        tiers = adv["tiers"]
        weights = adv["weights"]

        for step, (inc, delta, tier, weight) in enumerate(zip(increments, deltas, tiers, weights), start=1):
            all_episode_idx.append(episode_idx)
            all_step_idx.append(step)
            all_increment.append(float(inc))
            all_delta.append(float(delta))
            all_weight.append(float(weight))
            all_tier_id.append(TIER_TO_ID[tier])

        episode_summaries.append(
            {
                "episode_idx": episode_idx,
                "instruction": effective_instruction,
                "camera_key": camera_key,
                "metadata": metadata,
                "num_steps": len(weights),
                "tier_counts": {
                    "good": int(sum(t == "good" for t in tiers)),
                    "neutral": int(sum(t == "neutral" for t in tiers)),
                    "bad": int(sum(t == "bad" for t in tiers)),
                },
            }
        )

    tensors = {
        "episode_idx": torch.tensor(all_episode_idx, dtype=torch.int32),
        "step_idx": torch.tensor(all_step_idx, dtype=torch.int32),
        "increment": torch.tensor(all_increment, dtype=torch.float32),
        "delta": torch.tensor(all_delta, dtype=torch.float32),
        "weight": torch.tensor(all_weight, dtype=torch.float32),
        "tier_id": torch.tensor(all_tier_id, dtype=torch.int8),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output))

    summary_path = output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "tier_to_id": TIER_TO_ID,
                "config": vars(config),
                "num_entries": len(all_weight),
                "episodes": episode_summaries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
