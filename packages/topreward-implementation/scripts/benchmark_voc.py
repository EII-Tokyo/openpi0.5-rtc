from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm


def _add_local_paths() -> None:
    this_file = Path(__file__).resolve()
    pkg_src = this_file.parents[1] / "src"
    if str(pkg_src) not in sys.path:
        sys.path.insert(0, str(pkg_src))


_add_local_paths()

from topreward.data.camera import select_camera  # noqa: E402
from topreward.data.lerobot_loader import load_episode_frames  # noqa: E402
from topreward.model import TOPRewardModel  # noqa: E402
from topreward.progress import estimate_progress  # noqa: E402
from topreward.utils.metrics import compute_voc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TOPReward VOC on LeRobot episodes")
    parser.add_argument("--dataset_repo", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--camera_key", type=str, default=None)
    parser.add_argument("--camera_strategy", type=str, default="first")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--subsample_fps", type=int, default=None)
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

    per_episode: list[dict] = []
    voc_values: list[float] = []

    for episode_idx in tqdm(range(num_episodes), desc="Benchmarking VOC"):
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

        progress = estimate_progress(model, frames, effective_instruction, k=args.K)
        voc = compute_voc(progress["normalized"], progress["timestamps"])
        voc_values.append(voc)

        per_episode.append(
            {
                "episode_idx": episode_idx,
                "camera_key": camera_key,
                "metadata": metadata,
                "voc": voc,
            }
        )

    summary = {
        "num_episodes": len(voc_values),
        "mean_voc": float(statistics.mean(voc_values)) if voc_values else float("nan"),
        "median_voc": float(statistics.median(voc_values)) if voc_values else float("nan"),
        "min_voc": float(min(voc_values)) if voc_values else float("nan"),
        "max_voc": float(max(voc_values)) if voc_values else float("nan"),
        "episodes": per_episode,
    }

    print(json.dumps(summary, indent=2))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
