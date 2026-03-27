from __future__ import annotations

import argparse
import json
from pathlib import Path
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
from topreward.success_detector import detect_success  # noqa: E402
from topreward.utils.metrics import compute_voc  # noqa: E402


def _parse_episode_indices(raw: str) -> list[int]:
    indices: list[int] = []
    for chunk in raw.split(","):
        item = chunk.strip()
        if not item:
            continue
        try:
            indices.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid episode index '{item}' in --episode_indices='{raw}'") from exc
    if not indices:
        raise ValueError(f"No valid episode indices found in --episode_indices='{raw}'")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline TOPReward evaluation on LeRobot episodes")
    parser.add_argument("--dataset_repo", type=str, required=True)
    parser.add_argument("--camera_key", type=str, default=None)
    parser.add_argument("--camera_strategy", type=str, default="first")
    parser.add_argument("--episode_index", type=int, default=None, help="Evaluate only this single episode index.")
    parser.add_argument(
        "--episode_indices",
        type=str,
        default=None,
        help="Comma-separated episode indices to evaluate (e.g. '3,10,42').",
    )
    parser.add_argument("--start_episode", type=int, default=0, help="Starting episode index when using --num_episodes.")
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Optional override for all episodes. If omitted, instruction is auto-read from the LeRobot dataset.",
    )
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument(
        "--prefix_stride_frames",
        type=int,
        default=None,
        help="If set, run reward inference every N frames (1, 1+N, ... , T). Overrides --K.",
    )
    parser.add_argument("--n_final_frames", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--score_mode",
        type=str,
        default="true_log_prob",
        choices=["true_log_prob", "true_minus_false"],
        help="Reward scoring mode: logP(True) or margin logP(True)-logP(False).",
    )
    parser.add_argument("--true_token", type=str, default="True")
    parser.add_argument("--false_token", type=str, default="False")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=100)
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

    if args.episode_index is not None and args.episode_indices is not None:
        raise ValueError("Use either --episode_index or --episode_indices, not both.")

    if args.episode_index is not None:
        episode_indices = [int(args.episode_index)]
    elif args.episode_indices is not None:
        episode_indices = _parse_episode_indices(args.episode_indices)
    else:
        start = max(0, int(args.start_episode))
        start = min(start, total_episodes)
        stop = min(start + int(args.num_episodes), total_episodes)
        episode_indices = list(range(start, stop))

    if not episode_indices:
        raise ValueError("No episodes selected. Check --start_episode/--num_episodes or explicit episode arguments.")

    for episode_idx in episode_indices:
        if episode_idx < 0 or episode_idx >= total_episodes:
            raise IndexError(f"episode_idx={episode_idx} out of range [0, {total_episodes - 1}]")

    model = TOPRewardModel(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    results: list[dict] = []
    for episode_idx in tqdm(episode_indices, desc="Evaluating episodes"):
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

        progress = estimate_progress(
            model,
            frames,
            effective_instruction,
            k=args.K,
            prefix_stride=args.prefix_stride_frames,
            score_mode=args.score_mode,
            true_token=args.true_token,
            false_token=args.false_token,
        )
        voc = compute_voc(progress["normalized"], progress["timestamps"])
        success = detect_success(
            model,
            frames,
            effective_instruction,
            n_final_frames=args.n_final_frames,
            threshold=args.threshold,
            score_mode=args.score_mode,
            true_token=args.true_token,
            false_token=args.false_token,
        )

        results.append(
            {
                "episode_idx": episode_idx,
                "instruction": effective_instruction,
                "camera_key": camera_key,
                "metadata": metadata,
                "score_mode": args.score_mode,
                "true_token": args.true_token,
                "false_token": args.false_token,
                "progress": progress,
                "voc": voc,
                "success": success,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
