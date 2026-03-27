from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys


def _add_local_paths() -> None:
    this_file = Path(__file__).resolve()
    pkg_src = this_file.parents[1] / "src"
    if str(pkg_src) not in sys.path:
        sys.path.insert(0, str(pkg_src))


_add_local_paths()

from topreward.model import TOPRewardModel  # noqa: E402
from topreward.serving.real_time import RealTimeConfig  # noqa: E402
from topreward.serving.real_time import RealTimeScorer  # noqa: E402
from topreward.serving.websocket_client import ProgressWebSocketBridge  # noqa: E402
from topreward.serving.websocket_client import WebSocketBridgeConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve real-time TOPReward scores over websocket")
    parser.add_argument("--uri", type=str, required=True, help="Existing websocket URI to connect to")
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--frame_field", type=str, default="frame")
    parser.add_argument("--outgoing_type", type=str, default="progress_score")
    parser.add_argument("--score_interval", type=int, default=5)
    parser.add_argument("--max_buffer_frames", type=int, default=64)
    parser.add_argument("--subsample_factor", type=int, default=3)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = TOPRewardModel(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    rt_config = RealTimeConfig(
        score_interval=args.score_interval,
        max_buffer_frames=args.max_buffer_frames,
        subsample_factor=args.subsample_factor,
    )
    scorer = RealTimeScorer(model, args.instruction, rt_config)

    bridge = ProgressWebSocketBridge(
        scorer,
        WebSocketBridgeConfig(
            uri=args.uri,
            frame_field=args.frame_field,
            outgoing_type=args.outgoing_type,
        ),
    )

    asyncio.run(bridge.run_forever())


if __name__ == "__main__":
    main()
