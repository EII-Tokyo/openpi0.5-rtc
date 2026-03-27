import argparse
import logging
import pathlib
import sys

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start TOPReward inference server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind to")

    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--attn-implementation", type=str, default="flash_attention_2")

    parser.add_argument(
        "--instruction",
        type=str,
        default="",
        help="Default instruction if request observations do not include prompt/instruction text.",
    )
    parser.add_argument(
        "--camera-key",
        type=str,
        default="observation/exterior_image_1_left",
        help="Preferred observation image key used for progress scoring.",
    )
    parser.add_argument("--score-interval", type=int, default=5)
    parser.add_argument("--max-buffer-frames", type=int, default=64)
    parser.add_argument("--subsample-factor", type=int, default=3)
    parser.add_argument("--no-reset-on-prompt-change", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    return parser.parse_args()


def main() -> None:
    # Local path fallback for running from repo root without installation step.
    topreward_src = pathlib.Path(__file__).resolve().parent.parent / "packages" / "topreward" / "src"
    if str(topreward_src) not in sys.path:
        sys.path.insert(0, str(topreward_src))

    from topreward.model import TOPRewardModel
    from topreward.serving.policy import TOPRewardPolicy
    from topreward.serving.real_time import RealTimeConfig

    from openpi.serving import websocket_policy_server

    args = parse_args()

    logger.info("Loading TOPReward model %s on %s", args.model_name, args.device)
    model = TOPRewardModel(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    logger.info("Resolved True token id: %s", model.true_token_id)

    rt_config = RealTimeConfig(
        score_interval=args.score_interval,
        max_buffer_frames=args.max_buffer_frames,
        subsample_factor=args.subsample_factor,
    )
    policy = TOPRewardPolicy(
        model=model,
        instruction=args.instruction,
        camera_key=args.camera_key,
        config=rt_config,
        reset_on_prompt_change=not args.no_reset_on_prompt_change,
    )

    if not args.skip_warmup:
        logger.info("Running warmup inference...")
        warmup_obs = {
            args.camera_key: np.zeros((224, 224, 3), dtype=np.uint8),
            "prompt": args.instruction or "warmup",
            "reset": True,
        }
        _ = policy.infer(warmup_obs)
        logger.info("Warmup complete.")

    metadata = {
        "server": "topreward",
        "model_name": args.model_name,
        "camera_key": args.camera_key,
        "score_interval": args.score_interval,
        "subsample_factor": args.subsample_factor,
        "max_buffer_frames": args.max_buffer_frames,
    }

    logger.info("Starting TOPReward server on %s:%s", args.host, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
