import dataclasses
import enum
import logging
import json
import socket

import numpy as np
import tyro

from openpi.data import transforms as _transforms
from openpi.serving import policy as _policy
from openpi.serving import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name.
    config: str
    # Checkpoint directory.
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA
    # Preferred entrypoint: infer configuration from checkpoint metadata.
    model_dir: str | None = None
    config: str | None = None
    # Validate files, transforms and input shapes without loading weights or serving.
    check_only: bool = False

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False
    # Warm up the RTC low-level path.
    warmup_rtc: bool = True
    # Warm up the non-RTC low-level path.
    warmup_non_rtc: bool = True
    # Warm up infer_subtask for hierarchical/high-level usage.
    warmup_subtask: bool = True
    # Number of flow denoising steps to use for policy sampling.
    denoising_steps: int = 10
    # RTC scheduling parameters. These must match the robot runtime.
    rtc_replan_start_step: int = 25
    rtc_handoff_delay_steps: int = 10
    # Legacy arguments; checkpoint training config is authoritative.
    video_memory_num_frames: int = 1
    video_memory_stride_seconds: float = 1.0
    # Capture action-token attention over the three model camera views.
    attention_debug: bool = False
    attention_debug_dir: str = "/app/attention_debug"
    attention_debug_every_n: int = 1

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="twist_off_the_bottle_cap",
        dir="./checkpoints/20260205/39999",
    ),
}


def selected_checkpoint(args: Args):
    if args.model_dir is not None:
        return args.model_dir, args.config
    if isinstance(args.policy, Checkpoint):
        return args.policy.dir, args.policy.config
    checkpoint = DEFAULT_CHECKPOINT[args.env]
    return checkpoint.dir, checkpoint.config


def _make_dummy_obs(metadata: dict) -> dict:
    video = metadata["runtime"]["video_input"]
    obs = _transforms.make_aloha_example()
    mosaic = video["mosaic"]
    temporal_cameras = set(mosaic["cameras"]) if mosaic is not None else set(video["camera_keys"])
    obs["images"] = {
        camera: np.zeros(
            ((video["num_frames"],) if video["num_frames"] > 1 and camera in temporal_cameras else ())
            + tuple(video["image_resolution"]) + (3,), dtype=np.uint8)
        for camera in video["camera_keys"]
    }
    return obs


def check_checkpoint(args: Args):
    from openpi.serving.checkpoint_config import checkpoint_config, runtime_metadata
    root, config = checkpoint_config(*selected_checkpoint(args))
    metadata = {"runtime": runtime_metadata(config, root)}
    obs = _make_dummy_obs(metadata)
    # Exercise the actual training-config input transforms, including norm stats
    # and prompt/state tokenization, without loading model parameters.
    transformed = _transforms.compose(config.data.transform_pipeline.policy_input_transforms())(obs)
    print(json.dumps({"checkpoint": str(root), "runtime": metadata["runtime"],
                      "input_shapes": {key: list(value.shape) for key,value in obs["images"].items()},
                      "token_shape": list(transformed["tokenized_prompt"].shape)}), flush=True)
    return root, config


def create_policy(args: Args) -> _policy.Policy:
    from openpi.serving.checkpoint_config import checkpoint_config
    root, config = checkpoint_config(*selected_checkpoint(args), verify=False)
    # create_trained_policy performs the full verification before loading weights.
    return _policy_config.create_trained_policy(config, root, sample_kwargs={
        "denoising_steps": args.denoising_steps,
        "replan_start_step": args.rtc_replan_start_step,
        "handoff_delay_steps": args.rtc_handoff_delay_steps,
    })


def main(args: Args) -> None:
    if args.check_only:
        check_checkpoint(args)
        return
    policy = create_policy(args)
    policy_metadata = policy.metadata
    dummy_obs = _make_dummy_obs(policy_metadata)
    runtime = policy_metadata["runtime"]
    dummy_prev_action = np.zeros((runtime["action_horizon"], runtime["action_dim"]), dtype=np.float32)
    if args.rtc_replan_start_step + args.rtc_handoff_delay_steps > runtime["action_horizon"]:
        raise ValueError("RTC handoff exceeds checkpoint action horizon")
    logging.info("Warmup input shapes=%s action_shape=%s", {k: v.shape for k, v in dummy_obs["images"].items()}, dummy_prev_action.shape)
    dummy_action_prefix = np.zeros_like(dummy_prev_action)
    rtc_handoff_step = args.rtc_replan_start_step + args.rtc_handoff_delay_steps
    dummy_action_prefix[: args.rtc_handoff_delay_steps] = dummy_prev_action[
        args.rtc_replan_start_step : rtc_handoff_step
    ]
    if args.warmup_rtc:
        policy.infer(dummy_obs, prev_action=dummy_prev_action, chunking_mode="inference_time")
        policy.infer(
            dummy_obs,
            chunking_mode="training_time",
            action_prefix=dummy_action_prefix,
            handoff_delay_steps=args.rtc_handoff_delay_steps,
        )
    else:
        logging.info("Skipping RTC warmup by request.")
    if args.warmup_non_rtc:
        policy.infer(dummy_obs, chunking_mode="sync")
    else:
        logging.info("Skipping non-RTC warmup by request.")
    if not args.warmup_subtask:
        logging.info("Skipping infer_subtask warmup by request.")
    else:
        try:
            policy.infer_subtask(dummy_obs)
        except (AttributeError, NotImplementedError):
            logging.info("Skipping infer_subtask warmup because the current policy does not support it.")
    if args.attention_debug:
        policy.enable_attention_recording(
            args.attention_debug_dir,
            every_n=args.attention_debug_every_n,
        )
        policy.warmup_attention_recording(dummy_obs)
    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
