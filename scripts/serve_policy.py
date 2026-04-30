import dataclasses
import enum
import logging
import socket
import dataclasses as dc

import tyro
import numpy as np

from openpi.policies import aloha_policy as _aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"


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

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

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
    # Override temporal image history used by the training data config at inference time.
    video_memory_num_frames: int = 1
    video_memory_stride_seconds: float = 1.0

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="twist_off_the_bottle_cap",
        dir="./checkpoints/20260205/39999",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi05_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
}


def create_default_policy(
    env: EnvMode,
    *,
    default_prompt: str | None = None,
    video_memory_num_frames: int = 1,
    video_memory_stride_seconds: float = 1.0,
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        train_config = _config.get_config(checkpoint.config)
        return _policy_config.create_trained_policy(
            dc.replace(
                train_config,
                data=dc.replace(
                    train_config.data,
                    video_memory_num_frames=video_memory_num_frames,
                    video_memory_stride_seconds=video_memory_stride_seconds,
                ),
            ),
            checkpoint.dir,
            default_prompt=default_prompt,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    def _override_history(train_config: _config.TrainConfig) -> _config.TrainConfig:
        return dc.replace(
            train_config,
            data=dc.replace(
                train_config.data,
                video_memory_num_frames=args.video_memory_num_frames,
                video_memory_stride_seconds=args.video_memory_stride_seconds,
            ),
        )

    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _override_history(_config.get_config(args.policy.config)),
                args.policy.dir,
                default_prompt=args.default_prompt,
            )
        case Default():
            return create_default_policy(
                args.env,
                default_prompt=args.default_prompt,
                video_memory_num_frames=args.video_memory_num_frames,
                video_memory_stride_seconds=args.video_memory_stride_seconds,
            )


def _make_dummy_obs(num_frames: int) -> dict:
    obs = _aloha_policy.make_aloha_example()
    if num_frames <= 1:
        return obs
    obs["images"] = {
        cam_name: np.stack([img] * num_frames, axis=0)
        for cam_name, img in obs["images"].items()
    }
    return obs


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata
    dummy_obs = _make_dummy_obs(args.video_memory_num_frames)
    dummy_prev_action = np.random.rand(50, 32)
    if args.warmup_rtc:
        policy.infer(dummy_obs, dummy_prev_action, use_rtc=True)
    else:
        logging.info("Skipping RTC warmup by request.")
    if args.warmup_non_rtc:
        policy.infer(dummy_obs, dummy_prev_action, use_rtc=False)
    else:
        logging.info("Skipping non-RTC warmup by request.")
    if not args.warmup_subtask:
        logging.info("Skipping infer_subtask warmup by request.")
    else:
        try:
            policy.infer_subtask(dummy_obs)
        except (AttributeError, NotImplementedError):
            logging.info("Skipping infer_subtask warmup because the current policy does not support it.")
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
