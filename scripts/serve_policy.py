import dataclasses
import enum
import logging
import os
import socket
import dataclasses as dc

import tyro
import numpy as np

from openpi.policies import aloha_policy as _aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies import same_forward_rl_token as _same_forward_rl_token
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


def configure_jax_persistent_cache_from_env() -> bool:
    """Apply optional JAX persistent cache settings before first compilation."""
    cache_dir = os.getenv("JAX_COMPILATION_CACHE_DIR")
    if not cache_dir:
        return False

    import jax

    jax.config.update("jax_compilation_cache_dir", cache_dir)
    if min_compile_time := os.getenv("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"):
        jax.config.update("jax_persistent_cache_min_compile_time_secs", float(min_compile_time))
    if min_entry_size := os.getenv("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"):
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", int(min_entry_size))
    if xla_caches := os.getenv("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"):
        jax.config.update("jax_persistent_cache_enable_xla_caches", xla_caches)
    logging.info("Using JAX persistent compilation cache: %s", cache_dir)
    return True


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
            same_forward_rl_token_encoder=create_same_forward_rl_token_encoder_from_env(),
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
                same_forward_rl_token_encoder=create_same_forward_rl_token_encoder_from_env(),
            )
        case Default():
            return create_default_policy(
                args.env,
                default_prompt=args.default_prompt,
                video_memory_num_frames=args.video_memory_num_frames,
                video_memory_stride_seconds=args.video_memory_stride_seconds,
            )


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def create_same_forward_rl_token_encoder_from_env():
    if not _env_bool("RLT_SAME_FORWARD_RL_TOKEN_ENABLED", "0"):
        return None
    config_name = os.getenv(
        "RLT_SAME_FORWARD_RL_TOKEN_CONFIG",
        "eii_rinse_11repo_cam4_fullft_rl_token_lower_right_query_4layer",
    )
    checkpoint = os.getenv(
        "RLT_SAME_FORWARD_RL_TOKEN_CHECKPOINT_PATH",
        os.getenv("RLT_RL_TOKEN_CHECKPOINT_PATH", ""),
    )
    if not checkpoint:
        raise ValueError(
            "RLT_SAME_FORWARD_RL_TOKEN_ENABLED=1 requires "
            "RLT_SAME_FORWARD_RL_TOKEN_CHECKPOINT_PATH or RLT_RL_TOKEN_CHECKPOINT_PATH"
        )
    return _same_forward_rl_token.load_same_forward_rl_token_encoder(
        config_name=config_name,
        checkpoint_dir=checkpoint,
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
    configure_jax_persistent_cache_from_env()
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
