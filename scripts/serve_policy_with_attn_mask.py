"""
Policy server that captures and visualizes attention weights.

This script runs a policy server that:
1. Captures attention weights during inference
2. Saves frames with attention overlays
3. Generates videos showing what the policy is looking at

Usage:
    uv run scripts/serve_policy_with_attn_mask.py policy:checkpoint \\
        --policy.config=pi05_droid \\
        --policy.dir=checkpoints/3000 \\
        --output_dir=attention_videos \\
        --save_every_n=1
"""

import dataclasses
import datetime
import enum
import logging
import pathlib
import socket
from collections import defaultdict
from typing import Any

import cv2
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tyro

from openpi.models import model as _model
from openpi.policies import aloha_policy as _aloha_policy
from openpi.policies import droid_policy as _droid_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import nnx_utils
from openpi.training import config as _config

logger = logging.getLogger(__name__)


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi05_droid").
    config: str
    # Checkpoint directory (e.g., "checkpoints/3000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy_with_attn_mask script."""

    # Environment to serve the policy for.
    env: EnvMode = EnvMode.DROID

    # Default prompt if not provided in requests.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000

    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # === Attention visualization settings ===
    # Directory to save attention visualizations and videos
    output_dir: str = "attention_outputs"

    # Save attention overlay every N frames (1 = every frame, 10 = every 10th frame)
    save_every_n: int = 1

    # Colormap for attention heatmap
    colormap: str = "jet"

    # Heatmap overlay transparency (0=transparent, 1=opaque)
    alpha: float = 0.4

    # Whether to average attention across heads
    average_heads: bool = True

    # Use last diffusion step (True) or average across steps (False)
    use_last_step: bool = True

    # Generate video at end of session
    generate_video: bool = True

    # Video FPS
    video_fps: int = 10


# Default checkpoints
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="./checkpoints/20251014/30000",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_heatmap_overlay(
    image: np.ndarray,
    attention_map: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.4,
) -> np.ndarray:
    """Create a heatmap overlay on the image."""
    # Normalize attention map to [0, 1]
    attn_min = attention_map.min()
    attn_max = attention_map.max()
    if attn_max > attn_min:
        attention_map = (attention_map - attn_min) / (attn_max - attn_min)
    else:
        attention_map = np.zeros_like(attention_map)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(attention_map)[:, :, :3]  # Drop alpha, get RGB
    heatmap = (heatmap * 255).astype(np.uint8)

    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay


def extract_attention_for_cameras(
    attention_weights: jax.Array,
    camera_names: list[str],
    grid_size: int = 16,
    image_size: int = 224,
) -> dict[str, np.ndarray]:
    """
    Extract and process attention weights for each camera.

    Args:
        attention_weights: Attention tensor with various possible shapes
        camera_names: List of camera names in the observation
        grid_size: Spatial grid size (16x16 for SigLIP)
        image_size: Target image size for upsampling

    Returns:
        Dictionary mapping camera names to upsampled attention maps
    """
    tokens_per_camera = grid_size * grid_size
    num_cameras = len(camera_names)

    # Handle different attention shapes by averaging over all dimensions except sequence length
    # The sequence length dimension is always the last one
    if attention_weights.ndim >= 2:
        # Average over all dimensions except the last (sequence length)
        axes_to_average = tuple(range(attention_weights.ndim - 1))
        action_attn = jnp.mean(attention_weights, axis=axes_to_average)  # (S,)
    else:
        # Shape: (S,) - already fully processed
        action_attn = attention_weights

    # Extract attention to image tokens (they come first in the sequence)
    total_image_tokens = tokens_per_camera * num_cameras
    image_attn = action_attn[:total_image_tokens]  # (total_image_tokens,)

    # Split by camera
    attention_maps = {}

    for i, cam_name in enumerate(camera_names):
        start_idx = i * tokens_per_camera
        end_idx = start_idx + tokens_per_camera
        cam_attn = image_attn[start_idx:end_idx]  # (tokens_per_camera,)

        # Reshape to 2D grid (H, W)
        attn_2d = jnp.reshape(cam_attn, (grid_size, grid_size))

        # Convert to numpy with float32 dtype (OpenCV doesn't support bfloat16)
        attn_np = np.array(attn_2d, dtype=np.float32)

        # Upsample to match image size
        attn_upsampled = cv2.resize(
            attn_np,
            (image_size, image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        attention_maps[cam_name] = attn_upsampled

    return attention_maps


class PolicyWithAttentionVisualization(_policy.Policy):
    """Policy wrapper that captures and visualizes attention."""

    def __init__(
        self,
        *args,
        output_dir: str = "attention_outputs",
        save_every_n: int = 1,
        colormap: str = "jet",
        alpha: float = 0.4,
        use_last_step: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_every_n = save_every_n
        self.colormap = colormap
        self.alpha = alpha
        self.use_last_step = use_last_step

        self.frame_count = 0
        self.session_frames = defaultdict(list)  # Store frames per session

        logger.info(f"Attention visualization enabled. Saving to: {self.output_dir}")

    def infer(
        self,
        obs: dict,
        prev_action: np.ndarray | None = None,
        use_rtc: bool = True,
        noise: np.ndarray | None = None,
    ) -> dict:
        """Run inference with attention capture."""
        # Get current session ID or create new one
        session_id = obs.get("session_id", "default")

        # Make a copy of inputs
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        if not self._is_pytorch_model:
            # Convert to JAX arrays with batch dimension
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng = jax.random.split(self._rng)
        else:
            raise NotImplementedError("PyTorch models not yet supported for attention visualization")

        # Prepare observation
        observation = _model.Observation.from_dict(inputs)

        # Run inference with attention
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        if use_rtc and prev_action is not None:
            prev_action = jnp.asarray(prev_action)[np.newaxis, ...]
            origin_actions, attention_weights = self._guided_inference(
                sample_rng, prev_action, observation, **sample_kwargs
            )
        else:
            origin_actions, attention_weights = self._sample_actions(
                sample_rng, observation, **sample_kwargs
            )

        # Process attention and save visualization
        if self.frame_count % self.save_every_n == 0:
            self._save_attention_visualization(
                observation, attention_weights, session_id
            )

        self.frame_count += 1

        # Prepare outputs (same as parent class)
        outputs = {
            "state": inputs["state"],
            "actions": origin_actions,
            "origin_actions": origin_actions,
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)

        return outputs

    def _save_attention_visualization(
        self,
        observation: _model.Observation,
        attention_weights: list,
        session_id: str,
    ):
        """Save attention visualization for current frame."""
        if len(attention_weights) == 0:
            logger.warning("No attention weights captured")
            return

        # Select which attention to visualize
        if self.use_last_step:
            attn = attention_weights[-1]
        else:
            attn = jnp.mean(jnp.stack(attention_weights), axis=0)

        logger.debug(f"Attention shape: {attn.shape}")

        # Extract images from observation
        images = {}
        if hasattr(observation, 'images') and observation.images is not None:
            # Get image dict from observation
            image_dict = observation.images
            # Convert JAX arrays to numpy and remove batch dimension
            for cam_name, img in image_dict.items():
                img_np = np.array(img)
                if img_np.ndim == 4:  # (B, H, W, C)
                    img_np = img_np[0]
                images[cam_name] = img_np

        # Determine number of cameras
        if len(images) == 0:
            logger.warning("No images found in observation")
            return

        # Extract attention maps using actual camera names
        camera_names = list(images.keys())
        attention_maps = extract_attention_for_cameras(
            attn,
            camera_names=camera_names,
            grid_size=16,
            image_size=224,
        )

        # Create overlays
        camera_names = list(images.keys())
        overlays = []

        for cam_name in camera_names:
            img = images[cam_name]
            attn_map = attention_maps[cam_name]

            # Create overlay
            overlay = create_heatmap_overlay(
                img, attn_map, self.colormap, self.alpha
            )
            overlays.append(overlay)

        # Combine side-by-side
        if len(overlays) == 1:
            combined = overlays[0]
        else:
            combined = np.concatenate(overlays, axis=1)

        # Store frame for video generation
        self.session_frames[session_id].append(combined)

        # Save individual frame
        session_dir = self.output_dir / session_id
        session_dir.mkdir(exist_ok=True, parents=True)

        frame_path = session_dir / f"frame_{self.frame_count:06d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

        if self.frame_count % 10 == 0:
            logger.info(f"Saved attention frame {self.frame_count} to {frame_path}")

    def generate_video(self, session_id: str = "default", fps: int = 10):
        """Generate video from saved frames."""
        frames = self.session_frames.get(session_id, [])
        if len(frames) == 0:
            logger.warning(f"No frames found for session {session_id}")
            return

        session_dir = self.output_dir / session_id
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = session_dir / f"attention_video_{timestamp}.mp4"

        # Get frame dimensions
        height, width = frames[0].shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(video_path),
            fourcc,
            fps,
            (width, height),
        )

        # Write frames
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        logger.info(f"Generated video: {video_path} ({len(frames)} frames @ {fps} fps)")

        return video_path


def create_default_policy(
    env: EnvMode, *, default_prompt: str | None = None
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config),
            checkpoint.dir,
            default_prompt=default_prompt,
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> PolicyWithAttentionVisualization:
    """Create a policy with attention visualization from the given arguments."""
    # First create the base policy
    match args.policy:
        case Checkpoint():
            base_policy = _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
            )
        case Default():
            base_policy = create_default_policy(
                args.env, default_prompt=args.default_prompt
            )

    # Wrap with attention visualization
    attention_policy = PolicyWithAttentionVisualization(
        model=base_policy._model,
        rng=base_policy._rng if hasattr(base_policy, "_rng") else None,
        transforms=base_policy._input_transform.transforms
        if hasattr(base_policy._input_transform, "transforms")
        else [],
        output_transforms=base_policy._output_transform.transforms
        if hasattr(base_policy._output_transform, "transforms")
        else [],
        sample_kwargs=base_policy._sample_kwargs,
        metadata=base_policy._metadata,
        output_dir=args.output_dir,
        save_every_n=args.save_every_n,
        colormap=args.colormap,
        alpha=args.alpha,
        use_last_step=args.use_last_step,
    )

    return attention_policy


def main(args: Args) -> None:
    """Main function to run the attention-enabled policy server."""
    policy = create_policy(args)
    policy_metadata = policy.metadata
    config_name = (
        args.policy.config if isinstance(args.policy, Checkpoint) else None
    )

    # Warmup
    if config_name and ("droid" in config_name.lower() or args.env == EnvMode.DROID):
        dummy_obs = _droid_policy.make_droid_example()
    else:
        dummy_obs = _aloha_policy.make_aloha_example()

    dummy_prev_action = np.random.rand(16, 32)
    logger.info("Warming up policy with dummy inference...")
    # policy.infer(dummy_obs, dummy_prev_action, use_rtc=True)
    policy.infer(dummy_obs, dummy_prev_action, use_rtc=False)
    logger.info("Warmup complete")

    # Record if requested
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    logger.info(f"Attention visualizations will be saved to: {args.output_dir}")

    # Create and start server
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")

        # Generate videos for all sessions
        if args.generate_video and isinstance(policy, PolicyWithAttentionVisualization):
            logger.info("Generating videos from captured frames...")
            for session_id in policy.session_frames.keys():
                policy.generate_video(session_id, fps=args.video_fps)

        logger.info("Server stopped")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
