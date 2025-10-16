"""
Visualize attention weights overlaid on camera videos.

This script captures attention weights from the model during inference and
creates heatmap visualizations overlaid on the input camera frames.

Usage:
    python scripts/visualize_attention.py --help
"""

import dataclasses
import logging
from pathlib import Path
from typing import Any

import cv2
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import tyro
from PIL import Image

from openpi.models import pi0, pi0_config
from openpi.policies import droid_policy, policy_config
from openpi.shared import image_tools
from openpi.training import checkpoints

logger = logging.getLogger("openpi")


@dataclasses.dataclass
class Args:
    """Arguments for attention visualization."""

    # Model checkpoint
    checkpoint_path: str = "path/to/checkpoint"
    config_name: str = "droid_policy"  # or "aloha_policy", "libero_policy"

    # Input data
    image_exterior: str | None = None  # Path to exterior camera image
    image_wrist: str | None = None  # Path to wrist camera image
    prompt: str = "pick up the object"

    # For state, you can provide dummy values if you just want to visualize attention
    dummy_state: bool = True  # Use dummy state values

    # Output
    output_path: str = "attention_visualization.png"

    # Visualization options
    colormap: str = "jet"  # matplotlib colormap for heatmap
    alpha: float = 0.4  # transparency of heatmap overlay (0=transparent, 1=opaque)
    average_heads: bool = True  # average across attention heads
    average_steps: bool = False  # average across diffusion steps (or show last step)
    image_size: int = 224  # size of input images


def load_model_and_policy(checkpoint_path: str, config_name: str):
    """Load the model and policy from checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Get policy config
    policy_cfg = policy_config.get_policy_config(config_name)

    # Load checkpoint
    checkpoint_data = checkpoints.load_checkpoint(checkpoint_path)

    # Create model
    model_cfg = pi0_config.Pi0Config(
        action_dim=policy_cfg.action_dim,
        action_horizon=policy_cfg.action_horizon,
        max_token_len=policy_cfg.max_token_len,
    )

    # Initialize model
    rngs = jax.random.PRNGKey(0)
    model = pi0.Pi0(model_cfg, rngs)

    # Load weights
    # This is simplified - you may need to adapt based on your checkpoint structure
    # model = checkpoints.restore_model(model, checkpoint_data)

    return model, policy_cfg


def extract_image_token_attention(
    attn_weights: jax.Array,
    num_image_tokens_per_camera: int,
    num_cameras: int = 2,
    action_token_idx: int = -1,
) -> tuple[jax.Array, ...]:
    """
    Extract attention from action tokens to image tokens.

    Args:
        attn_weights: Attention weights with shape (B, K, G, T, S) where
            B = batch size
            K = num_kv_heads
            G = num_query_heads_per_kv_head
            T = query_length (suffix tokens including action tokens)
            S = key_length (prefix + suffix tokens)
        num_image_tokens_per_camera: Number of tokens per camera (typically 256 for 16x16 grid)
        num_cameras: Number of cameras (typically 2: exterior + wrist)
        action_token_idx: Which action token to use for visualization (-1 for last)

    Returns:
        Tuple of attention maps, one per camera
    """
    # attn_weights shape: (B, K, G, T, S)
    # We want attention from the last action token to image tokens

    # Take the action token's attention (last token in the query sequence)
    action_attn = attn_weights[:, :, :, action_token_idx, :]  # (B, K, G, S)

    # Average across K and G dimensions (attention heads)
    action_attn = jnp.mean(action_attn, axis=(1, 2))  # (B, S)

    # Extract attention to image tokens (they come first in the sequence)
    total_image_tokens = num_image_tokens_per_camera * num_cameras
    image_attn = action_attn[:, :total_image_tokens]  # (B, total_image_tokens)

    # Split by camera
    camera_attns = []
    for i in range(num_cameras):
        start_idx = i * num_image_tokens_per_camera
        end_idx = start_idx + num_image_tokens_per_camera
        cam_attn = image_attn[:, start_idx:end_idx]  # (B, num_tokens)
        camera_attns.append(cam_attn)

    return tuple(camera_attns)


def reshape_attention_to_2d(attn_1d: jax.Array, grid_size: int = 16) -> jax.Array:
    """
    Reshape 1D attention weights to 2D spatial grid.

    Args:
        attn_1d: 1D attention weights (B, num_tokens) where num_tokens = grid_size^2
        grid_size: Size of the spatial grid (default 16 for 16x16 = 256 tokens)

    Returns:
        2D attention map (B, grid_size, grid_size)
    """
    batch_size = attn_1d.shape[0]
    attn_2d = jnp.reshape(attn_1d, (batch_size, grid_size, grid_size))
    return attn_2d


def upsample_attention_map(attn_2d: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Upsample 2D attention map to match image resolution.

    Args:
        attn_2d: 2D attention map (H, W) or (B, H, W)
        target_size: Target image size (default 224x224)

    Returns:
        Upsampled attention map
    """
    if attn_2d.ndim == 3:
        # Batch dimension present, process first image
        attn_2d = attn_2d[0]

    # Use bilinear interpolation
    attn_upsampled = cv2.resize(
        attn_2d,
        (target_size, target_size),
        interpolation=cv2.INTER_LINEAR,
    )
    return attn_upsampled


def create_heatmap_overlay(
    image: np.ndarray,
    attention_map: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Create a heatmap overlay on the image.

    Args:
        image: Original image (H, W, 3) in RGB, values in [0, 255]
        attention_map: Attention heatmap (H, W), values in [0, 1]
        colormap: Matplotlib colormap name
        alpha: Transparency of heatmap (0=transparent, 1=opaque)

    Returns:
        Image with heatmap overlay (H, W, 3) in RGB
    """
    # Normalize attention map to [0, 1]
    attn_min = attention_map.min()
    attn_max = attention_map.max()
    if attn_max > attn_min:
        attention_map = (attention_map - attn_min) / (attn_max - attn_min)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(attention_map)[:, :, :3]  # Drop alpha channel, get RGB
    heatmap = (heatmap * 255).astype(np.uint8)

    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

    # Blend image and heatmap
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return overlay


def visualize_attention_on_images(
    images: dict[str, np.ndarray],
    attention_maps: dict[str, np.ndarray],
    output_path: str,
    colormap: str = "jet",
    alpha: float = 0.4,
):
    """
    Create visualization with attention overlays and save to file.

    Args:
        images: Dictionary of camera name -> image (H, W, 3)
        attention_maps: Dictionary of camera name -> attention map (H, W)
        output_path: Path to save output image
        colormap: Matplotlib colormap
        alpha: Heatmap transparency
    """
    n_cameras = len(images)

    fig, axes = plt.subplots(2, n_cameras, figsize=(6 * n_cameras, 10))
    if n_cameras == 1:
        axes = axes[:, np.newaxis]

    for i, (cam_name, image) in enumerate(images.items()):
        attn_map = attention_maps[cam_name]

        # Original image
        axes[0, i].imshow(image)
        axes[0, i].set_title(f"{cam_name} - Original")
        axes[0, i].axis("off")

        # Overlay
        overlay = create_heatmap_overlay(image, attn_map, colormap, alpha)
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"{cam_name} - Attention Overlay")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved visualization to {output_path}")
    plt.close()


def main(args: Args):
    """Main function to run attention visualization."""
    logger.info("Starting attention visualization...")

    # Load images
    images = {}
    if args.image_exterior:
        img = Image.open(args.image_exterior).convert("RGB")
        img = img.resize((args.image_size, args.image_size))
        images["exterior"] = np.array(img)

    if args.image_wrist:
        img = Image.open(args.image_wrist).convert("RGB")
        img = img.resize((args.image_size, args.image_size))
        images["wrist"] = np.array(img)

    if not images:
        logger.error("No images provided. Use --image_exterior and/or --image_wrist")
        return

    logger.info(f"Loaded {len(images)} images")

    # TODO: Load model and run inference with return_attention=True
    # This is a placeholder - you'll need to:
    # 1. Load your model checkpoint
    # 2. Prepare observation dict with images, state, prompt
    # 3. Run model.sample_actions(..., return_attention=True)
    # 4. Extract attention weights from the returned list

    logger.warning("Model loading and inference not implemented yet.")
    logger.warning("Creating dummy attention maps for demonstration...")

    # Create dummy attention maps for demonstration
    grid_size = 16
    dummy_attention = {}
    for cam_name in images.keys():
        # Create a dummy attention pattern (e.g., center focus)
        y, x = np.ogrid[:grid_size, :grid_size]
        center = grid_size // 2
        attn = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (grid_size / 2) ** 2)
        attn_upsampled = upsample_attention_map(attn, args.image_size)
        dummy_attention[cam_name] = attn_upsampled

    # Create visualization
    visualize_attention_on_images(
        images=images,
        attention_maps=dummy_attention,
        output_path=args.output_path,
        colormap=args.colormap,
        alpha=args.alpha,
    )

    logger.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    main(args)
