"""
Example script showing how to visualize attention weights during inference.

This demonstrates how to:
1. Run inference with attention capture enabled
2. Extract attention weights for each camera
3. Create heatmap visualizations overlaid on camera frames

Usage:
    python examples/visualize_attention_example.py
"""

import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

# Add parent directory to path to import scripts module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from visualize_attention import (
    extract_image_token_attention,
    reshape_attention_to_2d,
    upsample_attention_map,
    visualize_attention_on_images,
)

logger = logging.getLogger(__name__)


def run_inference_with_attention(model, observation, num_steps=10):
    """
    Run model inference and capture attention weights.

    Args:
        model: Pi0 model instance
        observation: Observation dict with images, state, prompt
        num_steps: Number of diffusion steps

    Returns:
        actions: Predicted actions
        attention_weights: List of attention weights from each diffusion step
    """
    rng = jax.random.PRNGKey(0)

    # Run inference with attention capture enabled
    actions, attention_weights = model.sample_actions(
        rng=rng,
        observation=observation,
        num_steps=num_steps,
        return_attention=True,  # This is the key parameter!
    )

    logger.info(f"Got {len(attention_weights)} attention weight arrays")
    if len(attention_weights) > 0:
        logger.info(f"Attention weight shape: {attention_weights[0].shape}")

    return actions, attention_weights


def process_attention_for_visualization(
    attention_weights_list,
    num_cameras=2,
    image_size=224,
    use_last_step=True,
    average_steps=False,
):
    """
    Process raw attention weights into heatmaps for visualization.

    Args:
        attention_weights_list: List of attention weight arrays from diffusion steps
        num_cameras: Number of cameras in the input
        image_size: Target size for upsampling
        use_last_step: Use only the last diffusion step
        average_steps: Average attention across all steps

    Returns:
        Dictionary mapping camera names to attention heatmaps
    """
    if len(attention_weights_list) == 0:
        logger.warning("No attention weights available")
        return {}

    # SigLIP processes 224x224 images with patch size 14 -> 16x16 = 256 tokens per image
    grid_size = 16
    tokens_per_camera = grid_size * grid_size

    # Select which attention to use
    if use_last_step:
        attn_weights = attention_weights_list[-1]
    elif average_steps:
        # Average attention across all steps
        attn_weights = jnp.mean(jnp.stack(attention_weights_list), axis=0)
    else:
        attn_weights = attention_weights_list[0]

    # Extract attention from action tokens to image tokens
    camera_attentions = extract_image_token_attention(
        attn_weights=attn_weights,
        num_image_tokens_per_camera=tokens_per_camera,
        num_cameras=num_cameras,
        action_token_idx=-1,  # Use last action token
    )

    # Process each camera's attention
    camera_names = ["exterior", "wrist"][:num_cameras]
    attention_maps = {}

    for cam_name, cam_attn in zip(camera_names, camera_attentions):
        # Reshape to 2D grid
        attn_2d = reshape_attention_to_2d(cam_attn, grid_size=grid_size)

        # Convert to numpy and remove batch dimension
        attn_2d_np = np.array(attn_2d[0])

        # Upsample to match image resolution
        attn_upsampled = upsample_attention_map(attn_2d_np, target_size=image_size)

        attention_maps[cam_name] = attn_upsampled

    return attention_maps


def visualize_from_video_frames(video_path, model, frame_indices=None, output_dir="attention_outputs"):
    """
    Extract frames from video and visualize attention on them.

    Args:
        video_path: Path to video file
        model: Pi0 model
        frame_indices: List of frame indices to process (or None for all)
        output_dir: Directory to save outputs
    """
    import cv2

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_indices is None:
        # Process every 10th frame by default
        frame_indices = range(0, frame_count, 10)

    logger.info(f"Processing {len(frame_indices)} frames from video...")

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            logger.warning(f"Could not read frame {frame_idx}")
            continue

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # TODO: Prepare observation dict with frame(s) and run inference
        # This depends on your specific setup (single camera vs multi-camera, etc.)
        # observation = prepare_observation(frame_rgb, ...)
        # actions, attention_weights = run_inference_with_attention(model, observation)

        # Process and visualize attention
        # attention_maps = process_attention_for_visualization(attention_weights)
        # ...

        logger.info(f"Processed frame {frame_idx}")

    cap.release()
    logger.info(f"Saved visualizations to {output_dir}")


def example_with_dummy_data():
    """
    Example using dummy data to demonstrate the workflow.
    Replace this with your actual model and data loading.
    """
    logger.info("Running example with dummy data...")

    # Create dummy images
    image_size = 224
    exterior_img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
    wrist_img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)

    images = {
        "exterior": exterior_img,
        "wrist": wrist_img,
    }

    # Create dummy attention weights with the expected shape
    # Shape: (B, K, G, T, S) where:
    #   B = batch_size = 1
    #   K = num_kv_heads = 1 (from gemma config)
    #   G = num_query_heads_per_kv_head = 8 (8 query heads / 1 kv head)
    #   T = query_length (suffix tokens, includes action tokens)
    #   S = key_length (prefix + suffix tokens)

    batch_size = 1
    num_kv_heads = 1
    num_query_heads_per_kv = 8
    action_horizon = 10
    state_tokens = 0  # For pi0.5, no separate state token
    tokens_per_image = 256  # 16x16 grid
    num_images = 2

    query_length = action_horizon + state_tokens
    key_length = num_images * tokens_per_image + query_length

    # Create dummy attention (shape: B, K, G, T, S)
    dummy_attn = jnp.ones((batch_size, num_kv_heads, num_query_heads_per_kv, query_length, key_length))

    # Add some spatial pattern (center focus)
    for img_idx in range(num_images):
        start_idx = img_idx * tokens_per_image
        # Create center-focused attention
        grid_size = 16
        y, x = np.ogrid[:grid_size, :grid_size]
        center = grid_size // 2
        spatial_attn = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (grid_size / 2) ** 2)
        spatial_attn_flat = spatial_attn.flatten()

        # Apply to all action tokens, all heads
        for t in range(query_length):
            for k in range(num_kv_heads):
                for g in range(num_query_heads_per_kv):
                    dummy_attn = dummy_attn.at[0, k, g, t, start_idx:start_idx + tokens_per_image].set(
                        spatial_attn_flat
                    )

    attention_weights_list = [dummy_attn]  # Single step for this example

    # Process attention
    attention_maps = process_attention_for_visualization(
        attention_weights_list,
        num_cameras=2,
        image_size=image_size,
    )

    # Visualize
    output_path = "attention_visualization_example.png"
    visualize_attention_on_images(
        images=images,
        attention_maps=attention_maps,
        output_path=output_path,
        colormap="jet",
        alpha=0.4,
    )

    logger.info(f"Created example visualization: {output_path}")


def example_with_real_model():
    """
    Example showing how to use with a real model.
    You'll need to customize this based on your setup.
    """
    logger.info("Example with real model (customize this for your setup)...")

    # 1. Load your model
    # from openpi.policies import droid_policy
    # policy = droid_policy.DroidPolicy.from_checkpoint("path/to/checkpoint")
    # model = policy.model

    # 2. Prepare observation
    # observation = {
    #     "images": {
    #         "exterior": jnp.array(exterior_image),  # (1, 224, 224, 3)
    #         "wrist": jnp.array(wrist_image),         # (1, 224, 224, 3)
    #     },
    #     "image_masks": {
    #         "exterior": jnp.ones((1,), dtype=bool),
    #         "wrist": jnp.ones((1,), dtype=bool),
    #     },
    #     "state": jnp.array(state_vector),           # (1, state_dim)
    #     "tokenized_prompt": tokenized_prompt,       # From tokenizer
    #     "tokenized_prompt_mask": prompt_mask,
    # }

    # 3. Run inference with attention
    # actions, attention_weights = run_inference_with_attention(model, observation)

    # 4. Process and visualize
    # attention_maps = process_attention_for_visualization(attention_weights)
    # visualize_attention_on_images(images, attention_maps, "output.png")

    logger.info("Customize this function with your model loading and data preparation code")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the dummy data example
    example_with_dummy_data()

    # Uncomment to use with real model
    # example_with_real_model()
