"""Test script for comparing SigLIP 1 and SigLIP 2 weights.

This script tests the backward compatibility claim that SigLIP 2 weights
can be swapped in as a drop-in replacement for SigLIP 1 weights.
"""

import urllib.request
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

# Import the SigLIP module from openpi
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

import openpi.models.siglip as _siglip


# Checkpoint URLs
SIGLIP2_SO400M14_224_URL = "https://storage.googleapis.com/big_vision/siglip2/siglip2_so400m14_224.npz"


def download_checkpoint(url: str, cache_dir: Path = Path("./checkpoints")) -> Path:
    """Download checkpoint from URL if not already cached."""
    cache_dir.mkdir(exist_ok=True)
    filename = url.split("/")[-1]
    filepath = cache_dir / filename

    if not filepath.exists():
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded to {filepath}")
    else:
        print(f"Using cached checkpoint: {filepath}")

    return filepath


def load_siglip2_weights(checkpoint_path: Path) -> dict:
    """Load SigLIP 2 weights from .npz file."""
    print(f"Loading weights from {checkpoint_path}...")
    weights = np.load(checkpoint_path)

    # Print structure to understand the checkpoint format
    print(f"\nTotal checkpoint keys: {len(weights.keys())}")
    print(f"First 20 keys:")
    for i, key in enumerate(list(weights.keys())[:20]):
        print(f"  {i}: {key} - shape: {weights[key].shape}")

    return dict(weights)


def map_siglip2_to_model_params(siglip2_weights: dict) -> dict:
    """Map SigLIP 2 checkpoint weights to model's expected parameter structure.

    SigLIP 2 checkpoint format: params/img/embedding/kernel
    Model expects: embedding/kernel

    We just strip the 'params/img/' prefix.
    """
    print("\n" + "="*80)
    print("Mapping SigLIP 2 weights to model format")
    print("="*80)

    params = {}

    for key, value in siglip2_weights.items():
        # Strip 'params/img/' prefix
        if key.startswith('params/img/'):
            new_key = key[11:]  # Remove 'params/img/'

            # Build nested dict structure
            parts = new_key.split('/')
            current = params
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the leaf value
            current[parts[-1]] = jnp.array(value)

    print(f"Mapped {len([k for k in siglip2_weights.keys() if k.startswith('params/img/')])} parameters")
    print(f"Top-level parameter groups: {list(params.keys())}")

    return {'params': params}


def create_test_image(batch_size: int = 2, size: int = 224) -> jnp.ndarray:
    """Create a test image with the expected shape."""
    # SigLIP expects [batch, height, width, channels]
    return jnp.ones((batch_size, size, size, 3), dtype=jnp.float32)


def print_param_structure(params, prefix="", max_depth=3, current_depth=0):
    """Recursively print parameter structure."""
    if current_depth >= max_depth:
        return

    for key, value in params.items():
        if isinstance(value, dict):
            print(f"{'  ' * current_depth}{prefix}{key}/")
            print_param_structure(value, prefix="", max_depth=max_depth, current_depth=current_depth + 1)
        else:
            print(f"{'  ' * current_depth}{prefix}{key}: {value.shape}")


def test_siglip2_weight_loading():
    """Test loading SigLIP 2 weights into the same architecture."""
    print("\n" + "="*80)
    print("Testing SigLIP 2 Weight Loading")
    print("="*80)

    # Download SigLIP 2 checkpoint
    checkpoint_path = download_checkpoint(SIGLIP2_SO400M14_224_URL)

    # Load weights
    siglip2_weights = load_siglip2_weights(checkpoint_path)

    print(f"\nTotal parameters in checkpoint: {len(siglip2_weights)}")

    return siglip2_weights


def test_siglip2_forward_pass(siglip2_weights: dict, test_image: jnp.ndarray):
    """Test forward pass with SigLIP 2 weights loaded into same architecture."""
    print("\n" + "="*80)
    print("Testing SigLIP 2 Forward Pass")
    print("="*80)

    # SigLIP 2 was trained with MAP pooling, not pool_type="none"
    # The checkpoint has MAPHead_0 but no "head" parameters
    # We'll use pool_type="map" and num_classes=None to match the checkpoint
    model = _siglip.Module(
        num_classes=None,  # No classification head in the checkpoint
        variant="So400m/14",
        pool_type="map",  # SigLIP 2 uses MAP pooling
        scan=True,
        dtype_mm="float32",
    )

    # Map SigLIP 2 weights to model format
    variables = map_siglip2_to_model_params(siglip2_weights)

    print("\nSigLIP 2 Loaded Parameter Structure:")
    print_param_structure(variables['params'], max_depth=3)

    # Forward pass with SigLIP 2 weights
    print("\nRunning forward pass with SigLIP 2 weights...")
    output, aux = model.apply(variables, test_image, train=False)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")

    if 'pre_logits_2d' in aux:
        print(f"pre_logits_2d shape: {aux['pre_logits_2d'].shape}")

    return {
        'model': model,
        'variables': variables,
        'output': output,
        'aux': aux,
    }


def main():
    """Run all tests."""
    print("SigLIP 1 -> SigLIP 2 Weight Switching Test")
    print("=" * 80)

    # Test 2: Load SigLIP 2 weights
    siglip2_weights = test_siglip2_weight_loading()
    
    test_image = create_test_image()

    # Test 3: SigLIP 2 forward pass with same architecture
    siglip2_results = test_siglip2_forward_pass(
        siglip2_weights,
        test_image
    )
    
    outputs = siglip2_results.output
    
    print(outputs)
    
    

if __name__ == "__main__":
    main()
