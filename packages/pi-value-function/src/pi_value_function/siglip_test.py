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


def test_siglip1_forward_pass():
    """Test forward pass with SigLIP 1 (current setup)."""
    print("\n" + "="*80)
    print("Testing SigLIP 1 (Current Setup - Random Initialization)")
    print("="*80)

    # For comparison, use same config as SigLIP 2 (MAP pooling, no classification head)
    # This way we can compare pre_logits fairly
    model = _siglip.Module(
        num_classes=None,
        variant="So400m/14",
        pool_type="map",
        scan=True,
        dtype_mm="float32",
    )

    # Create dummy image
    test_image = create_test_image()
    print(f"Input image shape: {test_image.shape}")

    # Initialize model parameters
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, test_image, train=False)

    # Print parameter structure
    print("\nSigLIP 1 Model Parameter Structure:")
    print_param_structure(variables['params'], max_depth=3)

    # Forward pass
    output, aux = model.apply(variables, test_image, train=False)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
    print(f"Auxiliary outputs keys: {list(aux.keys())}")

    # Check pre_logits_2d which is what we use in Pi0
    if 'pre_logits_2d' in aux:
        print(f"pre_logits_2d shape: {aux['pre_logits_2d'].shape}")

    return {
        'model': model,
        'variables': variables,
        'output': output,
        'aux': aux,
        'test_image': test_image,
    }


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


def compare_outputs(siglip1_results: dict, siglip2_results: dict):
    """Compare outputs from SigLIP 1 and SigLIP 2."""
    print("\n" + "="*80)
    print("Comparing Outputs")
    print("="*80)

    # Note: When num_classes=None, output is just the pooled pre_logits
    output1 = siglip1_results['output']
    output2 = siglip2_results['output']

    print(f"Output type: {'pre_logits' if output1.ndim == 2 else 'logits'}")
    print(f"Shape match: {output1.shape == output2.shape}")
    print(f"Random init shape: {output1.shape}")
    print(f"SigLIP 2 shape: {output2.shape}")

    if output1.shape == output2.shape:
        # Compare actual output values
        diff = jnp.abs(output1 - output2)
        print(f"\nOutput Difference Statistics:")
        print(f"  Mean absolute difference: {diff.mean():.6f}")
        print(f"  Max absolute difference: {diff.max():.6f}")
        print(f"  Min absolute difference: {diff.min():.6f}")

        # Compute cosine similarity for each example in batch
        print(f"\nCosine Similarity (per example, random vs pretrained):")
        for i in range(output1.shape[0]):
            # Flatten each example's output
            if output1.ndim == 2:
                o1_flat = output1[i]
                o2_flat = output2[i]
            else:
                o1_flat = output1[i].reshape(-1)
                o2_flat = output2[i].reshape(-1)

            # Compute cosine similarity
            cosine_sim = jnp.sum(o1_flat * o2_flat) / (jnp.linalg.norm(o1_flat) * jnp.linalg.norm(o2_flat))
            print(f"  Example {i}: {cosine_sim:.6f}")

    # Compare pre_logits_2d (what we use in Pi0)
    if 'pre_logits_2d' in siglip1_results['aux'] and 'pre_logits_2d' in siglip2_results['aux']:
        pre1 = siglip1_results['aux']['pre_logits_2d']
        pre2 = siglip2_results['aux']['pre_logits_2d']

        print(f"\npre_logits_2d Comparison:")
        print(f"  SigLIP 1: {pre1.shape}, mean={pre1.mean():.4f}, std={pre1.std():.4f}")
        print(f"  SigLIP 2: {pre2.shape}, mean={pre2.mean():.4f}, std={pre2.std():.4f}")

        diff = jnp.abs(pre1 - pre2)
        print(f"  Mean absolute difference: {diff.mean():.6f}")
        print(f"  Max absolute difference: {diff.max():.6f}")


def main():
    """Run all tests."""
    print("SigLIP 1 -> SigLIP 2 Weight Switching Test")
    print("=" * 80)

    # Test 1: SigLIP 1 forward pass
    siglip1_results = test_siglip1_forward_pass()

    # Test 2: Load SigLIP 2 weights
    siglip2_weights = test_siglip2_weight_loading()

    # Test 3: SigLIP 2 forward pass with same architecture
    siglip2_results = test_siglip2_forward_pass(
        siglip2_weights,
        siglip1_results['test_image']
    )

    # Test 4: Compare outputs
    compare_outputs(siglip1_results, siglip2_results)

    print("\n" + "="*80)
    print("Test Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
