"""Example: Load Gemma 3 270M weights without gemma package dependencies.

This example shows how to load Gemma checkpoint weights directly using Orbax,
bypassing the kauldron/TensorFlow dependency chain that causes protobuf conflicts.
"""

import sys
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pi_value_function.training.direct_checkpoint_loader import (
    load_gemma_checkpoint,
    DirectGemma3WeightLoader
)


def example_basic_loading():
    """Example 1: Basic checkpoint loading."""
    print("\n" + "="*80)
    print("Example 1: Basic Checkpoint Loading")
    print("="*80)

    # Load checkpoint directly
    params = load_gemma_checkpoint("checkpoints/gemma-3-270m")

    print(f"Loaded {len(params)} top-level parameter groups")
    print(f"Keys: {list(params.keys())[:5]}...")

    # Count total parameters
    from flax.traverse_util import flatten_dict
    flat_params = flatten_dict(params, sep="/")
    total = sum(v.size for v in flat_params.values() if hasattr(v, 'size'))
    print(f"Total parameters: {total:,} ({total/1e6:.1f}M)")


def example_with_weight_loader():
    """Example 2: Using the WeightLoader interface for model integration."""
    print("\n" + "="*80)
    print("Example 2: WeightLoader Interface")
    print("="*80)

    # Create weight loader that nests Gemma params under 'language_model' key
    loader = DirectGemma3WeightLoader(
        checkpoint_path="checkpoints/gemma-3-270m",
        param_key="language_model"
    )

    # Simulate existing model params (e.g., vision encoder, other components)
    existing_params = {
        "vision_encoder": {"dummy": "params"},
        "head": {"linear": "layer"}
    }

    print("Existing params:", list(existing_params.keys()))

    # Load and merge Gemma weights
    merged_params = loader.load(existing_params)

    print("After loading:", list(merged_params.keys()))
    print("Gemma sub-keys:", list(merged_params['language_model'].keys())[:5], "...")


def example_inspect_structure():
    """Example 3: Inspect checkpoint structure."""
    print("\n" + "="*80)
    print("Example 3: Inspect Checkpoint Structure")
    print("="*80)

    params = load_gemma_checkpoint("checkpoints/gemma-3-270m")

    # Show layer structure
    layer_keys = [k for k in params.keys() if k.startswith('layer_')]
    print(f"Transformer has {len(layer_keys)} layers")

    # Show first layer structure
    print(f"\nLayer 0 parameters:")
    from flax.traverse_util import flatten_dict
    layer_flat = flatten_dict(params['layer_0'], sep="/")
    for key, value in list(layer_flat.items())[:10]:
        print(f"  {key:40s} shape: {value.shape}")


if __name__ == "__main__":
    # Run examples
    example_basic_loading()
    example_with_weight_loader()
    example_inspect_structure()

    print("\n" + "="*80)
    print("✓ All examples completed successfully!")
    print("="*80)
