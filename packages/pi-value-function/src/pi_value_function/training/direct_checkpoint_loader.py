"""Direct Orbax checkpoint loader for Gemma 3 weights.

This module provides a lightweight checkpoint loading solution that avoids
the kauldron/TensorFlow dependency chain. It uses Orbax directly to load
Gemma 3 270M weights from Kaggle checkpoints.
"""

import dataclasses
from pathlib import Path

import jax
from orbax import checkpoint as ocp


def load_gemma_checkpoint(
    checkpoint_path: str | Path,
    target_params: dict | None = None,
) -> dict:
    """Load Gemma 3 checkpoint directly using Orbax.

    Bypasses gemma package's kauldron dependencies to avoid protobuf conflicts.

    Args:
        checkpoint_path: Path to the Gemma checkpoint directory
        target_params: Optional target parameter structure for sharding.
                      If None, loads as replicated arrays on first device.

    Returns:
        Dictionary of loaded parameters in nested format
    """
    checkpoint_path = Path(checkpoint_path).resolve()  # Convert to absolute path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Create Orbax checkpointer
    checkpointer = ocp.StandardCheckpointer()

    # Get checkpoint metadata to understand the structure
    metadata = checkpointer.metadata(checkpoint_path)

    # Create target structure from metadata for restoration
    if target_params is None:
        target_structure = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(
                shape=x.shape,
                dtype=x.dtype,
                sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0])
            ),
            metadata.tree
        )
    else:
        target_structure = target_params

    # Restore the checkpoint
    restored = checkpointer.restore(checkpoint_path, target_structure)

    # Convert from flat format (transformer/layer_0/...) to nested
    return _flat_to_nested(restored)


def _flat_to_nested(params: dict) -> dict:
    """Convert flat checkpoint format to nested dictionary structure.

    Converts: {'transformer/layer_0/attn/kernel': array}
    To: {'layer_0': {'attn': {'kernel': array}}}
    """
    return _unflatten_dict(params, 'transformer')


def _unflatten_dict(flat_dict: dict, prefix: str) -> dict:
    """Convert flat dictionary to nested structure.

    Removes 'prefix/' from keys and builds nested dict from '/' separators.
    """
    nested = {}
    for key, value in flat_dict.items():
        # Remove prefix
        parts = key[len(prefix) + 1:].split('/') if key.startswith(f'{prefix}/') else key.split('/')

        # Build nested structure
        current = nested
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value

    return nested


@dataclasses.dataclass(frozen=True)
class DirectGemma3WeightLoader:
    """Loads Gemma 3 270M weights using direct Orbax loading.

    Bypasses gemma package's kauldron dependencies to avoid protobuf conflicts.

    Attributes:
        checkpoint_path: Path to Gemma checkpoint directory
        param_key: Optional key to nest Gemma params under (e.g., 'language_model')
    """

    checkpoint_path: str
    param_key: str | None = None

    def load(self, params: dict) -> dict:
        """Load Gemma weights and merge with existing params."""
        gemma_params = load_gemma_checkpoint(self.checkpoint_path)

        # Nest params if key specified
        loaded = {self.param_key: gemma_params} if self.param_key else gemma_params

        # Merge with existing params
        return {**params, **loaded}


if __name__ == "__main__":
    import sys
    from flax.traverse_util import flatten_dict

    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/gemma-3-270m"

    print(f"Loading checkpoint from: {checkpoint_path}")
    params = load_gemma_checkpoint(checkpoint_path)

    # Print structure
    flat_params = flatten_dict(params, sep="/")
    total = sum(v.size for v in flat_params.values() if hasattr(v, 'size'))

    print(f"\nTop-level keys: {list(params.keys())}")
    print(f"Total parameters: {total:,} ({total/1e6:.1f}M)")

    print("\nFirst 20 parameter keys:")
    for i, (key, value) in enumerate(list(flat_params.items())[:20]):
        print(f"  {key:50s} shape: {value.shape}, dtype: {value.dtype}")

    print("\n✓ Success!")
