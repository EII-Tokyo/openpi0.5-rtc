#!/usr/bin/env python3
"""Standalone test for Gemma 3 weight loading."""

import sys
from pathlib import Path

# Add gemma package to path
gemma_path = Path("packages/gemma")
sys.path.insert(0, str(gemma_path))

import jax
import jax.numpy as jnp
from gemma import gm

print("="*80)
print("Testing Gemma 3 270M Weight Loading")
print("="*80)

# Initialize Gemma 3 270M model
print("\n1. Initializing Gemma 3 270M model...")
model = gm.nn.Gemma3_270M()

# Create dummy input to initialize model
dummy_tokens = jnp.ones((1, 10), dtype=jnp.int32)
rng = jax.random.PRNGKey(0)
variables = model.init(rng, dummy_tokens)

print(f"✓ Initialized model with {len(variables['params'])} top-level param groups")

# Load weights from local checkpoint
print("\n2. Loading weights from checkpoint...")
checkpoint_path = "checkpoints/gemma-3-270m"
loaded_params = gm.ckpts.load_params(checkpoint_path)

print(f"✓ Loaded {len(loaded_params)} param groups from checkpoint")

# Print structure
print("\n3. Checkpoint structure:")
flat_params = __import__('flax').traverse_util.flatten_dict(loaded_params, sep="/")
print(f"   Total keys: {len(flat_params)}")
print(f"   First 10 keys:")
for i, key in enumerate(list(flat_params.keys())[:10]):
    print(f"   {i}: {key} - shape: {flat_params[key].shape}")

print("\n" + "="*80)
print("✓ Weight loading test successful!")
print("="*80)
