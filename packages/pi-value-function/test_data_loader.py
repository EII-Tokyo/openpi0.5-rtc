#!/usr/bin/env python3
"""Simple test script for the value function data loader.

This script demonstrates how to use the value function data loader and
verifies that it produces correctly formatted batches.

Usage:
    # Update the repo_ids below with your actual LeRobot dataset repo IDs, then run:
    python test_data_loader.py
"""

import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pi_value_function.training.data_loader import create_value_dataloader


def test_data_loader():
    """Test the value function data loader."""
    print("Testing Value Function Data Loader")
    print("=" * 60)

    success_repo_ids = [
        "michios/droid_xxjd",
        "michios/droid_xxjd_2"
    ]
    failure_repo_ids = [
        "michios/droid_xxjd_3",
        "michios/droid_xxjd_4"
    ]

    # Configuration
    batch_size = 4
    failure_cost_json = Path(__file__).parent / "configs" / "failure_costs.json"

    print(f"\nConfiguration:")
    print(f"  Success repos: {success_repo_ids}")
    print(f"  Failure repos: {failure_repo_ids}")
    print(f"  Batch size: {batch_size}")
    print(f"  Failure cost JSON: {failure_cost_json}")
    print(f"  Failure cost JSON exists: {failure_cost_json.exists()}")

    # Create data loader
    print(f"\n{'Creating data loader...'}")
    try:
        dataloader = create_value_dataloader(
            success_repo_ids=success_repo_ids,
            failure_repo_ids=failure_repo_ids,
            batch_size=batch_size,
            failure_cost_json=str(failure_cost_json) if failure_cost_json.exists() else None,
            default_c_fail=100.0,
            success_sampling_ratio=0.5,
            num_workers=0,  # Use 0 workers for testing to avoid multiprocessing issues
            seed=42,
        )
        print("✓ Data loader created successfully")
    except Exception as e:
        print(f"✗ Failed to create data loader: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test loading a few batches
    print(f"\n{'Testing batch loading...'}")
    try:
        dataloader_iter = iter(dataloader)

        for i in range(3):
            print(f"\nBatch {i+1}:")
            batch = next(dataloader_iter)

            # Check keys
            expected_keys = {"image", "image_mask", "state", "prompt", "returns"}
            actual_keys = set(batch.keys())
            print(f"  Keys: {actual_keys}")

            if not expected_keys.issubset(actual_keys):
                print(f"  ✗ Missing keys: {expected_keys - actual_keys}")
            else:
                print(f"  ✓ All expected keys present")

            # Check shapes
            print(f"  Image shapes:")
            for camera_name, image in batch["image"].items():
                print(f"    {camera_name}: {image.shape} (dtype: {image.dtype})")

            print(f"  Image masks:")
            for camera_name, mask in batch["image_mask"].items():
                print(f"    {camera_name}: {mask}")

            print(f"  State shape: {batch['state'].shape} (dtype: {batch['state'].dtype})")
            print(f"  Prompt (first sample): {batch['prompt'][0] if isinstance(batch['prompt'], list) else batch['prompt']}")
            print(f"  Returns shape: {batch['returns'].shape} (dtype: {batch['returns'].dtype})")
            print(f"  Returns range: [{batch['returns'].min():.4f}, {batch['returns'].max():.4f}]")

            # Verify returns are in [-1, 0] range
            if (batch['returns'] >= -1.0).all() and (batch['returns'] <= 0.0).all():
                print(f"  ✓ Returns are in [-1, 0] range")
            else:
                print(f"  ✗ Returns are NOT in [-1, 0] range!")

            # Verify batch size
            if batch['returns'].shape[0] == batch_size:
                print(f"  ✓ Batch size is correct ({batch_size})")
            else:
                print(f"  ✗ Batch size mismatch: expected {batch_size}, got {batch['returns'].shape[0]}")

        print(f"\n{'✓ All batches loaded successfully!'}")

    except Exception as e:
        print(f"\n✗ Error loading batches: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n{'=' * 60}")
    print("Test completed successfully!")
    print("\nNext steps:")
    print("1. Update the repo_ids in this script with your actual LeRobot datasets")
    print("2. Update the failure_costs.json with your actual prompts and costs")
    print("3. Integrate this data loader into your training script")


if __name__ == "__main__":
    test_data_loader()
