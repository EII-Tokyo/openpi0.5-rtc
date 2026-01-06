#!/usr/bin/env python3
"""Check value function data distribution using ALL datasets.

This script analyzes the full training data distribution to verify that
the normalization produces good variance and coverage.

Run this on your training server with:
    uv run python packages/pi-value-function/check_full_data_distribution.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
from pathlib import Path

from pi_value_function.training.data_loader import create_value_dataloader
from openpi.models.tokenizer import Gemma3Tokenizer
from pi_value_function.training.checkpoint_downloader import download_gemma_from_kaggle


def main():
    print("="*70)
    print("VALUE FUNCTION DATA DISTRIBUTION CHECKER")
    print("="*70)

    # 1. Initialize tokenizer
    print("\n[1/4] Loading tokenizer...")
    _, tokenizer_path = download_gemma_from_kaggle()
    tokenizer = Gemma3Tokenizer(path=tokenizer_path, max_len=48)
    print("✓ Tokenizer loaded")

    # 2. Create dataloader with ALL datasets
    print("\n[2/4] Creating dataloader with ALL datasets...")
    print("This may take a few minutes to download datasets from HuggingFace...")

    dataloader = create_value_dataloader(
        tokenizer=tokenizer,
        success_repo_ids=[
            "michios/droid_xxjd",
            "michios/droid_xxjd_2",
            "michios/droid_xxjd_3",
            "michios/droid_xxjd_4",
            "michios/droid_xxjd_5",
            "michios/droid_xxjd_6",
            "michios/droid_xxjd_7",
        ],
        failure_repo_ids=[
            "michios/droid_xxjd_fail_1"
        ],
        batch_size=128,  # Larger batch for efficiency
        failure_cost_json="configs/failure_costs.json",
        default_c_fail=100.0,
        success_sampling_ratio=0.5,  # 50/50 mix
        num_workers=4,  # Parallel loading
        seed=42,
    )
    print("✓ Dataloader created")

    # 3. Sample data points
    num_samples = 10000  # Large sample for accurate statistics
    print(f"\n[3/4] Sampling {num_samples} data points...")
    print("This will take a few minutes...")

    returns = []
    data_iter = iter(dataloader)

    num_batches = num_samples // dataloader.batch_size
    for i in range(num_batches):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_batches} batches ({len(returns)} samples)")
        batch = next(data_iter)
        returns.extend(batch['returns'].tolist())

    returns = np.array(returns)
    print(f"✓ Collected {len(returns)} samples")

    # 4. Analyze and visualize
    print(f"\n[4/4] Analyzing distribution...")

    # Print statistics
    print("\n" + "="*70)
    print("DATA STATISTICS")
    print("="*70)
    print(f"Sample size:     {len(returns):,}")
    print(f"\nValue statistics:")
    print(f"  Mean:          {returns.mean():.4f}")
    print(f"  Std:           {returns.std():.4f}")
    print(f"  Min:           {returns.min():.4f}")
    print(f"  Max:           {returns.max():.4f}")
    print(f"  Median:        {np.median(returns):.4f}")
    print(f"  25th percentile: {np.percentile(returns, 25):.4f}")
    print(f"  75th percentile: {np.percentile(returns, 75):.4f}")

    # Print histogram
    print(f"\nHistogram (20 bins):")
    hist, bin_edges = np.histogram(returns, bins=20)
    max_count = hist.max()
    for i in range(len(hist)):
        bar_len = int(50 * hist[i] / max_count) if max_count > 0 else 0
        pct = 100 * hist[i] / len(returns)
        print(f"  [{bin_edges[i]:6.3f} to {bin_edges[i+1]:6.3f}]: {'█' * bar_len} {hist[i]:5d} ({pct:4.1f}%)")

    # Diagnosis
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)

    if returns.std() < 0.15:
        print("❌ CRITICAL: Very low standard deviation!")
        print(f"   Std = {returns.std():.3f} < 0.15")
        print("   The data has insufficient variance for effective learning.")
        print("   Model will likely just predict the mean.")
    elif returns.std() < 0.25:
        print("⚠️  WARNING: Low standard deviation")
        print(f"   Std = {returns.std():.3f} < 0.25")
        print("   The data has some variance but could be better.")
        print("   Model may struggle to learn diverse behaviors.")
    else:
        print("✅ GOOD: Sufficient variance for learning")
        print(f"   Std = {returns.std():.3f} >= 0.25")
        print("   The data has good variance for effective learning.")

    print(f"\n   Expected model prediction (if predicting mean): ~{returns.mean():.3f}")

    # Save visualization
    output_dir = Path("data_analysis")
    output_dir.mkdir(exist_ok=True)

    # Create detailed histogram plot
    plt.figure(figsize=(12, 8))

    # Main histogram
    plt.subplot(2, 1, 1)
    plt.hist(returns, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(returns.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {returns.mean():.3f}')
    plt.axvline(np.median(returns), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(returns):.3f}')
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Value Distribution ({len(returns):,} samples)\nStd: {returns.std():.3f}',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # CDF plot
    plt.subplot(2, 1, 2)
    sorted_values = np.sort(returns)
    cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    plt.plot(sorted_values, cdf, linewidth=2, color='steelblue')
    plt.axhline(0.25, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(0.50, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(0.75, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title('Cumulative Distribution Function (CDF)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / 'value_distribution_full.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved visualization to: {output_path}")

    # Save statistics to text file
    stats_path = output_dir / 'statistics.txt'
    with open(stats_path, 'w') as f:
        f.write("VALUE FUNCTION DATA STATISTICS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Sample size: {len(returns):,}\n\n")
        f.write(f"Mean:     {returns.mean():.4f}\n")
        f.write(f"Std:      {returns.std():.4f}\n")
        f.write(f"Min:      {returns.min():.4f}\n")
        f.write(f"Max:      {returns.max():.4f}\n")
        f.write(f"Median:   {np.median(returns):.4f}\n")
        f.write(f"25th %:   {np.percentile(returns, 25):.4f}\n")
        f.write(f"75th %:   {np.percentile(returns, 75):.4f}\n")
        f.write(f"95th %:   {np.percentile(returns, 95):.4f}\n")

    print(f"📄 Saved statistics to: {stats_path}")

    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print("  - value_distribution_full.png  (visualization)")
    print("  - statistics.txt               (detailed stats)")

    # Final recommendation
    print("\n📋 RECOMMENDATION:")
    if returns.std() >= 0.25:
        print("   ✅ Your data looks good! Proceed with training.")
    elif returns.std() >= 0.15:
        print("   ⚠️  Data variance is acceptable but could be better.")
        print("   Consider further tuning normalization if model performance is poor.")
    else:
        print("   ❌ Data variance is too low for effective learning!")
        print("   You must fix the normalization before training.")
        print("   Try using 50th percentile (median) instead of 75th.")


if __name__ == "__main__":
    main()
