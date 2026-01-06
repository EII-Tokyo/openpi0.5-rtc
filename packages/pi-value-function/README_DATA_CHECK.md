# Value Function Data Distribution Check

This guide explains how to verify your value function training data has good variance and coverage.

## Problem We're Solving

Your model was stuck outputting constant values (~-0.2) because the training data had very low variance (std=0.10). This happened due to poor normalization that compressed the value range.

**We fixed this by:**
- Changing from `max episode length + max failure cost` normalization
- To `75th percentile episode length + mean failure cost` normalization
- This increased data variance from std=0.10 → **std=0.31** (3x improvement!)

## Quick Test (Local - Already Done)

You already ran this locally:
```bash
cd /home/eii/Desktop/openpi0.5-rtc
uv run python packages/pi-value-function/quick_data_check_small.py
```

Results: ✅ std=0.31 (good variance)

## Full Data Check (Run on Server)

### Prerequisites

1. SSH into your training server
2. Navigate to the project directory
3. Ensure you have the latest code:
   ```bash
   git pull origin value_func
   ```

### Run the Full Analysis

This checks ALL 7 success datasets + 1 failure dataset (~10,000 samples):

```bash
cd /path/to/openpi0.5-rtc
uv run python packages/pi-value-function/check_full_data_distribution.py
```

**What it does:**
1. Downloads all datasets from HuggingFace (first time only, ~5-10 min)
2. Samples 10,000 data points with the new normalization
3. Computes detailed statistics (mean, std, percentiles)
4. Saves visualization and statistics to `data_analysis/` directory

**Expected output:**
```
============================================================
DATA STATISTICS
============================================================
Sample size:     10,000

Value statistics:
  Mean:          -0.4279
  Std:           0.31XX   ← Should be >= 0.25 ✅
  Min:           -1.0000
  Max:           -0.0026
  Median:        -0.3908

...

✅ GOOD: Sufficient variance for learning
   Std = 0.31XX >= 0.25
```

### Download Results

After the script finishes, download the results from your server:

```bash
# On your local machine
scp your-server:/path/to/openpi0.5-rtc/packages/pi-value-function/data_analysis/* ./
```

You'll get:
- `value_distribution_full.png` - Histogram and CDF visualization
- `statistics.txt` - Detailed numerical statistics

## Interpreting Results

### ✅ Good (Ready to Train)
```
Std >= 0.25
Histogram shows values spread across range
```
→ **Proceed with training!**

### ⚠️ Acceptable
```
0.15 <= Std < 0.25
Some clustering but usable
```
→ **Can train, but monitor model performance**
→ If model still outputs constant values, reduce to 50th percentile (median)

### ❌ Bad (Fix Before Training)
```
Std < 0.15
Values heavily clustered in narrow range
```
→ **DO NOT train yet!**
→ **Fix:** Edit `data_loader.py` line 356:
```python
typical_episode_length = int(np.percentile(all_lengths, 50))  # Use median
```
