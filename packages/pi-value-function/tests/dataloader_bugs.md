# Data Loader Bug Report

Bugs found in `pi_value_function/training/data_loader.py` before training round.

---

## Bug 1: Train/Val Split Not Enforced (Critical)

**Location:** `_build_episode_index` — lines 448-477

**Problem:** `compute_episode_splits` correctly computes disjoint train/val episode lists, but `_build_episode_index` ignores the split entirely. It iterates over `ds.meta.episodes`, which always contains **all** episodes regardless of what was requested.

**Root cause:** LeRobot's `episodes=` parameter only controls what gets *downloaded*. Once data is cached locally, `load_hf_dataset()` loads all parquet files. `meta.episodes` always returns all episode metadata. So both train and val datasets end up indexing every episode.

**Verified with `test_train_val_split.py`:**
```
STEP 4: _build_episode_index (CURRENT — iterates ALL meta.episodes)
  ... (281 total episodes indexed)
  Episodes that are train: 252
  Episodes that are val:   29  ← these should NOT be here
```

**Impact:** Model trains and validates on the same data. Val loss looks artificially good — no way to detect overfitting.

**Fix:** Filter by `ds.episodes` in `_build_episode_index`:
```python
requested_set = set(ds.episodes) if ds.episodes else None
for ep in ep_meta:
    if requested_set is not None and ep["episode_index"] not in requested_set:
        cumulative_idx += length  # still advance — hf_dataset has these frames
        continue
    # ... append EpisodeMetadata as before
```

---

## Bug 2: Redundant Metadata Loading (Medium)

**Location:** `compute_episode_splits` — lines 86-91

**Problem:** `compute_episode_splits` constructs `LeRobotDatasetMetadata` for every repo just to read `total_episodes`. Then `_load_multi_dataset` constructs it again via `MultiLeRobotDataset`. The `LeRobotDatasetMetadata.__init__` calls `get_safe_version` which hits the HuggingFace API — this network round-trip happens twice per repo.

**Fix:** Read `total_episodes` directly from the cached `info.json`:
```python
info_path = LEROBOT_CACHE / repo_id / "meta" / "info.json"
with open(info_path) as f:
    total_episodes = json.load(f)["total_episodes"]
```

---

## Bug 3: Non-deterministic Worker Seeds (Low)

**Location:** `_worker_init_fn` — line 628

**Problem:** Uses Python's `hash()` which is randomized across processes (PYTHONHASHSEED). Training is not reproducible across runs.

**Fix:** Replace with deterministic arithmetic:
```python
worker_seed = (dataset.base_seed * 131 + worker_id) % (2**32)
```

---

## Minor Issues

- **`_parse_image` CHW detection** (line 42): Checks `image.shape[0] == 3` which could false-positive on a 3-pixel-tall HWC image. Unlikely with real camera data.
- **75th percentile normalization** (line 511): Episodes longer than the 75th percentile have early timesteps saturated at `value_min=-1.0` with no gradient signal to differentiate them.
- **`__getitem__` ignores `idx`** (line 552): Combined with `RandomSampler`, this works but `__len__` is misleading.
