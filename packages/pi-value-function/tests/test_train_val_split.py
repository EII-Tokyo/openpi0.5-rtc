from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset
from pi_value_function.training.data_loader import compute_episode_splits

import torch

REPO_ID = "michios/droid_xxjd_7"

# ── Step 1: Show what compute_episode_splits produces ──
print("=" * 70)
print("STEP 1: compute_episode_splits")
print("=" * 70)
train_eps, val_eps = compute_episode_splits([REPO_ID], train_ratio=0.9, seed=42)
print(f"Train episodes: {len(train_eps[REPO_ID])} episodes")
print(f"  First 10: {train_eps[REPO_ID][:10]}")
print(f"Val episodes:   {len(val_eps[REPO_ID])} episodes")
print(f"  All: {val_eps[REPO_ID]}")
overlap = set(train_eps[REPO_ID]) & set(val_eps[REPO_ID])
print(f"Overlap between train/val: {len(overlap)} (should be 0)")

# ── Step 2: Load MultiLeRobotDataset with train-only episodes ──
# This mirrors _load_multi_dataset in ValueFunctionDataset.__init__
print("\n" + "=" * 70)
print("STEP 2: Load MultiLeRobotDataset with ONLY train episodes")
print("=" * 70)
mds_train = MultiLeRobotDataset([REPO_ID], episodes=train_eps)
print(f"Number of sub-datasets: {len(mds_train._datasets)}")
for i, ds in enumerate(mds_train._datasets):
    print(f"\n  Sub-dataset {i} ({ds.repo_id}):")
    print(f"    ds.episodes (requested):  {len(ds.episodes)} episodes")
    print(f"    ds.episodes first 10:     {ds.episodes[:10]}")
    print(f"    meta.episodes count:      {len(ds.meta.episodes)}")
    print(f"    hf_dataset length:        {len(ds.hf_dataset)}")
    print(f"    meta.total_episodes:      {ds.meta.total_episodes}")
    print(f"    meta.total_frames:        {ds.meta.total_frames}")

# ── Step 3: Check which episodes are actually accessible per sub-dataset ──
print("\n" + "=" * 70)
print("STEP 3: Which episodes does each sub-dataset's hf_dataset ACTUALLY contain?")
print("=" * 70)
for i, ds in enumerate(mds_train._datasets):
    unique_eps_in_data = ds.hf_dataset.unique("episode_index")
    unique_set = {e.item() if isinstance(e, torch.Tensor) else e for e in unique_eps_in_data}
    print(f"\n  Sub-dataset {i} ({ds.repo_id}):")
    print(f"    Unique episode indices in hf_dataset: {len(unique_set)}")
    val_eps_in_train_data = unique_set & set(val_eps[REPO_ID])
    print(f"    Val episodes present in train hf_dataset: {len(val_eps_in_train_data)}")
    if val_eps_in_train_data:
        print(f"    These val episodes leaked in: {sorted(val_eps_in_train_data)[:20]}...")

# ── Step 4: Simulate _build_episode_index (current buggy version) ──
# This is the exact logic from data_loader.py lines 448-477
print("\n" + "=" * 70)
print("STEP 4: _build_episode_index (CURRENT — iterates ALL meta.episodes)")
print("=" * 70)
train_set = set(train_eps[REPO_ID])
val_set = set(val_eps[REPO_ID])

for ds in mds_train._datasets:
    ep_meta = ds.meta.episodes
    cumulative_idx = 0
    built_episode_ids = []

    for i, ep in enumerate(ep_meta):
        ep_id = ep["episode_index"]
        length = ep["length"]
        built_episode_ids.append(ep_id)
        if i < 5:
            marker = "TRAIN" if ep_id in train_set else "VAL ← LEAKED!"
            print(f"  Episode {ep_id:>3d}: start_idx={cumulative_idx:>6d}, length={length:>4d}  [{marker}]")
        cumulative_idx += length

    print(f"  ... ({len(built_episode_ids)} total episodes indexed)")
    built_train = set(built_episode_ids) & train_set
    built_val = set(built_episode_ids) & val_set
    print(f"\n  Episodes that are train: {len(built_train)}")
    print(f"  Episodes that are val:   {len(built_val)}  ← these should NOT be here")
    print(f"  Total built:             {len(built_episode_ids)}")

# ── Step 5: Show what the FIXED version would do ──
print("\n" + "=" * 70)
print("STEP 5: _build_episode_index (FIXED — filters by ds.episodes)")
print("=" * 70)
for ds in mds_train._datasets:
    requested_set = set(ds.episodes) if ds.episodes else None
    ep_meta = ds.meta.episodes
    cumulative_idx = 0
    fixed_episode_ids = []
    skipped = 0

    for ep in ep_meta:
        ep_id = ep["episode_index"]
        length = ep["length"]
        if requested_set is not None and ep_id not in requested_set:
            skipped += 1
            cumulative_idx += length  # still advance — hf_dataset has these frames
            continue
        fixed_episode_ids.append(ep_id)
        if len(fixed_episode_ids) <= 5:
            print(f"  Episode {ep_id:>3d}: start_idx={cumulative_idx:>6d}, length={length:>4d}  [TRAIN]")
        cumulative_idx += length

    print(f"  ... ({len(fixed_episode_ids)} episodes indexed, {skipped} skipped)")
    fixed_val_leak = set(fixed_episode_ids) & val_set
    print(f"\n  Val episodes leaked: {len(fixed_val_leak)}  ← should be 0")
    print(f"  Total indexed:       {len(fixed_episode_ids)}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
