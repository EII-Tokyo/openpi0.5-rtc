# DROID LeRobot Conversion

This directory contains multiple DROID-to-LeRobot conversion scripts because the
project has used two DROID LeRobot schemas over time. Prefer the newer
canonical format for new DROID datasets.

## Recommended Path: Canonical DROID

Use `convert_raw_droid_to_canonical_lerobot.py` for new raw DROID datasets:

```bash
uv run python -m examples.droid.convert_raw_droid_to_canonical_lerobot \
  --data-dir /path/to/droid/raw/success \
  --repo-id your-org/your-droid-canonical \
  --annotations-path /path/to/annotations.jsonl \
  --output-root /tmp/your-droid-canonical \
  --overwrite
```

Add `--push-to-hub` when the local dataset has been checked and should be
uploaded.

The canonical schema stores DROID data with explicit observation, action, and
metadata namespaces:

- `observation.images.wrist_left`
- `observation.images.exterior_1_left`
- `observation.images.exterior_2_left`
- `observation.state.joint_position`
- `observation.state.gripper_position`
- `observation.state.cartesian_position`
- `action.joint_velocity`
- `action.joint_position`
- `action.gripper_position`
- `action.source_joint_velocity_gripper`
- `language_instruction`
- `building`, `collector_id`, `datetime`
- `environment.conveyor_speed`
- `subtask_index`

For DROID fine-tuning, the canonical training configs read
`action.source_joint_velocity_gripper`, which is 7 joint velocity dimensions plus
1 gripper position dimension. This matches the DROID action convention used by
the provided DROID checkpoints.

The main canonical scripts are:

- `convert_raw_droid_to_canonical_lerobot.py`: raw DROID `trajectory.h5` plus
  `recordings/MP4` to canonical LeRobot.
- `convert_legacy_lerobot_to_canonical.py`: older LeRobot DROID datasets to
  canonical LeRobot.
- `rebuild_canonical_droid_dataset.py`: wrapper that can combine legacy
  migration and raw conversion into one destination dataset.
- `backfill_canonical_subtasks_from_mongo.py`: refreshes `subtask_index`,
  `meta/subtasks.parquet`, and subtask stats for an existing canonical dataset
  using Mongo slice annotations already linked in `meta/episode_migration.parquet`.
- `canonical_lerobot.py`: shared writer, feature definitions, metadata helpers,
  and validation utilities. It is a library module, not the normal entry point.

Use `CanonicalLeRobotDROIDDataConfig` or
`CanonicalLeRobotDROIDConveyorDataConfig` in `src/openpi/training/config.py` for
training on canonical DROID datasets.

## Older Simple DROID Format

`convert_droid_data_to_lerobot.py` is the older/simple converter referenced by
the original custom DROID fine-tuning guide:

```bash
uv run examples/droid/convert_droid_data_to_lerobot.py --data_dir /path/to/droid/raw/data
```

It expects DROID raw episode directories containing:

- `trajectory.h5`
- `recordings/MP4/*.mp4`
- an `annotations.jsonl` file somewhere under `--data_dir`

It writes a flatter schema:

- `exterior_image_1_left`
- `exterior_image_2_left`
- `wrist_image_left`
- `joint_position`
- `gripper_position`
- `actions`
- `task`

This format still works with `LeRobotDROIDDataConfig`, but it is not preferred
for new DROID datasets. It has less metadata, uses older key names, and is more
likely to need follow-up migration if the dataset is combined with newer DROID
data.

Before using this script, change its hardcoded `REPO_NAME`.

## Special-Purpose Scripts

Most other conversion scripts in this directory are maintenance tools:

- `convert_droid7_to_canonical.py`: one-off migration for an old
  `michios/droid_xxjd_7` dataset stored as PNG bytes in parquet.
- `convert_droid_failures_to_lerobot.py`: downloads DROID failure episodes from
  GCS, filters short/static episodes, and writes the older/simple LeRobot
  schema.
- `rebuild_combined_from_csv.py`: builds a combined LeRobot dataset from
  selected episode IDs in a CSV.
- `swap_exterior_cameras.py`: repairs canonical datasets whose exterior camera
  labels were swapped.
- `run_raw_canonical_conversion_20260202.sh`: local wrapper for one raw
  canonical conversion run with hardcoded paths and MongoDB settings.
- `run_legacy_canonical_backfill.sh`: local wrapper for backfilling old
  datasets into canonical format with hardcoded repo/date settings.

Do not start from these scripts for a new DROID dataset unless the script name
matches a specific repair or backfill task you need.

## Format Choice

Use canonical format when:

- collecting or converting a new DROID dataset;
- combining several DROID datasets;
- preserving task, episode identity, conveyor speed, subtasks, or camera
  extrinsics;
- training with the current canonical DROID LoRA configs.

Use the older simple format only when:

- reproducing the original small custom DROID fine-tuning tutorial as written;
- working with an existing config that already expects the simple schema;
- doing a quick compatibility experiment where metadata is not needed.

For full official DROID-scale training, use the RLDS path described in
`README_train.md`; LeRobot conversion is intended for smaller custom DROID
fine-tuning datasets.
