# OpenPI RTC Project Notes

## Checkpoint aliases

- `正式演示版` refers to:
  - Policy config: `eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo`
  - Checkpoint: `/home/eii/openpi0.5-rtc/checkpoints/eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo/no_rinse_cam3_fullft_return_home_29repo_bs256_nw64_fsdp4_20260520/19000`
  - Runtime assets: `assets/trossen/norm_stats.json` inside that checkpoint
# Host 200 OpenPI training contract

Applies to all platform users and to launches performed by Codex or manually.

- Source of truth: host 103, `/home/eii/openpi0.5-rtc`, including current tracked working-tree changes and normalization assets. Never assume matching Git HEAD means matching code.
- All new training on 200 uses `/data/eii-platform/training-code/openpi`. Every start/resume verifies file SHA-256 values against 103; SSH/verification failures block startup. Synchronize changes only when no training is using the shared source. Never edit it underneath an active run.
- Use plain host Python (`/opt/h200-venv/bin/python scripts/train.py CONFIG_NAME`). Do not submit new training through Slurm, Apptainer or Docker. The system service only manages process lifetime and CPU/RAM limits; no wall-time limit.
- Each platform account has separate data under `/data/eii-platform/training-users/ACCOUNT_ID/datasets` and separate outputs under its `runs` directory, with a dedicated `h109_ACCOUNT_ID` Linux user. Do not replace personal data with symlinks to another user's cache. Copy-on-write physical storage is permitted while keeping independent files.
- Write a complete, named native TrainConfig in `src/openpi/training/config.py` BEFORE launch. The training CLI must load that name directly. Never use launch.py, dataclasses.replace overrides, exec/recompiled training source, DataLoader monkeypatches or checkpoint-handler monkeypatches to change its behavior. Platform-generated entries are the sole permitted append-only exception to source equality; preserve the exact 103 baseline above the marker.
- Record the actual source directory, account dataset directory, config name/file/line, code commit, verification time and executable command in the run provenance and expose them in the Runtime rules tab. Retain source file hashes in `/var/lib/h200-unified/native/source.json`.
- Preserve 103's shared-memory PyTorch DataLoader transport and native asynchronous Orbax saving. Do not reintroduce forced `wait_until_finished()` after every periodic checkpoint, forced prefetch_factor=1 or save_concurrent_gb=8. Wait for saving at pause/end boundaries only.
- Pause cooperatively at an optimizer-update boundary, save a complete checkpoint, then release resources. Honor intentional stops. Never overwrite/delete checkpoints to make resume succeed.
- Verify selected GPUs are idle (including non-platform processes), enforce 1–2 H200s, at most 64 CPU per instance/account, 96 GiB per GPU and 200 GiB total per account. Do not start an extra test training on GPUs already in use.
- Existing historical jobs may retain their old runtime until stopped; do not pretend their launch.py overrides are literal config.py values. Do not interrupt an active job just to migrate it.

## Robot deployment

- Always use the root `docker-compose.yml` for robot deployment. Do not create model-specific Compose files or deployment wrappers.
- Select weights with `OPENPI_CHECKPOINT_DIR` (container path under `/app/checkpoints`). Both policy server and runtime use this same value.
- Read camera, resolution, video memory and mosaic settings from the checkpoint training metadata through the generic entrypoints. Preserve training configurations.
- Set `ALOHA_TASK_1_PROMPT` and `ALOHA_TASK_2_PROMPT` to the selected model's training prompts when needed; do not introduce a Compose overlay.
