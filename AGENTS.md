# OpenPI RTC Project Notes

## Checkpoint aliases

- `正式演示版` refers to:
  - Policy config: `eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo`
  - Checkpoint: `/home/eii/openpi0.5-rtc/checkpoints/eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo/no_rinse_cam3_fullft_return_home_29repo_bs256_nw64_fsdp4_20260520/19000`
  - Runtime assets: `assets/trossen/norm_stats.json` inside that checkpoint

## Robot deployment

- Always use the root `docker-compose.yml` for robot deployment. Do not create model-specific Compose files or deployment wrappers.
- Select weights with `OPENPI_CHECKPOINT_DIR` (container path under `/app/checkpoints`). Both policy server and runtime use this same value.
- Read camera, resolution, video memory and mosaic settings from the checkpoint training metadata through the generic entrypoints. Preserve training configurations.
- Set `ALOHA_TASK_1_PROMPT` and `ALOHA_TASK_2_PROMPT` to the selected model's training prompts when needed; do not introduce a Compose overlay.
