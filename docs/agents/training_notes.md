# Training Notes

Read this for historical training facts and benchmark context that should not crowd the root `AGENTS.md`.

## 2026-05-06 LoRA Benchmark On Single RTX PRO 6000 Blackwell
- Remote machine:
  - `ssh -p 31483 root@147.185.60.9`
- Launcher/config:
  - Base config: `eii_rinse_cam4_lora`
  - 6 repos:
    - `lyl472324464/2026-05-04_direction-lerobot-with-rinse`
    - `lyl472324464/2026-05-04_direction-twist-water-lerobot-with-rinse`
    - `lyl472324464/2026-05-01_water1-lerobot-with-rinse`
    - `lyl472324464/2026-05-04_turn_over-lerobot-with-rinse`
    - `lyl472324464/2026-05-03_turn_over-lerobot-with-rinse`
    - `lyl472324464/2026-05-01_turn_over-lerobot-with-rinse`
  - 4 cameras from `eii_rinse_cam4_lora`:
    - `cam_high`
    - `cam_left_wrist`
    - `cam_right_wrist`
    - `cam_low`
  - `batch_size=32`
  - `log_interval=10`
  - `num_train_steps=40000`
  - `fsdp_devices=1`
  - `video_memory_num_frames=1`
  - `video_memory_stride_seconds=1.0`
- Verified first batch tensor structure:
  - `base_0_rgb`, `base_1_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` all present with shape `(32, 224, 224, 3)`
  - `tokenized_prompt` log shape `(32, 200)` is text-only; total multimodal token count is about `1024 image + 200 text = 1224`
- Dataloader/step timing on this exact setup:
  - `num_workers=0`:
    - `data_wait_time ~= 1.10s`
    - `train_step_time ~= 3.20s`
    - wall clock step time `~= 4.31s`
  - `num_workers=16`:
    - `data_wait_time ~= 0.03s`
    - `train_step_time ~= 3.20s`
    - wall clock step time `~= 3.23s`
  - Conclusion:
    - On this machine and dataset mix, `num_workers=16` is clearly better than `0`; model time is unchanged and the gain comes from removing dataloader wait.
