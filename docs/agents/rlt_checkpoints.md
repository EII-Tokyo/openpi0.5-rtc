# RLT Checkpoints And Runtime Constraints

Read this before training critic/actor, re-encoding `z_rl`, starting `openpi_server` / `rlt_warmup_runtime`, selecting VLA/RLToken checkpoints, or running robot actor tests.

## Correct Full-Camera VLA
- Strong constraint: for rinse / bottle-mouth insertion work that needs `cam_low`, do not use the `cam3` VLA or any RLToken checkpoint derived from it. The `cam3` checkpoint does not include `cam_low`, so it cannot be treated as a full camera checkpoint for judging bottle-mouth and pipe alignment.
- Correct full-camera VLA checkpoint with `cam_low`:
  - Config/checkpoint family: `eii_rinse_11repo_cam4_fullft`
  - Host path on `192.168.1.103`: `/home/eii/openpi0.5-rtc-reward-learning/checkpoints/eii_rinse_11repo_cam4_fullft/rinse_11repo_insertx5_fullft_bs256_nw64_fsdp8_20260513/9000`
  - Local copied path when present: `/home/eii/project/openpi0.5-rtc-reward-learning/checkpoints/eii_rinse_11repo_cam4_fullft/rinse_11repo_insertx5_fullft_bs256_nw64_fsdp8_20260513/9000`
  - Container path: `/app/checkpoints/eii_rinse_11repo_cam4_fullft/rinse_11repo_insertx5_fullft_bs256_nw64_fsdp8_20260513/9000`
  - Cameras: `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist`.

## Historical Cam4 Small-Query RLToken
- Historical cam4 RLToken checkpoint derived from the cam4 VLA above, but no longer the default for new data collection:
  - Config: `eii_rinse_11repo_cam4_fullft_rl_token_small_query`
  - Checkpoint: `rinse_11repo_rl_token_small_query_512_from_9000_20260615/9999`
  - Host path on `192.168.1.103`: `/home/eii/openpi0.5-rtc-reward-learning/checkpoints/eii_rinse_11repo_cam4_fullft_rl_token_small_query/rinse_11repo_rl_token_small_query_512_from_9000_20260615/9999`
  - Container path: `/app/checkpoints/eii_rinse_11repo_cam4_fullft_rl_token_small_query/rinse_11repo_rl_token_small_query_512_from_9000_20260615/9999`
  - This is the correct 2026-06-15 cam4 small-query RLToken checkpoint for historical cam4 work; it was initialized from `eii_rinse_11repo_cam4_fullft/.../9000/params`.

## Active Lower+Right 4-Layer RLToken
- Active RLToken checkpoint for new RLT data collection, replay re-encoding, critic training, and actor training:
  - Config: `eii_rinse_11repo_cam4_fullft_rl_token_lower_right_query_4layer`
  - Checkpoint family: `rlt_lower_right_rl_token_ablation_20260701`
  - Host path on `192.168.1.103`: `/home/eii/openpi0.5-rtc-reward-learning/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint`
  - Local path when present: `/home/eii/project/openpi0.5-rtc-reward-learning/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint`
  - Container path: `/app/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint`
  - Cameras used for visual information: `cam_low`, `cam_right_wrist`.
  - Output z dimension: `2048`.
- Historical training fact: this 4-layer lower+right RLToken was trained as an offline `rl_token_only=True` autoencoder on the 11-repo LeRobot rinse dataset, initialized from the frozen cam4 VLA checkpoint `eii_rinse_11repo_cam4_fullft/.../9000/params`. It was not trained from robot runtime `/rlt_policy_forward_events` and must not be described as "trained from same-forward runtime tokens".
- Evidence: `scripts/vast_train_lower_right_rl_token.sh` ran `uv run scripts/train.py eii_rinse_11repo_cam4_fullft_rl_token_lower_right_query_4layer`; the config uses `_make_rl_token_autoencoder_config`, `decoder_mode="query"`, `camera_keys=("cam_low", "cam_right_wrist")`, `output_camera_slots=("base_0_rgb", "base_1_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")`; model training uses `Pi0.compute_loss(... rl_token_only=True)` and `embed_prefix_hidden(observation, drop_language=True)`.
- Runtime same-forward is a later inference/data-collection mechanism that applies this trained RLToken encoder to hidden states from the running VLA forward pass. Do not conflate the RLToken checkpoint's training source with the actor/critic replay `z_rl` source.
- Strong constraint: new key-region data collection on `192.168.1.103` must set `RLT_RL_TOKEN_CHECKPOINT_PATH=/app/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint` before starting `openpi_server` or `rlt_warmup_runtime`.
- Strong constraint: do not collect new RLT replay with the old 512-dim small-query RLToken unless the user explicitly requests a controlled ablation. Mixing 512-dim and 2048-dim `z_rl` replay in one critic/actor training run is invalid.
- If existing 512-dim replay must be reused, re-encode it into a separate lower-right directory such as `/data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions_lower_right_z2048_4layer` or `/data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions_clean_lower_right_z2048_4layer`; never overwrite the original replay shards.

## Wrong Cam3-Derived Checkpoint
- Wrong checkpoint for rinse / bottle-mouth insertion if `cam_low` is required:
  - VLA family: `eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo`
  - RLToken config: `eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo_rl_token_query`
  - RLToken checkpoint: `rl_token_2048_enc4_dec4_query_from_19000_20260528/12000`
  - Container path: `/app/checkpoints/eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo_rl_token_query/rl_token_2048_enc4_dec4_query_from_19000_20260528/12000`
  - This checkpoint was derived from the cam3 VLA `no_rinse_cam3_fullft_return_home_29repo_bs256_nw64_fsdp4_20260520/19000`; do not use it to train or evaluate critic/actor for tasks that depend on `cam_low`.

## Audit Note
- `2026-05-28`: the cam3-derived RLToken configs were added.
- `2026-05-29` through `2026-06-15 09:50 +0900`: RLT defaults used the cam3-derived RLToken path.
- `2026-06-15 09:50 +0900`: defaults moved to the cam4 VLA base checkpoint.
- `2026-06-15 16:34 +0900`: defaults moved to the correct cam4 RLToken small query checkpoint.
- `2026-07-02`: defaults moved to the lower+right 4-layer RLToken checkpoint for new RLT data collection and training.

## Runtime Constraints
- Before training critic/actor, re-encoding `z_rl`, or starting `openpi_server`/`rlt_warmup_runtime`, verify the active `--policy.config`, `--policy.dir`, `--model-dir`, and `RLT_RL_TOKEN_CHECKPOINT_PATH` are from the active lower+right 4-layer RLToken family unless the user explicitly requests a controlled ablation.
- Strong runtime constraint for robot actor tests: the required online `z_rl` path is the B-group VLA same-forward method. Main `openpi_server` must keep the cam4 VLA policy (`eii_rinse_11repo_cam4_fullft`) for action inference and enable `RLT_SAME_FORWARD_RL_TOKEN_ENABLED=1` so `z_rl` is encoded from the same VLA forward pass using the lower+right autoencoder.
- Hard ban for normal robot actor tests: do not enable the sidecar RLToken path (`rlt_token_server`, `--rlt-token-port 8002`, or any fallback that calls `infer_rl_token()` after `policy.infer()` misses `z_rl`). If actor startup or intervention cannot get `z_rl` from the VLA same-forward path, treat it as a configuration/runtime error and stop to investigate; do not silently fall back to sidecar re-encoding.
