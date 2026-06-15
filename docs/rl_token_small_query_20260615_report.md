# RL Token Small Query Training Report

Date: 2026-06-15

## Setup

- Machine: rented GPU `root@136.175.252.90 -p 40127`
- GPU: 1x NVIDIA RTX PRO 6000 Blackwell Server Edition
- Repo: `/workspace/openpi0.5-rtc`
- Commit: `907239d feat: add small query rl token config`
- Config: `eii_rinse_11repo_cam4_fullft_rl_token_small_query`
- Base checkpoint: `eii_rinse_11repo_cam4_fullft/rinse_11repo_insertx5_fullft_bs256_nw64_fsdp8_20260513/9000/params`
- Decoder mode: `query`
- Cameras: `cam_high`, `cam_left_wrist`, `cam_right_wrist`, `cam_low`
- Subtask input: disabled
- Train steps: `10000`
- Save policy: final checkpoint only, saved as step `9999`

## Outputs

- GPU checkpoint:
  `checkpoints/eii_rinse_11repo_cam4_fullft_rl_token_small_query/rinse_11repo_rl_token_small_query_512_from_9000_20260615/9999`
- S3 checkpoint:
  `s3://aloha-checkpoints/rewarding/eii_rinse_11repo_cam4_fullft_rl_token_small_query/rinse_11repo_rl_token_small_query_512_from_9000_20260615/9999`
- S3 eval:
  `s3://aloha-checkpoints/rewarding/eii_rinse_11repo_cam4_fullft_rl_token_small_query/evals/rl_token_small_query_cam4/9999.json`
- Local downloaded checkpoint:
  `checkpoints/eii_rinse_11repo_cam4_fullft_rl_token_small_query/rinse_11repo_rl_token_small_query_512_from_9000_20260615/9999`

## Ablation Result

Evaluation command used `20` batches, batch size `16`, and the same 4-camera config.

| experiment | step | real | shuffled | zero | shuffled/real | zero/real | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| small_query | 9999 | 0.354590 | 0.548633 | 3.727344 | 1.548 | 10.516 | good |
| small_teacher_forced | 9999 | 0.337500 | 0.382192 | 0.459184 | 1.132 | 1.361 | weak z_rl dependence |
| old_query | 10000 | 0.293701 | 0.552734 | 5.460938 | 1.892 | 18.844 | best historical reference |
| old_query_margin | 10000 | 0.301514 | 0.594971 | 4.605469 | 1.977 | 15.438 | strong historical reference |

Raw result:

```json
{
  "real_loss": 0.35458984375,
  "real_vs_shuffled_gap": 0.19404296875,
  "real_vs_zero_gap": 3.371875,
  "shuffled_loss": 0.5486328125,
  "shuffled_over_real": 1.54765625,
  "z_rl_cosine_mean": 0.7939453125,
  "z_rl_cosine_std": 0.0421630859375,
  "zero_loss": 3.72734375,
  "zero_over_real": 10.515625
}
```

## Interpretation

The `query` decoder fixed the main failure mode of the small teacher-forced run. `zero/real` improved from `1.36` to `10.52`, and `shuffled/real` improved from `1.13` to `1.55`, so the decoder can no longer reconstruct well without meaningful `z_rl`.

This run is still weaker than the older query/query-margin references, whose `zero/real` was around `15-19`. The current result is usable as an RL Token checkpoint, but the historical query variants remain stronger by ablation separation.
