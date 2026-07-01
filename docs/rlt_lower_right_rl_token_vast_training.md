# Lower + Right Wrist RLToken Vast Training Plan

## Goal

Train two new RLToken checkpoints that only use the two camera views that can see bottle-mouth / nozzle alignment clearly:

- `cam_low`
- `cam_right_wrist`

Then compare them against the existing full cam4 RLToken checkpoint:

| Name | Effective cameras | Structure | Decoder | Purpose |
|---|---|---|---|---|
| existing cam4 small query | `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist` | 2-layer, `z_dim=512` | `query` | current baseline |
| lower_right_small_query | `cam_low`, `cam_right_wrist` | 2-layer, `z_dim=512` | `query` | camera ablation with same capacity |
| lower_right_query_4layer | `cam_low`, `cam_right_wrist` | 4-layer, `z_dim=2048` | `query` | camera ablation with larger capacity |

Do not train a teacher-forced RLToken for this run. Previous ablation reports showed the teacher-forced shortcut made `z_rl` weak:

- `small_teacher_forced`: `zero/real = 1.361`
- `small_query`: `zero/real = 10.516`
- historical `query`: `zero/real ~= 18-19`

The healthy target remains:

```text
real_loss < shuffled_loss << zero_loss
```

This means the decoder reconstructs well with the correct `z_rl`, worse with shuffled `z_rl`, and much worse with zero `z_rl`.

## Dataset

Use the same repo weighting as the existing cam4 VLA/RLToken training.

The unique 11 Hugging Face repos are:

```text
lyl472324464/2026-05-01_turn_over-lerobot-with-rinse
lyl472324464/2026-05-01_water1-lerobot-with-rinse
lyl472324464/2026-05-03_turn_over-lerobot-with-rinse
lyl472324464/2026-05-04_direction-twist-water-lerobot-with-rinse
lyl472324464/2026-05-04_turn_over-lerobot-with-rinse
lyl472324464/2026-05-04_direction-lerobot-with-rinse
lyl472324464/2026-05-05_direction-water-lerobot-with-rinse
lyl472324464/2026-05-05_water-lerobot-with-rinse
lyl472324464/2026-05-07_water-lerobot-with-rinse
lyl472324464/2026-05-12_insert-to-nozzle_realign-lerobot-with-rinse
lyl472324464/2026-05-13-insert-to-nozzle-no-cap-with-rinse
```

The actual training config uses `insert x5` weighting:

```text
ordinary repos: weight 1
insert-to-nozzle_realign: weight 5
insert-to-nozzle-no-cap: weight 5
```

This preserves comparability with the existing cam4 RLToken and keeps nozzle insertion overrepresented.

## Base Checkpoint

Download this VLA checkpoint on the vast.ai machine:

```text
s3://openpi-tokyo/checkpoints/eii_rinse_11repo_cam4_fullft/rinse_11repo_insertx5_fullft_bs256_nw64_fsdp8_20260513/9000/
```

Expected local path:

```text
/workspace/openpi0.5-rtc-reward-learning/checkpoints/eii_rinse_11repo_cam4_fullft/rinse_11repo_insertx5_fullft_bs256_nw64_fsdp8_20260513/9000
```

The configs load:

```text
/workspace/openpi0.5-rtc-reward-learning/checkpoints/eii_rinse_11repo_cam4_fullft/rinse_11repo_insertx5_fullft_bs256_nw64_fsdp8_20260513/9000/params
```

## New Configs

### 2-layer lower/right

```text
eii_rinse_11repo_cam4_fullft_rl_token_lower_right_small_query
```

Main settings:

```text
effective cameras: cam_low, cam_right_wrist
output slots: base_0_rgb, base_1_rgb, left_wrist_0_rgb, right_wrist_0_rgb
masked slots: base_0_rgb, left_wrist_0_rgb
decoder_mode: query
z_dim: 512
encoder_layers: 2
decoder_layers: 2
train steps: 10000
save interval: 5000
```

### 4-layer lower/right

```text
eii_rinse_11repo_cam4_fullft_rl_token_lower_right_query_4layer
```

Main settings:

```text
effective cameras: cam_low, cam_right_wrist
output slots: base_0_rgb, base_1_rgb, left_wrist_0_rgb, right_wrist_0_rgb
masked slots: base_0_rgb, left_wrist_0_rgb
decoder_mode: query
z_dim: 2048
encoder_layers: 4
decoder_layers: 4
train steps: 10000
save interval: 5000
```

The masked slot design keeps the cam4 checkpoint-compatible input layout while preventing `cam_high` and `cam_left_wrist` from contributing valid prefix tokens.

## Vast Machine Setup

Do not put tokens in git. Set them only as shell environment variables on the rented machine:

```bash
export HF_TOKEN='<set on vast only>'
export WANDB_API_KEY='<set on vast only>'
```

AWS credentials should be copied or exported from the local machine into the vast session. One safe pattern is:

```bash
aws configure export-credentials --format env
```

Then paste the exported `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optional `AWS_SESSION_TOKEN` into the vast shell. Do not commit those values.

## Run Commands

Clone this repo on vast as:

```bash
cd /workspace
git clone <private github repo url> openpi0.5-rtc-reward-learning
cd /workspace/openpi0.5-rtc-reward-learning
git checkout paper_actor_sample
```

Run both lower/right trainings:

```bash
bash scripts/vast_train_lower_right_rl_token.sh
```

Or run manually:

```bash
aws s3 sync \
  s3://openpi-tokyo/checkpoints/eii_rinse_11repo_cam4_fullft/rinse_11repo_insertx5_fullft_bs256_nw64_fsdp8_20260513/9000/ \
  /workspace/openpi0.5-rtc-reward-learning/checkpoints/eii_rinse_11repo_cam4_fullft/rinse_11repo_insertx5_fullft_bs256_nw64_fsdp8_20260513/9000/

uv sync --frozen

uv run scripts/train.py eii_rinse_11repo_cam4_fullft_rl_token_lower_right_small_query
uv run scripts/train.py eii_rinse_11repo_cam4_fullft_rl_token_lower_right_query_4layer
```

## Evaluation

Evaluate checkpoints at step `4999` or `5000` depending on saved directory naming, and final step `9999`.

Example:

```bash
uv run scripts/eval_rl_token.py \
  --config-name eii_rinse_11repo_cam4_fullft_rl_token_lower_right_small_query \
  --checkpoint-dir /workspace/openpi0.5-rtc-reward-learning/checkpoints/eii_rinse_11repo_cam4_fullft_rl_token_lower_right_small_query/rinse_11repo_rl_token_lower_right_small_query_512_from_9000_20260701/9999 \
  --num-batches 20 \
  --batch-size 16
```

The comparison report should include:

```text
real_loss
shuffled_loss
zero_loss
shuffled_over_real
zero_over_real
z_rl_cosine_mean
z_rl_cosine_std
```

Use the checkpoint for downstream RLT only if:

```text
real_loss < shuffled_loss
zero_loss >> shuffled_loss
zero_over_real is clearly larger than the current cam4 small query baseline
```

If lower/right has lower real loss but poor `zero_over_real`, it may reconstruct from shortcuts and should not replace the current RLToken.
