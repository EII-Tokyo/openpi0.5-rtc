# 2026-06-29 Full Cam4 Provenance Q Replay Critic 对比报告

## 结论

这次已经解决了 no-actor clean 数据无法在本机重编码的问题：本机因 RAM/swap 不足无法加载正确 cam4 RLToken，我把最小必要数据复制到 `192.168.1.103` 的正确项目目录 `/home/eii/openpi0.5-rtc-reward-learning` 临时执行，成功用正确 cam4 RLToken 重编码 109 条 no-actor clean，并拉回本机。

最终结果发生了关键变化：

- 普通 Q replay 的 full-cam4 critic 最好，最佳 checkpoint 是 `5000` step。
- 最佳 AUC = `0.8068`，q_gap = `0.1110`。
- Dense + hard-negative 版本没有更好，最佳 AUC = `0.6933`。
- 但是两个 critic 仍都有一个风险：`failure_actor_advantage_mean > success_actor_advantage_mean`。因此可以认为普通版 critic 已经能较好地区分 success/failure，但还不能无条件直接用于 actor 训练；actor 训练前需要处理 advantage 风险，或至少不要用当前随机/未训练 actor 的 advantage 作为部署决策。

## 本次解决的限制

原问题：本机执行正确 cam4 RLToken 重编码时，probe 加载模型被系统 kill，exit code `137`。

定位结果：

- 不是 GPU 显存不足。本机 3090Ti 当时有约 `22GB` 空闲显存。
- 是系统内存压力。本机只有约 `9GB` 可用 RAM，swap 已满。
- 本机 sudo 密码不可用，无法临时扩 swap。

解决方式：

1. 在本机 dry-run 得到 109 条独立 no-actor clean 输入。
2. 只复制最小必要数据到 103 项目目录内：
   - clean shard: 109 个；
   - 对应 rollout: 109 个；
   - 正确 cam4 RLToken checkpoint。
3. 在 103 上使用：
   - `USE_TF=0`
   - `TRANSFORMERS_NO_TF=1`
   - `XLA_PYTHON_CLIENT_PREALLOCATE=false`
4. probe 成功：`z_rl shape=(512,) dtype=float32 finite=True`。
5. full run 成功：`planned=109 converted=109 skipped={}`。
6. 结果拉回本机：
   - `/home/eii/data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions_clean_z512_cam4_provenance_20260629`

## 数据产物

新 no-actor clean cam4 z replay：

- `/home/eii/data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions_clean_z512_cam4_provenance_20260629`
- 数量：`109`
- z dim：`512`
- 全部写入正确 provenance：
  - `rl_token_config_name=eii_rinse_11repo_cam4_fullft_rl_token_small_query`
  - `rl_token_checkpoint_path=checkpoints/eii_rinse_11repo_cam4_fullft_rl_token_small_query/rinse_11repo_rl_token_small_query_512_from_9000_20260615/9999`

新 manifest：

- 普通版：`local_rlt_manifests/combined_no_actor_clean109_cam4_plus_expert59_q_cam4_provenance_20260629.jsonl`
- Dense + hard-negative：`local_rlt_manifests/no_actor_dense_hardneg_q_full_cam4_provenance_20260629.jsonl`

数据量：

| 数据集 | shards | success | failure | provenance |
|---|---:|---:|---:|---|
| 普通 full-cam4 Q replay | 168 | 89 | 79 | 168/168 |
| Dense + hard-negative full-cam4 | 257 | 89 | 168 | 257/257 |

## 训练设置

两套 critic 都使用：

- `num_train_steps=5000`
- `critic_burn_in_steps=5001`
- `batch_size=64`
- `save_interval=1000`
- `holdout_ratio=0.2`
- `holdout_seed=42`
- `wandb disabled`

因为 `critic_burn_in_steps > num_train_steps`，actor 没有更新，日志中 `actor_updated=0`。

## 普通 full-cam4 Q replay critic

训练目录：

- `local_rlt_runs/no_actor_clean109_cam4_expert59_q_full_cam4_provenance_critic5000_20260629`

holdout split 后训练集：

- train shards: `134`
- train transitions: `5578`
- success episodes: `71`
- failure episodes: `63`

最佳 checkpoint：

- `local_rlt_runs/no_actor_clean109_cam4_expert59_q_full_cam4_provenance_critic5000_20260629/snapshots/inference_actor/00005000`

| step | AUC | q_gap | success_q_mean | failure_q_mean | holdout_bellman_loss | success_adv | failure_adv | usable |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1000 | 0.6757 | 0.0359 | -0.2219 | -0.2578 | 0.0177 | -0.0030 | -0.0005 | false |
| 2000 | 0.7411 | 0.0534 | -0.1719 | -0.2254 | 0.0157 | -0.0027 | -0.0005 | false |
| 3000 | 0.6806 | 0.0427 | -0.1884 | -0.2311 | 0.0164 | -0.0028 | -0.0006 | false |
| 4000 | 0.6296 | 0.0340 | -0.1873 | -0.2213 | 0.0165 | -0.0028 | -0.0002 | false |
| 5000 | 0.8068 | 0.1110 | -0.1177 | -0.2287 | 0.0174 | -0.0075 | -0.0026 | false |

分析：

- 5000 step 的 AUC 已经超过 `0.80`，说明 critic 对 success/failure 的排序能力明显改善。
- q_gap 为 `0.1110`，且 success_q_mean 显著高于 failure_q_mean。
- 这个结果说明：之前没有重编码 109 条 no-actor clean 时，critic 评价被旧 z_rl 表示拖累；full cam4 z 后，普通 Q replay 反而变成最好。
- 仍然标记 `usable=false` 的原因是：failure 上的 actor advantage 高于 success。

这里的 warning 要谨慎解释：本次 actor 没有训练，`a_actor` 主要是随机/初始化 actor 输出；所以 `failure_actor_advantage>success_actor_advantage` 更像是在说“当前 critic 对随机 actor 动作的 advantage 还不适合作为 actor training gate”，不等价于 critic 完全不能区分好坏动作。

## Dense + hard-negative full-cam4 critic

训练目录：

- `local_rlt_runs/no_actor_dense_hardneg_q_full_cam4_provenance_critic5000_20260629`

holdout split 后训练集：

- train shards: `206`
- train transitions: `9005`
- success episodes: `68`
- failure episodes: `138`

最佳 checkpoint：

- `local_rlt_runs/no_actor_dense_hardneg_q_full_cam4_provenance_critic5000_20260629/snapshots/inference_actor/00002000`

| step | AUC | q_gap | success_q_mean | failure_q_mean | holdout_bellman_loss | success_adv | failure_adv | usable |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1000 | 0.6797 | 0.0881 | -0.1115 | -0.1997 | 0.1444 | 0.0032 | 0.0156 | false |
| 2000 | 0.6933 | 0.1135 | -0.0456 | -0.1590 | 0.1557 | 0.0021 | 0.0150 | false |
| 3000 | 0.6844 | 0.1286 | -0.0129 | -0.1415 | 0.1807 | -0.0010 | 0.0139 | false |
| 4000 | 0.6758 | 0.0858 | -0.1117 | -0.1975 | 0.1495 | 0.0012 | 0.0103 | false |
| 5000 | 0.6651 | 0.1097 | -0.0375 | -0.1472 | 0.1600 | 0.0003 | 0.0101 | false |

分析：

- Dense + hard-negative 的 q_gap 不小，但 AUC 没有超过 `0.70`。
- 它的 holdout Bellman loss 明显更高，说明 hard-negative/dense target 增加了拟合难度。
- 在 full-cam4 表示下，hard-negative 没有带来更好的 success/failure 排序，反而弱于普通 Q replay。
- 可能原因：当前 hard-negative 是 action mismatch 人造负样本，和真实失败动作分布不完全一致；它让 critic 学到“错配动作很坏”，但没有更好地区分真实成功/失败边界。

## 最终建议

当前最值得继续的 critic 是：

- `local_rlt_runs/no_actor_clean109_cam4_expert59_q_full_cam4_provenance_critic5000_20260629/snapshots/inference_actor/00005000`

它是目前唯一 AUC 超过 `0.80` 的 critic。

但我不建议立刻用它无约束训练 actor。建议下一步：

1. 固定普通 full-cam4 critic 5000 step 作为 critic candidate。
2. 先做 actor-training 前的动作级诊断：
   - 对 VLA reference action；
   - 对历史成功 action；
   - 对历史失败 action；
   - 对当前 actor/random actor action；
   分别计算 Q 分布。
3. 如果历史成功 action 的 Q 明显高于历史失败 action，而随机 actor 的 failure advantage 问题只出现在未训练 actor 上，可以进入保守 actor 训练。
4. Actor 训练建议先用 AWBC/advantage-weighted BC 或受限 TD3，不要直接强力最大化 Q。
5. 暂时不要使用 dense + hard-negative 版本作为主 critic。
