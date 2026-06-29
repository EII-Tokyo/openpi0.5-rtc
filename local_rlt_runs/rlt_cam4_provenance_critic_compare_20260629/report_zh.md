# 2026-06-29 Cam4 Provenance Q Replay Critic 对比报告

## 结论

本次已经删除明确错误的 cam3/2048 中间产物，并重新生成了带 provenance 的 Expert Q replay。随后训练了两套 critic：

1. 普通 Q replay：109 条 no-actor clean + 59 条 expert，不做 dense reward，不做人造 hard-negative。
2. Dense + hard-negative Q replay：在同一组合数据基础上加入 dense terminal-progress reward 和 action-mismatch hard-negative。

当前结果不能直接作为最终 actor 训练依据。主要原因不是训练没跑完，而是 critic 的 holdout success/failure 区分能力仍不够：

- 普通 Q replay 最佳 AUC 只有 `0.5313`，几乎接近随机。
- Dense + hard-negative 最佳 AUC 提升到 `0.6936`，q_gap 明显更大，但仍低于可用阈值 `0.70`，并且 failure 上的 actor advantage 高于 success，这是危险信号。
- 109 条 no-actor clean shard 目前仍来自旧 shard，z_rl 是 512 维，但 shard 内没有写明 RLToken checkpoint provenance。为了完全闭环，必须在内存更充足的环境重新用正确 cam4 RLToken 重编码这 109 条。

## 已删除的污染产物

已从本机数据目录删除：

- `/home/eii/data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions_clean_z2048_INVALID_missing_cam_low_20260629`
- `/home/eii/data/openpi0.5-rtc-reward-learning/replay/human_expert_no_actor_q_20260629`
- `/home/eii/data/openpi0.5-rtc-reward-learning/replay/no_actor_dense_hardneg_q_20260629`

已从 git 中移除旧 manifest：

- `local_rlt_manifests/human_expert_no_actor_q_20260629.jsonl`
- `local_rlt_manifests/combined_no_actor_clean109_plus_expert59_q_20260629.jsonl`
- `local_rlt_manifests/no_actor_dense_hardneg_q_20260629.jsonl`

## 重新生成的数据

Expert z cache 已确认使用正确 cam4 RLToken：

- config: `eii_rinse_11repo_cam4_fullft_rl_token_small_query`
- checkpoint: `checkpoints/eii_rinse_11repo_cam4_fullft_rl_token_small_query/rinse_11repo_rl_token_small_query_512_from_9000_20260615/9999`
- z dim: `512`
- cache 文件数: `58`

新生成 replay / manifest：

- Expert Q replay: `/home/eii/data/openpi0.5-rtc-reward-learning/replay/human_expert_no_actor_q_cam4_provenance_20260629`
- Expert manifest: `local_rlt_manifests/human_expert_no_actor_q_cam4_provenance_20260629.jsonl`
- Combined manifest: `local_rlt_manifests/combined_no_actor_clean109_plus_expert59_q_cam4_provenance_20260629.jsonl`
- Dense + hard-negative replay: `/home/eii/data/openpi0.5-rtc-reward-learning/replay/no_actor_dense_hardneg_q_cam4_provenance_20260629`
- Dense + hard-negative manifest: `local_rlt_manifests/no_actor_dense_hardneg_q_cam4_provenance_20260629.jsonl`

数据量：

| 数据集 | shards | success | failure | 说明 |
|---|---:|---:|---:|---|
| 普通 Q replay combined | 168 | 89 | 79 | 109 no-actor clean + 59 expert |
| Dense + hard-negative | 257 | 89 | 168 | 168 copied/dense + 89 hard-negative |

注意：Expert 59 条及其 hard-negative 都已写入 RLToken checkpoint/config provenance；109 条 no-actor clean 仍需重编码后才能做到同样严格。

## 训练设置

两套 critic 都使用：

- `num_train_steps=5000`
- `critic_burn_in_steps=5001`
- `batch_size=64`
- `save_interval=1000`
- `holdout_ratio=0.2`
- `holdout_seed=42`
- `wandb disabled`

因为 `critic_burn_in_steps` 大于训练总步数，所以 actor 没有更新，日志中 `actor_updated=0`，本次只比较 critic。

## 普通 Q replay critic

训练目录：

- `local_rlt_runs/no_actor_clean109_expert59_q_cam4_provenance_critic5000_20260629`

holdout split：

- total shards: `168`
- train shards: `134`
- holdout shards: `34`
- train transitions: `4936`

最佳 checkpoint：

- `local_rlt_runs/no_actor_clean109_expert59_q_cam4_provenance_critic5000_20260629/snapshots/inference_actor/00001000`

| step | AUC | q_gap | success_q_mean | failure_q_mean | holdout_bellman_loss | success_adv | failure_adv | usable |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1000 | 0.5313 | 0.0129 | -0.1523 | -0.1652 | 0.0191 | 0.0016 | 0.0006 | false |
| 2000 | 0.5075 | 0.0024 | -0.1705 | -0.1728 | 0.0200 | 0.0001 | 0.0009 | false |
| 3000 | 0.3660 | -0.0261 | -0.1303 | -0.1042 | 0.0184 | -0.0005 | -0.0009 | false |
| 4000 | 0.3995 | -0.0255 | -0.2014 | -0.1759 | 0.0222 | -0.0013 | -0.0016 | false |
| 5000 | 0.3718 | -0.0409 | -0.1755 | -0.1346 | 0.0213 | -0.0005 | -0.0002 | false |

分析：

- 1000 step 是普通版里最好的 checkpoint，但 AUC 只有 `0.5313`，不能可靠区分成功和失败。
- 3000 step 以后 q_gap 变成负数，即 failure_q_mean 高于 success_q_mean，这是 critic 排序方向错误。
- Bellman loss 低不代表 critic 好。这里 holdout Bellman loss 在 `0.018-0.022`，但 success/failure 分布没有分开。

## Dense + hard-negative critic

训练目录：

- `local_rlt_runs/no_actor_dense_hardneg_q_cam4_provenance_critic5000_20260629`

holdout split：

- total shards: `257`
- train shards: `206`
- holdout shards: `51`
- train transitions: `8008`

最佳 checkpoint：

- `local_rlt_runs/no_actor_dense_hardneg_q_cam4_provenance_critic5000_20260629/snapshots/inference_actor/00001000`

| step | AUC | q_gap | success_q_mean | failure_q_mean | holdout_bellman_loss | success_adv | failure_adv | usable |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1000 | 0.6936 | 0.0948 | -0.0712 | -0.1660 | 0.1305 | -0.0020 | 0.0077 | false |
| 2000 | 0.6479 | 0.0845 | -0.0624 | -0.1469 | 0.1450 | -0.0036 | 0.0101 | false |
| 3000 | 0.5924 | 0.0544 | -0.0669 | -0.1213 | 0.1336 | -0.0030 | 0.0052 | false |
| 4000 | 0.6189 | 0.0530 | -0.0703 | -0.1233 | 0.1456 | -0.0049 | 0.0108 | false |
| 5000 | 0.6014 | 0.0606 | -0.0990 | -0.1596 | 0.1363 | -0.0067 | 0.0035 | false |

分析：

- Dense + hard-negative 明显优于普通版：最佳 AUC 从 `0.5313` 提升到 `0.6936`，q_gap 从 `0.0129` 提升到 `0.0948`。
- 但是它仍未达到可用标准：AUC 没有超过 `0.70`，更没有达到理想的 `0.80`。
- 最大问题是 actor advantage：

  $$
  A(s)=Q(s,a_{actor})-Q(s,a_{VLA})
  $$

  在最佳 checkpoint 上：

  $$
  A_{success}=-0.0020,\quad A_{failure}=0.0077
  $$

  failure 上的 actor advantage 更高，说明如果直接用这个 critic 训练 actor，critic 可能会鼓励某些失败状态上的 actor 动作。

## 为什么 dense + hard-negative 有帮助但还不够

hard-negative 构造的是：

$$
(s, a_{human}) \rightarrow r=1
$$

以及：

$$
(s, a_{mismatch}) \rightarrow r=0
$$

这迫使 critic 学习“同一个状态下，不同动作有好坏差异”，所以 q_gap 和 AUC 明显改善。

但当前 hard-negative 仍然比较粗糙：它把别的成功片段动作错配到当前状态，能制造明显坏动作，但不一定覆盖真实机器人失败时的坏动作分布，例如：

- 瓶口已经接近水管但推进方向不对；
- 末端横向抖动；
- wrist/forearm roll 突然变化；
- actor 接入后产生的历史非平滑动作。

所以 critic 学到了部分排序，但还没有稳定学到“真实失败动作一定差于真实成功动作”。

## 本次 no-actor 重编码尝试

dry-run 结果：

- clean no-actor 文件数: `186`
- 去重后独立 key region: `109`

probe-only 第一次失败原因：

- `transformers` 导入 TensorFlow；
- 本机 TensorFlow 与 NumPy 2.4 ABI 不兼容。

使用 `TRANSFORMERS_NO_TF=1 USE_TF=0` 后，能进入正确 cam4 RLToken checkpoint 加载，但进程被系统 kill：

- exit code: `137`
- GPU 显存剩余约 `22.7GB`
- 系统 RAM 可用约 `9GB`
- swap 已满

判断：当前本机不适合强行重编码 109 条 no-actor clean。重编码应放到 103 或更大 RAM 环境执行，执行前仍要先 probe。

## 推荐下一步

1. 不要用本次两个 critic 直接训练/部署 actor。
2. 在 103 或大内存机器上，用正确 cam4 small-query RLToken 重编码 109 条 no-actor clean，输出新目录：

   `rlt_key_regions_clean_z512_cam4_provenance_20260629`

3. 用“重编码后的 109 条 no-actor + 59 条 cam4 provenance expert”重新生成普通 Q replay。
4. 再从这套完全 provenance 的普通 Q replay 生成 dense + hard-negative replay。
5. 重新训练两套 critic，再比较：

   - AUC 是否稳定超过 `0.70`；
   - q_gap 是否持续为正；
   - failure actor advantage 是否不再高于 success actor advantage；
   - checkpoint 1000/2000/3000/4000/5000 排序是否稳定。

只有满足这些条件后，才建议进入 actor 训练。

