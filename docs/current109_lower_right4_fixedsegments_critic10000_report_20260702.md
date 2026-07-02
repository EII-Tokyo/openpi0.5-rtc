# current109+37 lower+right 4-layer fixed-segments critic 10000 报告

## 结论

这次修复了旧数据转换新 `z_rl=2048` 时的核心错位问题，但重新训练 critic 后仍然没有得到可用 critic。因此没有继续训练 actor，也没有接入机器人。

最佳 holdout checkpoint 是 step `5000`，但：

- holdout AUC：`0.4172`
- holdout q_gap：`-0.0310`
- holdout success_q_mean：`-0.2620`
- holdout failure_q_mean：`-0.2310`

`q_gap = success_q_mean - failure_q_mean` 仍为负，表示 critic 给 failure 的 Q 反而高于 success。train 集复核也一样，best train AUC 只有 `0.4624`，train q_gap 为 `-0.0091`。

## 修复内容

之前错误转换的问题是：新 `z_rl` 每一行都变，但 `proprio` 仍保留旧采集时的 chunk 边界，导致 `(z_rl, proprio)` 不是同一个状态语义。

本次新增两类修复工具：

- `scripts/repair_reencoded_rlt_proprio.py`：有 rollout 时，可用 policy input transform 重算 `proprio/next_proprio`。
- `scripts/repair_reencoded_rlt_z_segments.py`：无 rollout 时，按旧 `proprio/next_proprio` 的 chunk 边界对齐新 `z_rl/next_z_rl`，保证 critic 输入中的 token 和 proprio 共享同一时间边界。

最终采用 fixed-segments 数据，因为它覆盖全部 146 个 shard。

## 数据

远端 fixed 数据：

`/home/eii/openpi0.5-rtc-reward-learning/local_rlt_reencoded/current109_37_actor6000_20260630_lower_right_z2048_4layer_fixed_segments`

数据验证结果：

| split | shards | transitions | success | failure | 问题数 |
|---|---:|---:|---:|---:|---:|
| train | 117 | 4253 | 36 | 81 | 0 |
| holdout | 29 | 972 | 7 | 22 | 0 |

验证项包括：`z_rl=2048`、`proprio=32`、`z_rl` 变化边界等于 `proprio` 变化边界、`next_z_rl` 变化边界等于 `next_proprio` 变化边界。

## 训练

critic-only 训练：

- script：`scripts/train_rlt_offline.py`
- stage：`critic_only`
- steps：`10000`
- batch size：`64`
- seed：`0`
- save interval：`1000`
- actor_updated：全程 `0`

远端输出：

`/home/eii/openpi0.5-rtc-reward-learning/local_rlt_runs/current109_37_lower_right4_fixedsegments_critic10000_actor10000_20260702/critic_only_10000`

本机同步：

`local_rlt_runs/current109_37_lower_right4_fixedsegments_critic10000_actor10000_20260702/critic_only_10000`

## Holdout 评估

| step | AUC | q_gap | success_q_mean | failure_q_mean |
|---:|---:|---:|---:|---:|
| 1000 | 0.3082 | -0.0680 | -0.3259 | -0.2579 |
| 2000 | 0.3320 | -0.0538 | -0.3079 | -0.2542 |
| 3000 | 0.3518 | -0.0605 | -0.2864 | -0.2259 |
| 4000 | 0.3732 | -0.0370 | -0.2728 | -0.2358 |
| 5000 | 0.4172 | -0.0310 | -0.2620 | -0.2310 |
| 6000 | 0.3822 | -0.0483 | -0.2926 | -0.2442 |
| 7000 | 0.3210 | -0.0647 | -0.2928 | -0.2281 |
| 8000 | 0.3748 | -0.0581 | -0.2814 | -0.2233 |
| 9000 | 0.4081 | -0.0357 | -0.2767 | -0.2409 |
| 10000 | 0.3944 | -0.0344 | -0.2814 | -0.2470 |

## Train 复核

最佳 train checkpoint 也是 step `5000`：

- train AUC：`0.4624`
- train q_gap：`-0.0091`
- warning：`failure_q_mean>=success_q_mean;auc<=0.60`

这说明问题不是 holdout 切分偶然，而是 current109+37 这批数据在 lower+right 4-layer token 下仍没有训练出正确排序的 critic。

## 决策

不训练 actor，不接入机器人。

原因：actor 会优化当前 critic 的 Q。如果 critic 已经把 failure 评得比 success 高，继续 actor-only 会把 actor 推向错误方向。

下一步应先做数据层排查或换训练配比，而不是继续 actor。
