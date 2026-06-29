# 2026-06-29 Ref-only vs Mixed RLT Critic/Actor 对比报告

## 实验定义

本次比较两种训练方式。

### A. Ref-only critic -> actor

- 数据：`action == reference_action` 的普通 full-cam4 Q replay。
- manifest: `local_rlt_manifests/combined_no_actor_clean109_cam4_plus_expert59_q_cam4_provenance_20260629.jsonl`
- critic：使用之前训练好的 5000 step critic 初始化。
- actor：TD3 方式训练 6000 step。
- 关键设置：`critic_lr=0`，因此 critic 固定不更新；actor 从新初始化开始训练。
- 输出目录：`local_rlt_runs/ref_equal_critic5000_fixed_actor6000_td3_20260629`
- actor checkpoints：
  - `local_rlt_runs/ref_equal_critic5000_fixed_actor6000_td3_20260629/snapshots/inference_actor/00003000`
  - `local_rlt_runs/ref_equal_critic5000_fixed_actor6000_td3_20260629/snapshots/inference_actor/00006000`

### B. Mixed critic + actor

- 数据：`action == reference_action` 168 条 + `action != reference_action` 去重后 377 条。
- manifest: `local_rlt_manifests/mixed_ref168_plus_actor_diff377_20260629.jsonl`
- 总数据：545 shards，success 212，failure 333，全部 z_dim=512。
- 训练：critic burn-in 5000 step，然后 actor 训练 6000 step，总步数 11000。
- 输出目录：`local_rlt_runs/mixed_ref168_actor_diff377_critic5000_actor6000_td3_20260629`
- actor checkpoints：
  - actor 3000 step 对应 `local_rlt_runs/mixed_ref168_actor_diff377_critic5000_actor6000_td3_20260629/snapshots/inference_actor/00008000`
  - actor 6000 step 对应 `local_rlt_runs/mixed_ref168_actor_diff377_critic5000_actor6000_td3_20260629/snapshots/inference_actor/00011000`

## Holdout Critic 对比

| 方法 | checkpoint step | actor step | AUC | q_gap | success_Q | failure_Q | success_adv | failure_adv | usable | warning |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Ref-only fixed critic | 3000 | 3000 | 0.8068 | 0.1110 | -0.1177 | -0.2287 | 0.1066 | 0.0322 | True |  |
| Ref-only fixed critic | 6000 | 6000 | 0.8068 | 0.1110 | -0.1177 | -0.2287 | 0.1067 | 0.0322 | True |  |
| Mixed burn-in critic | 3000 | 0 | 0.7215 | 0.0352 | -0.1666 | -0.2018 | -0.0004 | -0.0013 | True |  |
| Mixed burn-in critic | 5000 | 0 | 0.7096 | 0.0315 | -0.1682 | -0.1997 | 0.0034 | 0.0033 | True |  |
| Mixed TD3 actor | 8000 | 3000 | 0.6858 | 0.1989 | 0.8211 | 0.6222 | 0.2121 | 0.2671 | False | auc<=0.70;failure_actor_advantage>success_actor_advantage |
| Mixed TD3 actor | 11000 | 6000 | 0.6245 | 0.0484 | 0.6490 | 0.6007 | 0.0495 | 0.0558 | False | auc<=0.70;failure_actor_advantage>success_actor_advantage |

## 动作级诊断

| 方法 | checkpoint | actual_Q_AUC | actual_q_gap | actor_Q_AUC | actor_q_gap | success actor advantage | failure actor advantage | actor_delta_norm | actor_chunk_smoothness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ref-only | 6000 | 0.8067 | 0.1110 | 0.7998 | 0.1855 | 0.1067 | 0.0322 | 0.7142 | 0.1589 |
| Mixed | 8000 | 0.6858 | 0.1989 | 0.6673 | 0.1437 | 0.2121 | 0.2671 | 1.0389 | 0.2243 |
| Mixed | 11000 | 0.6245 | 0.0484 | 0.6119 | 0.0421 | 0.0495 | 0.0558 | 0.6821 | 0.1376 |

## 结论

### 1. 当前更好的 actor 是 Ref-only 6000

Ref-only 6000 的优点：

- critic holdout AUC = `0.8068`，明显高于 Mixed 11000 的 `0.6245`。
- success actor advantage = `0.1067`，failure actor advantage = `0.0322`，满足：

$$
A_{success} > A_{failure}
$$

这说明在这个 critic 下，actor 对成功片段的增益高于失败片段，方向是合理的。

### 2. Mixed critic 在 burn-in 早期可用，但 actor 训练后变差

Mixed 的最佳 holdout critic 是 3000 step：AUC = `0.7215`，q_gap = `0.0352`。这说明加入 `action != reference_action` 数据后，critic 不是完全不能学，但它的排序能力低于 Ref-only critic。

更关键的是，Mixed 在 actor 训练后出现退化：

- actor 3000 step：AUC 降到 `0.6858`，并出现 `failure_actor_advantage > success_actor_advantage`。
- actor 6000 step：AUC 降到 `0.6245`，仍有 `failure_actor_advantage > success_actor_advantage`。

这说明直接把 actor-influenced 数据混入 TD3 critic+actor 训练，会让 critic/actor 在失败动作附近也得到较高 advantage，风险较大。

### 3. 为什么 Mixed 变差

`action != reference_action` 数据确实补充了动作敏感性，但这批数据里失败更多：训练 split 中 failure episodes = 267，success episodes = 169。更重要的是，这些 action 来自旧 actor 和不同 runtime 结构，包含抖动、勉强成功、失败探索等混合分布。

TD3 actor loss 会最大化：

$$
Q(s, \pi(s, \tilde a)) - \beta \|\pi(s, \tilde a)-\tilde a\|^2
$$

如果 critic 没有可靠区分“好的 actor 修正”和“坏的 actor 修正”，actor 会利用 critic 的错误高分区域。Mixed 8000/11000 的 failure actor advantage 更高，正是这个问题的表现。

## 建议

1. 当前不要部署 Mixed 8000 或 Mixed 11000 actor。
2. 如果要选一个离线 actor 继续测试，优先使用：
   - `local_rlt_runs/ref_equal_critic5000_fixed_actor6000_td3_20260629/snapshots/inference_actor/00006000`
3. Mixed 数据不应该直接用 TD3 混训 actor。它可以继续用于 critic，但更适合：
   - 只训练 critic；
   - 或者 AWBC，只模仿 `Q(s,a)-Q(s,reference)>0` 且 episode_success 的样本；
   - 或者对 `action != reference_action` 数据加权/过滤，避免失败 actor 动作进入 actor imitation。
4. 下一步如果要继续用这 377 条 diff 数据，建议训练 AWBC 版本，而不是直接 TD3 mixed actor。

## 输出文件

- Ref-only run: `local_rlt_runs/ref_equal_critic5000_fixed_actor6000_td3_20260629`
- Mixed run: `local_rlt_runs/mixed_ref168_actor_diff377_critic5000_actor6000_td3_20260629`
- 本报告：`local_rlt_runs/rlt_actor_ref_vs_mixed_compare_20260629/report_zh.md`
