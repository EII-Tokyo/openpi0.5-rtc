# Critic Holdout Report

## 数据概况

- 评估 checkpoint 数量: 1
- holdout transitions: 972
- success transitions: 272
- failure transitions: 700
- skipped shards: 0

## 最佳 checkpoint

- path: `/app/rlt_online/run/candidates/round_000001/critic/inference_actor/00000500`
- step: 500
- AUC: 0.6852
- q_gap: 0.020367
- success_q_mean: -0.112507
- failure_q_mean: -0.132874
- holdout_bellman_loss: 0.008859
- critic 判断: **不可靠**
- warning_reason: `auc<=0.70;failure_actor_advantage>success_actor_advantage`

## Actor advantage

- actor_q_mean: -0.116758
- reference_q_mean: -0.127282
- actor_advantage_mean: 0.010524
- success_actor_advantage_mean: 0.008702
- failure_actor_advantage_mean: 0.011232

## 解释

如果 success Q 均值高于 failure Q, 且 AUC 大于 0.70, 说明 critic 在 holdout 数据上具备基本排序能力。若 failure actor advantage 高于 success, 则说明 critic 可能在错误鼓励失败动作, 本报告会标记为不可靠。

时间曲线使用 replay shard 内的 `transition_index` 和 normalized progress。若原始 replay 没有真实 `episode_id` / `timestep` 字段, 这不是完整 episode 时间, 只是 key region 内部传播检查。

## 下一步建议

优先使用最佳 checkpoint 进行 actor 训练或部署前评估; 如果 critic 被标记为不可靠, 应先增加可区分 success/failure 的数据、检查 reward 标注, 或缩短/重切关键区域后重新训练 critic。
