# current109+37 lower+right 4-layer critic 10000 训练报告

## 结论

本次没有继续训练 actor，也没有把新模型接入机器人容器。

原因是：critic-only 训练到 10000 step 后，固定 holdout 上最好的 checkpoint 仍然不可靠。best AUC 只有 `0.3945`，并且 `q_gap = success_q_mean - failure_q_mean = -0.0458`。这表示 critic 给 failure 的 Q 反而高于 success。根据之前确定的安全约束，这种 critic 不能用于 actor 训练和机器人测试。

## 数据

使用新的 lower+right 4-layer RLToken 重新编码后的 `z_rl=2048` 数据。

- remote 数据根目录：`/home/eii/openpi0.5-rtc-reward-learning/local_rlt_reencoded/current109_37_actor6000_20260630_lower_right_z2048_4layer`
- local 同步目录：`local_rlt_runs/current109_37_lower_right4_critic10000_actor10000_20260702`
- train：117 shards，4253 transitions，success/failure episodes = `36 / 81`
- holdout：29 shards，972 transitions，success/failure episodes = `7 / 22`

train 细分：

| 批次 | success | failure | transitions |
|---|---:|---:|---:|
| manual | 22 | 3 | 744 |
| 2026-06-17 | 3 | 59 | 1863 |
| 2026-06-30 | 11 | 19 | 1646 |

holdout 细分：

| 批次 | success | failure | transitions |
|---|---:|---:|---:|
| manual | 4 | 3 | 255 |
| 2026-06-17 | 0 | 14 | 388 |
| 2026-06-24 | 1 | 0 | 6 |
| 2026-06-30 | 2 | 5 | 323 |

## 训练设置

- script：`scripts/train_rlt_offline.py`
- stage：`critic_only`
- steps：10000
- save interval：1000
- batch size：64
- seed：0
- critic loss：TD3
- actor：未训练，`actor_updated=0`

训练输出：

- remote：`/home/eii/openpi0.5-rtc-reward-learning/local_rlt_runs/current109_37_lower_right4_critic10000_actor10000_20260702/critic_only_10000`
- local：`local_rlt_runs/current109_37_lower_right4_critic10000_actor10000_20260702/critic_only_10000`

## Holdout 评估

固定 holdout manifest：

`/home/eii/openpi0.5-rtc-reward-learning/local_rlt_manifests/online_bootstrap_current109_37_actor6000_20260630_lower_right_z2048_4layer/current109_37_actor6000_holdout_20260630_lower_right_z2048.remote.jsonl`

best checkpoint：

`/app/local_rlt_runs/current109_37_lower_right4_critic10000_actor10000_20260702/critic_only_10000/snapshots/inference_actor/00004000`

| step | AUC | q_gap | success_q_mean | failure_q_mean | Bellman loss |
|---:|---:|---:|---:|---:|---:|
| 1000 | 0.3225 | -0.0596 | -0.2916 | -0.2320 | 0.0134 |
| 2000 | 0.3555 | -0.0532 | -0.2996 | -0.2464 | 0.0139 |
| 3000 | 0.3286 | -0.0725 | -0.2994 | -0.2269 | 0.0145 |
| 4000 | 0.3945 | -0.0458 | -0.2906 | -0.2448 | 0.0151 |
| 5000 | 0.3537 | -0.0593 | -0.2681 | -0.2088 | 0.0141 |
| 6000 | 0.2772 | -0.0736 | -0.3122 | -0.2386 | 0.0144 |
| 7000 | 0.3035 | -0.0676 | -0.3109 | -0.2433 | 0.0146 |
| 8000 | 0.3698 | -0.0579 | -0.2920 | -0.2341 | 0.0160 |
| 9000 | 0.3769 | -0.0520 | -0.3001 | -0.2481 | 0.0150 |
| 10000 | 0.3364 | -0.0725 | -0.3022 | -0.2297 | 0.0156 |

所有 holdout checkpoint 的 q_gap 都是负的，所以没有一个 checkpoint 可以作为 actor 训练基础。

## Train 集复核

为了排除 holdout 切分或路径错误，又把同一批 snapshots 在 train manifest 上评估了一遍。

train best：

- step：1000
- AUC：`0.4661`
- q_gap：`-0.0084`
- warning：`failure_q_mean>=success_q_mean;auc<=0.60;failure_actor_advantage>success_actor_advantage`

train 集也没有学出正确排序，因此问题不是 holdout 偶然切错，而是这套 `lower+right 4-layer z_rl=2048 + current109+37` 数据训练出的 critic 本身不可用。

## 决策

停止 actor 训练。

如果继续 actor-only，actor 会最大化一个已经反向的 Q 函数，等价于鼓励 failure 方向。历史实验已经出现过这种模式：critic 退化后继续训练 actor，会让 actor 的动作评分和实际成功方向相反。

因此本次不执行：

- actor-only 10000 训练
- best actor 选择
- 新 actor 接入机器人容器
- 启动机器人进行测试

## 建议

1. 不把这次 lower+right 4-layer current109+37 critic 用作部署模型。
2. 若要继续验证 lower+right 4-layer，应先做数据/目标核查，而不是继续 actor：
   - 检查 success 样本是否集中在 manual，而 failure 是否集中在 2026-06-17，导致批次偏差；
   - 对比同一 train/holdout split 下旧 `z_rl=512` 的 AUC；
   - 尝试 balanced sampler 或按 batch 分层采样；
   - 尝试把 expert/handout success 只作为 holdout，避免 success 主要来自不同分布。
3. 当前机器人测试应继续使用之前已经人工确认的 actor，不要切到本次新 critic 派生模型。
