# Paper-Anchor Bootstrap + Expert Critic 20000-Step Report

## Technical Summary

继续训练到累计 20000 步后，holdout AUC 从累计 10000 步的 `0.6223` 提升到 `0.6453`，success/failure 的平均 Q 间隔也从 `0.0494` 提升到 `0.0523`。这说明继续训练有正向信号，但还没有达到这套评测里 `AUC > 0.70` 的可用门槛。

结论是：**累计 20000 是这三个累计步数里 AUC 最好的 critic，但还不建议直接用它作为 actor 训练/上机筛选依据。** 主要风险是 floor violation rate 在累计 20000 步升到 `0.3066`，明显高于累计 15000 步的 `0.1101`，说明更长训练提高了排序能力，但 calibration/floor 约束压力变差。

本次没有训练 actor。这里评估的是 critic-only 继续训练结果。

![critic metrics](figures/paper_anchor_bootstrap117_expert59_critic20000_metrics.png)

## Key Findings

### Cumulative 20000 improves ranking, but still fails the usable-critic gate

| cumulative critic step | holdout AUC | q_gap | success_q_mean | failure_q_mean | floor violation | holdout Bellman loss | eval warning |
|---:|---:|---:|---:|---:|---:|---:|---|
| 10000 | 0.6223 | 0.0494 | 0.1366 | 0.0872 | 0.1975 | 0.006770 | `auc<=0.70` |
| 15000 | 0.6162 | 0.0490 | 0.1282 | 0.0792 | 0.1101 | 0.006788 | `auc<=0.70` |
| 20000 | 0.6453 | 0.0523 | 0.1031 | 0.0508 | 0.3066 | 0.006950 | `auc<=0.70` |

按 **AUC** 看，累计 `20000` 最好。按评测脚本内置的 critic 选择规则看，最佳 critic 被选为累计 `15000`，因为脚本优先看 `q_propagation_score`，然后才看 q_gap、Bellman loss、actor advantage ordering、AUC。两种口径都没有把 critic 标成可靠：`is_critic_usable=false`。

### The longer run lowers absolute Q while slightly widening the success/failure gap

从累计 10000 到 20000，success 和 failure 的平均 Q 都下降了：

| comparison | success_q_mean | failure_q_mean | q_gap |
|---|---:|---:|---:|
| 10000 | 0.1366 | 0.0872 | 0.0494 |
| 20000 | 0.1031 | 0.0508 | 0.0523 |

这不是单纯“越训越高”。更准确地说，critic 在 20000 步时把两类样本整体往低值压，但 failure 被压得更多，所以垂直分离度略微变大。这个方向对排序有帮助，但因为 floor violation 变高，说明它可能也在破坏 reference floor 的校准约束。

### Floor violation is the main reason not to proceed directly

累计 15000 步 floor violation rate 最低，为 `0.1101`；累计 20000 步升到 `0.3066`。这意味着继续训练到 20000 虽然让 AUC 更高，但也让更多 transition 违反 reference floor。对于 Cal-QL/RLT 这种依赖 reference policy 校准的设置，这个副作用不能忽略。

## Scope And Data

训练数据使用同一份 paper-anchor 语义的 manifest：

- Train manifest: `local_rlt_manifests/paper_anchor_bootstrap_expert_20260703/train_bootstrap117_expert59.jsonl`
- Holdout manifest: `local_rlt_manifests/paper_anchor_bootstrap_expert_20260703/holdout_bootstrap29.jsonl`
- Train data: 117 个 paper-anchor bootstrap shards + 59 个 z-only expert shards
- Train transitions: 7149
- Train episodes: 95 success / 81 failure
- Holdout data: 29 个 bootstrap key-region shards
- Holdout transitions: 972
- Holdout transitions by label: 272 success / 700 failure
- State/action shape: `z_rl=2048`, `proprio=32`, `action_horizon=10`, `action_dim=14`

累计 10000 步来自原始 critic-only run；累计 15000/20000 来自从 10000 checkpoint 继续训练的 run。继续训练 run 内部 step 重新从 0 计数，因此：

- original `00010000` = cumulative 10000
- continuation `00005000` = cumulative 15000
- continuation `00010000` = cumulative 20000

## Methodology

继续训练命令从原始 10000 步 critic checkpoint 初始化，只训练 critic，不更新 actor：

- Init critic checkpoint: `local_rlt_runs/paper_anchor_bootstrap117_expert59_critic10000_20260703/critic_only_10000/checkpoints/00010000`
- Continue output: `local_rlt_runs/paper_anchor_bootstrap117_expert59_critic20000_20260703/critic_only_continue_from_10000`
- Training stage: `critic_only`
- Additional train steps: 10000
- Batch size: 64
- Save interval: 1000
- Actor training: not run. The run is `training_stage=critic_only`, and the exported actor path remained `inference_actor/00000000`.

评测目录：

`local_rlt_runs/paper_anchor_bootstrap117_expert59_critic20000_20260703/critic_only_continue_from_10000/holdout_eval_cumulative_10000_15000_20000`

关键评测文件：

- `critic_holdout_metrics.json`
- `critic_holdout_metrics.csv`
- `critic_holdout_transitions.csv`
- `critic_holdout_report.md`

## Limitations And Robustness

本报告只说明 critic 在 holdout replay 上的离线评分行为，不等价于上机成功率。

AUC 是 success/failure transition 的排序指标，不能单独证明 critic 学到了正确的 reward 传播。这里必须同时看 q_gap、Q 曲线、floor violation、Bellman loss 和 actor advantage。当前结果里 AUC 有提升，但 floor violation 同时恶化，所以证据不足以支持直接进入 actor 训练。

Holdout 只包含 bootstrap key-region，不包含 expert holdout。因此它主要检验从旧 actor 数据重建后的 paper-anchor key-region 是否可分，而不是检验 expert 分布上的泛化。

## Recommended Next Steps

1. 先不要基于这个 critic 直接启动 actor 训练。
2. 继续查看累计 10000/15000/20000 的单轨迹 Q 曲线，确认 20000 的 AUC 提升是不是来自合理的成功轨迹后段传播，而不是对失败轨迹或 floor 的错误压制。
3. 如果要选一个 checkpoint 做下一轮诊断：按 AUC 选累计 20000；按当前脚本的综合 critic selection 选累计 15000。二者都只能用于诊断，不应视为可靠可部署 critic。
4. 下一次训练应加入更直接的 propagation 可视化评估，把“成功轨迹是否从后向前形成连续 Q 曲线”和“失败轨迹是否保持低值”作为主监控，不再只看 AUC。

## Further Questions

- 累计 20000 的 floor violation 为什么升高：是 Cal-QL floor 权重不足，还是 bootstrap/expert 混合后 reference action 分布不一致？
- 当前 holdout 是否太偏向 bootstrap failure，导致 AUC 对 actor 后续训练质量的解释力有限？
- actor 训练前是否需要重新定义 best critic selection，把用户关心的轨迹传播曲线指标放在 AUC 之前？
