# Paper-Anchor Bootstrap + Expert Critic 40000-Step Report

## Technical Summary

这次把 critic 从累计 `20000` 继续训练到累计 `40000`。Holdout AUC 从 `0.6453` 提到 `0.6741`，说明排序能力继续变好；但它仍然低于当前评测门槛 `0.70`，所以 **40000 步 critic 仍不应被视为可靠可用 critic**。

更重要的是，训练到更久并没有单调改善所有信号。`q_gap` 在累计 `30000` 达到最高 `0.0620`，到 `40000` 回落到 `0.0561`；holdout Bellman loss 从 `0.00695` 升到 `0.00855`。这说明继续训练确实改变了 critic，但不是简单的“越训越好”。

本轮只继续训练 critic，没有训练 actor。下面的结论只说明离线 holdout replay 上的 critic 行为。

![critic metrics](../local_eval_assets/paper_anchor_bootstrap117_expert59_40000_rewarding_20260703/figures/overview_metrics.png)

## 40000 AUC Is Best, But The Separation Is Not Stable Enough

| cumulative critic step | holdout AUC | q_gap | success_q_mean | failure_q_mean | floor violation | holdout Bellman loss | eval warning |
|---:|---:|---:|---:|---:|---:|---:|---|
| 20000 | 0.6453 | 0.0523 | 0.1031 | 0.0508 | 0.3066 | 0.006950 | `auc<=0.70` |
| 30000 | 0.6723 | 0.0620 | 0.1195 | 0.0575 | 0.3138 | 0.008325 | `auc<=0.70` |
| 40000 | 0.6741 | 0.0561 | 0.1022 | 0.0461 | 0.2994 | 0.008549 | `auc<=0.70` |

按 AUC 看，累计 `40000` 最好。按 success/failure 的平均 Q 垂直间隔看，累计 `30000` 最好。按 Bellman loss 看，累计 `20000` 最好。三个指标没有同时指向同一个 checkpoint，这就是不能只用“继续训到 40000”直接进入 actor 的原因。

`40000` 的改善主要体现在 failure 的平均 Q 被压低到 `0.0461`，但 success 的平均 Q 也从 `30000` 的 `0.1195` 回落到 `0.1022`。这意味着分离度没有继续扩大，只是排序 AUC 略有增加。

## Rewarding Curves Show The Missing Propagation Problem

Rewarding 曲线页面：

`local_eval_assets/paper_anchor_bootstrap117_expert59_40000_rewarding_20260703/index.html`

每条 holdout 轨迹横向展示累计 `20000 / 30000 / 40000` 三个 checkpoint：

- 绿色实线：critic predicted Q
- 灰色虚线：TD target
- 横轴：key-region 内部 normalized progress
- 纵轴：Q / TD target

成功样例里，TD target 在末端能看到奖励尖峰，但 predicted Q 没有稳定形成“后面高、前面逐渐被抬高”的连续传播曲线。失败样例没有末端奖励尖峰，但 predicted Q 仍会出现局部正峰值。因此这批曲线支持一个更谨慎的判断：critic 学到了一些 success/failure 排序信号，但还没有学到足够稳定、可解释的 reward 反向传播结构。

## Scope And Data

训练使用同一份 paper-anchor 语义的 manifest，没有扩大数据量：

- Train manifest: `local_rlt_manifests/paper_anchor_bootstrap_expert_20260703/train_bootstrap117_expert59.jsonl`
- Holdout manifest: `local_rlt_manifests/paper_anchor_bootstrap_expert_20260703/holdout_bootstrap29.jsonl`
- Train data: 117 个 paper-anchor bootstrap shards + 59 个 z-only expert shards
- Train transitions: 7149
- Train episodes: 95 success / 81 failure
- Holdout data: 29 个 bootstrap key-region shards
- Holdout transitions: 972
- Holdout transitions by label: 272 success / 700 failure
- State/action shape: `z_rl=2048`, `proprio=32`, `action_horizon=10`, `action_dim=14`

累计步数映射如下：

- cumulative `20000`: `paper_anchor_bootstrap117_expert59_critic20000_20260703/.../snapshots/inference_actor/00010000`
- cumulative `30000`: `paper_anchor_bootstrap117_expert59_critic40000_20260703/.../snapshots/inference_actor/00010000`
- cumulative `40000`: `paper_anchor_bootstrap117_expert59_critic40000_20260703/.../snapshots/inference_actor/00020000`

注意：continuation run 内部 step 会重新计数，所以本报告和 rewarding 图都使用修正后的 `cumulative_step`，不是原始评测 CSV 里的内部 `step`。

## Methodology

本次训练从累计 `20000` 的 train-state checkpoint 初始化，只继续训练 critic：

- Init critic checkpoint: `local_rlt_runs/paper_anchor_bootstrap117_expert59_critic20000_20260703/critic_only_continue_from_10000/checkpoints/00010000`
- Continue output: `local_rlt_runs/paper_anchor_bootstrap117_expert59_critic40000_20260703/critic_only_continue_from_20000`
- Additional train steps: 20000
- Batch size: 64
- Save interval: 1000
- Actor training: not run

评测目录：

`local_rlt_runs/paper_anchor_bootstrap117_expert59_critic40000_20260703/critic_only_continue_from_20000/holdout_eval_cumulative_20000_30000_40000`

Rewarding 曲线资产目录：

`local_eval_assets/paper_anchor_bootstrap117_expert59_40000_rewarding_20260703`

关键文件：

- `critic_holdout_metrics.json`
- `critic_holdout_transitions.csv`
- `local_eval_assets/.../critic_holdout_transitions_cumulative.csv`
- `local_eval_assets/.../cards.json`
- `local_eval_assets/.../index.html`

## Limitations And Robustness

这个报告只说明 critic 在 holdout replay 上的离线行为，不等价于上机成功率。AUC 是 success/failure transition 的全局排序指标，不直接证明 reward 传播正确。

Rewarding 曲线比 AUC 更接近当前要验证的问题：成功轨迹是否从末端奖励向前传播，失败轨迹是否保持低值。当前曲线还没有达到这个直觉目标，因此即使 `40000` 的 AUC 最高，也不能把它直接当作 actor 训练的可靠评分器。

Holdout 仍只包含 bootstrap key-region，不包含 expert holdout。因此它主要验证 paper-anchor bootstrap 分布上的泛化，不覆盖 expert 分布。

## Recommended Next Steps

1. 不建议直接用累计 `40000` critic 启动 actor 训练。
2. 下一轮选择 checkpoint 时，不要只看 AUC：同时看 `q_gap`、Bellman loss、floor violation 和 rewarding 曲线。
3. 如果必须从这三个里选一个做诊断：按 AUC 选 `40000`；按垂直分离选 `30000`；按 Bellman loss 选 `20000`。三者冲突说明这批 critic 还需要继续诊断。
4. 后续训练应把“成功轨迹 predicted Q 是否形成可解释的后高前低传播曲线”作为主监控之一，而不是只用 AUC 决策。

## Further Questions

- 为什么 `30000` 的 q_gap 最好，但 `40000` 的 AUC 略高？需要看排序改善来自哪些 transition。
- 为什么成功轨迹 TD target 末端尖峰没有稳定传播到 predicted Q 的前半段？
- 当前 paper-anchor bootstrap + expert 的配比是否让 critic 更擅长压低 failure，而不是稳定抬高 success？
