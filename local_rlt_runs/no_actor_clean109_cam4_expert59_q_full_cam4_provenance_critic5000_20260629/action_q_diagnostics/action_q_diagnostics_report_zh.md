# 2026-06-29 Full Cam4 Critic 动作级诊断

## 评估对象

- critic/actor checkpoint: `local_rlt_runs/no_actor_clean109_cam4_expert59_q_full_cam4_provenance_critic5000_20260629/snapshots/inference_actor/00005000`
- actor step: `5000`
- holdout manifest: `local_rlt_runs/no_actor_clean109_cam4_expert59_q_full_cam4_provenance_critic5000_20260629/holdout_split/holdout_manifest.jsonl`
- holdout episodes: `34`，success `18`，failure `16`
- holdout transitions: `1309`，success `702`，failure `607`
- skipped shards: `0`

## 结论

这个 critic 在 holdout 上对 success/failure 的排序已经明显成立：

- all transitions: actual Q AUC = `0.8067`，q_gap = `0.1110`
- terminal transitions: actual Q AUC = `0.8646`，q_gap = `0.2598`

但是，这个诊断也暴露一个限制：`q_actual` 与 `q_reference` 完全相同。这说明当前 Q replay 中的 `action` 基本等于 `reference_action`，critic 暂时主要被验证为“能评价状态/片段是否成功”，还没有充分验证“同一个状态下，不同动作谁更好”。

## 全部 transition 的 Q

| score | AUC | success mean | failure mean | gap |
|---|---:|---:|---:|---:|
| actual action | 0.8067 | -0.1177 | -0.2287 | 0.1110 |
| VLA reference | 0.8067 | -0.1177 | -0.2287 | 0.1110 |
| checkpoint actor | 0.8037 | -0.1252 | -0.2313 | 0.1061 |

Actor advantage:

$$
A(s)=Q(s,a_{actor})-Q(s,a_{reference})
$$

- success mean advantage: `-0.0075`
- failure mean advantage: `-0.0026`

这里 failure advantage 比 success advantage 更高，因为两者都是负数但 failure 更接近 0。这不是好信号，但本次 actor 没有训练，checkpoint 内 actor 接近初始化输出，所以它更适合作为风险提示，而不是直接否定 critic 的 success/failure 判别能力。

## terminal transition 的 Q

| score | AUC | success mean | failure mean | gap |
|---|---:|---:|---:|---:|
| actual action | 0.8646 | 0.0820 | -0.1777 | 0.2598 |
| VLA reference | 0.8646 | 0.0820 | -0.1777 | 0.2598 |
| checkpoint actor | 0.8681 | 0.0643 | -0.1817 | 0.2460 |

Terminal 上分离更明显，说明 Q 的成功信号确实沿时间向关键区域末端传播了：success terminal Q 明显高于 failure terminal Q。

Terminal actor advantage:

- success mean advantage: `-0.0177`
- failure mean advantage: `-0.0039`

## 重要解释

这次普通版 AUC 变好，主要不是因为 dense/hard-negative，而是因为 no-actor clean 的 109 条数据被重新用正确 cam4 RLToken 统一编码。训练输入从：

$$
Q_\theta(z_{old}, p, a)
$$

变为统一的：

$$
Q_\theta(z_{cam4}, p, a)
$$

其中 `z_cam4` 来自包含 `cam_low` 的正确 RLToken checkpoint。以前 no-actor 和 expert 的 z 表示来源不一致，critic 容易学到 encoder/domain 差异；现在状态表示一致，所以 success/failure 排序明显改善。

## 下一步

1. 不建议使用 dense + hard-negative 版本作为主 critic。
2. 普通 full-cam4 5000 step critic 可以作为当前最佳 critic candidate。
3. 真正进入 actor 训练前，还需要构造“同状态不同动作”的评估集，例如：
   - 当前状态 + 历史成功动作；
   - 当前状态 + 历史失败动作；
   - 当前状态 + actor 输出动作；
   - 当前状态 + reference 动作。
4. 如果 critic 在这些同状态动作比较中仍能给成功动作更高 Q，才说明它适合驱动 actor 学习。

## 输出文件

- `per_transition_q.csv`
- `summary.json`
- `q_actual_mean_curve.png`
- `q_reference_mean_curve.png`
- `q_actor_mean_curve.png`
- `q_actual_hist.png`
- `q_reference_hist.png`
- `q_actor_hist.png`
