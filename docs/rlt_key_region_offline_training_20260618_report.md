# RLT Key Region 离线训练研究报告 - 2026-06-18

## 技术结论

本机已经使用你标注完成的 clean key regions 完成一次离线 RLT actor-critic 训练，并用最终导出的 `critic.msgpack` 对成功/失败插入瓶口动作做了 Q 曲线评估。

核心结论：

- 本次训练只使用 SQLite ledger 中 `committed` 状态的 clean key region shard，没有使用目录里残留的 orphan/旧 crop npz。实际加载 `247` 条 shard、`8021` 个 transition，其中成功 `56` 条、失败 `191` 条。
- 训练配置为 `critic_burn_in_steps=5000`、`beta=12`、`num_train_steps=10000`、`batch_size=64`、`train_action_horizon=10`。前 5000 步只训练 critic/Q，5000 步以后才开始训练 actor。
- 在训练样本内，最终 Q 网络可以区分成功和失败插入动作。`q_actual` 全部 transition 的 AUC 为 `0.757`，terminal transition 的 AUC 为 `0.827`。
- Q 曲线显示成功轨迹整体 Q 高于失败轨迹，并且越接近 key region 末段，成功/失败差距越大。最后 25% key-region progress 的 `q_actual` 成功-失败均值差为 `0.213`。
- 这是 in-sample 评估，也就是在训练使用的 committed clean 样本上测试。它说明当前 key-region 标注和 RLT 表征里有可学习信号，但还不能单独证明对新轨迹的泛化能力。

## 产物位置

- 训练目录：`local_rlt_runs/rinse_insert_key_regions_20260618`
- 最终 actor/critic：`local_rlt_runs/rinse_insert_key_regions_20260618/inference_actor/00010000`
- 训练摘要：`local_rlt_runs/rinse_insert_key_regions_20260618/training_summary.json`
- 评估摘要：`local_rlt_runs/rinse_insert_key_regions_20260618/eval_committed/summary.json`
- 每个 transition 的 Q 值：`local_rlt_runs/rinse_insert_key_regions_20260618/eval_committed/per_transition_q.csv`
- 平均 Q 曲线：`local_rlt_runs/rinse_insert_key_regions_20260618/eval_committed/q_actual_mean_curve.png`
- Q 分布图：`local_rlt_runs/rinse_insert_key_regions_20260618/eval_committed/q_actual_hist.png`
- 评估脚本：`scripts/eval_rlt_critic_curves.py`

## 数据口径

你在前端看到约 `248` 条 clean key region，这是因为前端以 segment ledger 为准。磁盘目录中有 `266` 个 clean npz，是因为里面还残留 orphan/旧 crop 版本。为了避免把你已经删除或替换的样本混进训练，本次给离线训练脚本补了 `--segment-db-path` 参数，只加载 `committed` 的 shard。

实际训练数据如下：

| 指标 | 数值 |
|---|---:|
| Replay root | `local_rlt_data/raw_from_103/replay/rlt_key_regions_clean` |
| Segment DB | `local_rlt_data/raw_from_103/state/segments.sqlite3` |
| 加载 shard | 247 |
| Transition | 8021 |
| 成功 episode | 56 |
| 失败 episode | 191 |
| `z_rl` 维度 | 512 |
| `proprio` 维度 | 32 |
| Action horizon | 10 |
| Action dim | 14 |

为什么不是 248 条：ledger 里有 1 条 committed shard 路径在本机缺失，所以最终可训练的 ledger-backed shard 是 `247` 条。

## 训练配置

实际训练命令：

```bash
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=.35 \
.venv/bin/python scripts/train_rlt_offline.py \
  --replay-dir local_rlt_data/raw_from_103/replay/rlt_key_regions_clean \
  --recursive-scan \
  --segment-db-path local_rlt_data/raw_from_103/state/segments.sqlite3 \
  --output-dir local_rlt_runs/rinse_insert_key_regions_20260618 \
  --num-train-steps 10000 \
  --batch-size 64 \
  --critic-burn-in-steps 5000 \
  --beta 12 \
  --log-interval 100 \
  --save-interval 2500 \
  --no-wandb-enabled \
  --overwrite
```

保存的训练 checkpoint：`0`、`2500`、`5000`、`7500`、`10000`。

最终部署用 actor/critic export：`inference_actor/00010000`。

训练健康情况：

- step 5000 之前：`actor_enabled=0`，actor 没有更新，只训练 critic/Q。
- step 5000：同步 target actor/critic，actor 开始训练。
- step 10000 最终指标：
  - `critic_loss ~= 0.042`
  - `actor_delta_norm ~= 0.327`
  - `actor_q_value ~= 0.548`
  - `reference_q_value ~= 0.488`
  - `q_advantage ~= 0.060`
  - `beta = 12`

解释：actor 接入以后 critic loss 从约 `0.02` 上升到约 `0.04-0.09`，随后回落并稳定到 `0.04` 左右；actor 的 Q 优势为正，说明 actor 学到的动作在当前 critic 下比 reference action 得分更高。

## Q 区分能力

这里的 critic 每个 transition/action chunk 输出一个标量 Q。也就是说，Q 曲线是沿 key-region transition progress 的曲线，不是 action chunk 内每个动作维度的单独 Q。

全部 transition 的结果：

| 分数 | AUC | 成功均值 | 失败均值 | 成功-失败差 |
|---|---:|---:|---:|---:|
| `q_actual` | 0.757 | 1.137 | 0.952 | 0.186 |
| `q_reference` | 0.756 | 1.138 | 0.953 | 0.185 |
| `q_actor` | 0.731 | 1.228 | 1.063 | 0.165 |

terminal transition 的结果：

| 分数 | AUC | 成功均值 | 失败均值 | 成功-失败差 |
|---|---:|---:|---:|---:|
| `q_actual` | 0.827 | 0.992 | 0.779 | 0.213 |
| `q_reference` | 0.826 | 0.994 | 0.780 | 0.213 |
| `q_actor` | 0.804 | 1.217 | 1.000 | 0.218 |

按 key-region 进度分段看，`q_actual` 的成功/失败差距越到末段越大：

| Progress 区间 | 成功均值 | 失败均值 | 成功-失败差 |
|---|---:|---:|---:|
| 0-25% | 1.178 | 1.010 | 0.168 |
| 25-50% | 1.173 | 1.000 | 0.173 |
| 50-75% | 1.145 | 0.959 | 0.186 |
| 75-100% | 1.057 | 0.844 | 0.213 |

## 图像解读

`q_actual_mean_curve.png` 显示：

- 成功曲线整体高于失败曲线。
- 成功和失败在前段已有差距，但末段差距更明显。
- 这和任务现象一致：瓶口对准和细管插入的成败，越靠近插入末段越清晰。

`q_actual_hist.png` 显示：

- 成功样本的 Q 分布整体向右偏移。
- 失败样本仍和成功样本有重叠，因此这个 Q 不是完美分类器。
- 但 AUC 已明显高于随机的 `0.5`，说明它确实学到了成功/失败相关信号。

## 对当前问题的回答

“能不能区分成功和失败插入瓶口的动作？”

在当前训练样本内，答案是：可以区分，而且 terminal 附近区分更明显。

最有代表性的数字是：

- `q_actual` 全 transition AUC：`0.757`
- `q_actual` terminal transition AUC：`0.827`
- terminal 成功均值 Q：`0.992`
- terminal 失败均值 Q：`0.779`
- terminal 成功-失败差：`0.213`

这说明：当把录好的 transition 输入 critic 后，它给成功插入相关 transition 的评分显著高于失败 transition，尤其是 key region 的后半段和末端。

## 限制和风险

- 这是训练集内评估，不是 holdout。它验证“能学到信号”，但还没有验证“对新轨迹泛化”。
- 成功/失败样本不平衡：成功 `56`，失败 `191`。AUC 比简单准确率更适合这个场景，但后续最好做 episode-level holdout。
- 当前 Q 是 10-step action chunk 的价值，不是单帧二分类概率。用于在线控制时要看相对变化、`q_advantage` 和 gating 逻辑，不应该直接当作绝对成功概率。
- actor 的 `q_actor` 绝对值高于 `q_actual/reference`，但全 transition AUC 稍低。这说明 actor 已经学会寻找 critic 更喜欢的动作，但离线指标更能证明 critic 的区分能力，不能直接证明机器人上 actor 一定提升成功率。

## 建议下一步

1. 把本次 run 作为第一个可用的本地离线 baseline。
2. 下一轮做 episode-level holdout：按 shard 切分训练集/测试集，例如 80% 训练、20% 测试，再看 held-out Q 曲线和 AUC。
3. 机器人上线前先只启用 critic gate metrics，观察真实在线数据中的 `reference_q`、`actor_q`、`q_advantage` 是否稳定。
4. 如果 held-out 的全 transition AUC 仍能高于约 `0.75`，terminal AUC 仍能高于约 `0.80`，再进入小步 actor 介入测试。
