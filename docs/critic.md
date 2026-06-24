你现在在我的 RLT / OpenPI 项目代码仓库中工作。请按照下面要求实现一个 critic 训练与评估流程。

目标不是只把 critic loss 训练到很低，而是要判断 critic 是否真的能够区分成功动作和失败动作，并判断这个 critic 能不能用于后续 actor 训练。

# 背景

我使用 RLT 方法训练机器人关键阶段任务，例如：

* 瓶口对准冲水口
* 瓶口插入冲水口
* 其他关键阶段精细操作任务

数据中包含 replay buffer transitions、action chunk、VLA reference action、success / failure terminal reward。

当前 reward 可能比较稀疏，主要是：

[
r_{\text{terminal}} = 1
]

或者：

[
r_{\text{terminal}} = 0
]

也就是说，最终成功给 1，最终失败给 0。

# 核心 TD backup 公式

单步 TD backup 版本：

[
Q(s_t,a_t)
\leftarrow
r_t+\gamma Q(s_{t+1},a_{t+1})
]

action chunk 版本：

[
\hat Q_t
========

\sum_{i=0}^{C-1}\gamma^i r_{t+i}
+
\gamma^C Q(s_{t+C},a_{t+C})
]

其中：

* (C)：action chunk 长度
* (\gamma)：discount factor
* (r_{t+i})：chunk 内部每一步 reward
* (Q(s_{t+C},a_{t+C}))：chunk 执行完成后的 bootstrap value

---

# 第一部分：critic 训练

请复用现有 RLT critic / TD3 critic 训练代码，不要重写整个训练系统。

请实现以下功能：

1. 从 replay buffer 中划分 train set 和 holdout validation set，例如：

[
80% \text{ train}
\quad
20% \text{ holdout}
]

2. holdout 数据绝对不能参与训练。

3. 支持固定随机种子，保证划分可复现。

4. 训练过程中每隔固定 step 保存 checkpoint，例如每 500 或 1000 step 保存一次 critic。

5. 每个 checkpoint 都要在 holdout set 上做评估。

6. 训练时仍然记录以下指标：

```text
train_critic_loss
holdout_bellman_loss
target_q_mean
target_q_std
predicted_q_mean
predicted_q_std
```

---

# 第二部分：holdout success / failure 判别评估

请实现一个评估脚本，例如：

```bash
python scripts/evaluate_rlt_critic_holdout.py \
  --checkpoint_dir <critic_checkpoint_dir> \
  --replay_buffer <buffer_path> \
  --output_dir <eval_output_dir>
```

或者，如果项目中已有训练入口，也可以加参数实现：

```bash
--eval_holdout_critic=true
--holdout_ratio=0.2
--eval_every_steps=1000
```

评估时，对 holdout 数据按照 success / failure 分组。

对于每个 critic checkpoint，计算以下指标：

```text
success_q_mean
success_q_std
failure_q_mean
failure_q_std
q_gap
auc
```

其中：

[
q_{\text{gap}}
==============

## \mathbb E[Q_{\text{success}}]

\mathbb E[Q_{\text{failure}}]
]

AUC 的含义是：

> 随机抽一个成功样本和一个失败样本，critic 给成功样本更高 Q 的概率。

如果 sklearn 可用，可以使用：

```python
sklearn.metrics.roc_auc_score
```

如果 sklearn 不可用，请实现一个简单 fallback。

---

# 第三部分：actor action vs VLA reference action advantage

对于同一个 holdout 状态 (s)，比较 actor 动作和 VLA reference 动作：

[
A(s)
====

## Q(s,a_{\text{actor}})

Q(s,\tilde a_{\text{VLA}})
]

其中：

* (a_{\text{actor}})：当前 actor 输出动作，或者 checkpoint actor 输出动作
* (\tilde a_{\text{VLA}})：VLA reference action chunk
* (Q(s,a_{\text{actor}}))：critic 对 actor 动作的评分
* (Q(s,\tilde a_{\text{VLA}}))：critic 对 VLA reference 动作的评分

请计算：

```text
actor_q_mean
reference_q_mean
actor_advantage_mean
actor_advantage_std
success_actor_advantage_mean
failure_actor_advantage_mean
```

重点检查：

[
A_{\text{success}}
\ge
A_{\text{failure}}
]

如果 failure 上 actor advantage 系统性高于 success，要在报告中标记为危险信号，因为这说明 critic 可能在错误地鼓励失败动作。

---

# 第四部分：时间传播合理性评估

如果数据中有：

```text
episode_id
trajectory_id
timestep
```

请按 episode 画出 Q 随时间变化曲线。

对成功轨迹，越接近成功终点，Q 值通常应该更高，或者至少不能乱跳。

对失败轨迹，临近失败终点的 Q 不应该异常升高。

请输出：

```text
每条成功轨迹的 Q(t) 曲线
每条失败轨迹的 Q(t) 曲线
success mean Q(t)
failure mean Q(t)
```

如果没有 episode_id 或 timestep，请在报告中说明无法做时间传播评估，不要静默失败。

---

# 第五部分：图形化输出

请在 output_dir 下生成以下图像。

## 1. Q 分布图

```text
q_distribution_success_failure.png
```

显示 holdout success Q 和 failure Q 的分布。

可以使用 histogram 或 KDE。

要求能清楚看到两者是否分开。

## 2. Q 箱线图

```text
q_boxplot_success_failure.png
```

显示 success / failure 的 Q 值箱线图。

## 3. AUC 随 checkpoint 变化图

```text
auc_over_checkpoints.png
```

横轴：

```text
checkpoint step
```

纵轴：

```text
AUC
```

## 4. Q gap 随 checkpoint 变化图

```text
q_gap_over_checkpoints.png
```

横轴：

```text
checkpoint step
```

纵轴：

[
\mathbb E[Q_{\text{success}}]
-----------------------------

\mathbb E[Q_{\text{failure}}]
]

## 5. actor advantage 图

```text
actor_advantage_success_failure.png
```

比较 success 和 failure 上的 actor advantage：

[
A(s)=Q(s,a_{\text{actor}})-Q(s,\tilde a_{\text{VLA}})
]

## 6. Q 随时间变化图

如果有 trajectory 信息，请输出：

```text
q_over_time_success.png
q_over_time_failure.png
q_over_time_mean_success_failure.png
```

---

# 第六部分：机器可读报告

请输出 JSON 和 CSV 报告，例如：

```text
critic_holdout_metrics.csv
critic_holdout_metrics.json
```

每个 checkpoint 一行，包含以下字段：

```text
step
train_critic_loss
holdout_bellman_loss
success_q_mean
success_q_std
failure_q_mean
failure_q_std
q_gap
auc
actor_q_mean
reference_q_mean
actor_advantage_mean
actor_advantage_std
success_actor_advantage_mean
failure_actor_advantage_mean
is_critic_usable
warning_reason
```

---

# 第七部分：自动判断 critic 是否可用

请根据以下规则自动生成判断。

critic 可用的基本标准：

[
\mathbb E[Q_{\text{success}}]

>

\mathbb E[Q_{\text{failure}}]
]

并且：

[
\text{AUC} > 0.70
]

比较理想的情况是：

[
\text{AUC} > 0.80
]

同时要求：

[
A_{\text{failure}}
]

不能系统性高于：

[
A_{\text{success}}
]

如果出现以下任意情况，请标记 critic 不可靠：

1. holdout 上：

[
\text{failure_q_mean}
\ge
\text{success_q_mean}
]

2. AUC 过低：

[
\text{AUC} \le 0.60
]

3. failure actor advantage 高于 success actor advantage：

[
\text{failure_actor_advantage_mean}

>

\text{success_actor_advantage_mean}
]

4. holdout Bellman loss 下降，但 success / failure Q 分布没有分开。

5. 不同 checkpoint 的排序结果剧烈变化。

---

# 第八部分：自然语言分析报告

请输出：

```text
critic_holdout_report.md
```

报告中必须包含：

1. 本次训练数据量
2. train / holdout 划分数量
3. success / failure 样本数量
4. 最佳 checkpoint
5. 最佳 checkpoint 的 AUC、q_gap、success_q_mean、failure_q_mean
6. critic 是否可用
7. 如果不可用，说明原因
8. 如果可用，说明为什么可以用于 actor training
9. 给出下一步建议

---

# 第九部分：选择最佳 critic checkpoint

不要默认选择最后一个 checkpoint。

请选择 holdout 上综合效果最好的 checkpoint。

选择优先级如下：

1. AUC 最高
2. q_gap 为正且较大
3. holdout Bellman loss 不异常
4. actor advantage 在 success 上合理高于 failure
5. checkpoint 排序相对稳定

请输出：

```text
best_critic_checkpoint.txt
```

内容包含：

```text
best checkpoint path
selection reason
AUC
q_gap
success_q_mean
failure_q_mean
actor_advantage_summary
warning_reason, if any
```

---

# 第十部分：代码质量要求

请遵守以下要求：

1. 尽量复用现有项目结构。
2. 不要破坏现有训练流程。
3. 新增功能尽量放在以下类似位置：

```text
scripts/evaluate_rlt_critic_holdout.py
src/openpi/training/rlt_eval.py
src/openpi/training/rlt_training.py
```

4. 添加必要注释，尤其解释每个指标的意义。
5. 所有图表保存到 output_dir。
6. 如果缺少某些字段，例如：

```text
episode_id
done
success label
reference action
```

请给出清楚报错或 warning。

7. 不要静默返回空图。
8. 尽量保证 JAX / NumPy 数据转换正确，避免 device array 导致 matplotlib 或 sklearn 出错。
9. 如果项目已有 wandb logging，请可选支持把这些 metrics 和图片记录到 wandb。
10. 请优先实现最小可运行版本，然后再做美化和 wandb 集成。

---

# 第十一部分：最终输出说明

完成后请告诉我：

1. 修改了哪些文件
2. 新增了哪些命令
3. 如何训练 critic
4. 如何运行 holdout 评估
5. 如何查看图像结果
6. 如何判断 critic 是否可以用于 actor
7. 给出一个完整示例命令

完整示例命令可以类似：

```bash
python scripts/evaluate_rlt_critic_holdout.py \
  --checkpoint_dir ./checkpoints/rlt_critic \
  --replay_buffer ./data/replay_buffer \
  --output_dir ./outputs/critic_holdout_eval \
  --holdout_ratio 0.2 \
  --seed 42
```

请开始实现。
