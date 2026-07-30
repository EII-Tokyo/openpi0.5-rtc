# 现场部署基础模型 Checkpoint 只读审计

## 选择依据

实际停止容器的机器人运行端和模型服务端均指向同一目录：

`/data/openpi0.5-rtc/checkpoints/eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo/no_rinse_cam3_fullft_return_home_29repo_bs256_nw64_fsdp4_20260520/19000`

因此本审计选择该目录，而不是根据当前 compose 默认值、README 或最近修改时间
猜测。审计只读取小型 metadata 与归一化 JSON，没有加载或修改 12 GB 参数数组。

## 可确认信息

| 项目 | 审计结果 |
|---|---|
| 格式 | Orbax OCDBT，纯参数 Checkpoint |
| 目录 step | 19,000 |
| 目录大小 | 12,440,702,849 字节 |
| 参数叶节点 | 51 |
| 参数总量 | 838,358,468 |
| optimizer state | 不存在 |
| EMA 独立参数 | 未发现 |
| normalization | state/action 均为 14 维，含 mean/std/q01/q99 |
| metadata metrics | 空 |
| 完整数值数组加载 | 未执行；只审计 metadata |

参数总量由 metadata 中每个 `write_shape` 的维度乘积求和得到，并在远端使用
独立脚本再次计算，两个结果均为 838,358,468。

## 模型配置交叉核验

W&B 原始训练配置给出：

- 输入图像分辨率 224×224；
- 动作 horizon 50；
- 内部 action dimension 32，输出变换取前 14 维控制双臂与夹爪；
- 视觉语言主干为 2B 级，动作专家为 300M 级；
- bfloat16；
- batch size 256；
- 四张 H200 上分布式全参数微调；
- seed 42；
- 学习率 warmup 10,000 步，峰值 2.5e-5，40,000 步余弦衰减至 2.5e-6；
- Adam 系数 0.9/0.95，梯度范数裁剪为 1；
- EMA decay 0.99。

W&B 运行从 2026-05-20 08:05:45 UTC 到 2026-05-22 11:29:28 UTC，
记录到第 59,990 步，共 6,059 行历史。该运行存在训练 loss，但没有 validation、
test 或机器人成功率指标。

## 必须保留的冲突

1. 现场部署只保留第 19,000 步目录；W&B 运行后来继续到约 60,000 步。
2. W&B 初始配置为 29 个采样入口；当前代码为后续 51 入口配置。
3. W&B 配置仍显示初始 `num_train_steps=40000`，但历史连续到 59,990；
   当前代码中的 60,000 步续训设置形成旁证，但不能证明 40k 后每一步的实际
   数据权重。
4. params-only Checkpoint 没有 trainability mask。W&B 的全参数微调配置支持
   “训练时未冻结参数”的结论，但不能从 Checkpoint 单独计算“可训练参数量”。

因此报告不得把当前代码配置当作 19,000 步 Checkpoint 的完整历史 metadata，
也不得把 59,990 步最终 loss 作为部署 Checkpoint 的 loss。

机器可读详情见：

- `artifacts/baseline_policy_audit.json`
- `artifacts/baseline_wandb_audit.json`
