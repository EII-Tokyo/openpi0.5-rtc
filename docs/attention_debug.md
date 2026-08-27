# 三路相机 Attention 诊断

当前 `docker-compose.yml` 已启用采集。为了不阻塞 Real-Time Chunking 的动作交接，attention
在后台队列中异步计算，默认每 5 次 policy 重规划保存一次：

- `cam_high`、`cam_left_wrist`、`cam_right_wrist` 原图
- 18 层、50 个未来动作 token、16×16 视觉 patch 的原始 attention
- 三路相机 attention 占比
- 最后 6 层、前 10 个动作 token 的快速叠加图

数据写入：

```text
attention_debug/<启动时间>/
├── manifest.jsonl
└── sample_000000/
    ├── attention.npz
    ├── metadata.json
    ├── overview.jpg
    ├── cam_high.jpg
    ├── cam_left_wrist.jpg
    └── cam_right_wrist.jpg
```

生成整段运行的 HTML 报告：

```bash
cd /home/eii/openpi0.5-rtc
python3 scripts/render_attention_report.py
```

报告默认选择 `attention_debug` 下最新的一次运行，输出到该运行目录的 `report.html`。

## 诊断含义

导出的是 action expert 的动作查询 token 对视觉前缀 token 的真实 Transformer attention。它对 KV
head 和 query head 取平均，但保留全部网络层和 50 个未来动作位置。快速图为便于观察，进一步对最后
6 层和前 10 个动作位置取平均；`attention.npz` 保留未做该平均的完整数据。

比较“瓶盖已脱落但右爪未抓住”和“右爪抓住瓶盖”两类片段时，重点看：

1. `cam_high` 的瓶口区域是否在脱落瞬间持续高亮。
2. 三路相机占比是否突然转向 `cam_right_wrist`。
3. 前几个动作 token 与后续动作 token 的注意区域是否不同。
4. 成功和失败片段在相同动作阶段的差异，而不是只比较整段平均。

Attention 能说明模型内部的信息路由，但不等同于因果贡献。若热图显示瓶口有 attention，仍建议再做
遮挡测试：分别遮挡 `cam_high` 瓶口区域和右腕爪内区域，比较动作输出是否发生明显变化。
