# 103 推理性能对照（2026-09-16）

## 方法与边界

RTX 5090 32GB；每组使用全新、独立的 compose one-off 模型容器，顺序执行。
没有启动 ROS、runtime 或下发机械臂动作。生产 compose 和权重未修改。
四类配置各选择一份代表 checkpoint，交叉测试 224/448 和 3/4 相机，共 16 组。
8 组成功；8 组跨分辨率组合由于视觉位置编码 shape 不兼容，在加载阶段失败。
没有插值或改写权重。同一配置下无 rinse 的其他步数 checkpoint 未重复测试。

固定：batch=1，bfloat16 参数，action horizon=50，denoising_steps=10，单帧历史，文本槽位200。
输入：固定随机合成 RGB 图像，每路640×480，零机器人状态；经过正式预处理变成目标分辨率。
统一 prompt：`twist off the bottle cap and rinse the bottle`；subtask=`good action`。
普通推理不带 prev_action；RTC 带50×32上一段动作，replan_start_step=25、handoff_delay_steps=10。
按照生产 serve_policy 的顺序预热 inference-time RTC、training-time RTC、普通推理。
预热后每条路径额外丢弃2次，统计20次；计时包围完整 policy.infer 并同步结果。
时间包括预处理、模型执行和输出转回主机，不含 WebSocket 通信、相机采集或机器人控制。
不是任务成功率测试；改变相机数的非原生组合不能据此用于部署。

## 结果

显存为 nvidia-smi 整卡稳定占用，含185 MiB桌面基线；GiB=1024 MiB。

| 权重 | 输入 | 原生组合 | Prefix token | 普通平均ms | RTC平均ms | 显存MiB |
|---|---|---|---:|---:|---:|---:|
| 正式演示版19000 | 224×224 / 3路 | 是 | 968 | 73.52 | 107.30 | 9003 |
| 正式演示版19000 | 224×224 / 4路 | 否 | 1224 | 85.61 | 117.58 | 8991 |
| 无rinse 20000 | 448×448 / 3路 | 是 | 3272 | 166.33 | 204.07 | 8995 |
| 无rinse 20000 | 448×448 / 4路 | 否 | 4296 | 219.52 | 258.98 | 17185 |
| 三相机rinse 5000 | 448×448 / 3路 | 是 | 3272 | 167.41 | 205.63 | 9069 |
| 三相机rinse 5000 | 448×448 / 4路 | 否 | 4296 | 220.64 | 260.63 | 17187 |
| 四相机rinse 3000 | 448×448 / 3路 | 否 | 3272 | 167.09 | 203.89 | 8999 |
| 四相机rinse 3000 | 448×448 / 4路 | 是 | 4296 | 219.80 | 257.60 | 17403 |

448每路1024视觉token，224每路256；上表包含200个文本/状态槽位（含padding）。
后六组同时记录了实际输入shape和相机mask，均为true。第一组未开启INFO日志，token数由模型规格计算；第二组及后续成功组有PREFIX_TOKEN_PROBE形状日志。

## 显存增加的证据

以同一份四相机3000权重对照：

| 指标 | 448三路 | 448四路 |
|---|---:|---:|
| nvidia-smi整卡GiB | 8.79 | 17.00 |
| JAX活跃分配峰值GiB（含预热） | 7.391 | 7.788 |
| JAX稳定活跃分配GiB | 6.646 | 6.564 |
| JAX保留内存池GiB | 8.010 | 16.217 |

当前设置已经是 XLA_PYTHON_CLIENT_PREALLOCATE=false。
四路448在第一次RTC预热后，内存池扩容并保留；后续training-time RTC和普通推理预热没有再次增加整卡占用。
因此本次17 GiB不是参数量翻倍，也不是额外预热多条路径累计导致的再次翻倍。
更多token增加了临时内存需求；整卡占用跳跃主要表现为分配池扩容。具体碎片布局未做逐块分析，不能仅由上述统计断言哪次分配触发扩容。
不同进程分配顺序有轻微变化，因此同结构的显存和活跃分配统计不要求完全一致。
JAX官方说明：关闭预分配后仍按需分配并复用内存；platform allocator可释放不用的内存，但有性能代价。本次未更改allocator。
参考：https://docs.jax.dev/en/latest/gpu_memory_allocation.html

## 不兼容组合

- 演示版224权重→448（3/4路）均失败：pos_embedding期望(1,1024,1152)，权重为(1,256,1152)。
- 其余三类448权重→224（各3/4路）均失败：反向shape mismatch。
- 因此224与448的延迟比较是不同原生权重间的比较，不是同一权重严格控制分辨率的因果实验。

## 配置映射与原始记录

- 正式演示版19000：eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo
- 无rinse 20000：eii_data_system_without_rinse_cam3_fullft_h200_have_cap_water_return_home_drop_empty_x1_original_45repo
- 三相机rinse 5000：eii_data_system_with_rinse_cam3_fullft_h200_original_59repo
- 四相机rinse 3000：eii_data_system_with_rinse_cam4_fullft_h200_from_april_original_30repo

各checkpoint完整路径、均值/中位数/P95、显存阶段统计在results.jsonl和对应.log中。
results.jsonl保留了首次容器挂载路径失败的16条记录；以每个label的最后一条为本次最终结果。
103脚本：/home/eii/openpi0.5-rtc/benchmarks/benchmark_inference_matrix.py
103日志：/home/eii/openpi0.5-rtc/benchmarks/inference_matrix_20260916/
完成时无运行中的Docker容器，GPU占用恢复185 MiB。
