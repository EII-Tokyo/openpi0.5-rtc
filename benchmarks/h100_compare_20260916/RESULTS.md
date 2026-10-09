# 单卡 H100 对比 RTX 5090：四路 448 推理

日期：2026-09-16。

## 测试范围

- 同一份 rinse 四相机 3000 步权重：`eii_data_system_with_rinse_cam4_fullft_h200_from_april_original_30repo`。
- 实验名：`rinse_cam4_april_30repo_44entries_448_bs64_ga4_scan_nw96_fsdp4_fd65536_20260915`。
- 从 openpi-tokyo 下载 params/assets，19 个文件，12,443,883,411 字节，逐文件 SHA256 验证通过；不含 train state。
- 四路 448×448，视觉 token 每路 1024，文本长度 200，prefix 4296，action horizon 50，去噪 10 步。
- 固定合成图片/状态，不连接机械臂，不评价动作质量。普通采样与 inference-time RTC 分开测量。
- RTC：replan_start_step=25，handoff_delay_steps=10。
- 使用与 5090 基准相同的脚本，仅增加 repo-root 参数以适配新机器路径。没有修改模型算法或精度。

## 机器与环境

| 项目 | 103 / RTX 5090 | 新机器 / H100 |
|---|---|---|
| GPU | RTX 5090 32GB | H100 80GB HBM3 |
| 驱动 | 580.159.03 | 610.57.04 |
| CPU | Core Ultra 5 225 | Xeon Platinum 8568Y+ |
| Python | 3.11.9 | 3.11.16 |
| JAX / jaxlib | 0.5.3 | 0.5.3 |
| Flax | 0.10.2 | 0.10.2 |
| NumPy | 2.4.0 | 2.4.0 |
| cuBLAS pip 包 | 12.6.4.1 | 12.6.4.1 |
| cuDNN pip 包 | 9.5.1.17 | 9.5.1.17 |

H100 功耗上限保持 599W，未调整驱动、时钟或功耗。其容器 CPU quota 为 23.04 核等效。
两端 `XLA_PYTHON_CLIENT_PREALLOCATE=false`，`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`。
这是两台实际机器的比较，不是完全隔离 CPU、驱动等变量的纯硬件实验。

## GPU 驻留输入：模型调用延迟

每个 batch 独立进程，预热 3 次，计时 30 次；显式等待结果完成。
不含图片预处理、网络和机器人控制，包含模型调用及主机 dispatch/synchronization，不能称为纯 kernel 时间。

| Batch | 5090 普通 ms/批 | H100 普通 ms/批 | 加速比 | 5090 RTC ms/批 | H100 RTC ms/批 | 加速比 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 189.956 | 103.037 | 1.84× | 227.350 | 149.750 | 1.52× |
| 2 | 360.942 | 215.961 | 1.67× | 434.444 | 289.978 | 1.50× |
| 4 | 713.737 | 408.642 | 1.75× | 839.871 | 527.832 | 1.59× |
| 8 | 1396.637 | 815.205 | 1.71× | 1677.003 | 1015.519 | 1.65× |

| Batch | H100 普通样本/秒 | H100 RTC 样本/秒 |
|---:|---:|---:|
| 1 | 9.705 | 6.678 |
| 2 | 9.261 | 6.897 |
| 4 | 9.789 | 7.578 |
| 8 | 9.813 | 7.878 |

全部成功，无 OOM。batch 1→8，普通吞吐仅增加约 1.1%，RTC 吞吐增加约 18.0%。
没有证据据此断言 Tensor Core 理论峰值已经耗尽。

### 显存和功耗记录

H100 batch 1/2/4 的采样峰值分别为 17011/17013/17017 MiB；batch 8 为 33397 MiB。
batch 1 的 JAX 活跃分配峰值约 8.22 GB、内存池约 17.18 GB；batch 8 分别约 16.62 GB、34.36 GB。
池占用不是权重或活跃 tensor 的大小，不能从 33GB 显存占用推导模型实际需要 33GB。
采样功耗最大值分别为 563.22/593.06/598.98/595.68 W，采样覆盖加载、编译、预热及正式测量。
大 batch 采样接近 599W 上限；没有进一步实验分离功耗限制对延迟的贡献。

## 完整 policy.infer

另跑 100 次，预热并丢弃前 5 次，包含原始图片预处理、传输、模型调用及输出处理，不含网络与机械臂。
初次 H100 结果为普通 154.065 ms、RTC 206.869 ms。随后发现 OpenCV/Pillow 等版本未完全对齐，因此保留该结果为初测，不作为最终完整调用结果。
复测对齐 OpenCV 4.11.0.86、Pillow 11.2.1、numpydantic 1.6.9、tyro 0.9.22；其余核心模型库此前已对齐。

最终完整调用复测（两端各 100 次）：

| 模式 | 5090 平均 ms | H100 平均 ms | 加速比 | H100 P95 ms |
|---|---:|---:|---:|---:|
| 普通 | 219.409 | 160.009 | 1.37× | 171.182 |
| RTC | 259.818 | 203.858 | 1.27× | 212.292 |

同步模型调用/其他部分拆解：

| 模式 | 5090 模型调用 ms | 5090 其他 ms | H100 模型调用 ms | H100 其他 ms |
|---|---:|---:|---:|---:|
| 普通 | 200.272 | 19.137 | 101.899 | 58.110 |
| RTC | 240.574 | 19.244 | 148.940 | 54.918 |

这里“其他”包括输入转换、图片预处理、部分数据传输、输出处理等，未进一步逐项拆开，不能全部归因于 CPU 或某一个库。
主机 CPU、容器限制、驱动及完整依赖环境仍不完全一致。版本对齐后，这部分开销差异依然存在。
完整调用 H100 显存最终 17085 MiB，无 OOM；JAX 内存池约 17.24 GB，活跃分配峰值约 8.27 GB。

结论：当前 H100 更快。单样本 GPU 驻留模型调用普通/RTC 分别约 1.84×/1.52×；包含预处理的完整调用约 1.37×/1.27×。
不能把 GPU 驻留输入的加速比直接视为部署端到端收益；也不应把该结果推广成所有 H100 的固定加速比。
测试结束进程自然退出，机器保持开机，环境和权重保留。未连接机械臂、未修改原远端训练。

## 官方算力口径

| RTX 5090 精度/累加方式 | 非稀疏 | 稀疏 |
|---|---:|---:|
| 普通 FP32，非 Tensor | 104.8 TFLOPS | 不适用 |
| BF16 Tensor，FP32 累加 | 209.5 TFLOPS | 419 TFLOPS |
| FP16 Tensor，FP32 累加 | 209.5 TFLOPS | 419 TFLOPS |
| FP16 Tensor，FP16 累加 | 419 TFLOPS | 838 TFLOPS |
| FP4 Tensor，FP32 累加 | 1676 TFLOPS | 3352 TFLOPS / AI TOPS |

来源：[NVIDIA RTX Blackwell 白皮书，第 46–48 页](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)。
H100 SXM 官方 BF16 Tensor 标称 1979 TFLOPS 为稀疏口径，非稀疏约 989.5 TFLOPS；官方显存带宽 3.35 TB/s。
来源：[NVIDIA H100 规格](https://www.nvidia.com/en-us/data-center/h100/)。
5090 的 BF16 非稀疏、FP32 累加口径不是 3352，也不是 838 TFLOPS。理论速率不等于整段推理实际速率。
换成 H100 同时改变计算、缓存、显存带宽、编译后的 kernel 和主机，因此本实验不能单独证明 5090 是纯算力瓶颈。

## 原始资料与复现

本目录：`results.jsonl`、`batch1/2/4/8.log`、对应 GPU 采样 CSV；完整调用保留 `policy.json/.log`（初测）和 `policy_matched.json/.log`（版本对齐复测）。
5090 基线：相邻 `batch_profile_20260916/results.jsonl`、`resolution_scaling_20260916/results.jsonl` 中 448 条目。
5090 完整调用基线的 resize-pos 开关开启，但 448→448 时没有改动 position embedding。
H100 远端目录：`/workspace/benchmarks/h100_compare`；模型代码与权重：`/workspace/openpi0.5-rtc`。
独立环境：`/workspace/benchmarks/h100_compare/venv`。
脚本：`benchmark_batch_profile.py`、`run_h100_compare.py`、`benchmark_inference_matrix.py`、`run_h100_policy.py`。
传输使用临时签名 URL，不向新机器复制 AWS 密钥；保存的权重 manifest 不含签名 URL。
