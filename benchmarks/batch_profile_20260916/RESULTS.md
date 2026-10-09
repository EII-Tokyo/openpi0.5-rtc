# RTX 5090 四路448推理吞吐与性能分析

日期：2026-09-16。主机：103，RTX 5090 32GB，驱动580.159.03，JAX/jaxlib 0.5.3。
权重：eii_data_system_with_rinse_cam4_fullft_h200_from_april_original_30repo，3000步。
四路448、prefix4296、action horizon50、10步去噪，普通采样与inference-time RTC分别测试。
输入是GPU驻留的合成观测，batch维真实扩展；每组独立容器，预热3次，再同步测量30次。
下述时间是模型调用时间，不含图像预处理、网络、机器人控制；不应与之前完整policy.infer时间直接比较。

## 原生CUDA Graph路径，不挂profiling工具的延迟

| Batch | 普通平均ms | 普通样本/秒 | RTC平均ms | RTC样本/秒 |
|---:|---:|---:|---:|---:|
| 1 | 189.96 | 5.264 | 227.35 | 4.399 |
| 2 | 360.94 | 5.541 | 434.44 | 4.604 |
| 4 | 713.74 | 5.604 | 839.87 | 4.763 |
| 8 | 1396.64 | 5.728 | 1677.00 | 4.770 |

batch1→8吞吐仅提高约8.8%（普通）/8.5%（RTC）；RTC batch4→8仅提高约0.16%。
四组均成功，无OOM。结论是当前实现接近batch扩展吞吐平台，而不是已经证明计算硬件峰值耗尽。

## Nsight Systems 2025.3.2：原生CUDA Graph路径

预热后采集3次batch1 RTC；必须启用--cuda-graph-trace=node。
第一次默认graph级报告没有展开图内部kernel，不能用其kernel总和代表全部计算；有效报告为rtc_batch1_nodes。
SQLite统计：48357次kernel事件，kernel时间和636.043ms，时间区间并集635.921ms，首个kernel开始至最后结束679.583ms，覆盖率93.575%。
这只是kernel时间覆盖率，不是SM/Tensor Core利用率；剩余时间包括空隙及其他活动，不能全部归因于CPU。
三个主要GEMM占kernel时间33.5%；input_transpose_fusion_1占7.8%，fusion_113占4.7%。
以上五类合计约46%的kernel时间。

## Nsight Compute 2025.3.1：诊断采样

首次原生路径，以及第二次禁用CUDA Graph的尝试，均在预热阶段segfault，未取得有效计数器。
第三次先脱离分析器建立JAX持久编译缓存，再在禁用CUDA Graph的独立进程中加载缓存，成功采集。
使用clock-control=none、cache-control=none，未改变机器时钟设置；计数器为近似诊断值，不能当作恒定频率基准。
对上述五类kernel各采集2个实例（共10个），每个22次replay；不将profiling时间计入吞吐。
所采样kernel名称与Nsight Systems一致；没有采全所有kernel，也没有profile batch8。

| Kernel | SM计算吞吐% | DRAM吞吐% | L2吞吐% |
|---|---:|---:|---:|
| gemm_fusion_dot_111_0 | 47.80–47.81 | 10.81–11.00 | 77.97–78.07 |
| gemm_fusion_dot_112_0 | 47.80–47.81 | 10.89–11.21 | 78.01–78.09 |
| gemm_fusion_dot_113_0 | 44.43–44.44 | 15.39–15.82 | 71.93–71.95 |
| input_transpose_fusion_1 | 22.90–23.04 | 75.11–75.31 | 35.22–35.55 |
| fusion_113 | 11.26–11.27 | 89.68–89.75 | 29.24–29.44 |

三个GEMM的最高利用计算管线为Tensor，值约44.4%–47.8%，并非所有kernel的全局平均值。
分析器对三个GEMM提示L2瓶颈；理论occupancy为16.7%，受寄存器及shared memory限制。
对两个融合算子提示DRAM瓶颈。低occupancy不能单独证明性能差，也不能由48%推导必然能提速2倍。

## 结论与边界

GPU很忙、增batch收益很小，但关键采样kernel并未把Tensor计算管线推至接近100%。
现有证据更支持访存/缓存、kernel资源与实现共同限制性能，不能断言5090纯算力已耗尽。
优先研究主要GEMM的tiling/寄存器/shared memory使用，以及显存带宽受限的transpose/fusion数据布局与融合机会。
这只是优化方向，不保证任何具体提速幅度；本次没有修改生产推理算法或参数。

原始资料：results.jsonl、rtc_batch1_nodes.nsys-rep/.sqlite、rtc_topkernels_cached.ncu-rep、ncu_details.csv及各.log。
103路径：/home/eii/openpi0.5-rtc/benchmarks/batch_profile_20260916/
实验结束所有测试容器已退出，无GPU计算进程。此时显存913MiB来自Xorg、GNOME、浏览器等图形进程，未修改这些进程。
