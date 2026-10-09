# B300 软件兼容性与比较限制

2026-09-16，B300 主机 `185.200.1.11:22109`。

## JAX 0.5.3 未通过编译

先安装了与前三台 GPU 相同的 JAX 0.5.3 环境，BF16 2048×2048 矩阵乘法在编译阶段失败：

```text
Unknown compute capability 10.3. Defaulting to telling LLVM that we're compiling for sm_101
Instruction 'tcgen05.alloc' not supported on .target 'sm_101'
Instruction 'tcgen05.mma' not supported on .target 'sm_101'
ptxas fatal: Ptx assembly aborted due to errors
```

这不是 OOM，也不是权重损坏；旧环境未产生有效模型延迟数据。不能说 JAX 0.7.2 是最低可用版本，只能说本次旧版本失败、所选择的新版本跑通。

## 本次新环境

独立环境 `/workspace/benchmarks/b300_compare/venv_jax072`，旧环境保留。
主要版本：JAX/jaxlib/CUDA13 插件 0.7.2，Flax 0.11.2，Orbax 0.11.24，Equinox 0.13.1，Optax 0.2.5，ml-dtypes 0.5.3。
CUDA runtime 13.0.96、nvcc 13.0.88、cuBLAS 13.1.0.3、cuDNN 9.14.0.64、NCCL 2.27.7。
完整安装包清单见 requirements.txt/environment.json；pip 依赖检查通过。

选择参考：[NVIDIA JAX Release 25.10 说明](https://docs.nvidia.com/deeplearning/frameworks/pdf/JAX-Release-Notes.pdf)列出 JAX 0.7.2、Flax 0.11.2、CUDA 13 和 G/B300 支持。
本次使用的是 PyPI 软件包，并不是 NVIDIA NGC 镜像，也不声称包含 NGC 的全部补丁和优化。
实际编译日志仍有 `No SoL config found ... Using default config`，说明编译器使用默认成本模型；这不是执行失败，但本次没有做 B300 专项优化。

## 仅在 B300 测试副本中做的代码适配

1. `src/openpi/models/model.py`：新版 Orbax 的 metadata 返回 StepMetadata，先取 `item_metadata`，再按原逻辑取 params；不改权重值。
2. `src/openpi/models/pi0.py`：新版 Flax 移除了 nnx.Dict。原 Dict 的构造实质是 setattr，故改成 nnx.Module 容器并设置相同的 llm/img 属性；不改计算函数或参数路径。

参数加载仍执行原有的参数树结构和 shape 检查，dtype 仍按原推理流程恢复为 BF16。
权重文件未改、模型计算公式未改、103 的训练/部署源码未改。测试不连接机械臂。
前两次新版加载尝试的失败分别是 StepMetadata 接口和 nnx.Dict 接口错误，不能误记成 OOM 或推理超时。

## 不能把历史差距直接解释为硬件差距

5090、H100、B200 的历史基准均为 JAX 0.5.3；B300 本次是 0.7.2，CUDA、Flax 等也变化。
尝试连接之前的 B200 做同环境对照，但 `129.213.94.1:40015` 两次返回 Connection refused，未自行启动或更改该实例。
因此在取得同软件版本控制实验前，历史对照只能用于观察实际整套配置的延迟，不能作为纯 GPU 硬件加速比。
