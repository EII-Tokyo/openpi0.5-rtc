# OpenPI ALOHA 学习资源目录

本目录包含OpenPI ALOHA系统的完整技术文档，涵盖从数据到机器人控制的全流程。

## 📚 文档导航

### 1. **01-OpenPI-ALOHA-Complete-Guide.md** (完整指南)
**最全面的参考文档** - 适合第一次学习或需要深入理解

包含内容：
- ✅ **1. 数据格式** - LeRobot格式、ALOHA坐标转换、图像处理
- ✅ **2. 训练流程** - 配置、完整训练管道、数据加载细节
- ✅ **3. 推理系统** - 子任务识别、动作预测、RTC实时控制
- ✅ **4. 机器人控制** - Policy Server、Runtime执行流程、数据流速
- ✅ **5. 核心代码位置** - 所有关键文件的行号索引
- ✅ **6. 调试指南** - 快速诊断和问题解决

**建议阅读顺序**：
```
如果你有30分钟 → 读 1-2 节
如果你有1小时 → 读 1-4 节
如果你有2小时 → 完全阅读 + 附录
```

---

### 2. **02-Quick-Reference.md** (快速参考)
**实用命令速查** - 适合已经理解基础，需要快速查询

包含内容：
- ✅ **常用命令** - 训练、服务器、runtime、数据转换
- ✅ **常见问题** (FAQ) - 10+个常见错误和解决方案
- ✅ **配置对比表** - LORA vs 全量 vs 完整训练
- ✅ **性能目标** - 延迟、吞吐量、显存要求
- ✅ **数据集对照表** - LibEvo, DROID, ALOHA Real, LeRobot
- ✅ **调试工作流** - 新任务/机器人的完整开发流程

**最常用的部分**：
- `CUDA out of memory` 错误 → 看Q2
- 连接Policy Server失败 → 看Q3
- 关节抖动 → 看Q4

---

### 3. **03-Key-Code-Snippets.md** (代码详解)
**源代码注释版** - 适合想理解实现细节

包含内容：
- ✅ **1. 数据转换管道** - AlohaInputs, 坐标转换, 夹爪数学
- ✅ **2. 训练循环** - train_step 完整分析
- ✅ **3. 推理系统** - Flow Matching 采样, 自回归解码
- ✅ **4. RTC实时控制** - guided_inference 细节
- ✅ **5. Runtime集成** - ActionChunkBroker, AlohaRealEnvironment

**最有价值的部分**：
- **2.1** - 理解为什么需要坐标转换
- **3.1** - 理解ODE扩散如何生成动作
- **3.2** - 理解自回归如何生成子任务
- **4.1** - 理解RTC如何保证平滑控制

---

## 🎯 学习路径

### 新手路线 (第一次学习)
```
1. 阅读 01-Guide 的 1-2 节 (数据 + 训练配置)
   ↓
2. 尝试运行一个训练
   PYTHONPATH=src uv run scripts/train.py twist_off_the_bottle_cap_subtask_lora
   ↓
3. 查看 02-Reference 的常见问题，理解错误
   ↓
4. 阅读 01-Guide 的 3-4 节 (推理 + 机器人)
   ↓
5. 在模拟环境测试推理
   ↓
6. 深入 03-Snippets 理解关键算法
```
**预计时间**: 4-6小时

---

### 快速启动路线 (已有基础)
```
1. 用 02-Reference 快速复制一个训练命令
   ↓
2. 遇到问题 → 在 02-Reference FAQ 查询
   ↓
3. 需要深入 → 在 03-Snippets 找对应代码
```
**预计时间**: <30分钟

---

### 架构理解路线 (完整精通)
```
1. 01-Guide 第5节 - 核心代码位置
   ↓
2. 依次读 03-Snippets 的 5个部分
   ↓
3. 对照 01-Guide 的详细流程理解工程细节
   ↓
4. 修改参数，观察效果
```
**预计时间**: 8-10小时

---

## 🚀 快速启动检查清单

在运行任何东西前，检查：

### 环境检查
```bash
# 1. JAX和GPU
python -c "import jax; print(f'GPUs: {jax.device_count()}')"

# 2. 依赖
python -c "import flax; import lerobot; print('OK')"

# 3. ALOHA assets
gsutil ls gs://openpi-assets/checkpoints/
# 或离线模式：见 02-Reference Q1
```

### 数据准备
```bash
# 1. 数据集存在
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('lyl472324464/wipe_the_desk_with_a_blue_cloth')
print(f'Episodes: {len(ds)}')
"

# 2. 存储空间
df -h
# 建议 >100GB 用于checkpoints
```

### 第一次训练
```bash
# 最小化配置 (单GPU, 小batch)
PYTHONPATH=src uv run scripts/train.py twist_off_the_bottle_cap_subtask_lora \
  --exp-name test_run \
  --batch-size 1 \
  --num-train-steps 100 \
  --num-workers 0
```

如果这个跑成功，你已经搞定了基础设置！

---

## 📊 核心概念速查

| 概念 | 含义 | 重要性 | 详见 |
|------|------|--------|------|
| **Flow Matching** | 扩散模型训练方式 | ★★★★★ | 01-Guide 2.2, 03-Snippets 3.1 |
| **RTC** | 实时控制用前一步引导 | ★★★★★ | 01-Guide 3.3, 03-Snippets 4.1 |
| **adapt_to_pi** | 坐标系转换开关 | ★★★★★ | 01-Guide 1.2, 03-Snippets 1.1 |
| **KV缓存** | Transformer推理优化 | ★★★★☆ | 01-Guide 3.2, 03-Snippets 3.2 |
| **FSDP分布式** | 8GPU并行训练 | ★★★★☆ | 01-Guide 2.2 |
| **Norm Stats** | 数据归一化统计 | ★★★★☆ | 01-Guide 1.1 |

---

## 🔧 快速问题排查

**问题**: 模型加载失败
→ 看 02-Reference Q1

**问题**: 显存不足
→ 看 02-Reference Q2

**问题**: Policy Server连接失败
→ 看 02-Reference Q3

**问题**: 机器人运动不平滑
→ 看 02-Reference Q4

**问题**: 子任务识别不准
→ 看 02-Reference Q5

**问题**: 想理解ODE扩散怎么工作
→ 看 03-Snippets 3.1

**问题**: 想改进RTC参数
→ 看 01-Guide 3.3 + 03-Snippets 4.1

---

## 💾 文件大小和阅读时间

| 文件 | 大小 | 阅读时间 | 难度 |
|------|------|---------|------|
| 01-Complete-Guide | ~60KB | 60-90min | ⭐⭐⭐ |
| 02-Quick-Reference | ~40KB | 20-30min | ⭐⭐ |
| 03-Key-Code-Snippets | ~50KB | 40-60min | ⭐⭐⭐⭐ |

总计：~150KB，约2-3小时完整阅读

---

## 🎓 推荐深度学习路径

### Week 1: 基础理论
- 理解Diffusion Models（Flow Matching是新一代）
- 理解Transformer和KV缓存
- 理解FSDP分布式训练

### Week 2-3: 代码实现
- 读 01-Guide 完整
- 跑 twist_off_bottle_cap_subtask_lora 训练
- 修改参数，观察效果

### Week 4: 机器人集成
- 部署Policy Server
- 在模拟器中测试Runtime
- 理解ActionChunkBroker

### Week 5+: 优化和扩展
- 在真实机器人上运行
- 采集新任务数据
- 从头训练新模型

---

## 📖 相关资源

### 官方文档
- OpenPI 论文: https://ai.meta.com/research/robotics/
- LeRobot: https://huggingface.co/blog/lerobot
- JAX/Flax: https://flax.readthedocs.io/

### 代码仓库
- 主仓库: `/home/eii/project/openpi0.5-rtc/`
- 核心训练: `scripts/train.py`
- 模型: `src/openpi/models/pi0.py`
- ALOHA接口: `src/openpi/policies/aloha_policy.py`

### 快速实验
```bash
# 运行推理测试
PYTHONPATH=src uv run scripts/test_pi05_subtask_from_hdf5.py \
  --hdf5 /path/to/episode.hdf5 \
  --config pi05_aloha_pen_uncap \
  --checkpoint /path/to/checkpoint \
  --prompt "process bottles"

# 计算损失 (离线评估)
XLA_FLAGS='--xla_gpu_deterministic_ops=true' \
  PYTHONPATH=src uv run scripts/compute_loss_from_hdf5.py \
  --hdf5 /path/to/episode.hdf5 \
  --config pi05_aloha_pen_uncap \
  --checkpoint /path/to/checkpoint
```

---

## ❓ 常见问题 (简版)

**Q: 应该从哪里开始？**
A: 如果是第一次，从 01-Guide 的 1-2 节开始（数据格式和训练）。

**Q: 我只想快速运行一个训练？**
A: 复制 02-Reference 中的训练命令，改一下数据集和参数。

**Q: 为什么我的模型生成的动作抖动？**
A: 见 02-Reference Q4 - RTC参数调整。

**Q: 数据格式到底是什么？**
A: 见 01-Guide 1.1-1.3，有完整图解。

**Q: ODE扩散怎么工作的？**
A: 见 03-Snippets 3.1，有完整代码注释。

**Q: 如何部署到真实机器人？**
A: 见 01-Guide 4.3，有完整流程。

---

## 📝 更新日志

**2026-04-03** (当前)
- 完成全3份文档
- 包含 twist_off_the_bottle_cap_subtask_full_finetune 配置详解
- 包含RTC实时控制详解

---

## 🙏 贡献

如果你发现文档中的错误或有改进建议：
1. 修改对应的md文件
2. 验证内容准确性
3. 提交更新

---

**最后提醒**：
> 这不只是文档，而是完整的学习资源。
> 从 01-Guide 开始，按照学习路径走，你会逐步理解OpenPI从数据到机器人的全流程。
> 遇到问题先查 02-Reference，深入理解看 03-Snippets。
> 祝学习愉快！🚀

