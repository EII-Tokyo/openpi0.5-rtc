# ALOHA 机器人拧瓶盖项目现阶段技术报告

## 最终文件

- PDF：`aloha_bottle_cap_report.pdf`
- LaTeX 主文件：`aloha_bottle_cap_report.tex`
- 分章源文件：`sections/`
- 科学图与真实媒体图：`figures/`
- 原始绘图数据与机器可读统计：`artifacts/`
- 内部只读审计与结论—证据矩阵：`audit/`

公开 PDF 面向项目汇报和验收专家：不显示源代码、源文件名、函数名、远程路径或运行命令。精确技术追溯保存在 `audit/`，不进入公开正文。

## 数据与结论口径

报告明确区分：

1. 数据平台有效 ALOHA 总资产：51 个项目、2,413 条轨迹、约 16.15 小时；
2. 正式部署基础模型唯一训练子集：25 个仓库、1,051 条轨迹、879,852 帧、约 4.89 小时；
3. 过滤后实际可训练：844,102 帧、约 4.69 小时；
4. 强化学习研究数据：另一组 835 条冲洗轨迹，不能与基础分拣训练量相加。

现场“连续超过 1 小时、约 2 瓶/分钟”是观测估计，没有完整录像、逐瓶计数或固定测试协议；报告未把它换算成正式成功率。

## 重新生成机器可读统计

远程源证据的审计脚本位于 `scripts/`。这些脚本只读访问用户指定的 ALOHA 机器、数据中心、训练平台和公开固定版本数据。运行前应先阅读项目的远程操作与数据规则，禁止启动机器人或修改远程数据。

在已经存在原始审计产物时，重建报告规范化统计：

```bash
cd /home/eii/project/openpi0.5-rtc-reward-learning/reports/aloha_bottle_cap_report
../../.venv/bin/python scripts/build_public_artifacts.py
```

该命令生成：

- `artifacts/dataset_statistics.json`
- `artifacts/checkpoint_metadata.json`

RLT 历史统计分别保存在：

- `artifacts/rlt_dataset_statistics.json`
- `artifacts/rlt_checkpoint_metadata.json`

## 重新生成图形

```bash
cd /home/eii/project/openpi0.5-rtc-reward-learning/reports/aloha_bottle_cap_report
../../.venv/bin/python scripts/generate_annual_report_figures.py
```

脚本生成 17 个有明确问题目标的图形或媒体组合，并写出 `artifacts/plot_manifest.json`。每张科学图同时保存 PDF 和高分辨率 PNG；真实照片、训练关键帧和注意力组合使用 PNG。

图形证据规则：

- 正式设备照片明确标为静止休眠状态；
- 训练关键帧明确标为示教数据，不是自主评测；
- 注意力热图明确标为没有成功/失败标签；
- 封面为概念插画，不是实验照片；
- 不生成没有原始计数的成功率或失败率图；
- 加权采样不画成新增数据。

## 重新编译 PDF

环境：

- XeLaTeX / BibTeX，TeX Live 2023
- 中文 `ctexrep`
- Python 图形环境使用项目 `.venv`

完整构建：

```bash
cd /home/eii/project/openpi0.5-rtc-reward-learning/reports/aloha_bottle_cap_report
make all
```

或仅编译：

```bash
bash build.sh
```

构建顺序为 XeLaTeX、BibTeX、XeLaTeX、XeLaTeX，确保目录、图表、引用和参考文献稳定。

## 最终验证

```bash
cd /home/eii/project/openpi0.5-rtc-reward-learning/reports/aloha_bottle_cap_report
../../.venv/bin/python scripts/verify_report.py
```

验证脚本检查：

- PDF 可读、页数和文本可提取；
- 缺失引用、缺失图片、致命错误和 overfull；
- 潜在空白页；
- 所有正文引用图片是否存在；
- 栅格图片分辨率；
- 公开 PDF 是否泄露代码名、源文件名或内部绝对路径；
- 数据规模、正式保存点和参数量是否与规范化 JSON 一致；
- 结论—证据矩阵来源是否存在；
- 现场估计、训练误差和强化学习结论是否保留证据边界。

结果写入：

- `artifacts/verification_results.json`
- `audit/final_verification.md`

## 当前仍需补充的关键实验

- 固定场景的逐瓶计数、完整录像、起止时间和失败分类；
- 自动或双人复核的阶段/完整任务成功判定；
- 有盖/无盖、正向/反向瓶子的配对实验；
- 基础模型多个保存点的同条件比较；
- 独立验证/测试集和多随机种子；
- 全量视频逐帧解码与语义标签；
- 接触力、瓶盖转角和端到端延迟；
- 强化学习与基础模型的同条件真机对照；
- 空瓶插水管的真实几何误差预算。

数据中心和 Hugging Face 的浏览器视口截图仍缺失：受审 MCPJungle Chrome DevTools 工具组在两次访问时均返回 404。报告没有使用绕过方式，也没有把替代图伪装成网页截图。恢复网关后应将界面切换为日语，分别补采“数据集合总览”和“单集合四相机/虚拟演示/时间轴”两个浏览器可见区域，并按标注流程解释其训练用途。
