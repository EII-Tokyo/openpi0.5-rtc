# 逐图视觉与证据审查

- 状态：**PASS**
- 正文实际引用图：18
- 违规项：0

| 图文件 | 唯一问题 | 支撑结论 | 证据边界 | 状态 |
|---|---|---|---|---|
| figures/aloha_formal_photo_annotated.png | 正式 ALOHA 工作站由哪些主要部分组成？ | 正式设备包含示教臂、工作臂、相机和任务工作区。 | 照片拍摄时机器人处于静止休眠状态。 | PASS |
| figures/attention_camera_share.pdf | 8,223 个样本中三路视野的注意力份额如何分布？ | 腕部近景整体高于顶部总览。 | 注意力份额不是失败原因的因果解释。 | PASS |
| figures/attention_real_example.png | 一次真实运行中三路视野的关注位置是什么样？ | 样例展示顶部和双腕视野的真实注意力热区。 | 记录没有成功或失败标签。 | PASS |
| figures/baseline_episode_length_distribution.pdf | 正式训练轨迹的长度分布如何？ | 1,051 条轨迹覆盖短、中、长多种示教时长。 | 极长尾只为绘图可读性截到 99% 分位。 | PASS |
| figures/baseline_training_loss.pdf | 正式模型是否持续拟合示教动作？ | 完整训练历史显示动作拟合误差持续下降。 | 训练误差下降不能代替真机成功率。 | PASS |
| figures/condition_coverage.pdf | 正式训练数据覆盖了哪些困难条件？ | 数据包含方向、无盖、翻转和自由旋转等条件。 | 名称分类允许重叠且不表示各条件成功率。 | PASS |
| figures/cover_concept.png | 报告面向的完整双臂瓶子分拣任务是什么？ | 封面概括取瓶、旋盖和分类投放的完整任务链。 | 概念示意图不是实验照片。 | PASS |
| figures/data_scope.pdf | 数据总资产、正式训练子集和过滤后输入分别有多大？ | 三层数据口径必须分开统计。 | 总资产不能写成正式模型训练量。 | PASS |
| figures/evidence_grade.pdf | 当前成果分别具有怎样的证据强度？ | 可复核事实、现场观察和待量化指标需要分层陈述。 | 证据等级不是任务成功率。 | PASS |
| figures/experiment_funnel.pdf | 41 次训练尝试中有多少留下历史并运行到较远阶段？ | 训练谱系体现工程迭代量和运行稳定性。 | 运行状态不等同于模型真机任务成功。 | PASS |
| figures/model_dataflow.pdf | 三路视觉与机器人状态如何形成连续双臂动作？ | 正式部署模型输出未来 50 个控制时刻的双臂动作。 | 示意图只描述正式部署模型，不代表其他候选模型。 | PASS |
| figures/next_year_roadmap.pdf | 下一年度工作为什么按当前优先级推进？ | 路线从可信评估到基础能力、强化学习和高精度插入逐级推进。 | 插管目标尚未完成，毫米级要求仍需实测标定。 | PASS |
| figures/prompt_condition_gap.pdf | 无盖数据的任务指令是否明确要求跳过旋拧？ | 无盖条件仍普遍使用无条件旋盖指令。 | 指令缺口是风险证据，不是唯一原因证明。 | PASS |
| figures/rlt_offline_validation.pdf | 强化学习研究闭环目前达到什么程度？ | 连续轮次形成了可比较的离线动作与价值指标。 | 离线改善不能写成真机能力提升。 | PASS |
| figures/sampling_exposure.pdf | 困难样本如何在训练中获得更多出现机会？ | 重复采样提高重点场景的训练暴露。 | 等效采样数量不是新增独立轨迹。 | PASS |
| figures/software_data_pipeline.pdf | 第一年度形成了怎样的数据到部署闭环？ | 工作覆盖采集、编辑、训练、部署、观察和问题回流。 | 当前软件能力不代表历史数据都使用了全部后来功能。 | PASS |
| figures/task_timeline.pdf | 单个瓶子的完整处理链包含哪些阶段？ | 任务包含七个相互依赖的长流程阶段。 | 阶段划分不能代替逐阶段成功率。 | PASS |
| figures/training_demonstration_keyframes.png | 真实训练数据中的典型画面是什么样？ | 固定关键帧展示多种真实示教条件。 | 示教关键帧不是自主测试结果。 | PASS |
