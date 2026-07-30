# 缺失信息与补充方法

日期：2026-07-30

| 优先级 | 缺失内容 | 为什么需要 | 当前能否做出结论 | 补充方法 |
|---|---|---|---|---|
| P0 | 正式逐瓶测试记录、完整录像、起止时间 | 验证 >1h、约2瓶/min、任务成功率与中断 | 只能写现场估计 | 固定场景；每瓶编号；同步录像；记录成功/失败/恢复/中止 |
| P0 | 自动或双人复核成功判定 | 区分完整任务和部分完成 | 不能给正式成功率 | 分阶段检测抓瓶、抓盖、转动、脱离、两类进盒；人工双人抽检 |
| P0 | 有盖/无盖配对测试 | 验证指令缺口是否导致 air-unscrew | 只能说风险一致，不能证明因果 | 同瓶型/同位置成对测试；记录是否进入旋拧阶段 |
| P0 | 正向/反向/倒置配对测试 | 定位 upside-down grasp 环节 | 只能确认现场观察到 | 给瓶口端标签；固定姿态；记录选取端、抬升后方向、后续完成性 |
| P0 | 独立 validation/test split | 评估泛化与过拟合 | 不能判断 train/val gap 或 leakage | 冻结不进入训练的瓶型/位置/光照场景 |
| P1 | 保存点同条件比较 | 判断为何部署 19k，而不是更后保存点 | 不能说19k最佳 | 冻结3–5个保存点，同一测试清单评测 |
| P1 | 多随机种子 | 判断训练波动 | 当前只有 seed42 | 关键配置至少3 seeds，报告mean/std/CI |
| P1 | 全量视频 decode CRC | 确认无损坏帧 | 只确认代表视频 | 对所有正式训练视频全量 ffmpeg decode 和帧数核对 |
| P1 | 逐帧语义标签 | 定量分析 no-cap、orientation、phase | 文件名只支持启发式覆盖 | 标瓶盖存在、瓶口方向、active arm、contact phase、outcome |
| P1 | 力/扭矩/瓶盖角度 | 区分滑动、卡住、未转动 | 当前不能断言摩擦原因 | 记录 gripper current、视觉角标或 force/torque/tactile |
| P1 | 完整端到端延迟 | 判断50Hz与action chunk匹配 | attention capture overhead 不是 inference latency | 记录视觉、网络、推理、handoff、control p50/p95/p99 |
| P1 | 日语界面的数据中心/HF browser viewport screenshots | 满足界面证据与可视化数据展示 | 数据库/代码功能已核验，缺真实网页视口 | 修复 reviewed MCPJungle Chrome group；切换日语；分别截取“数据集合总览”和“左侧四相机、右侧虚拟演示、下方时间轴”的单集合页面 |
| P1 | RLT same-condition BC baseline | 验证 reinforcement learning gain | 只能确认 offline loop | 冻结 BC/RLT，同场景同次数比较 task/stage metrics |
| P2 | ROS/Isaac Sim actual version manifest | 环境复现 | 本报告没有使用其结果 | 从实际运行容器导出 immutable manifest |
| P2 | insertion geometry tolerance | 验证“毫米级”要求 | 只能作为目标 | 测瓶口/管口/外参，建立误差预算和重复插入基准 |

## Browser screenshot blocker

按项目 Gateway-only 规则调用 MCPJungle Chrome DevTools：

- tool discovery 成功；
- 两次调用 `navigate("https://ai.swm-eii.com/")` 均在初始化阶段失败；
- HTTP 404：`tool group not found: codex-research`。

未用本地 Selenium/Playwright 或其他直接 MCP 绕过。报告用数据库聚合、公开固定版本数据和真实训练关键帧继续完成，但没有把替代图伪装成网站截图。恢复网关后仍需补采日语界面的两张浏览器可见区域截图，并对数据集合、四相机、虚拟演示和时间轴作标注说明。
