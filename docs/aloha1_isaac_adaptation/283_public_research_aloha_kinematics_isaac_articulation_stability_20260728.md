# ALOHA1/VX300S 运动学可达性与 Isaac articulation 稳定性公开证据链

**调查日期：** 2026-07-28

**范围：** 只读公开资料与本地已保存的 URDF/运行证据；未连接 103、未启动新的 physics live、未修改 USD/gains。

**版本边界：** Isaac Sim 仅采用 5.1.0 文档，Omni Physics 仅采用 Kit 107.3 文档；不采用 `latest` 或 6.0 语义。

**ROLLOUT_USED=false**

## 调查目标

把当前问题拆成两条独立证据链：

1. ALOHA1/VX300S 的真实六关节串联运动学与受限可达性；
2. Isaac Sim 5.1 / PhysX articulation 的 root、joint frame、drive、reset 和稳定性语义。

结论必须避免把 ALOHA 当作 Franka 式冗余 7DOF 机械臂，也避免把“首帧异常”误判为抓取或碰撞问题。

## 环境与本地事实

当前本地最小 arm-only URDF：

`assets/isaac/original_stationary_aloha_arm_only/generated/puppet_left_vx300s_arm_only_resolved.urdf`

本地文件审计确认该 URDF 存在，六个 arm joint 的顺序为 `waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate`，轴依次为 Z、Y、Y、X、Y、X。工作会话另记录过 fresh 首帧最大 qvel 约 `12.4632282257 rad/s`，但本次有界仓库搜索没有找到承载该精确数值的独立原始 artifact；该数值因此标记为 `UNVERIFIED_LOCAL_NOTE`，不能单独作为物理验收证据。

## 一手来源

### ALOHA 与 VX300S

- Tony Zhao 等，ALOHA 开源仓库：[tonyzhaozh/aloha](https://github.com/tonyzhaozh/aloha)。README 说明它是双臂遥操作和数据采集硬件/软件仓库，并明确依赖 Interbotix 软件；仓库说明硬件扫描到每个机器人 9 个电机。该仓库不是 6D IK 证明，而是硬件控制与数据接口来源。
- 原始论文：[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware, arXiv:2304.13705](https://arxiv.org/abs/2304.13705)。论文证明 ALOHA 用低成本双臂系统进行精细双臂任务，但不声称任意末端 6D pose 对所有位置均可达。
- Trossen/Interbotix 官方规格：[ViperX-300 6DOF](https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/vx300s.html)。官方列出 6 DOF、约 750 mm reach、默认关节限制：waist ±180°、shoulder −101°/+101°、elbow −101°/+92°、wrist angle −107°/+130°、forearm roll ±180°、wrist rotate ±180°；同时列出 9 个舵机，说明 shoulder/elbow 等包含双电机结构。该页面是普通 `vx300s` 的公开规格，只能支持通用结构事实，不能证明 `aloha_vx300s` 的 mesh、gripper、质量、惯量或控制配置完全相同。
- Interbotix 官方描述包文档：[Arm Descriptions](https://docs.trossenrobotics.com/interbotix_xsarms_docs/ros2_packages/arm_descriptions.html)。文档说明 `interbotix_xsarm_descriptions` 提供 URDF/Xacro 和 mesh，`vx300s` 是受支持型号；URDF/Xacro 是机器人描述来源。
- Interbotix 官方运动学工具：[kinematics_from_description](https://github.com/Interbotix/kinematics_from_description)。该工具从 URDF 读取 `space_frame`、`body_frame` 和 screw axes，生成 product-of-exponentials 参数；它适合建立 FK/受限搜索基线，不等于物理仿真验证。

### Isaac Sim / PhysX

- NVIDIA Isaac Sim 5.1 physics fundamentals：[Physics Simulation Fundamentals](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/simulation_fundamentals.html)。官方明确 joint 必须配置 body0/body1 相对 joint frames；articulation 中 body0 应是 parent、body1 应是 child；USD 角度属性用 degree。文档还提供 residual reporting，可用于检查约束求解收敛。
- NVIDIA/Omni Physics Kit 107.3：[Articulations](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html)。PhysX articulation 使用 reduced coordinates，由 root body 与 joint angles 决定构型；articulation tree 由 joint 的 body0/body1 关系决定，而不是简单由 USD 层级决定；固定基 articulation 的 root API 可放在 world fixed joint 或其祖先；非 root link 不应直接写 pose/velocity。
- NVIDIA/Omni Physics Kit 107.3：[Articulation and Robot Simulation Stability Guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html)。官方列出 timestep、mimic compliance、max joint velocity、max drive force、drive stability、mass ratios、self-collision、solver order 等稳定性因素；指出 stiff drives、mimic 和 contact 组成竞争硬约束时可能造成大力/大速度；建议先减小 timestep 或增加 solver position iterations，并检查质量/惯量比。
- NVIDIA Isaac Sim articulation controller：[Articulation Controller](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_simulation/articulation_controller.html)。官方要求 simulation 开始后先 initialize articulation，再发送 action；action 的 joint indices 必须与 command 数量和顺序匹配；一个 joint 不能同时使用 position 与 torque 等不同 control method。
- NVIDIA PhysX 官方源码：[NVIDIA-Omniverse/PhysX](https://github.com/NVIDIA-Omniverse/PhysX)。这是 PhysX SDK 源码与 release 入口；本报告没有从 issue 或博客推断内部行为，而以官方 articulation/stability 文档为主。

## 已确认事实

### ALOHA/VX300S 运动学

1. VX300S 是 6DOF arm，不是 Franka 式 7DOF 冗余臂。官方规格给出的 reach 和 joint limits 只是几何/安全边界，不能把整个左半桌面宣称为可达域。
2. 六个 arm joint 是串联链：`waist → shoulder → elbow → forearm_roll → wrist_angle → wrist_rotate`。本地 URDF 与官方结构一致；本地 URDF 轴为 waist Z、shoulder Y、elbow Y、forearm_roll X、wrist_angle Y、wrist_rotate X。
3. `shoulder/elbow` 主要改变末端位置和肘形；`wrist_angle` 同时参与姿态和高度补偿；`forearm_roll/wrist_rotate` 主要是绕链轴的姿态自由度，不能当作任意位置补偿。
4. 关节限位和姿态要求必须同时检查。对平躺瓶子，瓶身轴线 AB 在桌面 XY；夹爪目标应在线构造，不能固定假设 AB 沿世界 X。若要求 approach 接近 −Z、opening 轴垂直 AB，部分瓶子 yaw/位置可能因肩肘限位或姿态约束不可达。
5. `kinematics_from_description` 的 POE/FK 输出可支持纯运动学验证；但官方 Interbotix/ALOHA 资料没有证明任意 pose 都有唯一或物理可执行解。

### Isaac articulation 稳定性

1. PhysX reduced-coordinate articulation 的构型由 root 与关节角决定；写非 root link pose 不是可靠初始化方式。初始化应在 articulation/root 与 joint state 层面完成。
2. body0/body1、joint localPos/localRot、axis 和 root 关系是结构核心；仅看 USD 层级或视觉位置不足以证明 articulation 正确。
3. NVIDIA stability guide 明确指出 stiff drive、mimic、contact、质量/惯量比、self-collision、solver order 和 timestep 都会影响稳定性。它没有授权用随意调 gains 掩盖错误 frame 或拓扑。
4. Isaac Sim 5.1 controller API 要求先 initialize articulation，再 apply action，并严格对齐 joint index/order；因此 target ownership 和 action mapping 必须单独审计。
5. 官方文档提供 articulation/joint residual reporting；残差是比“看起来没动”更有意义的约束收敛证据。

## 有来源的推断

1. 安全工程路径应把 FK/受限 pose solver 与 PhysX 稳定性拆开：先用 URDF/POE 验证瓶位和 yaw 的几何可达性，再用最小单关节/双刚体模型验证 importer 与 articulation semantics，最后才组合双臂场景。

## 待复核的本地工作记录

- `UNVERIFIED_LOCAL_NOTE`：工作会话记录称 fresh 首帧最大 qvel 约 `12.4632282257 rad/s`，并据临时 A/B 推测初始约束或投影状态可能不自洽。本次未找到独立原始 artifact，因此不能排除 drive、初始化顺序或其他原因。
- `UNVERIFIED_LOCAL_NOTE`：工作会话记录称离线一次性重建六个 joint 曾产生约 `18058 rad/s`，恢复原 schema 后回到 baseline。本次未找到独立原始 artifact，因此该记录只能提示重建流程风险，不能作为根因或修复证据。

## 工程建议

### 路线 A：立即继续（不依赖 physics）

建立 ALOHA 专用纯运动学工具链：

1. 从 resolved URDF 解析六个 joint 的 axis、origin、limit、effort、velocity；
2. 用 FK/POE 对 home 与每个候选 q 计算 gripper frame；
3. 对平躺瓶子在线构造 `approach=-world_Z`、`opening=normalize(cross(world_Z, AB))`；
4. 在左半桌面候选区域采样有限位置和 yaw；
5. 分类为 `reachable_and_pose_valid`、`ik_unreachable`、`joint_limit_violation`、`orientation_unreachable`，不把 IK 返回解直接当抓取通过；
6. 输出关节余量、姿态误差、桌面/底座/夹爪 base 间隙。

这条线完全不调用 policy/rollout，不执行 physics step，也不改正式 USD。

### 路线 B：单独定位 PhysX（禁止抓取）

1. 最小 1-joint、2-rigid-body revolute articulation；
2. 逐步加入 imported VX300S 的真实 body mass/inertia、joint frame、drive schema；
3. 每步做 fresh process、S0/S1、joint residual reporting；
4. 首个出现首帧速度或 residual 的 joint 才进入下一层；
5. 只在该层定位 frame/axis/root/schema 问题，不调 gains 作为遮蔽手段；
6. 稳定后再恢复六关节链，最后才合并 A23 场景。

### 暂停项

只要 physics articulation 仍为 `motion_anomaly=true`，就暂停 Grasp Editor dynamic、重力保持、lift、contact-pad 和水管测试。视觉/静态场景组织可以继续，但不得把视觉正确当物理正确。

## 未确认项

- 公开资料未给出当前这份 generated USD 在 Isaac Sim 5.1 中的具体 projection 首发 joint；需要本地最小 articulation 实验。
- NVIDIA 文档描述了 stability factors，但不能直接证明某一个因素就是当前异常的根因。
- Trossen 规格中的 reach/limits 不等于桌面上任意瓶位的可达性；仍需用本地 URDF FK 和碰撞/路径约束验证。
- ALOHA 原论文、Tony Zhao 仓库主要描述遥操作系统和任务，不是完整的 Isaac PhysX 参数规范。
- 当前工作会话中的两个 qvel 数值缺少可提交的独立原始 artifact；复测时必须保存 fresh-process 命令、Stage hash、S0/S1 状态和机器可读输出。

## 决策结论

当前推荐路线：**先走纯 URDF/FK/受限可达性链，同时以最小单关节 PhysX 模型独立定位 articulation 稳定性；暂停抓取动态。** 这是能继续推进、又不把错误首帧物理包装成抓取结论的最小安全边界。

**状态：**

```text
ALOHA_KINEMATICS_PUBLIC_EVIDENCE=CONFIRMED_PARTIAL
ISAAC_ARTICULATION_STABILITY_PUBLIC_EVIDENCE=CONFIRMED_GUIDANCE
CURRENT_PHYSICS_ARTICULATION=BLOCKED_PENDING_REPRODUCIBLE_RUNTIME_EVIDENCE
GRASP_EDITOR_ACCEPTANCE=NOT_RUN
GRAVITY_HOLD_ACCEPTANCE=NOT_RUN
LIFT_ACCEPTANCE=NOT_RUN
ROLLOUT_USED=false
```
