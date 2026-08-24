# Isaac Sim 远程开发环境

## 主机、连接方式与职责

### `mac`（当前计算机）

- 当前工作区：`/Users/mac/Documents/project/aruco-lab`。
- 运行 Isaac Sim WebRTC Streaming Client。
- 客户端连接到 `aloha` 上的 WebRTC Streaming Server，用于显示和操作远程 Isaac Sim 场景。
- 当前客户端通过本机 `127.0.0.1` 连接；本机 FRP 转发应监听 TCP `49100` 和 UDP `47998`。
- 从 Codex shell 启动 Electron 客户端时必须移除环境变量 `ELECTRON_RUN_AS_NODE`，例如：`env -u ELECTRON_RUN_AS_NODE open -n -a 'Isaac Sim WebRTC Streaming Client'`；否则客户端会立即退出。
- Mac 主要作为远程客户端；Isaac Sim 的实际进程和场景不在本机运行。

### `aloha`

- 从 Mac 使用 `ssh aloha` 连接。
- 运行 Isaac Sim 和 WebRTC Streaming Server。
- Isaac Sim 相关项目路径：`~/openpi0.5-rtc-reward-learning`，即 `/home/eii/openpi0.5-rtc-reward-learning`。
- 场景或远程运行状态的修改、验证和服务重启应在此主机上进行。

### `hxz`

- 从 Mac 使用 `ssh hxz` 连接。
- 保存项目源代码的原始版本。
- 原始项目路径：`~/project/openpi0.5-rtc-reward-learning`。
- 从 `hxz` 使用 `ssh eii@192.168.1.109` 可连接到 `109` 数据系统主机。

### `109` 数据系统主机

- 必须先连接 `hxz`，再从 `hxz` 执行 `ssh eii@192.168.1.109`。
- 采集、分割及标注数据的系统目录：`/home/eii/learn/eii-data-system-prod`。
- 数据采集网页：`https://ai.swm-eii.com`。
- 网页登录用户名：`hxz`。
- 网页登录密码：`123456`。

## 项目同步关系

- `aloha` 上使用的相关项目源代码由 `hxz` 的 `~/project/openpi0.5-rtc-reward-learning` 复制并同步而来。
- 修改或执行命令前，必须先确认目标是：
  - 当前 Mac 工作区；
  - `aloha` 上实际运行的 Isaac Sim 项目；或
  - `hxz` 上的源项目。
- 不要因为目录名称相似而默认三台计算机上的文件已经同步。
- 涉及当前运行场景时，优先检查 `aloha` 上的实际文件和进程，再决定是否需要同步回 `hxz`。

## 当前 Mac 工作区目录约束

- 项目根目录禁止再任意新建 `isaac_*`、`issac_*` 或 `server_runs` 一类目录来保存 Isaac Sim 截图、录像、JSON、CSV、服务日志、临时远程副本或运行证据；上述信息统一放入根目录 `issac_log/`。用户指定的目录名拼写为 `issac_log`，不要擅自改成 `isaac_log`。
- 2026-08-19 已将原根目录 `isaac_extensions`、`isaac_failure_captures`、`isaac_monitoring` 和 `server_runs` 整体迁移为 `issac_log/` 下的同名子目录。后续读取 Mac 本地归档时必须使用新路径，例如 `issac_log/isaac_failure_captures/...` 和 `issac_log/server_runs/...`。
- `remote_isaac_assets/` 是项目资源、Stage、loader 和运行配置，不是日志；必须保留在项目根目录，禁止移动到 `issac_log/`。
- 可直接在 Isaac Sim 中运行的独立 Python 脚本统一放入根目录 `isaac_script/`；不要把运行脚本散落到项目根目录或 `issac_log/`。
- 因 Isaac Sim 实际运行在 `aloha`，Mac 的 `isaac_script/` 应同步到 `aloha:/home/eii/openpi0.5-rtc-reward-learning/isaac_script/` 后再由 Script Editor 执行。脚本不得默认保存 Stage、重启服务、连接 ROS 或控制真实机器人。

## Isaac Sim 初始化基线（必须保持）

下次打开或重启场景时，必须恢复并验证本节中的机械臂 sleep 位姿和 Bottle 位姿，不要只加载模型而遗漏运行时初始化。

### 场景与初始化入口

- 当前运行的 BottleCap 阶段 1/2 诊断 Stage：`/home/eii/openpi0.5-rtc-reward-learning/remote_isaac_assets/aloha1_bottle_server/attempt1/remote_stream_cap_stage.usda`。
- 已生成但尚未切换为 Streaming Server 当前 Stage 的右旋螺纹 + PrepareUncap Stage：`/home/eii/openpi0.5-rtc-reward-learning/remote_isaac_assets/aloha1_bottle_server/attempt1/remote_stream_prepare_uncap_stage.usda`。
- 当前初始化 loader：`/home/eii/openpi0.5-rtc-reward-learning/remote_isaac_assets/aloha1_bottle_server/attempt1/tools/remote_stream_cap_stage_loader.py`。
- 用户服务 `isaac-sim-streaming.service` 的 drop-in 为 `/home/eii/.config/systemd/user/isaac-sim-streaming.service.d/10-writable-home.conf`。其 `ExecStart` 必须同时包含项目 loader 的 `--exec`、`--ext-folder /home/eii/Applications/isaacsim-5.1.0/extsUser` 和 `--enable aloha.lula_base_aligned`；否则 Stage 虽能初始化，但重启后自定义 Lula 面板不会自动加载。项目归档副本位于 `remote_isaac_assets/aloha1_bottle_server/attempt1/systemd/10-writable-home.conf`，SHA-256 为 `8a11b73e6de5f6a2c38d0ca706fb834ce9e8c57e2f69b8b37a1c4d8cd778f363`。
- 当前重启脚本：`/home/eii/openpi0.5-rtc-reward-learning/remote_isaac_assets/aloha1_bottle_server/attempt1/tools/restart_remote_isaac_stream_with_cap_stage.sh`。
- 原稳定 Bottle Stage 保留为 `remote_stream_grasp_stage.usda`，原 loader 和重启脚本分别为 `tools/remote_stream_stage_loader.py`、`tools/restart_remote_isaac_stream_with_stage.sh`；需要回退时使用这组文件，不要覆盖原 Stage。
- Stage 使用 Z-up、米制单位。

### 左右机械臂 sleep 位姿

- 左臂 articulation：`/World/follower_left/vx300s_left/root_joint`。
- 右臂 articulation：`/World/follower_right/vx300s_right/root_joint`。
- 两臂使用相同的前 6 关节 sleep 目标，单位为弧度。
- 关节顺序：`waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate`。
- sleep 目标：`[0.0, -1.8, 1.55, 0.0, -1.57, 0.0] rad`。
- 完整 DOF 顺序必须为：`waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper, left_finger, right_finger`。若顺序不一致，应停止初始化并报错，不要按错误索引写入角度。
- sleep 初始化只固定前 6 个手臂关节；`gripper`、`left_finger`、`right_finger` 不属于上述固定 sleep 向量，应保留场景初始化得到的值，不要擅自写入假定角度。
- 初始化时将关节速度归零，应用目标后暂停 timeline。
- 左右臂前 6 关节的最大回读误差必须不超过 `0.02 rad`。

### 左右机械臂运行时 IK Drive 基线

- 当前 ALOHA 机械臂关节使用 `acceleration` Drive。源 USD 中前六关节的 angular Drive 为 `stiffness=625`、`damping=0.1`；Isaac articulation API 以弧度制回读时约为 `stiffness=35809.863`、`damping=5.729578`，两者相差约 `57.2958` 是度/弧度单位换算，比较参数时必须同时注明 Drive 类型和单位。
- 源配置虽然数值看起来很大，但实际是高刚度、极低阻尼：按 acceleration Drive 近似，其固有频率约 `30.1 Hz`、阻尼比约 `0.015`。离线阶跃测试出现 `52–93` 次目标交叉和约 `73%–109%` 超调，因此不得继续把源参数作为 Lula IK 运行时基线。
- 参考 NVIDIA Isaac Lab 中 Franka、OpenArm、UR10/UR10e、Kinova 和 KUKA iiwa 的官方 actuator 配置，并对当前 ALOHA 同步六关节 `+Z 5 mm` 动作扫描后，选定前六个机械臂关节的运行时参数为：`drive_type=acceleration`、`stiffness=1600`、`damping=100`。该参数按 acceleration Drive 近似对应 `6.37 Hz`、阻尼比 `1.25`；它同时降低原运行时刚度并提高阻尼，不是仅靠增大 damping 压制振荡。
- 只对 `waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate` 应用上述参数；不得覆盖 `gripper`、`left_finger`、`right_finger`。左右臂还应关闭 articulation 自身重力，使控制行为等效于官方 Cortex `MotionCommandedRobot` 的机器人重力补偿；Bottle、BottleCap、桌面和其他环境刚体的重力不得关闭。
- `tools/remote_stream_cap_stage_loader.py` 会在每次服务启动时为左右臂应用并回读 `1600/100`，结果必须出现在启动报告的 `runtime_state.arm_runtime_profile`。项目扩展 `aloha.lula_base_aligned` 则在左臂每次 Load 或 Reset 时应用同一运行时参数。两处均使用 `save_to_usd=False`，当前 Stage 不得保存；完成更广泛路径和接触测试后，应把经审核的配置写入独立 USD 覆盖层。
- 2026-08-17 当前 Streaming Server 上的自动 `sleep -> Load -> Sync -> Validate -> +Z 5 mm` 验收已通过：共 `120` 个物理步；shoulder、elbow、wrist_angle 的超调均为 `0`、目标交叉均为 `0`，稳定时间分别为 `0.1500 s`、`0.1500 s`、`0.1667 s`，末端最终位置误差约 `0.000185 mm`、姿态误差约 `0.000011°`。该结果只证明左臂空载小位移 IK 响应通过，尚不等于抓瓶路径或接触动力学通过。
- 2026-08-17 又在同一 PID `61422` 上完成左臂空载 `+Z 20 mm` 录像复测，运行 ID 为 `aloha_left_pd_1600_100_z20_viewport_20260817_retest2`。shoulder、elbow 的关节位置全程单向到位；三项评分关节均为 `0` 次目标交叉，shoulder/elbow 超调为 `0`，wrist_angle 超调约 `0.59%`，最终关节误差为 `0`，末端最终位置/姿态误差约为 `0.000166 mm / 0.000025°`。9 帧服务器活动视口录像未见弹簧式往复或回弹，符合本轮人工“运动平滑、无可见振荡”的要求。
- ALOHA 默认控制频率按 `50 Hz` 配置，即一个控制周期为 `20 ms`。当前 `tools/remote_stream_cap_stage_loader.py` 已明确设置 `physics_dt=1/50 s`、`rendering_dt=1/50 s`，`hxz` 源项目与 `aloha` 运行 bundle 已核对一致；该配置将在下一次自然启动时生效。这里的 `50 Hz` 首先是机器人命令/策略更新频率，不应与 WebRTC 客户端实际显示 FPS 混为一谈。若未来为了接触稳定性使用高于 `50 Hz` 的 PhysX substeps，则必须在多个 substeps 之间保持同一控制命令，只在 `50 Hz` 控制边界更新 IK/关节目标。
- 上述两次 `+Z` 自动测试是修改控制调度前的历史证据：当时扩展在每个 physics step 更新 IK，日志 `physics_dt` 约为 `1/240 s`，因此不能把它们描述成已经按 ALOHA `50 Hz` 控制频率通过的测试。录像与关节曲线仍能证明 `1600/100` 没有可见往复振荡，但后续必须在真正的 `50 Hz` 命令更新下重新验收。
- `0.25 s` 是旧扩展中的硬编码秒数门槛，不是有效的 `50 Hz` 帧门槛：`0.25 s * 50 Hz = 12.5` 个控制周期，无法对应整数控制帧。上述 `+Z 20 mm` 历史结果仍保持原文件的 `FAILED_GATE`，不得事后改写；在按 `50 Hz` 重新验收前，应先明确整数稳定周期数 `N`，再使用 `N / 50 s` 作为门槛，并同时记录 `control_hz=50`、控制周期编号和实际 physics substeps。未预先确定 `N` 前，不得用新的时间阈值追认旧结果为自动 PASS。
- 自动验收结果保存在 Mac `issac_log/isaac_failure_captures/aloha_left_pd_tuning_20260817/auto_z5_acceptance_result.json`（SHA-256 `39eef760d0a42e653fcde74781ece34729fb0448f0f7b154ae0fa7d73d98586a`），关节 CSV 为同目录 `aloha_lula_joint_response_20260817_193136.csv`（SHA-256 `e30cb2466eb4c92d867eea40cf7a1ce5c2568fd23ba4760c069158b2ea3dd07c`），离线多关节候选汇总为 `offline_multi_joint_damping_summary.json`（SHA-256 `5a656b53d5bd7bf55033ae9d197aa8d306ecb2d9bf286b6dc206ffcab99e6a33`）。自动扫描脚本为 `tools/auto_tune_aloha_left_arm_damping.py`（SHA-256 `486a6cf201c297f66adfa26ff788facd29fd98e68dcdb410e0b4fde911877a8e`）。
- `+Z 20 mm` 复测证据位于 Mac `issac_log/isaac_failure_captures/aloha_left_pd_tuning_20260817/retest2/`：结果 JSON `auto_z20_acceptance_result.json`（SHA-256 `6b502ae269f8e98684f8440b94739255fe434ccbe0472e7855391cc028c9280d`）、关节 CSV `aloha_lula_joint_response_20260817_221350.csv`（SHA-256 `e15404aeadfc464e848292a99ce6c6dd48fade7662a5a9442993d41245ff1ff0`）、录像 `aloha_left_z20_retest2.mp4`（SHA-256 `d4ffa7b1b915dfff827bb2fbd279a68c4246c9547c0cc48fda626d074d7def40`）和九宫格 `contact_sheet_3x3.png`（SHA-256 `b2af7c7b5788c25afd2df41175cc552858b1ce25fd072b53738d63886117f7e7`）。录像来自 Isaac Sim Server 的活动 viewport，采集过程中没有切换 Mac 桌面。
- 2026-08-18 已通过用户服务完成干净启动并验证 loader 报告为 `PASS`；未启用 ROS，也没有触碰真实机器人。启动报告已同时回读左右臂 `1600/100`，但本轮 HOVER 动态轨迹仍只验收左臂，不能据此宣称右臂动态轨迹已通过。

### 左右夹爪运行时控制基线

- 本节是当前 Streaming Server 的有效运行时基线，优先于下文早期实验中记录的旧 drive/mimic 参数；左右两个 follower 夹爪均已完成人工空载全行程验收。
- 对每个 `side in {left,right}`，主动关节为 `/World/follower_{side}/vx300s_{side}/joints/left_finger`。其 linear Drive 使用 `type=acceleration`、`maxForce=5 N`、`stiffness=200`、`damping=50`。
- 对每个 `side in {left,right}`，被动关节为 `/World/follower_{side}/vx300s_{side}/joints/right_finger`。它保持 `PhysxMimicJointAPI:rotY`，reference joint 必须为同一 follower 的上述 `left_finger`；运行时参数为 `naturalFrequency=25 Hz`、`dampingRatio=1.0`、`gearing=1.0`、`offset=0`。
- 控制语义仍是“一台主动执行器 + 单标量 `left_finger` 命令 + `right_finger` 被动 Mimic 跟随”。禁止给 `right_finger` 增加独立 Drive，禁止把 `gearing` 改为 `-1`。
- 2026-08-17 人工空载全行程测试确认左右夹爪均符合开合运动要求。此前左夹爪出现的明显弹簧式往复振荡，是因为主动 `left_finger` 的 `damping` 未实际改为 `50`、仍为源资产的 `0.1`；补齐 `damping=50` 后振荡消失。左右侧测试均应使用 `Drive Target Position`，不能用直接写入 `Joint State Position` 的结果替代 Drive 验收。
- 已验收的双侧夹爪参数已写入独立、可复用的 USD 覆盖层：`assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/configuration/aloha1_human_accepted_gripper_control.usda`，SHA-256 为 `b0b1e9aaca78bef9e22e09dbe03119d680a9a8aa456e74a81ee50f2baaf0e965`。该层同时存在于 `hxz` 源资产和 `aloha` 远程 bundle 中，并作为最强子层加入 `aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_z_up_meters_diagnostic.usda`；更新后的顶层 SHA-256 为 `a68dd722d1451398652115a09cfff718962563a9bed3f3fd74292c454a387c14`。
- 当前 loader `tools/remote_stream_cap_stage_loader.py` 对夹爪 Drive、Mimic、引用关系和左右侧一致性仍只读验证 USD；它只对前六个机械臂关节应用上述运行时 IK Drive 参数。此外，loader 会在第一次 PhysX Play 前为左臂 articulation、左右 finger rigid body 和 Bottle 配置运行时 Contact Report API，并严格回读 GripperPad 的用户确认有效摩擦配置 `static=2.0`、`dynamic=1.5`、combine=`maximum`。夹爪验证结果位于 `runtime_state.gripper_usd_control.{left,right}`，摩擦回读位于 `runtime_state.gripper_material_profile`，接触报告路径位于 `runtime_state.contact_report_prim_paths`。当前 loader SHA-256 为 `fb435f1566dfed7c140d05a4c3aaffb343f7780182bc8dac926aaccd7b663d67`。启动物体位姿比较的位置容差为 `1 µm`，用于容纳 float32 Fabric/PhysX 回读噪声；不得恢复为不现实的 `1 nm` 门槛。
- 2026-08-18 干净启动后，`hxz`/`aloha` 的离线 USD 组合和 `aloha` 运行时 loader 报告均已通过；左右夹爪 USD 配置、Mimic 引用和启动 sleep 回读已由同一报告验证。
- USD 层中的 `200/50/5/acceleration` 与 Mimic `25/1/1/0` 是人工空载全行程验收后的仿真配置，不是 DYNAMIXEL 寄存器值，也不是已完成的 sim-to-real 标定结果。除非重新走证据和验收流程，不得改回源资产旧值或在 loader 中恢复重复覆盖。

### Grasp Editor 5.1 临时兼容补丁

- Isaac Sim 5.1 Grasp Editor 原实现从内部 `SingleArticulation.prim` 遍历 Gripper Frame；当前 ALOHA1 的 ArticulationRoot prim 为 link 层级的同级 joint，导致 `follower_left_ee_gripper_link` 不出现在下拉框。补丁改为从用户在 `Select Gripper` 中选择的 `/World/follower_left/vx300s_left` 开始遍历；不得通过移动 USD 中的 `ee_gripper_link` 来迎合该 UI。
- `aloha` 当前安装文件已经打补丁，普通机器或 Isaac 重启后仍保留；Isaac 升级或重装可能覆盖它。原始 SHA-256 为 `bd48e3c3c0e5587d9f790e86a5b1b843aa4b64cde26420af09f5f1f4dbf29823`，补丁后为 `1258241849257f2da3e82a20741785fc8eb9e392e9bdb134d991317e533901dc`。
- 补丁、README、原始文件和补丁后文件已归档到 `remote_isaac_assets/aloha1_bottle_server/attempt1/vendor_patches/isaacsim_grasp_editor_5_1/`，并同时保存在 `hxz` 与 `aloha` 项目目录中。不要依赖 `/tmp` 备份作为长期恢复来源。
- 此补丁不是 Isaac 启动参数；当前 Grasp Editor 会话仍保持打开。未点击 `Simulate Grasp`，未保存当前 Stage。
- 在正式写入 USD 前，不要在 Isaac Sim UI 中保存包含手工参数改动的当前 Stage；运行时基线应由 loader 可重复重建。

### ALOHA Lula 基座对齐扩展

- Isaac Sim 5.1 原生 `Lula Test Widget` 的 Kinematics 流程不会为当前非世界原点安装的左臂调用 `LulaKinematicsSolver.set_robot_base_pose()`；其 `/Lula/end_effector` 可视化因此不能作为当前 ALOHA1 世界位姿依据。
- 项目本地扩展 ID：`aloha.lula_base_aligned`。Mac 开发归档位于 `issac_log/isaac_extensions/aloha.lula_base_aligned/`；`hxz` 源项目和 `aloha` 运行 bundle 均同步到 `remote_isaac_assets/aloha1_bottle_server/attempt1/exts/aloha.lula_base_aligned/`。`aloha` 的启动搜索副本位于 `/home/eii/Applications/isaacsim-5.1.0/extsUser/aloha.lula_base_aligned/`，用户服务通过 `--ext-folder .../extsUser --enable aloha.lula_base_aligned` 自动启用它。更新扩展时必须同时同步项目 bundle 与该启动副本并核对 SHA-256；不得改写 NVIDIA 的 `exts/` 或 `extscache/`。
- 扩展从 `/World/follower_left/vx300s_left/follower_left_base_link` 读取世界位姿，并在每次 FK/IK 前同步 Lula base pose；末端 frame 固定为 `follower_left_ee_gripper_link`，articulation 固定为 `/World/follower_left/vx300s_left/root_joint`。
- 扩展默认 `SAFE IDLE`，不会自动控制机械臂。必须依次人工执行 `Load Left Arm`、`Sync Base Pose`、`Validate EE Alignment`、`Create Target At Current EE` 和 `Enable IK Follow`。位置误差门槛为 `1 mm`，姿态误差门槛为 `0.5°`。
- 扩展提供显式 `Reset Left Arm to Sleep (Paused)`，但加载或重载扩展本身绝不会无条件复位。点击该按钮会原子地暂停 Timeline、关闭 IK Follow、删除残留 `/World/ALOHAAlignedIKTarget`、清除 HOVER plan，把左臂前六关节及其 Drive target 设为 `[0.0, -1.8, 1.55, 0.0, -1.57, 0.0] rad`、清零速度并保留三个夹爪 DOF；随后短暂运行 30 个 physics update 以发布可见姿态并稳定 Drive，最终自动回到 Paused。只有正在记录且尚未保存的关节日志仍会阻止复位。最大回读误差必须不超过 `0.02 rad`，随后仍须重新执行 Load、Sync 和 Validate。
- IK 控制只允许六个臂关节 `waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate`；运行时若 c-space 或 DOF 顺序不匹配必须拒绝发送。禁止控制 `gripper`、`left_finger` 或 `right_finger`。
- 通用在线 IK 每个控制更新的目标位姿增量限制为 `5 mm / 2°`，六关节命令增量限制为 `0.02 rad`。已预检 HOVER 使用另一条确定性路径：按 `50 Hz`（`20 ms`）发送预先采样为不超过 `0.01 rad` 的关节参考，不再逐采样等待实际关节进入 `0.006 rad` 门槛，也不再在每个运动中的 physics substep 重复移动帧对齐门禁。静态起终点仍须通过 Lula/USD 对齐和 FK 验证，异常或 Bottle 明显扰动仍会失败并暂停。该扩展只是 IK 诊断，不提供碰撞规避；不得把它当作已经验证的完整抓取路径或据此连接 PrepareUncap Action Graph。
- 当前扩展源码 SHA-256 为 `134055837fc73547b4fa5eae369c8d9bed341b23d1dc99ed6c5135e83694f077`。扩展诊断日志同时记录 UI 原始 Target、每步限幅 Target、actual position、velocity、raw IK requested position 和 bounded joint position target，并增加累计 physics time；面板有常驻纵向滚动条，输出目录为 `reports/lula_joint_diagnostics/`。初始 Load/Sync/Validate 的严格门槛仍为 `1 mm / 0.5°`。扩展已加入一次性自动验收请求流程、Bottle 组件中点旋转请求流程、螺纹基础 Joint/右旋耦合创建请求流程、`THREADED → RELEASED` 测试请求流程和服务器活动 viewport 帧采集；基础 Joint 请求在干净 Stage 中若发现运行时 `BottleThreadSlider` 不存在，会用 BottleCap 的同父级局部 `translate + orient` 创建无碰撞 Dynamic Rigid Body 代理，再创建并验证两个基础 Joint，不能再依赖上一会话未保存的 Slider。服务启动时若扩展先于 `--exec` loader 初始化，会等待 articulation 与 Bottle Prim 出现，不能把启动顺序误报为 IK 无解。窗口会在 Kit 恢复 workspace 后的下一帧自动显示、聚焦并优先停靠左下；`Tools > ALOHA Lula Base Aligned` 使用弱引用回调切换可见性，避免热重载后旧菜单项持有失效扩展对象。面板内容宽度必须受当前 dock 宽度约束：根 `VStack`、输入框和主要按钮使用 `ui.Fraction(1)`；Guided target 的六个长步骤采用单列全宽按钮，禁止恢复成双列长标签 `HStack`，否则窄侧栏下按钮文字和命中区域会被挤到可视区外而无法点击。
- `reset_sleep_button_fix_20260818` 自动复测为 `PASS`：sleep 六关节回读为约 `[0,-1.79999995,1.54999995,0,-1.57000005,0] rad`，最大误差 `5.25e-8 rad`；Target 已删除，Timeline 最终 Paused，夹爪未命令，Stage 未保存，真实机器人未触碰。服务器结果为 `reports/lula_joint_diagnostics/auto_reset_sleep_result.json`；Mac 归档 SHA-256 为 `e267ada18b65db656241d1ae36c60e570e886a3ec048039d4799553e392b8d68`。
- 2026-08-18 面板已重组为四阶段操作向导：`Prepare robot -> Align and validate frames -> Define Bottle grasp -> Guided target sequence`，诊断和高级配置默认折叠；顶部常驻 `ABORT: Pause and Hold Current EE`。Bottle grasp 从 `configs/aloha1_grasps/bottle500_horizontal_body_grasp_isolated_20260817.yaml` 读取，严格验证 `O=/World/ALOHA1RemoteBottleSession/Bottle500`、`G=/World/follower_left/vx300s_left/follower_left_ee_gripper_link` 以及唯一主动夹爪 DOF `left_finger`，再显式应用已审计的 Bottle 局部修正 `(-5.5,-1.5,-10.0) mm`。
- 橘黄色 `/World/ALOHAAlignedIKTarget` 定义为供操作者理解的抓取工具坐标系，不再直接冒充 Lula/URDF 原生 EE 坐标系。ALOHA URDF 中原生 EE `+X` 从腕部指向夹指，顶抓时应沿世界 `-Z`；橘色 Target 则要求 `+Z` 与世界 `+Z` 完全一致。两者原点相同，固定姿态关系为 `R_W_EE = R_W_Target * RotY(+90°)`。扩展显示和读取 Target 时执行该双向变换，Lula 始终收到原生 EE 姿态；不得为了让橘色块看起来竖直而直接把同一个四元数送入 Lula。
- 旧 `HOVER +80 mm` 在纠正顶抓姿态后已由服务器 viewport 录像证实会让两根夹指进入 Bottle 周围/接触区，不能再视为 HOVER。当前 `PLAN VERTICAL HOVER (Bottle bottom L/3)` 在 Paused 状态一次求解终点 IK，并对关节插值路线执行无运动 FK 预检；持久化默认抓取轴向站位是从瓶底起 `L_bot/3 = 68.667 mm`，默认垂直 HOVER clearance 为 `160 mm`。`SAFE PREGRASP +120 mm` 已通过无接触下降验收；`NEAR +10 mm`、`GRASP POSE` 和接触边界尚未重新标定，必须从 `+120 mm` 以 `5 mm` 递减实验，不得直接跳转。
- 原 `corrected_top_down_hover180_50hz_20260818` 只验证了 EE 运动和夹指朝下，橘色块仍直接显示 EE 坐标，现已由下面的新验收取代，旧证据只保留用于回归比较。
- `world_z_aligned_target_hover180_50hz_20260818` 自动验收为 `PASS`：橘色 Target `+Z · World +Z = 1.0`，Lula 目标 EE 四元数为 `(0.70710678,0,0.70710678,0)`，即原生 EE `+X` 精确朝世界 `-Z`；320 个 50 Hz 关节参考样本，最小 EE 高度 `0.187275 m`，最大相邻 FK 步长 `3.169 mm`。最终目标位置误差 `0.0346 mm`、姿态误差约 `0.0356°`，Bottle 平移 `0.445 mm`；1712 个 physics samples 中六关节最终误差均小于 `0.000071 rad`，录像未见持续震荡。未命令夹爪、未保存 Stage、未连接真实机器人，Timeline 最终保持 Paused，Isaac Sim Server 未重启。Mac 证据位于 `issac_log/isaac_failure_captures/world_z_aligned_target_hover180_50hz_20260818/`，服务器结果位于 `reports/lula_joint_diagnostics/auto_hover_acceptance_result.json`。
- `plus_x_bottom_third_hover_clean_start_50hz_20260818` 是当前 `+X` Bottle 启动基线的权威 HOVER 验收，状态为 `PASS`：精确选择瓶底起 `1/3`（`68.667 mm`）、clearance `160 mm`、Target yaw `0°`，橘色 Target `+Z · World +Z = 1.0`；309 个 50 Hz 关节参考样本，最大相邻 FK 步长 `2.926 mm`。最终目标位置误差 `0.0043 mm`、姿态误差约 `0.0509°`，Bottle 平移和姿态变化均为 `0`。未命令夹爪、未保存 Stage、未连接 ROS/真实机器人，Timeline 最终保持 Paused。Mac 证据目录为 `issac_log/isaac_failure_captures/plus_x_bottom_third_hover_clean_start_50hz_20260818/`；结果 JSON、关节 CSV 和 50 Hz MP4 的 SHA-256 分别为 `0e8ed9ba4255cefa023b517d4e17cd7518acc629db9ab63e4e46e7cc034ff165`、`41871fca9189a249d5e033a222004a1dccf2c540417436ba3c8a47fb7290cbad`、`88cff4f2adfc3e498604cf4d9c9892541dbb75b893e97a4d261a2172951e7f6b`。
- 早先 `plus_x_bottom_third_ik_audit_20260818` 的“无解”结论已作废。诊断时 Bottle 世界位置实际已漂移到约 `(-5.689, 7.450, -0.344) m`：原因是运行中只旋转/平移父级 Bottle Xform，而其子级 kinematic rigid bodies 仍保留旧 PhysX 状态，重新 Play 后物理回写把组件推飞。禁止在活动物理 Stage 中只改该父级 Xform；需要改变持久姿态时必须编辑 USD 后干净重载，临时变换则必须原子同步整个刚体层级。
- `PREGRASP +30 mm` 旧按钮的第一次在线执行在 Play 后首个 `5 mm` 子目标即因 Lula 在线 IK 不收敛而安全停止；这不是终点无解。改成 Paused 终点求解和整路预检后，三个 warm start 均得到精确 `+30 mm` 解，但实际执行在物理接触处停住：shoulder/elbow/wrist_angle 的稳态目标残差约 `0.023/0.047/0.064 rad`。因此旧 `+30 mm` 已判定为接触区，禁止通过延长超时、提高 Drive 或放宽到达门槛硬压。
- `hover_to_safe_pregrasp120_preplanned_50hz_20260818` 自动验收为 `PASS`：从 `+160 mm` HOVER 到 `+120 mm` 安全 PREGRASP 使用 14 个 50 Hz 关节样本，最大相邻 FK 步长 `3.097 mm`、相对直线最大横向偏差 `0.590 mm`；终点误差 `0.0040 mm / 0.081°`，Bottle 位姿变化为零。Mac 证据位于 `issac_log/isaac_failure_captures/hover_to_safe_pregrasp120_preplanned_50hz_20260818/`，结果 JSON、关节 CSV、MP4 的 SHA-256 分别为 `04a79f4e0e36ba48d9d412cd2675de0409d04c44a6415b026a3ceac1adebefd6`、`8c22b38f454e6b9efa834a3d5b80089e746fc685fe4c18f4526f31fd0e48f5b7`、`ee10b5250cd98582548caed357c59ae41986b79ffc84b884b645157e6ec0ad68`。
- 左夹爪必须在安全 PREGRASP 下降前打开。面板 `5. Gripper preparation > 5A. OPEN LEFT GRIPPER (0.057 m)` 只命令主动 `left_finger`；`right_finger` 必须保持 Mimic，禁止独立发送目标。`open_left_gripper_safe_pregrasp120_20260818` 自动复测为 `PASS`：left 从 `0.021437` 到 `0.056886 m`，right 由 Mimic 从 `-0.021431` 到 `-0.056964 m`，六个手臂关节最大变化 `5.23e-6 rad`，Timeline 最终 Paused。结果归档为 `remote_isaac_assets/aloha1_bottle_server/attempt1/reports/lula_joint_diagnostics/auto_open_left_gripper_result.json`，SHA-256 `1c035053340847baaae4743b0d9c345a5d8e4a3c6f702d99ad8ed5543fa2ce7c`。
- 面板新增 `6. Grasp contact calibration`，仅在 Timeline 为 Paused、IK 已就绪且当前 waypoint 为 `APPROACH +0 mm / REACHED` 时允许操作。`6A. CLOSE 1 mm` 和 `6B. OPEN 1 mm` 只向主动 `left_finger` 发送 `1 mm` 小步进目标；`right_finger` 始终由原生 Mimic 跟随。`6C. AUTO CLOSE UNTIL BILATERAL CONTACT` 每次闭合 `1 mm`，连续 `5` 个 physics step 检测到左右指与 Bottle 的双侧接触即停；检测到非手指 Bottle 接触、Mimic 残差连续 `5` 步超过 `2.5 mm`、达到 `0.021 m` 下限或人工按下 `6D. ABORT GRIPPER MOTION` 时安全停止。状态行必须同时显示 commanded target、left/right actual、Mimic residual、left/right Bottle contact 和 non-finger contact；不得以“关节到达目标”代替“双侧物理接触”验收。Contact Report API 必须在第一次 PhysX Play 之前由 loader 同时应用到左臂 articulation、左右 finger rigid body 和 Bottle 根 Prim；运行中才动态添加可能漏报已存在的接触。该运行时配置不应保存 Stage。
- 第 6 节的 `6A/6B/6C` 在没有夹爪任务运行时保持可点击，安全门槛改为在回调内部权威校验；面板常驻显示 `READY/BLOCKED/BUSY` 及具体缺失门槛。每次点击先立即写入 `6A received`、`6B received` 或 `6C received`，随后才可能发送夹爪命令，避免“按钮被静默禁用”和“点击事件未到达”无法区分。安全门槛本身没有放宽，忙碌期间仍禁用新命令且只允许 `6D` 中止。
- 2026-08-19 自动准备与动态自定心门控均已通过。自动准备不再用 `1 mm` 相对目标反复逼近：在当前竖直腕姿下，有限刚度和负载会产生约 `0.77 mm` 静态跟随误差，使小步目标在到达 Bottle 前停住；自动流程改为一次命令经验证的闭合目标 `0.021 m`，Bottle 保持 Kinematic，并在首个稳定 finger 接触处暂停。准备结果为 `left=yes`、`right=no`、非指接触为 false、Mimic 残差 `0.133 mm`。
- 随后的动态门控将 Bottle 从 Kinematic 切为 Dynamic，并保持主动指闭合目标 `0.021 m`；Bottle 沿蓝指到橙指方向移动 `8.863 mm`，总位移 `9.084 mm`，最终双侧接触为 true、非指接触为 false，Timeline 最终 Paused。动态刚体位姿必须通过初始化后的 `SingleRigidPrim.get_world_pose()` 从 PhysX tensor view 读取；普通 USD `get_world_pose()` 以及当前 Stage 中的 `fabric=True` 查询都会继续返回 authored 启动位姿并错误报告零位移。
- 本次 loader SHA-256 为 `6c1e22834ea792bfc827f339546646e841106d0de63afb0cdf6b95c231274dc8`。Mac 权威证据目录为 `issac_log/server_runs/dynamic_self_center_attempt3_pass/`：动态结果、准备结果、关节 CSV 和 loader 报告的 SHA-256 分别为 `69f6428857ba686199a65439a4daea0d742d0ff7e29f4059b255241fdf56ae6e`、`367c3e30ca233fd321ae053d3b0fb0f0524879b8a2d0e694dc1320fc575f5f31`、`976637a42bc87a7d0663f8d0089172869acad6c02488837d4cb742bf657d4337`、`7b74b56861bb575a0cee8cc3bafbf31a6430be1f171d7c7c96b729a41c63ab55`。本轮仅同步到 `aloha` bundle 与 `extsUser`；`hxz` 当前网络不可达，尚未同步，恢复后必须补做且不得宣称三端一致。
- 2026-08-18 源码已同步至 Mac、`hxz` bundle、`aloha` bundle 和 `aloha` `extsUser`，三端扩展文件 SHA-256 均应为 `d19de0e439e97564cda48fa1c3057d6944f782f3f0a2fbae3d10b6041ba48096`。当次运行曾因服务器 inotify 观察器耗尽而无法自动热重载；释放外部文件观察器后，已在 Extensions 中禁用并重新启用 `aloha.lula_base_aligned`，第 6 节已加载，无需重启 Isaac Sim。若后续仍提示无法建立 change watch，不要据此否定已加载的功能；修改源码后仍需显式切换扩展或在维护窗口处理 inotify 上限。

### aloha Codex Isaac Sim 监控

- `aloha` 已安装官方 Codex CLI `0.147.0` 到 `/home/eii/.local/bin/codex`，并通过现有 ChatGPT 登录缓存完成认证；`~/.codex/auth.json` 属于凭据文件，禁止读取内容、提交到仓库或复制到报告。
- 用户级 systemd timer `aloha-codex-isaac-monitor.timer` 已启用，每 2 分钟运行一次 `aloha-codex-isaac-monitor.service`。监控脚本位于 `remote_isaac_assets/aloha1_bottle_server/attempt1/tools/codex_isaac_monitor.py`，不使用 inotify，只轮询 Kit 进程、`49100/8006` 端口、HTTP health/streaming readiness、Kit 新增错误日志和 inotify 使用率。
- 普通健康轮询只写 JSON 快照；仅首次基线、健康状态变化或发现新增匹配错误时调用只读、ephemeral 的 `codex exec`，非紧急诊断有 15 分钟冷却。Codex 监控不得修改文件、重启或 signal 进程、控制机械臂、保存 Stage 或输出凭据；任何修复都必须回到人工审批。
- 最新状态为 `remote_isaac_assets/aloha1_bottle_server/attempt1/reports/codex_monitor/latest_status.json`，Codex Markdown 诊断也写入该目录。2026-08-18 首次基线报告判定 Isaac Sim `healthy`：Kit PID `61422`、WebRTC `49100`、服务 `8006` 和 HTTP health 均正常，streaming session active，inotify 为约 `3602/65536`。
- 查看状态：`systemctl --user status aloha-codex-isaac-monitor.timer`；立即执行一次轻量轮询：`systemctl --user start aloha-codex-isaac-monitor.service`；查看日志：`journalctl --user -u aloha-codex-isaac-monitor.service -n 100 --no-pager`。不要为了查看报告重启 Isaac Sim。

### Bottle 位姿

- Bottle Prim：`/World/ALOHA1RemoteBottleSession/Bottle500`。
- Bottle 运行时设为 kinematic rigid body。
- Bottle 资产沿本地 Z 轴的长度：`L_bot ≈ 0.206 m`，几何范围为 `0` 到 `0.206 m`。
- 启动基线平移位置：`(-L_bot/2, 0, 0.034)`，值为 `(-0.103, 0, 0.034)`。
- USD 姿态四元数采用 `(w, x, y, z)` 顺序，启动基线值为 `(0.70710677, 0, 0.70710677, 0)`。
- `xformOpOrder`：`["xformOp:translate", "xformOp:orient"]`。
- 启动基线瓶口沿世界 X 轴正方向（`+X`）；在该朝向下，`x = -L_bot/2` 使瓶身几何中心在桌面 XY 平面上的投影位于桌面中心 `(0, 0)`。
- 该 `+X` 姿态已写入根 Stage，而非仅存在于匿名运行层。包含用户确认 GripperPad 摩擦配置后的 `remote_stream_cap_stage.usda` SHA-256 为 `41af2047922e10cc415c84bf57a2f1965ede9fb3f9e58cef27618d1bb9a6874b`；每次服务启动必须由 loader 回读 `bottle_mouth_world_axis=+X`、`bottle_center_world_m=[0,0,0.034]` 和上述精确位姿。除非用户明确要求改变任务基线，不要再次施加同样的 180° 旋转。

### BottleCap 阶段 1/2 诊断资产

- 当前 Streaming Server 的根 Stage 仍是阶段 1/2 文件，但 2026-08-19 已在未保存的活动运行层中创建后述三个螺纹 Joint；这不等于根 Stage 文件已经永久包含螺旋约束，Action Graph 仍未加载或连接。
- `hxz` 源资产目录：`/home/eii/project/openpi0.5-rtc-reward-learning/assets/bottle_cap/1.0`。
- `aloha` 已同步的新资产目录：`/home/eii/openpi0.5-rtc-reward-learning/assets/bottle_cap/1.0`；原阶段 1/2 的 bundle 内资产仍位于 `remote_isaac_assets/aloha1_bottle_server/attempt1/assets/bottle_cap/1.0`。
- BottleCap Prim：`/World/ALOHA1RemoteBottleSession/BottleCap`。
- BottleCap 启动基线平移：`(0.085, 0, 0.034) m`；姿态四元数 `(w, x, y, z)`：`(0.70710677, 0, 0.70710677, 0)`。该位姿使瓶盖本地 `+Z` 轴沿世界 `+X`，与启动基线 Bottle 瓶口同轴；该姿态也已写入根 Stage 并由 loader 每次启动回读。
- BottleCap 外径 `34.0 mm`、内径 `30.8 mm`、高度 `22.0 mm`；瓶盖基面对应 Bottle 本地 `z = 0.188 m`，顶面对应 `z = 0.210 m`，Bottle 瓶口位于本地 `z = 0.206 m`。
- BottleCap 质量暂设为 `0.004 kg`，碰撞体由 16 个侧壁 box 和 1 个顶部 cylinder 组成。
- BottleCap 使用深蓝色半哑光塑料主体材质，并带 clearcoat；外侧有 32 条亮蓝色竖向防滑纹，顶面有亮蓝色高光圆片，用于在远程画面中快速区分瓶盖与瓶身。
- 防滑纹和顶面高光仅为可视几何，不应用 CollisionAPI，不改变 17 个物理碰撞体。物理高度仍为 `22.0 mm`；顶面高光使可视范围增加到 `22.25 mm`，即 Bottle 本地 `z = 0.21025 m`。
- BottleCap 在可复用 USD 资产中是动态 rigid body；当前诊断 Stage 因尚无螺纹约束，运行时将 Bottle 和 BottleCap 都保持为 kinematic，并对瓶盖关闭 CCD。不要把这一临时固定误认为已经实现可拧下瓶盖的动力学。
- 外径来自 Bottle CAD 的 `support_OD = 34 mm`，内径来自 `thread_OD = 30 mm` 加每侧 `0.4 mm` 间隙；高度、顶厚、质量和摩擦系数均为 `TEMPORARY_UNCALIBRATED`，后续需要实测或标定。

### 临时摩擦材质

- Bottle：静摩擦 `0.65`、动摩擦 `0.50`。
- BottleCap：静摩擦 `0.90`、动摩擦 `0.75`。
- 桌面：静摩擦 `0.60`、动摩擦 `0.50`。
- 夹爪接触面：静摩擦 `2.0`、动摩擦 `1.5`、friction combine mode 为 `maximum`。这是用户在 `15 N` 夹爪驱动下确认能顺利抬起 Bottle 后明确选择停止进一步阈值扫描并采用的参数。
- 四类材质的 restitution 均为 `0`；Bottle、BottleCap 和桌面的 friction combine mode 仍为 `average`，GripperPad 为 `maximum`；restitution combine mode 均为 `min`。
- GripperPad 参数分类必须写为 `USER_ACCEPTED_SIMULATION_EFFECTIVE_FRICTION`，验证结果为 `BOTTLE_LIFT_SUCCEEDED`；不得描述为真实材料摩擦系数标定。其余物体摩擦参数仍是诊断初值。当前 loader 的绑定计数应为 Bottle `41`、BottleCap `17`、夹爪 `4`、桌面 `1`。

### BottleCap 测试证据

- FreeCAD 两次新鲜生成的 OBJ SHA-256 均为 `c2d6c67ce553e873d0af50b33a08a2b6152a4f3b802d679e22c279462329d747`，几何指标和 572 个三角面完全一致。
- BottleCap USD SHA-256：`14a1d0b77e37c97620a1dfdec9154c51cb7eaabe4bf0ed92be2391470c0725f2`。
- 当前持久化诊断 Stage SHA-256：`41af2047922e10cc415c84bf57a2f1965ede9fb3f9e58cef27618d1bb9a6874b`。
- 原稳定 Bottle Stage SHA-256 仍为 `d7da82416f7ac90a681d8fabe74c8402155670ce75b33c7bbbfaa84f4e90b9f8`。
- `hxz` 的 FreeCAD、静态 USD（含 32 条可视防滑纹与顶面高光）、Bottle+Cap 组合和 Isaac Sim 运行时加载报告均为 `PASS`，位于 `assets/bottle_cap/1.0/reports/`。

## 右旋螺纹等效约束

- 约束层：`assets/bottle_cap/1.0/usd/physics/bottle_cap_right_hand_helical_constraint.usda`。
- 实现由 Bottle 到隐藏 `BottleThreadSlider` 的 Prismatic Joint、Slider 到 BottleCap 的 Revolute Joint，以及 `PhysxPhysicsRackAndPinionJoint` 耦合组成。
- 正向定义：瓶盖绕 Bottle 本地 `+Z` 正向旋转时，沿 Bottle 本地 `+Z` 退出，符合右旋螺纹拆卸方向。
- 当前诊断螺距为 `0.003 m/turn`，轴向行程为 `0.012 m`，耦合比为 `-120000 deg/m`。螺距和行程均为 `TEMPORARY_UNCALIBRATED`；Bottle FreeCAD 中没有可作为真实螺距依据的螺旋几何。
- `hxz` 的全场景 Isaac Sim 5.1 动力学验证为 `PASS`：正向转角 `11.7494°`，正向退出 `9.5868e-05 m`，与螺距理论值残差约 `-2.044e-06 m`。
- 约束层 SHA-256：`0ab2a3d85807305f9ca45367e8981892b0be45195c1270ea0e7c5b56bcc3d0bc`。
- 报告：`assets/bottle_cap/1.0/reports/bottle_cap_helical_constraint_build.json` 和 `bottle_cap_helical_runtime_validation.json`。
- 2026-08-19 当前 `aloha` Streaming Server PID `2203966` 的活动 Stage 已在 Timeline Paused 下创建并回读通过：`/World/ALOHA1RemoteBottleSession/BottleThreadJoints/ThreadPrismatic`、`ThreadRevolute` 和 `RightHandThreadCoupling`。耦合关系为 Body0=`BottleCap`、Body1=`BottleThreadSlider`、hinge=`ThreadRevolute`、prismatic=`ThreadPrismatic`、ratio=`-120000 deg/m`；创建过程没有播放 Timeline、修改物体位姿/Kinematic 状态、保存 Stage、连接 ROS 或重启服务。
- 上述三个 Joint 当前只存在于活动 Stage 的未保存编辑状态；关闭或重启 Isaac Sim 后不能假定仍会存在。创建结果保存在 Mac `issac_log/server_runs/create_bottle_thread_coupling_result_20260819.json`（SHA-256 `0d8bc8332c858f24c657d3d766dff83360b52bd481306e53f44f868e318181ae`）。
- 第四阶段动态测试脚本为 Mac `isaac_script/bottle_thread_coupling_test.py`，并已同步到 `aloha:/home/eii/openpi0.5-rtc-reward-learning/isaac_script/`，SHA-256 为 `4ae25ba616ab6545ff443c9d94a51d3211338e6746d0b6a38e41a997f78d442d`。脚本必须从 Paused 开始，会临时固定 Bottle、以低力驱动瓶盖正转约 `2.4 s`、验证正向退出和螺距残差，然后自动 Pause 并恢复临时 Drive 与 Kinematic 状态；它不保存 Stage。
- `THREADED → RELEASED` 使用显式事件：当轴向退出量达到 `11.75 mm` 时，依次将 `RightHandThreadCoupling`、`ThreadRevolute` 和 `ThreadPrismatic` 的 `physics:jointEnabled` 设为 `false`，在 Joint Scope/Coupling 上写入 `threadState=RELEASED`，并确保 BottleCap 为 Dynamic、重力启用。当前阈值同 `12 mm` 行程一样仍为 `TEMPORARY_UNCALIBRATED`。
- 状态切换测试脚本为 `isaac_script/bottle_thread_release_test.py`（SHA-256 `41ff8a45a98ef09738438de26684f46ad401eeafd881aa629d22bc618c134bb7`）。测试夹具在 Paused 下把 Cap 与 Slider 一起注入 `11.80 mm` 边界，随后调用释放事件；该注入只验证边界逻辑，不是驱动力、轨迹或真实螺纹标定。释放后只沿 Bottle 本地 `+Z` 平移 `25 mm` 清除瓶颈碰撞包络，线速度和角速度均设为零，然后由重力/接触自然运行 `1.2 s`。禁止再把带侧向或向上初速度的清障探测混入主视觉测试。
- 2026-08-19 当前 Server 自然释放端到端测试为 `PASS`：释放位移 `11.800 mm`，三个 Joint 均 disabled，BottleCap Dynamic、重力启用；轴向清障前后命令线速度/角速度均为零，运行 `1.200 s` 后最终线速度和角速度为零，Timeline 最终 Paused，Stage 未保存。由于 Bottle 横放且 Cap 已紧邻桌面，Z 坐标几乎不变是桌面立即提供支持力的结果，不代表重力关闭。结果为 Mac `issac_log/server_runs/bottle_thread_release_transition_natural_settle_result_20260819.json`（SHA-256 `38245e26a20ccde39b304662d6a4f6ea9a028285f2ffed93b8d4e12df9e1394d`）。独立 RELEASED 复核脚本 `isaac_script/bottle_thread_release_verify.py`（SHA-256 `2e451579767f889a5fac5b4cac2c8049fa1e83592de07abf3ec8b90d7f4f7ec1`）仍保留高速度 6-DOF 清障探测，可能把瓶盖快速推出；它只用于诊断残余 Joint/碰撞限制，不用于自然落盖观察，旧复核结果 SHA-256 为 `f179c4cc3ac2f3b4f20d2a28e9b298a06e500b7e187aec9aed7c161be3c68852`。
- 2026-08-20 在干净重启后的 Server PID `2851078` 上重新执行完整恢复链并通过：自动重建 `BottleThreadSlider`、`ThreadPrismatic`、`ThreadRevolute`，创建 `RightHandThreadCoupling`，随后完成 `THREADED → RELEASED`。释放位移为 `11.800 mm`，三个 Joint 均 disabled，BottleCap Dynamic、重力启用，最终速度为零且 Timeline Paused；Stage 未保存、未使用 ROS、未触碰真实机器人。Mac 证据为 `issac_log/server_runs/create_bottle_thread_base_joints_result_20260820.json`、`create_bottle_thread_coupling_result_20260820.json` 和 `bottle_thread_release_transition_result_20260820.json`，SHA-256 分别为 `73ad49a3dd50d6173b82ec5f1056f20636ff4d76e74b62802ee0533d0de64e41`、`0d8bc8332c858f24c657d3d766dff83360b52bd481306e53f44f868e318181ae`、`b1b1fecc35f3e4eaa36b7f289fc5d20e4132fc6d26e453616133ee2589c40227`。
- 本次设计已保存为候选版本 `remote_isaac_assets/aloha1_bottle_server/attempt1/versions/thread_release_v1/`。入口为 `remote_stream_threaded_release_v1.usda`（SHA-256 `faf1ca14d8f0b0e5e845cfb7537a1631061993f279951235637be62dbb054cfc`），约束层为 `bottle_thread_constraint_v1.usda`（SHA-256 `96d038c711fbd8a7a40f3b052e4366b10c917ab6cc148a40b4e86ca0bcdd8a10`）。版本固定的是可复载的 `THREADED` 初始状态，不保存测试结束时的 `RELEASED` 瞬间；Isaac Sim 5.1 bundled USD 离线组合验证为 `PASS`，报告 `usd_composition_validation.json` SHA-256 为 `5be3a26a9e950aeb568dfd9b73cb487a97b18732dfe23c8cae0045086b4a3d97`。该版本仍为 `CANDIDATE_NOT_ACTIVE_SERVER_STAGE`，没有切换当前服务；螺距、行程和释放阈值继续标记 `TEMPORARY_UNCALIBRATED`。
- `isaac_script/bottle_force_pulse_y.py` 已改为低冲击接触诊断（SHA-256 `a225fd0059a319d2b171c9097dc91db7afae4f4fc55f17732c1954608656c3ee`）：默认只对 `Bottle500` 施加世界 `+Y` 的 `0.03 N × 0.10 s`，冲量 `0.003 N·s`，对 `0.025 kg` Bottle 的无阻力理论速度增量约 `0.12 m/s`；旧 `0.30 N × 0.10 s` 对应约 `1.2 m/s`，属于强冲击反例。新脚本会先自动 Pause，将 Bottle/Cap 设置为 Dynamic 后自动 Play，消除 Script Editor Run 与人工 Play 之间的异步竞态；随后观察 `2.0 s`，记录 Bottle/Cap 位姿、速度、Joint 状态和 Bottle 接触事件并自动 Pause。它不得编辑 Cap 位姿或改变 Joint，以保持单变量诊断。

## PrepareUncap Action Graph

- 候选 Stage：`remote_isaac_assets/aloha1_bottle_server/attempt1/remote_stream_prepare_uncap_stage.usda`，SHA-256 为 `a5260ee646b9f86498800b05bac3c5a225ddcead20b444e81d011316e318a0ed`；该文件已从 `hxz` 同步到 `aloha`，但当前 Streaming Server 尚未切换到它。
- Action Graph Prim：`/World/PrepareUncap`。
- 图中已有左右臂各自的 6-DOF `IsaacArticulationController`、sleep 命令数组，以及左右 `IsaacGripperController`、开合目标、速度和手指关节数组，共 17 个节点。
- 当前分类必须保持为 `SAFE_INERT_SCAFFOLD`：`prepare:executionEnabled=false`、`prepare:ready=false`、`prepare:state=BLOCKED_TARGETS_UNVALIDATED`。四个控制器的 `execIn` 均未连接，点击 Play 不会由该图驱动机械臂。
- 不得仅连接 `OnPlaybackTick` 就声称 PrepareUncap 已可执行。启用前必须分别验证当前场景的左臂预抓取、动态抓取、抬升及瓶口对齐世界 `+Z` 的轨迹，以及右臂接近瓶盖、抓盖轨迹、双臂碰撞和接触稳定性；不得用未经验证的左右镜像关节角替代 IK/碰撞验证。
- 状态清单：`assets/bottle_cap/1.0/config/prepare_uncap_state_machine.json`，依次为 `RESET`、`LEFT_PREGRASP`、`LEFT_GRASP`、`LIFT_ALIGN`、`RIGHT_PREGRASP`、`RIGHT_GRASP`、`READY`。
- `hxz` 的新鲜 Isaac Sim 5.1 复载验证为 `PASS`：17 个节点齐全、4 个执行输入均断开、螺旋约束四个 Prim 正确组合、timeline 已暂停。报告为 `prepare_uncap_action_graph_build.json` 和 `prepare_uncap_action_graph_runtime_validation.json`。
- `aloha` 上未并行启动第二个 Isaac Sim 验证实例：在当前 Streaming Server 占用运行资源时，独立验证进程未完成启动，已只终止该验证进程，当前 Streaming Server PID `2478467` 未被停止。后续应通过专用 loader 离线验证后再受控切换 Stage。

### 左夹爪动态抓取现状（下一次必须从这里继续）

- Grasp Editor/IK 坐标契约已经重新审计。`G` 必须是 `/World/follower_left/vx300s_left/follower_left_ee_gripper_link`，不是指尖接触中心；`C` 是 supplier-CAD 有效 pad/contact center，仅用于几何对齐，不能作为导出的 Grasp Editor 或 IK frame。矩阵约定为 `T_A_B` 将 B 系列向量映射到 A 系。
- `T_G_C` 为沿 `G` 本地 `+X` 的 `28.3208044428 mm` 平移；更上游 `gripper_link -> G` 为 `107.2 mm`，因此 `gripper_link -> C` 总长度为 `135.5208044428 mm`。必须用 `T_O_G = T_O_C @ inverse(T_G_C)` 和 `T_W_G = T_W_C @ inverse(T_G_C)` 反算，禁止把 `T_O_C` 的数字换名为 `T_O_G`。当前链闭合最大误差约 `2.78e-17`，门限为 `1e-9`。
- 已在 `hxz` 新鲜 Isaac Sim 5.1 进程中建立“0 个机械臂控制 DOF、只暴露一个夹爪标量”的隔离实验，并执行等效垂直接近、闭合以及动态无支撑保持。该实验确认坐标链正确，但也确认不能通过手工引用独立 link 重建 ALOHA1 的 passive mimic 机构。
- 原生 `acceleration / 5 N / stiffness 625 / damping 0.1` 最小夹具中，保留 CAD helper 径向偏移时 Bottle 动态掉落 `208.6 mm`；将 `C` 精确移到 Bottle 截面中心轴（抵消 `3.336382 mm, 0.444157 mm`）后仍掉落 `248.9 mm`。`gearing=-1` A/B 在接触阶段产生 `18.65 mm` mimic 残差且掉落 `141.8 mm`，已拒绝。
- 该阶段得出的后续要求是：直接复用经批准的完整 ALOHA1 articulation、原始 joint frame 和原生 passive mimic；六个臂关节保持固定并从 Grasp Editor 控制接口排除，只控制原生 `left_finger`。此要求已由下方“最新完整 articulation 结论”落实。不得继续在最小夹具中修补 mimic 符号，也不得把“接触过”当作成功抓取。Mac 证据目录为 `issac_log/isaac_failure_captures/prepare_uncap_isolated_gripper_20260815/`；`hxz` 隔离诊断脚本为 `assets/bottle_cap/1.0/scripts/validate_prepare_uncap_isolated_gripper.py`，SHA-256 为 `d490f780bb7c6089f9ded8098ba7d7e0c07a94a33e3bb169f55e000083dd449b`。
- 本轮只使用 `hxz` 的独立 Isaac 进程和匿名 session layer；没有切换或重启 `aloha` Streaming Server，没有改动其当前 Stage，也没有连接 PrepareUncap Action Graph。

#### 最新完整 articulation 结论

- 已进一步在 `hxz` 的新鲜 Isaac Sim 5.1 进程中直接复用完整 ALOHA1 articulation、原始 joint frame 和原生 passive mimic 做隔离 Grasp Editor 门控。左臂只在初始化时进入 `LEFT_REORIENT_VERTICAL_32` 抬升位姿；初始化后控制接口中唯一出现的关节索引为主动 `left_finger`（index 7），六个手臂关节不参与扫描，最大手臂漂移仅 `0.000266 rad`。
- 完整 articulation 的坐标契约已通过运行时交叉验证：Pinocchio FK 与 USD 实际 `G=/World/follower_left/vx300s_left/follower_left_ee_gripper_link` 的位置误差约 `2.02e-7 m`，旋转矩阵最大元素误差约 `7.60e-7`；`T_O_G=T_O_C@inverse(T_G_C)` 闭合误差约 `2.78e-17`。这确认夹爪底部/末端原点与接触中心 `C` 的 `28.320804 mm` 偏移已被正确处理，早期失败不是坐标原点换算错误。
- 使用 Grasp Editor 几何接触值 `q=0.048316875 m` 时，Bottle 切为 dynamic 后只在前 6 个 60 Hz 步保持双侧接触，随后掉落约 `90.89 mm` 并落桌；负载下原生 mimic 最大误差为 `7.08 mm`。前后可见网格采样均未发现夹指进入 Bottle 可见表面。
- 使用专家轨迹/硬件闭合区间 `q=0.021 m` 做不改变 `5 N` drive、摩擦或 mimic 参数的 A/B 时，末 60 步虽保持双侧物理接触，Bottle 仍掉落约 `93.87 mm` 并落桌，负载下原生 mimic 最大误差扩大到 `22.78 mm`。因此已排除“只需把 Grasp Editor 接触位姿当作更紧的闭合命令”这一解释；双侧接触本身也不得再作为抓取成功依据。
- 上述两次掉落是基线，不再是最新阻塞点。原生 PhysX mimic 的 `gearing=+1` 与其约束方程结合后等效于 URDF 的 `right=-left`，不得改成 `-1`；本轮只在匿名层将原生 mimic 的 `naturalFrequency/dampingRatio` 从导入值 `25/0.005` 改为诊断候选 `100/1.0`。在不绑定物理材质时，最大 mimic 误差已降至约 `2.07 mm`，但 Bottle 仍掉落约 `92.48 mm`，说明被动联动稳定性与摩擦/夹持中心是两个独立问题。
- 早期 PrepareUncap 验证使用 loader 临时绑定 Bottle `static/dynamic=0.65/0.50`、gripper `0.90/0.75`，均标记 `TEMPORARY_UNCALIBRATED`；当时确认 41 个 Bottle、17 个 Cap、1 个桌面和 4 个夹指 collider 解析到预期材质，且未修正中心时动态位移由约 `92.48 mm` 降至 `15.81 mm`。该条只保留为历史证据；当前 GripperPad 有效运行基线已由用户明确提升为 `2.0/1.5/maximum`，旧 `0.90/0.75/average` 不得再作为当前配置。
- 根据两侧接触点中心相对 Bottle 中心的实测偏差，只做了一个定向 `T_O_G` 修正：对象局部径向 `(-5.5, -1.5) mm`。该点连续两次新鲜 Isaac Sim 5.1 运行结果完全一致且全部门槛通过：4 秒动态无支撑保持位移 `5.720 mm`，最大 mimic 误差 `2.183 mm`，末 60 步双侧物理接触 `60/60`，无桌面/硬碰撞，前后均无采样到的可见夹指穿入 Bottle。
- 已围绕该重复通过点完成 20 个新鲜 Isaac 进程的局部径向包络：`15 PASS / 5 FAIL`。最佳采样点 `(-5.5, -2.0) mm` 位移 `1.418 mm`；`y=-2.5 mm` 的三个点和 `(-5.5,-3.0) mm` 因 mimic 误差超过 `2.5 mm` 失败，`(-7.0,-1.5) mm` 因位移 `15.261 mm` 失败；`(-4.0,-1.5) mm` 和 `(-5.5,0) mm` 仍通过。该结果只是离散采样，不得描述为连续安全矩形。
- 当前推荐的可复现实验中心仍为已重复验证的对象局部径向 `(-5.5,-1.5) mm`，不是尚未复跑的最低位移样本。该点的最终透明顶视、指间线侧视和斜视图已逐张审查：顶视图中两指终止于 Bottle 外轮廓两侧，没有早期截图中的明显整段穿体；透明侧壁会让侧视/斜视中的指垫看起来位于瓶内，不能单独作为穿透判据。
- 已把视觉表面门槛从“顶点+面中心”提高为每侧 `17,451` 点，包括所有顶点、面中心及每条面边的 `1/4、1/2、3/4` 采样。新鲜 Isaac Sim 5.1 运行在 4 秒动态保持前后均检测到 `0` 个夹指视觉表面采样点进入 Bottle body 圆柱体，同时全部动力学门槛保持 PASS（位移 `5.720 mm`、mimic `2.183 mm`、末 60 步双侧接触、无桌面/硬碰撞）。这已排除该重复中心点的“大块夹指穿过瓶身”问题，但不能替代真实软 Bottle 变形模型。
- 有效截图位于 Mac `issac_log/isaac_failure_captures/prepare_uncap_full_articulation_visual_gate_centered_v3_20260816/`，只有 `02_post_dynamic_hold__*.png` 的相机位姿标签有效；同目录 `01_pre_dynamic_closed__*.png` 的相机回读停留在旧斜视位姿，不得使用。不透明 Bottle 的独立 headless 诊断返回统一灰色空 RGB，已拒绝，不能作为通过证据。
- 已固定径向中心 `(-5.5,-1.5) mm` 完成 Bottle 轴向抓取位置粗扫、边界细化和推荐点复跑，共 16 个新鲜 Isaac Sim 5.1 进程，结果为 `12 PASS / 4 FAIL`。所有运行均保持动态无支撑、每侧 `17,451` 点密集穿透门槛及单主动指原生 mimic 语义。
- Bottle 本地接触中心 `z=39/49 mm` 虽满足位移和 mimic 门槛，但动态后分别有 `4/6` 个夹指视觉表面采样点进入 Bottle body 圆柱体，因此失败；`z=99/109 mm` 穿透为零但动态位移分别为 `24.430/19.669 mm`，超过 `10 mm` 门槛，因此失败。已测试的离散通过点覆盖 `z=51.5–96.5 mm`，不得把离散结果描述为连续安全区间。
- 当前推荐轴向中心为 Bottle 本地 `z=59 mm`，即相对原 `z=69 mm` 基准使用 `--object-axial-m=-0.010`。该点在三个新鲜进程中完全一致地 PASS：Bottle 位移 `1.846 mm`、mimic 最大误差 `2.178 mm`、末 60 步双侧接触 `60/60`、无桌面/硬接触且动态前后穿入点均为零。单次位移更低的 `z=54 mm`（`1.448 mm`）尚未复跑，不升级为推荐点。
- `z=59 mm` 使接触中心到 Bottle 本地瓶口 `z=206 mm` 的轴向距离从原基准的 `137 mm` 增至 `147 mm`；它仍是静态竖直保持门槛的推荐点，但不能升级为完整动态任务的推荐抓取。已在独立进程中执行左臂动态抬升与瓶口对齐世界 `+Z` 的 `LIFT_ALIGN` 门控，结果为 FAIL，详见下方最新条目。
- 轴向扫描的 Mac 证据目录为 `issac_log/isaac_failure_captures/prepare_uncap_full_articulation_axial_scan_20260816/`；远程粗扫和细化报告分别位于 `assets/bottle_cap/1.0/reports/prepare_uncap_full_articulation_axial_scan_8/` 与 `assets/bottle_cap/1.0/reports/prepare_uncap_full_articulation_axial_refine_8/`。粗扫脚本 SHA-256 为 `f5741fcf99f1902375e4004fa059ddb56e02bf296a6ae2c9d005ca36ca05322f`，细化脚本 SHA-256 为 `cf900fe96f5772871748435c6aafa949706d0fa2f0d84edf46fd2db542a34627`。
- 完整 articulation 验证脚本：`assets/bottle_cap/1.0/scripts/validate_prepare_uncap_full_articulation_grasp_editor.py`，当前 SHA-256 为 `b163914e9a1377491ec1bcd2c3903584017d0ec90daecc87452920fc6f060339`。密集穿透报告：`assets/bottle_cap/1.0/reports/prepare_uncap_full_articulation_dense_visual_penetration_gate_v2.json`，SHA-256 为 `c1933ca1e3fc002e0fdc856a4d9c12e2b84169f12b20a567e4ea2d6d5eb9609d`。20 次执行脚本 SHA-256 为 `284e45fbbe422280bc87e8919f01ba58f10dfc179513b6e66ac1459f09f3dd5e`；远程汇总报告 `prepare_uncap_full_articulation_envelope_20_summary.json` 的 SHA-256 为 `da6fb1f8eca5cfe24dc98f92735340be677ba41e105b7394d2094aa9583478f4`。

#### 最新 LIFT_ALIGN 动态门控结论

- 动态拾取验证器已扩展为支持 Bottle 物体局部三维抓取修正、每侧 `17,451` 点密集视觉穿透检测、loader 临时材质绑定，以及使用修正后 `T_O_G` 逐段求解的 32 段 `LEFT_REORIENT_VERTICAL`。当前脚本 `assets/bottle_cap/1.0/scripts/validate_prepare_uncap_dynamic_pickup_smoke.py` SHA-256 为 `6b6722a70e40bc9892647f481fe6daa70285ffdbb258c019017a00183df076f9`。
- 固定径向 `(-5.5,-1.5) mm`、轴向 `z=59 mm`、预压 `q=21 mm` 时，Bottle 可先抬升 `50.381 mm`，但仅 support-clear 后最低可视点仍在桌下 `21.290 mm`，且已有密集穿入事件。靠近瓶身质心的静态通过点 `z=94 mm` 在水平动态拾取中反而在指间滑动，最终抬升约为零并有 57 个控制步触发穿入，因此已拒绝作为替代点。
- 第一版完整旋转复现了旧绝对轨迹拼接错误：support-clear 后 Bottle 根部约在 `82 mm`，第一旋转目标却回到约 `74 mm`，造成向桌面下压。已改为把 32 段规划平移锚定到 support-clear 的实际 Bottle 位置；所有 IK 位置/姿态误差达到数值零量级，但锚定版 387 个旋转控制步中只有 183 步保持双侧接触、204 步失联，Bottle 仍在旋转中段掉落。说明当前阻塞点不是 IK。
- 闭合 A/B 已确认刚性接触冲突：预压到 `q=21 mm` 可先抬升，但旋转期间有 177 个控制步触发密集穿入，最大已记录径向深度至少 `3.321 mm`；取消预压、在首次双侧物理接触 `q≈45.706 mm` 停止时，mimic 最大误差降到 `0.702 mm`，但 318 个旋转步中只有 82 步保持双侧接触、236 步失联，Bottle 基本没有离桌。不得通过提高摩擦、驱动力或放宽穿透门槛解决。
- 当前 `LIFT_ALIGN` 状态为 `BLOCKED_RIGID_BOTTLE_CONTACT_MODEL`。真实软 Bottle 的夹持凹陷会改变接触面积和法向力分布，静态竖直 Grasp Editor 通过点不能代表动态旋转通过。下一步应只在匿名 session layer 建立标记为 `TEMPORARY_UNCALIBRATED` 的软瓶等效顺应接触 A/B，固定现有摩擦、`5 N / stiffness 625 / damping 0.1` drive、mimic `100/1.0`、轨迹和全部碰撞门槛；接触 stiffness/damping 必须有明确扫描范围与来源记录。在通过前 PrepareUncap Action Graph 保持断开。
- Mac 证据目录：`issac_log/isaac_failure_captures/prepare_uncap_lift_align_20260816/`。关键远程报告为 `prepare_uncap_dynamic_lift_align_axial59_anchored_v2.json` 和 `prepare_uncap_dynamic_lift_align_axial59_event_close_anchored_v3.json`，SHA-256 分别为 `94a3ca598a63023c3309334adb88ebb323315029f21b29bc612141e5de74483c`、`2a4a8af7e4a4b21c926c54bd06decc34985fae19b99729b43a02a30fc84ef508`。
- 已按上述要求完成刚性 Bottle 的 PhysX compliant-contact A/B。运行时回读 Bottle authored mass 为 `0.025 kg`；保持摩擦、`5 N / 625 / 0.1` drive、mimic `100/1.0`、抓取点、`q=21 mm` 预压、轨迹和原碰撞门槛不变。stiffness 扫描 `1000/2500/5000 N/m`，临界阻尼按 `c=2*sqrt(k*0.025)` 取 `10/15.811/22.361 N·s/m`，并在 `k=1000` 补测 `c=5/20`。参数只写匿名层并标记 `TEMPORARY_UNCALIBRATED_RIGID_COMPLIANT_CONTACT_SURROGATE`。
- 五组柔顺接触候选全部 `FAIL`：最终抬升为 `-0.252` 到 `-0.075 mm`，旋转期间仍有 `176–202` 步失去双侧接触，密集视觉门槛记录 `195–225` 个穿入步，最大刚性几何压入 `1.529–2.902 mm`，最终瓶轴与世界 `+Z` 点积仅 `0.00805–0.01276`。柔顺接触只把几何重叠解释为接触弹簧压缩，没有产生软瓶凹陷形状或扩大的接触斑；不得放宽零穿入门槛将其升级为通过，也不再扩大该 surrogate 的参数搜索。
- 当前 `LIFT_ALIGN` 状态升级为 `BLOCKED_NEEDS_DEFORMABLE_BOTTLE_CALIBRATION`。下一步是建立独立可变形 Bottle 标定夹具，以实瓶“夹持力—两侧凹陷量—卸载恢复”测量为输入，先验证显式形变、接触斑、单夹爪静态夹持和动态无支撑保持，再复用现有 `LIFT_ALIGN` 轨迹；在此之前 Action Graph 继续断开。
- 柔顺扫描 Mac 证据目录：`issac_log/isaac_failure_captures/prepare_uncap_compliant_contact_scan_20260816/`；远程报告目录：`assets/bottle_cap/1.0/reports/prepare_uncap_compliant_contact_scan_20260816/`。动态验证脚本现支持 Bottle compliant stiffness/damping、参数回读和全程最大压入统计，当前 SHA-256 为 `ff560e2d6c3f05a0c5a633db9c41b88218f7a20518b96fd6cfadc24253696a42`。

#### 早期对照证据（不可覆盖上方最新结论）

- Grasp Editor 候选：`configs/aloha1_grasps/bottle500_horizontal_body_grasp.isaac_grasp.yaml`；物体坐标系位置 `(-0.02473675, -0.00329309, 0.069) m`，左指 `cspace_position = 0.048316875 m`，`pregrasp = 0.057 m`。该文件的 `confidence = 0.0`，只能作为候选，不得当作抓取验收结果。
- `hxz` 的随机五位置实验必须区分两轮结果：早期 `five_position_acceptance_002` 为 `4/5`，其中 position 02 因最大离桌间隙仅约 `0.1984 m` 而未达到 `0.200 m` 高度门槛；后期 CAD-derived、Z-up 的 `phase8_five_pose_runtime_zup_attempt7` 才是 `5/5` 动态抬升并保持通过。
- 后期 `5/5` 不能作为当前原生 mimic 已正确的证据。它在匿名 session layer 中移除右指 PhysX mimic，用 `official_symmetric_adapter` 将一个夹爪标量命令分配为 `q_right=-q_left`，并给两指使用相同的 `force` drive（maxForce `5 N`、stiffness `625`、damping `0.1`）；报告明确标记 `DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING`、`promotion_authorized=false`。当时原生 mimic 的平均耦合残差为 `1.779 mm`（FAIL），适配器为 `0.0239 mm`（PASS）。这说明当前穿模/单侧跟随问题具有历史先例，但最终实现仍须保持“一台物理执行器、单标量命令、被动连杆等幅反向联动”的硬件语义，不能将适配器描述为两台独立执行器。
- 已在 `hxz` 的新鲜 Isaac Sim 5.1 世界中扫描 Grasp Editor 夹爪本地 `+Y` 指间轴投影到世界 XY 的中心线偏移。已测试的安全双侧物理接触区间为 `[-4.5, -3.0] mm`；`0/+3/+6 mm` 在形成双侧抓取前发生物理指尖/桌面接触，`-6 mm` 出现非手指夹爪基座/瓶身硬接触。
- 上述区间只是“物理接触包络”，不是“成功抓取包络”。最佳实测抬升仅 `+0.378 mm`，未达到单次诊断要求的 `20 mm`，因此不得开始 20 次验收，也不得启用 PrepareUncap Action Graph。
- 原生夹爪运行时左指驱动上限为 `5 N`、stiffness `625`，右侧 mimic 指显示驱动上限为 `0`；在负载下最大目标误差为 `5.77 mm`。在当前 Phase 97 场景条件下另做的匿名 session 复测，虽同样移除右指 mimic、复制不变的 `5 N`/`625` drive 并命令 `q_right=-q_left`，最大耦合残差仍为 `8.04 mm`，抬升为 `-0.530 mm`。该失败复测与历史五位置通过的具体 drive type、阻尼、场景和轨迹条件不同，不能互相替代；适配器仍仅限诊断，不得直接写入最终控制或 Action Graph。
- 该早期阶段曾将根因收敛为负载下双指控制/耦合映射未校准，而不是单纯的夹爪中心线问题；其“完成一次悬空门槛后再执行 20 次”的要求，以及顶视/侧视与密集视觉表面穿透门槛，均已由上方最新完整-articulation 测试完成。只有上方明确标记有效的 `02_post_dynamic_hold__*.png` 可作为带相机位姿标签的截图证据。
- 禁止用提高摩擦、未经来源支持地提高驱动力、降低双侧接触门槛、让夹爪穿桌、给 Bottle 加固定约束或直接移动 Bottle 的方式制造通过。
- 原中心线包络汇总报告：`assets/bottle_cap/1.0/reports/prepare_uncap_dynamic_grasp_editor_envelope.json`。验证 Stage SHA-256：`ba209deeacf8417d13ab3c360958c26dcc54db969526f6e4cd08b5396959ea30`；该报告记录的旧脚本 SHA-256 为 `860fc45c9eab269c7ff533c8d1b2d7475bbe70637a4f7ce900340d230b9682ae`，当前脚本哈希见下方最新校准条目。
- 已按 `hxz` 官方硬件来源链确认 follower 为 Trossen Interbotix ViperX-300 6DOF，夹爪执行器为 XM430-W350（ID 9）。不得把 DYNAMIXEL 寄存器增益直接复制为 PhysX 参数。
- ALOHA1 follower 的确切项目型号为 `aloha_vx300s`。每个 follower 夹爪只有一台物理执行器（XM430-W350，ID 9，官方 ALOHA 运行时为 `current_based_position`）；`right_finger` 不独立传感，其位置由 Interbotix 驱动器依据 `horn_radius=0.0275 m`、`arm_length=0.035 m` 的连杆模型派生。官方 URDF 使用 `right_finger` mimic `left_finger`、multiplier `-1`、offset `0`。最终仿真控制必须保持“单标量夹爪命令 + 等幅反向联动”语义，不得把两根指头描述成两台独立硬件执行器。
- 同血缘 Phase 97 Isaac 候选曾先验证“只命令主动 `left_finger`、保留 `right_finger` PhysX mimic”的控制语义：主动 drive `stiffness=200`、`damping=50`，原 acceleration type 和 `5 N` maxForce 不变；mimic `dampingRatio=1.0`。左右侧分别在关闭重力/碰撞的 2 mm 小位移测试中通过，最大无负载耦合残差约 `0.094 mm`、无超调。此条是早期微动证据；左右 follower 后续均已由 2026-08-17 人工空载全行程测试验收并提升为上文运行时基线。
- 在恢复重力/碰撞后，中心线偏移 `-4.115 mm` 首次形成安全的同一步双侧物理接触，无指尖/桌面或非手指硬碰撞；但只维持 3 个抬升步，最终保持为 0 步，Bottle 仅移动 `0.289 mm`，负载耦合残差扩大至 `7.92 mm`。这不是抓取通过。
- 人工截图已否定“当前接触包络安全”的前提：夹指/夹爪视觉几何进入 Bottle 可见表面，同时 Bottle 可见包围盒在桌面 `z=0` 下方达到 `-15.89 mm`，而接触报告仍给出近零 separation。当前首要硬门槛是核对 Bottle、夹指及夹爪基座的视觉网格与 PhysX collider 对齐，并建立可见表面穿透硬门槛；真实软 Bottle 变形标定只能在碰撞几何正确后继续。
- 双独立 drive 仅做过匿名 session 诊断并已拒绝：移除左夹爪右指 mimic、同步两侧 acceleration drive（maxForce `5 N`、stiffness `200`、damping `50`）并发送 `q_right=-q_left` 后，加载耦合残差反而为 `14.30 mm`，Bottle 抬升 `-0.225 mm`。证据位于 Mac 的 `issac_log/isaac_failure_captures/prepare_uncap_symmetric_n004115_20260815_v3/`；不得把该方案写入 Action Graph 或最终 Stage。
- 物理子步诊断也已拒绝：在保持 60 Hz 控制目标的情况下，120 Hz 和 240 Hz physics 均改变动态 settle 后的 Bottle 状态并导致后续物体相对 IK 失败，不能用来掩盖碰撞/支撑问题。
- 该早期阶段汇总报告：`assets/bottle_cap/1.0/reports/prepare_uncap_finger_control_calibration.json`。小位移脚本 SHA-256：`096379fbe636690807be1c30a3d394c678f880c0f54e0f1976cec81ed4f9b64f`；支持三阶段截图、双主动指反例和可选物理子步诊断的动态脚本 SHA-256：`10d0d1f2780e553a2cc67fc31c23ba6819b52990e1a10816dea434eab9b26fbf`。原生 mimic 失败近景位于 `issac_log/isaac_failure_captures/prepare_uncap_n004115_20260815_v2/`。

### 初始化验证

- 当前初始化报告：`/home/eii/openpi0.5-rtc-reward-learning/remote_isaac_assets/aloha1_bottle_server/attempt1/remote_cap_stage_loader_report.json`；原稳定 Stage 报告仍为 `remote_stage_loader_report.json`。
- 完成当前初始化后必须确认报告中 `status` 为 `PASS`，Bottle 和 BottleCap Prim 存在，`runtime_binding_ok` 为 `true`，四类材质绑定计数合格，`runtime_state.gripper_material_profile` 精确为 `2.0/1.5/maximum` 且分类正确，左右臂 sleep 回读误差合格且 timeline 已暂停。
- 重启远程 Streaming Server 后，Mac 上的 WebRTC Streaming Client 可能需要执行 Reload 才会重新连接。
