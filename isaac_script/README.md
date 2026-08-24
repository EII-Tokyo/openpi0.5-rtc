# Isaac Sim 可执行实验脚本

这些脚本面向 `aloha` 上运行的 Isaac Sim 5.1 Streaming Server，不保存 Stage，
也不连接 ROS 或真实机器人。

## 目录映射

- Mac 开发目录：`/Users/mac/Documents/project/aruco-lab/isaac_script/`
- aloha 执行目录：`/home/eii/openpi0.5-rtc-reward-learning/isaac_script/`

WebRTC Client 显示的是 aloha 上的 Isaac Sim，因此 Script Editor 必须读取 aloha
路径，不能使用 Mac 路径。

## 通用准备

1. Timeline Stop，确认 Bottle 回到启动位姿。
2. 让机械臂和夹爪远离 Bottle。
3. 选择 `/World/ALOHA1RemoteBottleSession/Bottle500`。
4. 取消 Geometry 和 Physics/Rigid Body 下的 `Kinematic Enabled`。
5. 确认 `Disable Gravity` 未勾选；不要保存 Stage。

## 在 Script Editor 中加载和执行

1. 打开 `Window > Script Editor`。
2. 在空白标签中粘贴下面三行，把 `SCRIPT_NAME` 换成要执行的文件名，
   然后点击 Run：

   ```python
   SCRIPT_NAME = "bottle_force_continuous_y.py"
   path = "/home/eii/openpi0.5-rtc-reward-learning/isaac_script/" + SCRIPT_NAME
   exec(compile(open(path, "rb").read(), path, "exec"))
   ```

3. 如果 Script Editor 有文件夹/Open File 按钮，也可以直接打开上述 aloha
   绝对路径中的 `.py` 文件并点击 Run。

## 脚本用途

- `bottle_force_continuous_y.py`：持续施加世界 `+Y` 方向 `0.20 N`；Paused
  状态运行脚本后点击 Play。
- `bottle_force_pulse_y.py`：低冲击 Bottle/Cap 接触诊断。只向 Bottle500 施加
  世界 `+Y` 方向 `0.03 N`、持续 `0.10 s`，随后继续观察到 `2.0 s`；逐步记录
  Bottle/Cap 位姿和速度、三个螺纹 Joint 状态及 Bottle 接触事件，最后自动
  Pause 并写入 `reports/lula_joint_diagnostics/bottle_force_pulse_y_diagnostic_*.json`。
  它会自动 Pause、把 Bottle/Cap 切成 Dynamic 并自动 Play，避免手工 Play 的
  异步竞态；不会编辑物体位姿、切换 Joint、保存 Stage 或连接真实机器人。
- `bottle_initial_velocity_y.py`：直接设置世界 `+Y` 初速度 `0.50 m/s`；必须先
  点击 Play，再运行脚本。
- `bottle_force_stop.py`：禁用并清零 Bottle 上的 PhysxForceAPI。持续力实验后
  必须执行。
- `bottle_thread_coupling_test.py`：第四阶段右旋螺纹耦合动态测试。它临时固定
  Bottle500，以低力驱动瓶盖绕局部 `+Z` 正转，验证瓶盖是否沿 Bottle 局部
  `+Z` 退出以及 `3 mm/turn` 螺距残差；结束后自动 Pause，并恢复临时 Drive
  和三个刚体原有的 Kinematic 状态。
- `bottle_thread_release_test.py`：`THREADED → RELEASED` 边界状态测试。它在
  Paused 状态用测试夹具把 Cap 与 Slider 一起放到 `11.80 mm`，超过
  `11.75 mm` 释放阈值后禁用 Rack-and-Pinion、Revolute 和 Prismatic，并将
  BottleCap 保持为 Dynamic、启用重力。随后仅沿瓶轴平移 `25 mm` 使瓶盖退出
  瓶颈碰撞区，明确把线速度和角速度清零，再让 PhysX/重力自然运行 `1.2 s`。
  该轴向平移代表右夹爪完成最后的拔离动作，不是抛掷速度；边界位姿是测试注入，
  不是驱动力或螺距标定结果。
- `bottle_thread_release_verify.py`：只验证当前已经处于 `RELEASED` 的场景，
  不重新连接螺纹。检查三个 Joint 均 disabled，并执行带明显初速度的独立运动
  探测；它可能把瓶盖快速推出，只用于排除残余约束，不能用于观察自然落盖。
- `bottle_thread_reset_tight.py`：执行 Timeline Stop 恢复启动位姿，然后把
  Prismatic 锁在 `0 mm`、Revolute 锁在 `0°`、禁用 Coupling，并移除遗留的
  Revolute angular Drive。Bottle500、BottleCap、Slider 都保持 Dynamic；脚本
  最终保持 Paused，不保存 Stage。
- `bottle_thread_begin_unthreading.py`：显式进入 `UNTHREADING`。Prismatic 开放为
  `0–12 mm`，Revolute 按 USD Physics 5.1 标准设为 `-inf/+inf`，再启用右旋
  Coupling；不会创建 Drive，也不会自动 Play。
- `bottle_thread_set_released.py`：在 Paused 状态显式进入 `RELEASED`，禁用三个
  Thread Joint，使 Dynamic BottleCap 完全独立；不移动瓶盖、不注入速度。
- `bottle_gripper_contact_points_diagnostic.py`：在抓取已经建立、Timeline 为
  Paused（不是 Stopped）时，只读观察 `2.0 s` 的 PhysX Contact Report。报告按
  左指、右指和非手指机器人几何分别给出每 physics step 的去重接触点数、接触
  存在比例、中位数、P95、最大值、冲量和代表性世界坐标；同时通过 PhysX tensor
  view 记录 Bottle/Cap 的世界转角、相对转角、角速度及绕瓶轴角速度，结束后自动
  回到 Paused。它不改变位姿、关节目标、材质或刚体模式。

## Bottle/Cap 低冲击接触诊断

该实验一次只改变推力强度，不自动改变 BottleCap 间距：

1. 如果目标是验证释放后的碰撞传力，先确认面板/报告中 `threadState=RELEASED`，
   且三个螺纹 Joint 均为 disabled。不要用肉眼位置替代这项检查。
2. 在 Script Editor 只运行一次 `bottle_force_pulse_y.py`。脚本会自行 Pause、
   将 Bottle500/BottleCap 设为 Dynamic 并自动 Play；不要再手工点击 Play。
3. 不再操作 Timeline。脚本在 `2.0 s` 物理时间后自动 Pause 并输出报告路径。
4. 把最新的 `bottle_force_pulse_y_diagnostic_latest.json` 交给 Codex 分析。关键
   字段为 `bottle_cap_contact_observed`、`preflight.joint_enabled`、两物体最大速度、
   位移以及 `contact_events` 中的碰撞路径和 impulse。

当前 Bottle 质量约 `0.025 kg`；新脉冲冲量为 `0.003 N·s`，忽略接触和摩擦时
理论速度增量约 `0.12 m/s`。旧值 `0.30 N × 0.10 s` 对应约 `1.2 m/s`，只可视为
强冲击反例，不再作为默认桌面推力测试。

## 第四阶段右旋耦合测试

1. 确认 Timeline 为 **Paused**，不要点击 Stop，也不要保存 Stage。
2. 确认 Stage 中存在以下三个 Joint：
   - `/World/ALOHA1RemoteBottleSession/BottleThreadJoints/ThreadPrismatic`
   - `/World/ALOHA1RemoteBottleSession/BottleThreadJoints/ThreadRevolute`
   - `/World/ALOHA1RemoteBottleSession/BottleThreadJoints/RightHandThreadCoupling`
3. 在 `Window > Script Editor` 中执行：

   ```python
   path = "/home/eii/openpi0.5-rtc-reward-learning/isaac_script/bottle_thread_coupling_test.py"
   exec(compile(open(path, "rb").read(), path, "exec"))
   ```

4. 脚本会自动 Play 约 `2.4 s` 物理时间，然后自动 Pause。测试期间不要拖动
   Bottle、BottleCap、Target 或 Timeline。
5. 预期可见现象：瓶盖绕自身/Bottle 轴正转，并沿 Bottle 局部 `+Z` 退出；当前
   Bottle 水平放置时，该退出方向对应世界 `+X`。
6. 在 Console 中检查 `status`：必须为 `PASS`。报告写入服务器
   `remote_isaac_assets/aloha1_bottle_server/attempt1/reports/lula_joint_diagnostics/`
   下的 `bottle_thread_coupling_test_*.json`。

该脚本只验证等效右旋螺旋约束是否成立；`3 mm/turn`、`12 mm` 行程仍是
`TEMPORARY_UNCALIBRATED`，不能据此宣称已经匹配真实瓶盖螺纹。

## THREADED → RELEASED 状态测试

保持 Timeline 为 Paused，在 Script Editor 中执行：

```python
path = "/home/eii/openpi0.5-rtc-reward-learning/isaac_script/bottle_thread_release_test.py"
exec(compile(open(path, "rb").read(), path, "exec"))
```

脚本自动完成测试并回到 Paused。通过条件包括：

- `extension_at_release_m >= 0.01175`；
- 三个螺纹 Joint 的 `physics:jointEnabled=false`；
- `threadState=RELEASED`；
- BottleCap 为 Dynamic 且重力启用；
- 轴向退出 `25 mm` 时命令线速度、角速度均为零；
- 仅靠物理运行 `1.2 s` 后瓶盖不再向上飞行且速度已稳定。

当前 Bottle 横放且 BottleCap 紧邻桌面，因此自然释放后 Z 方向下落距离可能很小：
桌面会立即提供支持力。判断是否正确应看“没有侧向/向上发射、最终稳定在桌面”，
不能要求它先出现明显的自由落体高度。

测试成功后会故意保留 `RELEASED` 状态供观察。再次运行同一脚本时，脚本会先用
Timeline Stop 恢复未保存的启动位姿并重新启用三个 Joint，再重复测试；它不会
重启 Server 或保存 Stage。

## 恢复仿真名义拧紧状态

先保持 Timeline 为 **Paused**，并确认左右机械臂和夹爪已经远离 Bottle。该操作
会执行 Timeline Stop，因此会把整段仿真的物理状态恢复到启动状态。然后执行：

```python
path = "/home/eii/openpi0.5-rtc-reward-learning/isaac_script/bottle_thread_reset_tight.py"
exec(compile(open(path, "rb").read(), path, "exec"))
```

脚本结束后必须看到 `status=PASS`、`readback.state=THREADED`、Prismatic 与
Revolute enabled、Coupling disabled、两个限位均为 `[0, 0]`，且
`angular_drive_present=false`。报告写入
`reports/lula_joint_diagnostics/bottle_thread_reset_tight_result.json`。若脚本失败，
会保持 Timeline Paused。不要通过手工修改 `threadState` 代替该恢复流程。

需要真正开始旋松时，再单独执行：

```python
path = "/home/eii/openpi0.5-rtc-reward-learning/isaac_script/bottle_thread_begin_unthreading.py"
exec(compile(open(path, "rb").read(), path, "exec"))
```

只有该脚本 `status=PASS` 后才应 Play 并施加旋松动作。到达释放位置并 Pause 后，
执行 `bottle_thread_set_released.py`；它会禁用全部三个螺纹 Joint。

## 查看夹爪与 Bottle 的接触点

先建立抓取，并保持 Timeline 为 **Paused**（不能是 Stopped）。如果要评估真实
悬持接触，Bottle 应为 Dynamic。执行：

```python
path = "/home/eii/openpi0.5-rtc-reward-learning/isaac_script/bottle_gripper_contact_points_diagnostic.py"
exec(compile(open(path, "rb").read(), path, "exec"))
```

脚本自动 Play `2.0 s` 后回到 Paused。报告写入
`reports/lula_joint_diagnostics/bottle_gripper_contact_points_diagnostic_latest.json`。
重点查看 `summary.left`、`summary.right`、`summary.nonfinger`、
`bilateral_contact.fraction` 和 `motion_summary`。接触点是 PhysX 每步生成的求解器 manifold 点，会随
碰撞特征变化；应比较中位数、P95 和接触存在比例，不要把某一帧点数当成永久的
物理接触面数量。

每轮只改变一个参数。重新实验前执行 `bottle_force_stop.py`，再点击 Timeline
Stop。若出现异常，先 Pause，再执行停止脚本。
