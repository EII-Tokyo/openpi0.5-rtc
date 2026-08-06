# ALOHA 桌面世界与 Bottle500 标定工作台

这个 React + FastAPI 工作台只解决当前最小闭环：保留四台 D405 的出厂内参，不做 ChArUco 重标定；仅用 `camera_high` 的 RGB 流把真实桌面坐标系和带刚性标签夹具的 Bottle500 位姿传到 Isaac Sim。深度流、机器人控制和 Isaac timeline 均不在本流程中启用。

默认 `npm run dev` 是零设备访问的 Preview。只有显式 `npm run dev:live` 才会显示可执行按钮，而且每一步仍受后端状态机约束。

## 三个实验

1. **世界锚点**：AprilTag 36h11 ID0 中心与真实桌面原点重合，标签印刷轴与桌面 `+X/+Y` 对齐。`camera_high` 连续采集 200 个不可覆盖 RGB 证据帧，求得 `camera_high_optical -> table_world`。
2. **9 点桌面交叉验证**：用钢尺和直角尺对 9 个彩色圆点各测两次。服务端先冻结物理真值；网页拍摄一张不可变桌面快照，然后逐点在图上点击圆心。6 点用于求解，非共线的 `P11/P23/P32` 只用于盲测。这个实验只验证桌面平面 **XY**，不宣称验证了离面 Z。
3. **Bottle500 刚性夹具传递**：冻结瓶长、直径、标签相对瓶体的变换、资产轴变换和垫块高度，再按服务端规定的 `B-A/B-B/B-C` 三种位置各采 150 帧。通过只表示“带标签刚性夹具的位姿传递通过”，不表示透明瓶无标签感知、碰撞或动力学通过。

每次重新采集都会生成新的编号 attempt，已有图像和 JSON 不覆盖。浏览器不能在求解请求中修改已经冻结的圆点真值或 Bottle expected pose。

## 本地 Preview

```bash
cd /home/eii/project/openpi0.5-rtc-reward-learning/aloha_calibration_workbench
npm install
npm run dev
```

打开 `http://127.0.0.1:4173`。Preview 不发出 API 请求。

## Live 服务拓扑

相机连接在哪台机器，capture-agent 就运行在哪台机器。当前 ALOHA 拓扑使用 `103` 上的 localhost-only capture-agent；`101` 上的 orchestrator 只经 SSH localhost 转发访问它，两个服务都没有机器人命令 API。

在相机主机启动 capture-agent：

```bash
cd /home/eii/openpi0.5-rtc-reward-learning/aloha_calibration_workbench/backend
/home/eii/.local/bin/uv sync --dev
/home/eii/.local/bin/uv run aloha-calibration-capture-agent
```

在 `101` 建立转发：

```bash
ssh -N -L 8017:127.0.0.1:8017 192.168.1.103
```

在 `101` 启动 orchestrator 和 live 前端：

```bash
cd /home/eii/project/openpi0.5-rtc-reward-learning/aloha_calibration_workbench/backend
/home/eii/.local/bin/uv sync --dev
/home/eii/.local/bin/uv run aloha-calibration-orchestrator

cd ..
npm run dev:live
```

网页中的执行顺序固定为：

`PREFLIGHT_READY -> FACTORY_INTRINSICS_FROZEN -> WORLD_ORIGIN_SOLVED -> TABLE_POINT_CONTRACT_FROZEN -> WORLD_REGISTRATION_VALIDATED -> BOTTLE_FIXTURE_CONTRACT_FROZEN -> TAGGED_FIXTURE_TRANSFER_PASS -> EXPORT_READY`

若门禁失败，不要改阈值绕过；根据页面错误重新摆放、消除遮挡或重新拍摄一个新 attempt。

### 复用 `aloha2-collect` 的 ROS2 相机流

当 `aloha2-collect` 已经运行四个 `realsense2_camera_node` 时，它必须是 D405 的唯一设备所有者。此时不要让 capture-agent 再通过 `pyrealsense2.pipeline.start()` 打开相机；使用容器内只读 ROS bridge：

```bash
cd /home/eii/openpi0.5-rtc-reward-learning
bash third_party/aloha_collection/scripts/run_calibration_camera_bridge.sh
```

bridge 只订阅四组 `sensor_msgs/msg/Image` 和 `sensor_msgs/msg/CameraInfo`，绑定 `127.0.0.1:8018`，不创建 ROS publisher、service/action client 或机器人命令 API。它通过 stdin 把项目内脚本送入已有容器，不修改容器挂载的 ALOHA 源码。

另一个终端在 103 启动 capture-agent：

```bash
cd /home/eii/openpi0.5-rtc-reward-learning/aloha_calibration_workbench/backend
ALOHA_CALIBRATION_CAMERA_SOURCE=ros_bridge \
ALOHA_CALIBRATION_ROS_BRIDGE_URL=http://127.0.0.1:8018 \
.venv/bin/aloha-calibration-capture-agent
```

该模式下预检必须显示 `exclusive_capture_required: false`，四台相机所有权为 `ROS_SOURCE`。出厂 K/D 来自各相机的 `CameraInfo`；正式证据帧来自 `Image`，经 bridge 无损编码为 PNG，并保留 ROS header 时间戳。`plumb_bob` 在宿主适配层显式映射为 OpenCV 的 `brown_conrady`，不修改数值。

## USD 导出契约

导出前固定核验：

- 源 Stage：`assets/Trossen/ALOHA1/1.0/diagnostics/table_support_alignment/1.0/aloha1_table_support_aligned_workcell.usda`
- 源 SHA-256：`2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c`
- Bottle 资产：`assets/bottle_500ml/isaac/bottle_500ml_sim.usd` 的 `/Bottle500`
- Bottle SHA-256：`16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e`

输出目录包含：

- `calibration.json`：列向量、米、行主序、`wxyz` 的显式变换契约和源文件冻结信息；
- `calibration.usda`：OpenCV optical 到 USD Camera 的 `Rx(π)` 轴转换、`CameraHigh` 和三个 Bottle ghost prim；
- `calibrated_review.usda`：校准层强于冻结源层的独立组合 Stage。

导出器不修改源 Stage。OpenUSD/Isaac 环境的可重复读回审计命令：

```bash
cd /home/eii/project/openpi0.5-rtc-reward-learning
PYLIB=$(dirname $(find /home/eii/.local/share/uv/python -name 'libpython3.11.so.1.0' -print | head -n 1))
USD_ROOT=.venv_issac/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
PYTHONPATH="$USD_ROOT" LD_LIBRARY_PATH="$USD_ROOT/bin:$PYLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  .venv_issac/bin/python aloha_calibration_workbench/backend/scripts/audit_calibration_bundle.py \
  <export-dir>/calibrated_review.usda <export-dir>/calibration.json \
  --json-output <export-dir>/openusd_audit.json
```

审计只有在组合 Stage 可打开、相机矩阵读回一致、Bottle 引用含 Mesh 且三者世界 AABB 有限时才输出 `CALIBRATION_BUNDLE_OPENUSD_AUDIT_PASS`。

## 验证

```bash
cd /home/eii/project/openpi0.5-rtc-reward-learning/aloha_calibration_workbench
npm test -- --run
npm run build
PYTHONPATH=$PWD/backend backend/.venv/bin/python -m pytest backend/tests -q
```

测试不会启动相机、机器人或 Isaac Sim GUI。
