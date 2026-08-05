# ALOHA 四相机标定工作台

交互式标定工作台目前实现阶段 0（只读预检）和阶段 1 的单相机内参验证采集。默认启动仍是纯 Preview；只有显式 `dev:live` 模式、预检通过且用户点击启动后，103 才会打开一台 RealSense 彩色流。

## 本地预览

```bash
npm install
npm run dev
```

在 `101` 本机打开 `http://127.0.0.1:4173`。

## 只读预检模式

阶段 0 使用两个 localhost-only 服务：`103` 的 capture-agent 枚举相机、生产 Profile 和设备占用，`101` 的 orchestrator 创建不可变预检会话。两个后端都不提供机器人命令。

```bash
cd backend
/home/eii/.local/bin/uv sync --dev
/home/eii/.local/bin/uv run aloha-calibration-capture-agent
```

capture-agent 只绑定 103 的 `127.0.0.1:8017`。在 101 建立本地转发，将它暴露为 101 的同一 localhost 端口：

```bash
ssh -N -L 8017:127.0.0.1:8017 192.168.1.103
```

随后在 101 分别启动 orchestrator 和显式 live 前端：

```bash
cd backend
/home/eii/.local/bin/uv run aloha-calibration-orchestrator

cd ..
npm run dev:live
```

默认 `npm run dev` 始终保持纯 Preview，不产生任何 API 请求。

如需先在 103 命令行验证设备而不启动常驻服务：

```bash
cd /home/eii/openpi0.5-rtc-reward-learning/aloha_calibration_workbench/backend
/home/eii/.local/bin/uv run aloha-calibration-preflight
```

该命令不会启动 RealSense pipeline。只有四台身份、生产 Profile 和独占权全部通过时才输出 `READY`；不能证明独占时输出 `BLOCKED`。

## 阶段 1：cam_high 内参与正式 Profile

当前标定板的冻结定义如下：

- `DICT_5X5_100`
- `7 × 5` squares
- `squareLength = 0.030 m`
- `markerLength = 0.022 m`
- 文件：`~/Downloads/ALOHA_D405_calibration_starter/01_charuco_7x5_square30_marker22_A4_landscape.pdf`
- 打印：A4 横向、100%/Actual size；开始前实测 PDF 底部检查线为 100 mm

操作顺序：

1. 在 live 页面运行只读预检。
2. 只有页面显示 `PREFLIGHT_READY` 后，点击“启动 cam_high 内参采集”。
3. 系统再次核验身份、正式 Profile 与独占权，然后只启动序列号 `130322270656` 的 `640 × 480 @ 60 Hz RGB8` 彩色流。
4. 面向 cam_high 手持刚性 ChArUco 板，先让全板位于中央并避免反光；实时画面会叠加 marker、ChArUco corners 和 factory K/D 求得的坐标轴。
5. 点击“采集当前 ChArUco 帧”写入不可覆盖的原始 PNG 与 JSON。每第五个样本预先分到 `HELD_OUT`，其余进入 `SOLVE`。
6. 结束时点击“停止 cam_high”，释放唯一活动 pipeline。

原始采集只保存在 103 的仓库内 `.calibration_captures/<session>/<role>/`，factory intrinsics 在启动时单独冻结。网页显示的角点数、清晰度、曝光裁剪与 factory K/D 重投影误差是诊断证据，不是 NVIDIA 官方 PASS 门限，也不会自动改写相机 EEPROM。

## 安全边界

- 不包含 WebSocket、机器人控制或自动建立 SSH 隧道的代码。
- 默认 Preview 不产生 API 请求；显式 live 模式经过预检门禁后才允许单相机采集。
- 不包含机器人运动按钮或命令接口。
- 阶段 1 只启用彩色流，深度流关闭；不会改变曝光、固件或 EEPROM。
- 同一时刻最多一台相机；当前流程只开放 `cam_high`，其它角色仍为待机。
- Solve、Validate 与 Export 仍为 Preview，不能把“已采样”解释为“内参已验证”或“系统标定成功”。

## 验证

```bash
npm test
npm run build
```
