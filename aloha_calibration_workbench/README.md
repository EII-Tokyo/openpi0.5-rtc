# ALOHA 四相机标定工作台（Preview）

这是交互式标定系统的第一阶段视觉原型。当前版本只使用浏览器内的模拟数据，不连接 `103`、RealSense、Isaac Sim 或真实机器人。

## 本地预览

```bash
npm install
npm run dev
```

在 `101` 本机打开 `http://127.0.0.1:4173`。

## 只读预检模式

阶段 0 使用两个 localhost-only 服务：`103` 的 capture-agent 仅枚举相机、生产 Profile 和设备占用，`101` 的 orchestrator 创建不可变预检会话。两个后端都不提供机器人命令或图像 pipeline API。

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

## 安全边界

- 不包含 WebSocket、机器人控制或自动建立 SSH 隧道的代码。
- 默认 Preview 不产生 API 请求；显式 live 模式只调用只读预检 API。
- 不包含机器人运动按钮或命令接口。
- 所有采集、求解、导出按钮均为 Preview 状态。
- 页面中的图像、指标、变换和状态均为合成示例，不得作为标定证据。

## 验证

```bash
npm test
npm run build
```
