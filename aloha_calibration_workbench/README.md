# ALOHA 四相机标定工作台（Preview）

这是交互式标定系统的第一阶段视觉原型。当前版本只使用浏览器内的模拟数据，不连接 `103`、RealSense、Isaac Sim 或真实机器人。

## 本地预览

```bash
npm install
npm run dev
```

在 `101` 本机打开 `http://127.0.0.1:4173`。

## 安全边界

- 不包含 API client、WebSocket、SSH 或设备发现代码。
- 不包含机器人运动按钮或命令接口。
- 所有采集、求解、导出按钮均为 Preview 状态。
- 页面中的图像、指标、变换和状态均为合成示例，不得作为标定证据。

## 验证

```bash
npm test
npm run build
```
