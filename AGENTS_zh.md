# AGENTS（中文）

## 图像流程（采集）
- `/home/eii/aloha-2.0` 当前采集链：
  - `/home/eii/aloha-2.0/aloha/robot_utils.py` 用 `CvBridge.imgmsg_to_cv2(..., desired_encoding='passthrough')` 取图。
  - 在 ROS2 采集环境里，我们核实过上游相机 topic 的编码是 `rgb8`，所以采集阶段内存中的图像是 RGB。
- HDF5 导出中的颜色问题：
  - `/home/eii/aloha-2.0/scripts/record_episodes_copy.py` 直接把 RGB numpy 数组传给 `cv2.imencode(".jpg", image, ...)`。
  - OpenCV 默认按 BGR 解释输入，因此保存出来的 JPEG 相对原始 RGB 场景存在通道语义错位。
- JPEG 质量：
  - `/home/eii/aloha-2.0/scripts/record_episodes_copy.py` 已改为 `JPEG_QUALITY = 100`。
  - `/home/eii/openpi0.5-rtc/examples/aloha_real/hdf5_utils.py` 也已改为 JPEG 质量 `100`。

## 图像流程（训练）
- 当前实际使用的 LeRobot 生成链：
  - 数据构建脚本会对 HDF5 中的 JPEG 字节做 `cv2.imdecode(..., cv2.IMREAD_COLOR)`。
  - OpenCV 返回 BGR 数组，这些数组随后直接写入 `LeRobotDataset.add_frame(...)`。
  - 结果：当前 LeRobot 训练数据集里，图像数值实际上是 BGR。
- LeRobot 读取：
  - 图片版数据：`lerobot.datasets.utils.hf_transform_to_torch()` 用 `ToTensor()` 转成 CHW float32，范围 `[0,1]`。
  - 视频版数据：`lerobot.datasets.video_utils.decode_video_frames_*()` 只负责解码，不会额外做 BGR/RGB 修正；编码时的通道顺序会原样保留为数值。
- OpenPI 训练变换：
  - `/home/eii/openpi0.5-rtc/src/openpi/policies/aloha_policy.py` 里的 `AlohaInputs` 只处理 dtype、layout、相机键名，不会交换 RGB/BGR。
  - `/home/eii/openpi0.5-rtc/src/openpi/training/config.py` 里的 `LeRobotAlohaDataConfig.image_size` 会传给 `ModelTransformFactory`，再应用 `ResizeImages(...)`。
- 模型侧图像归一化：
  - `/home/eii/openpi0.5-rtc/src/openpi/models/model.py::Observation.from_dict()`
  - 如果输入图片是 `uint8`，这里会把像素从 `[0,255]` 转成 `float32` 的 `[-1,1]`。
  - 这是图片第一次真正被数值归一化给模型使用的位置。

## 图像流程（推理）
- 当前实际推理走的是本仓库 `docker compose` + ROS1，而不是 `/home/eii/aloha-2.0` 的 ROS2 采集链。
- `docker compose` 路径：
  - `/home/eii/openpi0.5-rtc/docker-compose.yml`
  - 启动 `aloha_ros_nodes`，其 launch 文件是 `/home/eii/openpi0.5-rtc/third_party/aloha/launch/ros_nodes.launch`
  - launch 内实际起的是 `/home/eii/openpi0.5-rtc/third_party/aloha/aloha_scripts/realsense_publisher.py`
- 子模块说明：
  - `third_party/aloha` 是 git submodule，本地改动不会随主仓库一起 push，除非单独更新 submodule。
- 当前本地 `realsense_publisher.py` 的目标逻辑：
  - RealSense 输出设置为 `rs.format.rgb8`
  - 发布 `RGBGrayscaleImage.images[0]` 时标记 `encoding="rgb8"`
  - 本地文件里已经去掉了手动通道反转
- Runtime 订阅端：
  - `/home/eii/openpi0.5-rtc/examples/aloha_real/robot_utils.py`
  - 当前使用 `imgmsg_to_cv2(..., desired_encoding="passthrough")`
  - 所以订阅端会保留发布端的编码；配合上面的本地 publisher 修复，推理图像应保持 RGB。
- Environment / Policy 路径：
  - `/home/eii/openpi0.5-rtc/examples/aloha_real/env.py` 目前是原样转发 `obs["images"]`，旧的 resize/CHW 转换代码已经注释掉。
  - `/home/eii/openpi0.5-rtc/src/openpi/policies/policy_config.py` 中输入 transforms 顺序是：
    - `InjectDefaultPrompt`
    - `AlohaInputs`
    - `Normalize`
    - `ResizeImages`
    - `TokenizePrompt`
  - `AlohaInputs` 不会做 RGB/BGR 交换，只会在需要时做 CHW->HWC，并把：
    - `cam_high -> base_0_rgb`
    - `cam_left_wrist -> left_wrist_0_rgb`
    - `cam_right_wrist -> right_wrist_0_rgb`
- 当前预期的推理颜色语义：
  - 如果本地 `third_party/aloha` 的 publisher 修复已生效，并且 runtime 仍用 `passthrough`，那么图片应以 RGB 形式一路传到 `AlohaInputs`。

## 夹爪流程

### 采集
- 状态（gripper qpos）来源：
  - `aloha-2.0/aloha/real_env.py`
  - `FOLLOWER_GRIPPER_JOINT_NORMALIZE_FN(bot.gripper.get_gripper_position())`
  - 注意：`get_gripper_position()` 返回关节角度不是位置，这是已知问题。
- 动作（gripper）来源：
  - `aloha-2.0/aloha/real_env.py` 里 `LEADER_GRIPPER_JOINT_NORMALIZE_FN(robot.gripper.get_gripper_position())`。

### 训练
- 状态变换：
  - `/home/eii/openpi0.5-rtc/src/openpi/policies/aloha_policy.py::_decode_state`
  - `adapt_to_pi=True` 时：关节翻转 + `_gripper_to_angular`（索引 `[6, 13]`）。
- 动作变换：
  - `/home/eii/openpi0.5-rtc/src/openpi/policies/aloha_policy.py::_encode_actions_inv`
  - `adapt_to_pi=True` 时：关节翻转 + `_gripper_from_angular_inv`（索引 `[6, 13]`）。

### 推理
- 状态获取：
  - `/home/eii/openpi0.5-rtc/examples/aloha_real/real_env.py::get_qpos`
  - 用 `PUPPET_GRIPPER_POSITION_NORMALIZE_FN` 处理左右夹爪。
- 模型输出动作解码：
  - `/home/eii/openpi0.5-rtc/src/openpi/policies/aloha_policy.py::_encode_actions`
- 下发到机器人：
  - `/home/eii/openpi0.5-rtc/examples/aloha_real/real_env.py::set_gripper_pose`
  - `PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN` -> 发布到 `/puppet_*` gripper command。
