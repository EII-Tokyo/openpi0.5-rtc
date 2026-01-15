# AGENTS（中文）

## 图像流程（推理）
- 来源：`/home/eii/openpi0.5-rtc/examples/aloha_real/real_env.py` -> `RealEnv.get_observation()` -> `ImageRecorder.get_images()`（RealSense 的 HWC uint8）。
- Runtime 适配层：`/home/eii/openpi0.5-rtc/examples/aloha_real/env.py`
  - 过滤 `*_depth`。
  - `resize_with_pad` 到目标分辨率。
  - `einops.rearrange(img, "h w c -> c h w")` 转成 CHW。
- Policy server：
  - `serve_policy.py` -> `policy_config.create_trained_policy(...)` -> 输入 transforms 包含 `ResizeImages(224, 224)`（来自 `ModelTransformFactory`）。
  - `AlohaInputs`（`/home/eii/openpi0.5-rtc/src/openpi/policies/aloha_policy.py`）把 CHW 转成 HWC 供模型预处理。

## 图像流程（训练）
- 数据集生成：`/home/eii/openpi0.5-rtc/examples/aloha_real/convert_aloha_data_to_lerobot.py`
  - 写入 LeRobot 数据集时图像是 HWC uint8。
- LeRobot 加载（`lerobot.datasets.lerobot_dataset.LeRobotDataset`）：
  - `hf_transform_to_torch` 用 `torchvision.transforms.ToTensor()` 把 PIL 图像转成 **CHW float32、范围 [0,1]**。
- 训练 transforms（`/home/eii/openpi0.5-rtc/src/openpi/training/config.py`）：
  - `ModelTransformFactory` 对 PI0/PI05/FAST 添加 `ResizeImages(224, 224)`。
  - `AlohaInputs`（`/home/eii/openpi0.5-rtc/src/openpi/policies/aloha_policy.py`）把 CHW -> HWC。

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
