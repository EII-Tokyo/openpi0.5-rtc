# AGENTS

## Image flow (inference)
- Source: `/home/eii/openpi0.5-rtc/examples/aloha_real/real_env.py` -> `RealEnv.get_observation()` -> `obs["images"]` from `ImageRecorder.get_images()` (HWC uint8 from RealSense).
- Runtime adapter: `/home/eii/openpi0.5-rtc/examples/aloha_real/env.py`
  - Removes `*_depth` keys.
  - Resize + pad to target size via `openpi_client.image_tools.resize_with_pad`.
  - Convert to CHW via `einops.rearrange(img, "h w c -> c h w")`.
- Policy server:
  - `serve_policy.py` -> `policy_config.create_trained_policy(...)` -> input transforms include `ResizeImages(224, 224)` from `ModelTransformFactory`.
  - `AlohaInputs` (`/home/eii/openpi0.5-rtc/src/openpi/policies/aloha_policy.py`) converts CHW to HWC before model preprocessing.

## Image flow (training)
- Dataset creation: `/home/eii/openpi0.5-rtc/examples/aloha_real/convert_aloha_data_to_lerobot.py`
  - Stores images as HWC uint8 in LeRobot dataset.
- LeRobot loader (`lerobot.datasets.lerobot_dataset.LeRobotDataset`):
  - `hf_transform_to_torch` converts PIL images to `torchvision.transforms.ToTensor()`.
  - This yields CHW float32 in [0, 1].
- Training transforms (`/home/eii/openpi0.5-rtc/src/openpi/training/config.py`):
  - `ModelTransformFactory` adds `ResizeImages(224, 224)` for PI0/PI05/FAST configs.
  - `AlohaInputs` (`/home/eii/openpi0.5-rtc/src/openpi/policies/aloha_policy.py`) converts CHW -> HWC for model input.

## Gripper flow

### Data collection
- State (gripper qpos) source:
  - `aloha-2.0/aloha/real_env.py`
  - Uses `FOLLOWER_GRIPPER_JOINT_NORMALIZE_FN(bot.gripper.get_gripper_position())`.
  - `get_gripper_position()` returns joint angle (not linear position); this is a known issue.
- Action (gripper) source:
  - `aloha-2.0/aloha/real_env.py` uses `LEADER_GRIPPER_JOINT_NORMALIZE_FN(robot.gripper.get_gripper_position())`.

### Training
- State transform:
  - `/home/eii/openpi0.5-rtc/src/openpi/policies/aloha_policy.py::_decode_state`
  - If `adapt_to_pi=True`: joint flip + `_gripper_to_angular` on indices `[6, 13]`.
- Action transform:
  - `/home/eii/openpi0.5-rtc/src/openpi/policies/aloha_policy.py::_encode_actions_inv`
  - If `adapt_to_pi=True`: joint flip + `_gripper_from_angular_inv` on `[6, 13]`.

### Inference
- State acquisition:
  - `/home/eii/openpi0.5-rtc/examples/aloha_real/real_env.py::get_qpos`.
  - Uses `PUPPET_GRIPPER_POSITION_NORMALIZE_FN` on indices `[6]` for left and right gripper.
- Action decoding (model -> robot space):
  - `/home/eii/openpi0.5-rtc/src/openpi/policies/aloha_policy.py::_encode_actions`.
- Actuation:
  - `/home/eii/openpi0.5-rtc/examples/aloha_real/real_env.py::set_gripper_pose` uses
    `PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN` then publishes to `/puppet_*` gripper command topics.
