# AGENTS

## Image flow (data collection)
- Active collection code path for `/home/eii/aloha-2.0`:
  - `/home/eii/aloha-2.0/aloha/robot_utils.py` uses `CvBridge.imgmsg_to_cv2(..., desired_encoding='passthrough')`.
  - In the ROS2 collection setup we verified the source camera topic encoding is `rgb8`, so the in-memory image used by collection is RGB.
- HDF5 export bug in `/home/eii/aloha-2.0/scripts/record_episodes_copy.py`:
  - The script passes that RGB numpy array directly into `cv2.imencode(".jpg", image, ...)`.
  - OpenCV assumes BGR input, so the saved JPEG bytes have swapped color semantics relative to the original RGB scene.
- JPEG quality:
  - `/home/eii/aloha-2.0/scripts/record_episodes_copy.py` now uses `JPEG_QUALITY = 100`.
  - `/home/eii/openpi0.5-rtc/examples/aloha_real/hdf5_utils.py` also uses JPEG quality `100`.

## Image flow (training)
- LeRobot generation path currently used for export:
  - The dataset builder decodes HDF5 JPEG bytes with `cv2.imdecode(..., cv2.IMREAD_COLOR)`.
  - OpenCV returns BGR arrays, and those arrays are written directly into `LeRobotDataset.add_frame(...)`.
  - Result: current LeRobot training datasets effectively store BGR-valued images.
- LeRobot loading:
  - Image datasets: `lerobot.datasets.utils.hf_transform_to_torch()` converts PIL images to CHW float32 in `[0, 1]`.
  - Video datasets: `lerobot.datasets.video_utils.decode_video_frames_*()` decodes frames without any explicit BGR/RGB correction; whatever channel order was encoded is preserved numerically.
- OpenPI training transforms:
  - `AlohaInputs` in `/home/eii/openpi0.5-rtc/src/openpi/policies/aloha_policy.py` only normalizes dtype/layout and camera key names; it does not swap RGB/BGR.
  - `LeRobotAlohaDataConfig.image_size` in `/home/eii/openpi0.5-rtc/src/openpi/training/config.py` is passed into `ModelTransformFactory`, which applies `ResizeImages(...)`.
- Model-side image normalization:
  - `/home/eii/openpi0.5-rtc/src/openpi/models/model.py::Observation.from_dict()`
  - If image dtype is `uint8`, it converts image values from `[0, 255]` to `float32` in `[-1, 1]`.
  - This is the first place images are numerically normalized for the model.

## Image flow (inference)
- Active runtime path in this repo uses `docker compose` and ROS1:
  - `/home/eii/openpi0.5-rtc/docker-compose.yml` starts `aloha_ros_nodes`, which launches `/home/eii/openpi0.5-rtc/third_party/aloha/launch/ros_nodes.launch`.
  - That launch starts `/home/eii/openpi0.5-rtc/third_party/aloha/aloha_scripts/realsense_publisher.py`.
- Local submodule note:
  - `third_party/aloha` is a git submodule. Local changes there are not pushed with the main repo unless the submodule itself is updated separately.
- Current local `realsense_publisher.py` logic:
  - Configures RealSense with `rs.format.rgb8`.
  - Publishes `RGBGrayscaleImage.images[0]` with `encoding="rgb8"`.
  - No manual channel reversal remains in the local file.
- Runtime subscriber path:
  - `/home/eii/openpi0.5-rtc/examples/aloha_real/robot_utils.py` reads `data.images[0]` with `desired_encoding="passthrough"`.
  - So runtime now preserves whatever encoding the publisher set; with the local publisher change, inference images stay RGB.
- Environment and policy path:
  - `/home/eii/openpi0.5-rtc/examples/aloha_real/env.py` currently forwards `obs["images"]` as-is; the older resize/CHW conversion code is commented out.
  - `/home/eii/openpi0.5-rtc/src/openpi/policies/policy_config.py` builds transforms in this order:
    - `InjectDefaultPrompt`
    - `AlohaInputs`
    - `Normalize`
    - `ResizeImages`
    - `TokenizePrompt`
  - `AlohaInputs` does not swap RGB/BGR; it only converts CHW->HWC if needed and remaps:
    - `cam_high -> base_0_rgb`
    - `cam_left_wrist -> left_wrist_0_rgb`
    - `cam_right_wrist -> right_wrist_0_rgb`
- Current intended inference color semantics:
  - With the local ROS1 publisher fix plus runtime `passthrough`, images should remain RGB all the way into `AlohaInputs`.

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
