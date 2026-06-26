# AGENTS

## Remote project paths
- On `192.168.1.103`, the user's robot project is `~/openpi0.5-rtc-reward-learning` (`/home/eii/openpi0.5-rtc-reward-learning`).
- Strong constraint for `192.168.1.103`: do not modify code outside `/home/eii/openpi0.5-rtc-reward-learning` for this user's robot project. This includes editing files, applying patches, copying files, rsyncing, running formatters, running git commands that change files, or running project scripts with a working directory outside `/home/eii/openpi0.5-rtc-reward-learning`.
- Do not use or modify `/home/eii/openpi0.5-rlt` for this user's robot project; that path belongs to another project.
- Before copying files, restarting containers, or inspecting remote code on `192.168.1.103`, verify the working directory is `/home/eii/openpi0.5-rtc-reward-learning`.
- If a command on `192.168.1.103` would touch any path outside `/home/eii/openpi0.5-rtc-reward-learning`, stop and ask the user for explicit approval first.
- `uv` locations on `192.168.1.103`:
  - Host: `/home/eii/.local/bin/uv` exists but is not on the default non-interactive SSH `PATH`.
  - `openpi_server` container: `/usr/bin/uv`.
  - `rlt_warmup_runtime` container: `/usr/bin/uv`.
  - For compose commands that run `uv run ...`, assume container path `/usr/bin/uv`; do not rediscover this each time.
- When the user asks to stop all of their robot containers on `192.168.1.103`, do it in one compose command from the verified project directory:
  - `cd /home/eii/openpi0.5-rtc-reward-learning && docker compose --profile rlt --profile legacy --profile train stop`
  - The compose file sets `name: openpi_reward_learning_eii`. Include all profiles so `rlt_warmup_runtime`, legacy `runtime`, and `rlt_online_trainer` are stopped together with the non-profile services.
- To start the user's robot for actor testing on `192.168.1.103`, do not run a broad `docker compose --profile rlt up -d`. Start explicit services to avoid accidentally starting legacy runtime or online trainer:
  - `cd /home/eii/openpi0.5-rtc-reward-learning && docker compose --profile rlt up -d --no-build ros_master redis openpi_server aloha_ros_nodes eii_pilot_backend eii_pilot_frontend eii_pilot_webrtc_media rlt_warmup_runtime`
  - For fast runtime-only restarts after `openpi_server` is warm: `docker compose --profile rlt up -d --no-build --force-recreate --no-deps rlt_warmup_runtime`.
- `openpi_server` startup optimization:
  - The default compose command skips non-RTC warmup with `--no-warmup-non-rtc`; RTC warmup is still enabled because robot control uses the RTC path.
  - `openpi_server` uses a project-local trusted JAX persistent compilation cache mounted at `${OPENPI_JAX_CACHE_DIR:-./.jax_cache/openpi_server}:/app/.jax_cache`.
  - `.env` may set `OPENPI_JAX_CACHE_DIR=./.jax_cache/openpi_server`. Do not point this cache at a world-writable or untrusted shared directory.
  - The first cold start populates the cache; later starts can reuse compatible JAX/XLA compilations when code, shapes, XLA flags, jaxlib version, and GPU model are unchanged.
  - Measured on `192.168.1.103` after this optimization: first cache-fill start took about `62.7s` for `openpi_server` ready and `79.4s` for actor/critic ready; second cache-reuse start took about `19.7s` for `openpi_server` ready and `38.7s` for actor/critic ready.
  - For fastest repeated tests, keep `openpi_server` running and only recreate `rlt_warmup_runtime` with the fast runtime-only restart command above.
- The preferred actor/critic checkpoint for 103 robot tests is project-local and contains both `actor.msgpack` and `critic.msgpack`:
  - Host path: `/home/eii/openpi0.5-rtc-reward-learning/local_rlt_runs/rlt_unified_468_td3_burn5000_actor10000/inference_actor/00012000`
  - Container path: `/app/local_rlt_runs/rlt_unified_468_td3_burn5000_actor10000/inference_actor/00012000`
  - Set `.env`: `RLT_ACTOR_CHECKPOINT_PATH=/app/local_rlt_runs/rlt_unified_468_td3_burn5000_actor10000/inference_actor/00012000`
  - Do not rely on `/app/rlt_online/run/inference_actor/LATEST` for robot testing unless the user explicitly asks to use online-training output.
- `rlt_online` root ownership note:
  - `rlt_warmup_runtime` and `rlt_online_trainer` run as root inside their containers.
  - The old compose mount `/data/openpi0.5-rtc-reward-learning/rlt_online:/app/rlt_online` means container-created `/app/rlt_online/run` files become root-owned on the host.
  - Avoid using `/app/rlt_online/run/inference_actor/LATEST` as the default robot actor path; use the project-local `local_rlt_runs/.../00012000` path above.

## Local key region annotation on machine 101
- The local machine `101` is for offline key region data annotation only. Do not treat `http://127.0.0.1:3011/` as a robot-control UI, and do not expect local key presses there to control the robot on `192.168.1.103`.
- For local offline key region annotation, the correct data root is:
  - `/home/eii/data/openpi0.5-rtc-reward-learning`
- The local annotation service should use these mounts:
  - `/home/eii/data/openpi0.5-rtc-reward-learning/rollouts:/app/rollouts`
  - `/home/eii/data/openpi0.5-rtc-reward-learning/replay:/app/replay`
  - `/home/eii/data/openpi0.5-rtc-reward-learning/segment_db:/app/segment_db`
  - `/home/eii/data/openpi0.5-rtc-reward-learning:/home/eii/data/openpi0.5-rtc-reward-learning`
- The extra full-root mount is required because the segment DB may store absolute clean shard paths such as `/home/eii/data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions_clean/...`; without this mount, the backend may show clean/cropped records as missing.
- The 2026-06-22 third-batch cleaned key region data is under:
  - `/home/eii/data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions_clean/twist_off_the_bottle_cap/2026-06-22`
  - `/home/eii/data/openpi0.5-rtc-reward-learning/rollouts/key_regions/twist_off_the_bottle_cap/2026-06-22`
  - `/home/eii/data/openpi0.5-rtc-reward-learning/segment_db/segments.sqlite3`
- Do not assume `/home/eii/project/openpi0.5-rtc-reward-learning-local-data` is the active annotation dataset. It previously contained raw 2026-06-22 files but not the cleaned/cropped 2026-06-22 metadata, which made the UI appear to have no annotations.
- Do not pull, rsync, copy, or overwrite data from `192.168.1.103` unless the user explicitly asks for a data transfer. Before any data movement, first locate and inspect local data under `/home/eii/project/openpi0.5-*` and `/home/eii/data/openpi0.5-rtc-reward-learning`.
- Previous mistake to avoid: the local service was started against `/home/eii/project/openpi0.5-rtc-reward-learning-local-data`, then data was pulled from `192.168.1.103` before confirming the user's already-cleaned local dataset. The correct fix was only to switch the service mounts to `/home/eii/data/openpi0.5-rtc-reward-learning`.

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

## Training notes

### 2026-05-06 LoRA benchmark on single RTX PRO 6000 Blackwell
- Remote machine:
  - `ssh -p 31483 root@147.185.60.9`
- Launcher/config:
  - Base config: `eii_rinse_cam4_lora`
  - 6 repos:
    - `lyl472324464/2026-05-04_direction-lerobot-with-rinse`
    - `lyl472324464/2026-05-04_direction-twist-water-lerobot-with-rinse`
    - `lyl472324464/2026-05-01_water1-lerobot-with-rinse`
    - `lyl472324464/2026-05-04_turn_over-lerobot-with-rinse`
    - `lyl472324464/2026-05-03_turn_over-lerobot-with-rinse`
    - `lyl472324464/2026-05-01_turn_over-lerobot-with-rinse`
  - 4 cameras from `eii_rinse_cam4_lora`:
    - `cam_high`
    - `cam_left_wrist`
    - `cam_right_wrist`
    - `cam_low`
  - `batch_size=32`
  - `log_interval=10`
  - `num_train_steps=40000`
  - `fsdp_devices=1`
  - `video_memory_num_frames=1`
  - `video_memory_stride_seconds=1.0`
- Verified first batch tensor structure:
  - `base_0_rgb`, `base_1_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` all present with shape `(32, 224, 224, 3)`
  - `tokenized_prompt` log shape `(32, 200)` is text-only; total multimodal token count is about `1024 image + 200 text = 1224`
- Dataloader/step timing on this exact setup:
  - `num_workers=0`:
    - `data_wait_time ~= 1.10s`
    - `train_step_time ~= 3.20s`
    - wall clock step time `~= 4.31s`
  - `num_workers=16`:
    - `data_wait_time ~= 0.03s`
    - `train_step_time ~= 3.20s`
    - wall clock step time `~= 3.23s`
  - Conclusion:
    - On this machine and dataset mix, `num_workers=16` is clearly better than `0`; model time is unchanged and the gain comes from removing dataloader wait.
