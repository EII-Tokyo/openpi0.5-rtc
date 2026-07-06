# AGENTS

## Remote project paths
- On `192.168.1.103`, the user's robot project is `~/openpi0.5-rtc-reward-learning` (`/home/eii/openpi0.5-rtc-reward-learning`).
- Strong constraint for `192.168.1.103`: do not modify code outside `/home/eii/openpi0.5-rtc-reward-learning` for this user's robot project. This includes editing files, applying patches, copying files, rsyncing, running formatters, running git commands that change files, or running project scripts with a working directory outside `/home/eii/openpi0.5-rtc-reward-learning`.
- Do not use or modify `/home/eii/openpi0.5-rlt` for this user's robot project; that path belongs to another project.
- Before copying files, restarting containers, or inspecting remote code on `192.168.1.103`, verify the working directory is `/home/eii/openpi0.5-rtc-reward-learning`.
- If a command on `192.168.1.103` would touch any path outside `/home/eii/openpi0.5-rtc-reward-learning`, stop and ask the user for explicit approval first.
- For complex `192.168.1.103` inspections or statistics, do not embed Python/awk/jq/JSON-heavy logic in a nested one-line SSH command. Use a checked-in script, a single-quoted here-doc, or a `just`/Fabric wrapper that first runs `cd /home/eii/openpi0.5-rtc-reward-learning`.
- Any project `just` recipe, Fabric task, or Ansible playbook targeting `192.168.1.103` must preserve the same project boundary: start from `/home/eii/openpi0.5-rtc-reward-learning`, avoid `/home/eii/openpi0.5-rlt`, and avoid writes outside the project unless the user explicitly approves.
- Prefer this command shape for ad hoc multi-line 103 work:
  - `ssh 192.168.1.103 <<'REMOTE'`
  - `set -euo pipefail`
  - `cd /home/eii/openpi0.5-rtc-reward-learning`
  - project commands
  - `REMOTE`
- For reusable 103 tasks, prefer adding local scripts under this repository and syncing them to `/home/eii/openpi0.5-rtc-reward-learning` before execution. This avoids shell quoting bugs such as losing literal paths like `/app/replay`.
- Strong checkpoint constraint for `192.168.1.103`: user-trained checkpoints for this project must live under `/home/eii/openpi0.5-rtc-reward-learning/checkpoints` and be mounted into containers as `/app/checkpoints`.
- Do not load this project's VLA/RLToken checkpoints from `/home/eii/openpi0.5-rtc/checkpoints`; that path belongs outside this project boundary.
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

## Backend test environment
- Local backend pytest runs are not inside the 103 Docker containers, so they must not rely on container-only paths such as `/app/segment_db/segments.sqlite3`.
- Backend tests under `voice_assistant_web/backend/app` use a test-only `conftest.py` to set `RLT_SEGMENT_DB_PATH`, `RLT_STATE_PATH`, and `EII_PILOT_ENABLE_ROS` before importing backend modules.
- Keep those test defaults isolated and temporary. Do not change production defaults, `.env`, or compose mounts to make local tests pass.
- On `192.168.1.103`, `/app/segment_db/segments.sqlite3` is valid only inside the running containers because compose mounts `/data/openpi0.5-rtc-reward-learning/segment_db` there. The 103 host itself should use host paths under `/data/openpi0.5-rtc-reward-learning` when a host-side command needs the segment DB.

## VLA / RLToken checkpoint constraints
- Strong constraint: for rinse / bottle-mouth insertion work that needs `cam_low`, do not use the `cam3` VLA or any RLToken checkpoint derived from it. The `cam3` checkpoint does not include `cam_low`, so it cannot be treated as a full camera checkpoint for judging bottle-mouth and pipe alignment.
- Correct full-camera VLA checkpoint with `cam_low`:
  - Config/checkpoint family: `eii_rinse_11repo_cam4_fullft`
  - Host path on `192.168.1.103`: `/home/eii/openpi0.5-rtc-reward-learning/checkpoints/eii_rinse_11repo_cam4_fullft/rinse_11repo_insertx5_fullft_bs256_nw64_fsdp8_20260513/9000`
  - Local copied path when present: `/home/eii/project/openpi0.5-rtc-reward-learning/checkpoints/eii_rinse_11repo_cam4_fullft/rinse_11repo_insertx5_fullft_bs256_nw64_fsdp8_20260513/9000`
  - Container path: `/app/checkpoints/eii_rinse_11repo_cam4_fullft/rinse_11repo_insertx5_fullft_bs256_nw64_fsdp8_20260513/9000`
  - Cameras: `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist`.
- Historical cam4 RLToken checkpoint derived from the cam4 VLA above, but no longer the default for new data collection:
  - Config: `eii_rinse_11repo_cam4_fullft_rl_token_small_query`
  - Checkpoint: `rinse_11repo_rl_token_small_query_512_from_9000_20260615/9999`
  - Host path on `192.168.1.103`: `/home/eii/openpi0.5-rtc-reward-learning/checkpoints/eii_rinse_11repo_cam4_fullft_rl_token_small_query/rinse_11repo_rl_token_small_query_512_from_9000_20260615/9999`
  - Container path: `/app/checkpoints/eii_rinse_11repo_cam4_fullft_rl_token_small_query/rinse_11repo_rl_token_small_query_512_from_9000_20260615/9999`
  - This is the correct 2026-06-15 cam4 small-query RLToken checkpoint for historical cam4 work; it was initialized from `eii_rinse_11repo_cam4_fullft/.../9000/params`.
- Active RLToken checkpoint for new RLT data collection, replay re-encoding, critic training, and actor training:
  - Config: `eii_rinse_11repo_cam4_fullft_rl_token_lower_right_query_4layer`
  - Checkpoint family: `rlt_lower_right_rl_token_ablation_20260701`
  - Host path on `192.168.1.103`: `/home/eii/openpi0.5-rtc-reward-learning/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint`
  - Local path when present: `/home/eii/project/openpi0.5-rtc-reward-learning/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint`
  - Container path: `/app/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint`
  - Cameras used for visual information: `cam_low`, `cam_right_wrist`.
  - Output z dimension: `2048`.
  - Strong constraint: new key-region data collection on `192.168.1.103` must set `RLT_RL_TOKEN_CHECKPOINT_PATH=/app/checkpoints/rlt_lower_right_rl_token_ablation_20260701/BEST/checkpoint` before starting `openpi_server` or `rlt_warmup_runtime`.
  - Strong constraint: do not collect new RLT replay with the old 512-dim small-query RLToken unless the user explicitly requests a controlled ablation. Mixing 512-dim and 2048-dim `z_rl` replay in one critic/actor training run is invalid.
  - If existing 512-dim replay must be reused, re-encode it into a separate lower-right directory such as `/data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions_lower_right_z2048_4layer` or `/data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions_clean_lower_right_z2048_4layer`; never overwrite the original replay shards.
- Wrong checkpoint for rinse / bottle-mouth insertion if `cam_low` is required:
  - VLA family: `eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo`
  - RLToken config: `eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo_rl_token_query`
  - RLToken checkpoint: `rl_token_2048_enc4_dec4_query_from_19000_20260528/12000`
  - Container path: `/app/checkpoints/eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo_rl_token_query/rl_token_2048_enc4_dec4_query_from_19000_20260528/12000`
  - This checkpoint was derived from the cam3 VLA `no_rinse_cam3_fullft_return_home_29repo_bs256_nw64_fsdp4_20260520/19000`; do not use it to train or evaluate critic/actor for tasks that depend on `cam_low`.
- Historical audit note:
  - `2026-05-28`: the cam3-derived RLToken configs were added.
  - `2026-05-29` through `2026-06-15 09:50 +0900`: RLT defaults used the cam3-derived RLToken path.
  - `2026-06-15 09:50 +0900`: defaults moved to the cam4 VLA base checkpoint.
  - `2026-06-15 16:34 +0900`: defaults moved to the correct cam4 RLToken small query checkpoint.
  - `2026-07-02`: defaults moved to the lower+right 4-layer RLToken checkpoint for new RLT data collection and training.
- Before training critic/actor, re-encoding `z_rl`, or starting `openpi_server`/`rlt_warmup_runtime`, verify the active `--policy.config`, `--policy.dir`, `--model-dir`, and `RLT_RL_TOKEN_CHECKPOINT_PATH` are from the active lower+right 4-layer RLToken family unless the user explicitly requests a controlled ablation.
- Strong runtime constraint for robot actor tests: the required online `z_rl` path is the B-group VLA same-forward method. Main `openpi_server` must keep the cam4 VLA policy (`eii_rinse_11repo_cam4_fullft`) for action inference and enable `RLT_SAME_FORWARD_RL_TOKEN_ENABLED=1` so `z_rl` is encoded from the same VLA forward pass using the lower+right autoencoder.
- Hard ban for normal robot actor tests: do not enable the sidecar RLToken path (`rlt_token_server`, `--rlt-token-port 8002`, or any fallback that calls `infer_rl_token()` after `policy.infer()` misses `z_rl`). If actor startup or intervention cannot get `z_rl` from the VLA same-forward path, treat it as a configuration/runtime error and stop to investigate; do not silently fall back to sidecar re-encoding.

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

## Canonical RLT replay data
- Current canonical RLT replay data uses the lower+right 4-layer RLToken encoder with `z_rl` / `next_z_rl` dimension `2048`.
- Strong constraint: before any critic or actor training, newly collected or cleaned rollout/key-region data must first be converted offline into formal `paper_subsampled_anchor` replay shards. Do not train directly from online recorder outputs, runtime cache-block replay, cleaned crop NPZs, raw rollouts, or fixed/aligned segment artifacts.
- Do not train by scanning mixed legacy replay directories directly. Train from the canonical manifests unless the user explicitly requests an ablation.
- 2026-07-06 online RLT replay rescue:
  - Treat raw 2026-07-06 replay as source material only; do not train directly from its saved `z_rl/proprio`.
  - Split it into two source groups before rebuilding:
    - `base142_legacy_unmarked`: 09:06-10:27, 142 shards, the data used to train the first actor that day.
    - `actor93_runtime_cache_block`: 13:21-14:40, 93 shards, collected by the actor trained from the first group.
  - Rebuild each group separately with `scripts/rebuild_online_rollout_paper_anchor_replay.py --collection-group base142|actor93` so outputs are formal `paper_subsampled_anchor` replay with VLA same-forward lower/right RLToken.
  - Treat `actor93_runtime_cache_block` as high-risk off-policy data. It may help critic boundary learning after rebuild, but it must not be directly treated as high-quality actor imitation data.
  - Use `scripts/plan_20260706_data_rescue.py` to generate A-only / A+B rescue commands and combined manifests; do not hand-build mixed manifests for this batch.
- Legacy 512-dim replay roots such as `rlt_key_regions`, `rlt_key_regions_clean`, and `human_expert_no_actor_q_cam4_provenance_20260629` must not be mixed into a 2048 training run.
- Strong requirement: formal critic/actor training replay should match the RLT paper's subsampled transition semantics:
  - each replay row is `x_i, action[i:i+C], x_{i+C}`;
  - `z_rl/proprio` must be re-encoded from the row's own stride anchor frame `i`, not copied from an RTC/action-cache block;
  - regenerated shards should mark `replay_state_grain=paper_subsampled_anchor`;
  - `fixed_segments` / `rl_token_reencoded_aligned_to_proprio_segments` shards are only audit/ablation artifacts and must not be used for formal training manifests.
- Local canonical root:
  - Data: `/home/eii/data/openpi0.5-rtc-reward-learning/replay/canonical_2048`
  - Local-only manifest: `/home/eii/data/openpi0.5-rtc-reward-learning/manifests/canonical_2048`
  - 103 mirrored manifest with local filesystem paths: `/home/eii/data/openpi0.5-rtc-reward-learning/manifests/canonical_2048_from_103`
- `192.168.1.103` canonical root:
  - Data: `/data/openpi0.5-rtc-reward-learning/replay/canonical_2048`
  - Manifest: `/data/openpi0.5-rtc-reward-learning/manifests/canonical_2048`
- Historical note: the 2026-07-02 103 canonical manifest contained fixed-segment bootstrap data and is now considered stale for formal training under the paper-aligned replay requirement. Do not reuse it for new formal critic/actor training without rebuilding the bootstrap/clean shards from original rollouts.
- As of 2026-07-02, that stale 103 canonical manifest contained 384 replay shards and 18,721 transitions:
  - `bootstrap`: 146 shards (`train`: 117, `holdout`: 29), source `current109_37_actor6000_20260630`
  - `rlt_raw`: 178 shards, source 103 online/raw actor data
  - `rlt_clean`: 1 shard, source 103 cleaned data currently available under `/data`
  - `expert`: 59 shards, source human expert lower+right 4-layer encoding
- The canonical train and holdout split currently has zero `key_region_id` overlap. Preserve that invariant.
- If a new 512-dim or old cam4-small-query shard is discovered, re-encode it into a separate lower+right 2048 directory first; never overwrite the original shard.
- For human expert / Expert-for-D data, do not re-encode by rewriting old Q replay `z_rl` in place. The correct source-of-truth chain is:
  - Crop JSON: `/home/eii/data/openpi0.5-rtc-reward-learning/replay/discriminator_expert_crops`
  - Original LeRobot cache: `/home/eii/.cache/huggingface/lerobot/lyl472324464/<dataset_id>`
  - Encode frame-level z cache from original LeRobot parquet/video using the lower+right 4-layer RLToken checkpoint.
  - Convert crop JSON + parquet action/state + the new z cache into trainable Q replay.
- For human expert lower+right conversion, prefer the z-only deterministic code path `Policy.infer_rl_token()` over `Policy.infer()`. `Policy.infer()` may go through action sampling before returning `z_rl`, which is unnecessary for replay re-encoding and makes audits harder.
- Current fully regenerated human expert output from original LeRobot data:
  - z cache: `/home/eii/data/openpi0.5-rtc-reward-learning/replay/expert_crop_z_rl_cache_lower_right_4layer_from_raw_zonly_20260703`
  - Q replay: `/home/eii/data/openpi0.5-rtc-reward-learning/replay/human_expert_no_actor_q_lower_right_4layer_from_raw_zonly_20260703`
  - manifest: `/home/eii/project/openpi0.5-rtc-reward-learning/local_rlt_manifests/expert_from_raw_20260703/human_expert_no_actor_q_lower_right_4layer_from_raw_zonly_20260703.jsonl`
  - audit: `/home/eii/project/openpi0.5-rtc-reward-learning/local_rlt_manifests/expert_from_raw_20260703/audit_from_raw_zonly_20260703.json`
  - verified counts: 59 Q replay shards, 58 episode z-cache files, 2,896 transitions, all `z_rl` / `next_z_rl` dimension 2048, all `action == reference_action`.
- Training selection rule for human expert data:
  - Use only the z-only Q replay and manifest above for human expert lower+right 4-layer training.
  - Do not use deleted legacy / non-z-only expert outputs such as `human_expert_no_actor_q_lower_right_4layer_20260701`, `expert_crop_z_rl_cache_lower_right_4layer_20260701`, or `human_expert_no_actor_q_lower_right_4layer_from_raw_20260703`.
  - If the z-only output is missing or suspected stale, regenerate it from the crop JSON and original LeRobot parquet/video via `Policy.infer_rl_token()`; do not restore or train from the old `Policy.infer()`-generated expert replay.
  - Any canonical 2048 manifest that includes human expert data should point to the z-only manifest above, not to an old 20260701 or non-z-only manifest.

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
