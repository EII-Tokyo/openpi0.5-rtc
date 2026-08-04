# Remote 103 Operations

Read this before copying files, restarting containers, inspecting remote code, running robot services, or changing checkpoints on `192.168.1.103`.

## Project Boundary
- On `192.168.1.103`, the user's robot project is `~/openpi0.5-rtc-reward-learning` (`/home/eii/openpi0.5-rtc-reward-learning`).
- Strong constraint for `192.168.1.103`: do not modify code outside `/home/eii/openpi0.5-rtc-reward-learning` for this user's robot project. This includes editing files, applying patches, copying files, rsyncing, running formatters, running git commands that change files, or running project scripts with a working directory outside `/home/eii/openpi0.5-rtc-reward-learning`.
- Do not use or modify `/home/eii/openpi0.5-rlt` for this user's robot project; that path belongs to another project.
- Before copying files, restarting containers, or inspecting remote code on `192.168.1.103`, verify the working directory is `/home/eii/openpi0.5-rtc-reward-learning`.
- If a command on `192.168.1.103` would touch any path outside `/home/eii/openpi0.5-rtc-reward-learning`, stop and ask the user for explicit approval first.

## Command Hygiene
- For complex `192.168.1.103` inspections or statistics, do not embed Python/awk/jq/JSON-heavy logic in a nested one-line SSH command. Use a checked-in script, a single-quoted here-doc, or a `just`/Fabric wrapper that first runs `cd /home/eii/openpi0.5-rtc-reward-learning`.
- Any project `just` recipe, Fabric task, or Ansible playbook targeting `192.168.1.103` must preserve the same project boundary: start from `/home/eii/openpi0.5-rtc-reward-learning`, avoid `/home/eii/openpi0.5-rlt`, and avoid writes outside the project unless the user explicitly approves.
- Prefer this command shape for ad hoc multi-line 103 work:

```bash
ssh 192.168.1.103 <<'REMOTE'
set -euo pipefail
cd /home/eii/openpi0.5-rtc-reward-learning
project commands
REMOTE
```

- For reusable 103 tasks, prefer adding local scripts under this repository and syncing them to `/home/eii/openpi0.5-rtc-reward-learning` before execution. This avoids shell quoting bugs such as losing literal paths like `/app/replay`.

## Synchronized Home/Sleep Replay Gate
- The Stationary ALOHA follower-left synchronized real/simulation replay uses
  `/puppet_left/joint_states`, `/puppet_left/commands/joint_group`, and
  `/cam_high`; these names remain candidates until a separately authorized
  read-only inspection confirms their deployed types and publishers.
- Official Interbotix ROS1 Noetic defines `JointGroupCommand` as `string name`
  plus `float32[] cmd`. For the arm, `name` is `arm`; `cmd` order comes from the
  deployed motor-config group and must be read back rather than inferred.
- `commands/joint_group` is interpreted according to the group's current
  operating mode. Never use an all-zero group command as a generic stop: in
  position mode it is a position target, not an emergency-stop instruction.
- Do not import the live ROS adapter or create a command publisher until real
  access, real motion, workspace clearance, deployed joint order, camera,
  manifest identity, digital gate, and an operator-tested stop/hold path all
  pass. A read-only preflight must construct zero publishers and issue zero
  commands.
- Missing `Present_Current` support is recorded as `NOT_AVAILABLE`; it is not
  itself a reason to bypass the remaining live gates.

## Checkpoint Paths
- Strong checkpoint constraint for `192.168.1.103`: user-trained checkpoints for this project must live under `/home/eii/openpi0.5-rtc-reward-learning/checkpoints` and be mounted into containers as `/app/checkpoints`.
- Do not load this project's VLA/RLToken checkpoints from `/home/eii/openpi0.5-rtc/checkpoints`; that path belongs outside this project boundary.
- The preferred actor/critic checkpoint for 103 robot tests is project-local and contains both `actor.msgpack` and `critic.msgpack`:
  - Host path: `/home/eii/openpi0.5-rtc-reward-learning/local_rlt_runs/rlt_unified_468_td3_burn5000_actor10000/inference_actor/00012000`
  - Container path: `/app/local_rlt_runs/rlt_unified_468_td3_burn5000_actor10000/inference_actor/00012000`
  - Set `.env`: `RLT_ACTOR_CHECKPOINT_PATH=/app/local_rlt_runs/rlt_unified_468_td3_burn5000_actor10000/inference_actor/00012000`
  - Do not rely on `/app/rlt_online/run/inference_actor/LATEST` for robot testing unless the user explicitly asks to use online-training output.

## UV Locations
- Host: `/home/eii/.local/bin/uv` exists but is not on the default non-interactive SSH `PATH`.
- `openpi_server` container: `/usr/bin/uv`.
- `rlt_warmup_runtime` container: `/usr/bin/uv`.
- For compose commands that run `uv run ...`, assume container path `/usr/bin/uv`; do not rediscover this each time.

## Project-Owned ROS2 Collection Runtime

- The project-owned ROS2 collection source is
  `/home/eii/openpi0.5-rtc-reward-learning/third_party/aloha_collection`.
- `/home/eii/aloha-2.0` is the read-only source snapshot origin for this
  project; do not modify or launch it for new project collection sessions.
- Preview the resolved standalone collection container without touching robot
  hardware:

```bash
cd /home/eii/openpi0.5-rtc-reward-learning
third_party/aloha_collection/scripts/collect.sh --dry-run
```

- A real collection launch uses the same command without `--dry-run`, but only
  after explicit real-robot authorization and the hardware safety rules have
  been checked.
- This ROS2 `aloha2-collect` path is separate from the ROS1 inference services
  in the root `docker-compose.yml`; do not change the ROS1 mount as part of a
  recorder-only fix.

## Container Control
- When the user asks to stop all of their robot containers on `192.168.1.103`, do it in one compose command from the verified project directory:

```bash
cd /home/eii/openpi0.5-rtc-reward-learning && docker compose --profile rlt --profile legacy --profile train stop
```

- The compose file sets `name: openpi_reward_learning_eii`. Include all profiles so `rlt_warmup_runtime`, legacy `runtime`, and `rlt_online_trainer` are stopped together with the non-profile services.
- To start the user's robot for actor testing on `192.168.1.103`, do not run a broad `docker compose --profile rlt up -d`. Start explicit services to avoid accidentally starting legacy runtime or online trainer:

```bash
cd /home/eii/openpi0.5-rtc-reward-learning && docker compose --profile rlt up -d --no-build ros_master rosbridge redis openpi_server aloha_ros_nodes eii_pilot_backend eii_pilot_frontend eii_pilot_webrtc_media rlt_warmup_runtime
```

- For fast runtime-only restarts after `openpi_server` is warm:

```bash
docker compose --profile rlt up -d --no-build --force-recreate --no-deps rlt_warmup_runtime
```

## Startup Optimization
- The default compose command skips non-RTC warmup with `--no-warmup-non-rtc`; RTC warmup is still enabled because robot control uses the RTC path.
- `openpi_server` uses a project-local trusted JAX persistent compilation cache mounted at `${OPENPI_JAX_CACHE_DIR:-./.jax_cache/openpi_server}:/app/.jax_cache`.
- `.env` may set `OPENPI_JAX_CACHE_DIR=./.jax_cache/openpi_server`. Do not point this cache at a world-writable or untrusted shared directory.
- The first cold start populates the cache; later starts can reuse compatible JAX/XLA compilations when code, shapes, XLA flags, jaxlib version, and GPU model are unchanged.
- Measured on `192.168.1.103` after this optimization: first cache-fill start took about `62.7s` for `openpi_server` ready and `79.4s` for actor/critic ready; second cache-reuse start took about `19.7s` for `openpi_server` ready and `38.7s` for actor/critic ready.
- For fastest repeated tests, keep `openpi_server` running and only recreate `rlt_warmup_runtime` with the fast runtime-only restart command above.

## Root-Owned RLT Online Output
- `rlt_warmup_runtime` and `rlt_online_trainer` run as root inside their containers.
- The old compose mount `/data/openpi0.5-rtc-reward-learning/rlt_online:/app/rlt_online` means container-created `/app/rlt_online/run` files become root-owned on the host.
- Avoid using `/app/rlt_online/run/inference_actor/LATEST` as the default robot actor path; use the project-local `local_rlt_runs/.../00012000` path above.
