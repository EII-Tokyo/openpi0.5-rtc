# Isaac MCP Toolchain

Read this before using, changing, debugging, or reinstalling Isaac Sim / Isaac Lab MCP tooling on machine `101`.

## Setup Paths
- Setup root: `/home/eii/isaac_mcp_setup`
- Environment report: `/home/eii/isaac_mcp_setup/environment_report.md`
- Install report: `/home/eii/isaac_mcp_setup/INSTALL_REPORT.md`
- Security notes: `/home/eii/isaac_mcp_setup/SECURITY.md`
- Server config: `/home/eii/isaac_mcp_setup/config/servers.yaml`

## Management Scripts
- Start persistent MCP services: `/home/eii/isaac_mcp_setup/scripts/start_all.sh`
- Stop persistent MCP services: `/home/eii/isaac_mcp_setup/scripts/stop_all.sh`
- Show status: `/home/eii/isaac_mcp_setup/scripts/status_all.sh`
- Run safe verification: `/home/eii/isaac_mcp_setup/scripts/test_all.sh`

## Installed MCP Servers
- `nvidia-isaac-docs`: official NVIDIA Isaac Sim documentation/search MCP from `NVIDIA-Omniverse/kit-usd-agents`, running in Docker as `isaacsim-mcp`, bound only to `127.0.0.1:9904`.
- `isaacsim-control`: community scene-control MCP from `whats2000/isaacsim-mcp-server`, stdio MCP for Claude/Codex, waits for Isaac Sim extension socket `localhost:8766`.
- `isaacsim-python`: community Isaac Python execution/log MCP from `mochan-b/isaacsim-mcp`, stdio MCP for Claude/Codex, waits for Isaac Sim VS Code executor `127.0.0.1:8226`.
- `isaaclab`: local minimal read-only Isaac Lab MCP at `/home/eii/isaac_mcp_setup/repos/isaaclab-mcp-local`, using `/home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/bin/python`.
- ROS MCP is documented separately in `docs/agents/ros_mcp.md`; do not treat Isaac MCP setup as permission to connect MCP tools to the real ALOHA robot.

## Verified State
- `nvidia-isaac-docs` is healthy and responds to MCP initialize on `http://127.0.0.1:9904/mcp`.
- `isaacsim-control` and `isaacsim-python` are installed and registered, but are only partially testable until the corresponding Isaac Sim internal extensions are enabled.
- `isaaclab` local tests passed and the install probe sees Isaac Sim `5.1.0.0` and Isaac Lab `0.54.4`; task listing currently has a known `pxr` import issue.

## Claude Code And Codex Names
- `isaac-sim-mcp` for official NVIDIA docs/search MCP.
- `isaacsim-control` for Isaac scene manipulation.
- `isaacsim-python` for Isaac Python execution and Kit log reading.
- `isaaclab` for local safe Isaac Lab probes.

## Security Constraints
- Do not expose MCP ports to public interfaces; keep all network services bound to `127.0.0.1` / localhost.
- Do not print or commit `NVIDIA_API_KEY` or `NGC_API_KEY`; API keys are stored outside this repository.
- Treat `isaacsim-python` and `isaacsim-control` as high-privilege because they can execute Python or scene operations inside Isaac Sim.
- Do not connect MCP tools to the real ALOHA robot by default.
- Do not start or restart `aloha_ros_nodes`, `runtime`, `rlt_warmup_runtime`, or other real robot control containers as part of MCP work unless the user explicitly approves.

## ALOHA Isaac Scratch Workspace
- Scratch workspace: `/home/eii/isaac_mcp_setup/aloha_project`
- This is only a scratch workspace. The project source of truth remains `/home/eii/project/openpi0.5-rtc-reward-learning`, especially `examples/aloha_isaac`.
- Do not use `/home/eii/isaac_mcp_setup/aloha_project` as the selected workspace, output root, source root, or project root for this repository. It is MCP scratch only.
- For Isaac/ALOHA/USD work started from `/home/eii/project/openpi0.5-rtc-reward-learning`, the workspace root is the repository root unless the user explicitly provides a different existing project path. Do not let domain-named subdirectories such as `aloha_isaac_replay/` replace the repository root as the workspace root.

## Confirmed ALOHA Isaac Startup Stage
- As of 2026-07-15, the only user-confirmed ALOHA Isaac startup stage is:
  - Directory: `/home/eii/project/openpi0.5-rtc-reward-learning/local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/`
  - USD: `/home/eii/project/openpi0.5-rtc-reward-learning/local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/aloha2_menagerie_scene_deep_black_real_start_pose.usd`
- `examples/aloha_isaac/scripts/open_workcell_gui.py` defaults to this USD when launched without `--usd`; that default is intentional for the confirmed ALOHA scene.
- Normal startup should not pass `--usd`; use the script default so stale scratch stages cannot be loaded by accident.
- `open_workcell_gui.py` rejects noncanonical USD paths by default. Use `--allow-noncanonical-usd` only for an explicit experiment or conversion debug session.
- To inspect the official Trossen Stationary AI USD, do not use the normal canonical ALOHA1 startup semantics. Launch it with `--allow-noncanonical-usd --no-real-start-pose` so the ALOHA1 home/sleep pose controller is not applied to the different Trossen articulation structure:

```bash
cd /home/eii/project/openpi0.5-rtc-reward-learning
env PYTHONPATH=$PWD OMNI_KIT_ACCEPT_EULA=YES .venv_issac/bin/python \
  examples/aloha_isaac/scripts/open_workcell_gui.py \
  --usd external/trossen_ai_isaac/assets/robots/stationary_ai/stationary_ai.usd \
  --allow-noncanonical-usd \
  --no-real-start-pose
```

- When the user is comparing the original ALOHA1 Isaac scene against the original Trossen Stationary AI scene, do not proactively close either Isaac Sim window. These two GUI sessions are comparison references:
  - Original ALOHA1: launch `examples/aloha_isaac/scripts/open_workcell_gui.py` with its default canonical USD.
  - Original Trossen Stationary AI: launch the explicit `stationary_ai.usd` command above with `--allow-noncanonical-usd --no-real-start-pose`.
  - Only close these windows when the user explicitly asks to close them.

## Clean Rebuild USD Viewer Notes
- For the clean rebuild work under `aloha_isaac_rebuild/`, do not treat a GUI startup failure as evidence that the generated USD is invalid. First run the stage-specific static audit script and inspect its JSON report.
- On 2026-07-22, repeated attempts to open `aloha_isaac_rebuild/scenes/aloha_camera_semantic_skeleton.usda` with `aloha_isaac_rebuild/scripts/open_skeleton_gui.py` all reached Isaac `app ready`, moved the window, and printed `Opened skeleton stage`, but the Python process then exited with empty `stderr`. The same stage also exited when opened through `open_workcell_gui.py --allow-noncanonical-usd --no-real-start-pose`. Treat this as a viewer/process-lifetime issue, not an A3 camera semantic USD failure, unless the static audit also fails.
- Do not keep retrying the same A3 viewer command after this symptom. It creates noisy logs and does not add new evidence.
- For visual inspection of generated clean rebuild stages while this viewer issue is unresolved, use one of these approaches:
  - Preferred validation: keep the original ALOHA1 and original Trossen Stationary AI windows open as reference, and validate the generated clean stage with its stage-specific audit script plus USDA/Stage tree inspection.
  - Preferred scripted viewer: use `aloha_isaac_rebuild/scripts/open_skeleton_gui.py` for A3 and other clean rebuild skeleton stages. Before relaunching, verify that no previous A3/skeleton viewer process is still active.
  - Treat `Opened skeleton stage` in the viewer log as the stage-load signal, and separately verify viewport/window visibility because workspace moves and stale Isaac windows can be misleading.
  - If GUI inspection is still ambiguous, open the generated USD manually from an already-running Isaac Sim GUI with `File > Open` or `File > Open Recent`, so the viewer is not responsible for the application lifetime.
  - If a scripted GUI launcher is required later, fix and verify the launcher separately before using it as evidence for USD correctness.
- Do not use `examples/aloha_isaac/scripts/open_workcell_gui.py` as the normal viewer for A3 semantic skeleton stages. That script is for canonical `/scene` workcell startup; using it on `/aloha` skeleton stages requires `--allow-noncanonical-usd --no-real-start-pose` and can introduce `/scene/StartupViewCamera` assumptions into a clean `/aloha` inspection.
- Current A3 camera semantic skeleton audit command:

```bash
cd /home/eii/project/openpi0.5-rtc-reward-learning
PYLIB=$(dirname $(find /home/eii/.local/share/uv/python -name 'libpython3.11.so.1.0' -print | head -n 1))
USD_ROOT=.venv_issac/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
PYTHONPATH="$USD_ROOT" LD_LIBRARY_PATH="$USD_ROOT/bin:$PYLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  .venv_issac/bin/python aloha_isaac_rebuild/scripts/audit_aloha_camera_semantic_stage.py \
  aloha_isaac_rebuild/scenes/aloha_camera_semantic_skeleton.usda \
  --json-output aloha_isaac_rebuild/artifacts/validation/a3_camera_semantic_skeleton_audit.json
```

- On this desktop, normal non-headless Isaac startup must move the main Isaac Sim window to user-facing workspace 2. This keeps the current workspace usable for the user while agents inspect or screenshot Isaac on the second workspace. The launcher does this by default with `xdotool`; use `--no-move-to-startup-workspace` only when the user explicitly wants Isaac on the current workspace.
- When taking Isaac screenshots or doing visual inspection, do not move Isaac back to the user's active workspace. Prefer window-id based screenshots or operate on workspace 2 so the user's desktop remains usable.
- Do not substitute `examples/aloha_isaac/config/workcell_user_measured.yaml` for normal ALOHA Isaac startup. That measured workcell path was rejected by the user because it loads the wrong ALOHA configuration.
- The directory also contains `config.yaml`, but that file records MJCF conversion metadata. For GUI startup, the effective startup target is the USD above, not the YAML file name.
- Other `local_eval_assets/aloha_isaac*` directories are intermediate or scratch assets, not startup targets. Do not clean them by deletion unless the generator chain has been checked, because some are source layers for the confirmed stage.

## ALOHA1 Replay Validation Gate
- For Menagerie/Trossen `/scene` HDF5 left-arm replay, use the controller-ready mapping:
  - `configs/aloha/trossen_scene_base_link_aloha1_left_controller_mapping.yaml`
- When the user asks for the historical long "episode 19" replay, use this exact local HDF5 path:
  - `/home/eii/project/bottles_data/episode_19.hdf5`
  - It has 3642 frames at 50 Hz, about 72.84 seconds.
  - Do not substitute short key-region HDF5 files or 103-synced candidates unless the user explicitly asks for a different replay.
- When the user asks for the user-confirmed bottle grasp window, use `episode_18`, not `episode_19`:
  - `/home/eii/project/bottles_data/episode_18.hdf5`
  - User-confirmed window: frame `208` through `244` inclusive. Use end-frame `245` for exclusive-end scripts.
  - Semantics: open gripper -> clamp bottle -> lift onset. Do not substitute episode 19 for this grasp-window calibration task.
- Phase132 tabletop modes have different meanings:
  - `--tabletop-mode diagnostic_shift_to_open_finger` is a diagnostic-only contact-proxy mode. It moves `/World/Table` to match the replay open-finger height and must not be treated as an RL-ready fixed-workcell pass.
  - `--tabletop-mode fixed_reference` keeps `/World/Table` fixed and places the bottle on the table collider. This is the RL-ready semantics for tabletop replay validation.
  - As of 2026-07-22, fixed-reference episode_18 close-frame replay reports a height residual between the loaded clamp anchor and the fixed table object center. Treat that as a table/base calibration problem, not a friction, CCD, or contactOffset problem.
- This mapping preserves recorded ALOHA1 joint values and only renames left-arm DOFs into the `/scene` articulation. It intentionally does not apply FK rigid-alignment offsets, because those offsets can push runtime DOF targets outside PhysX limits.
- A successful replay validation must report all of these gates:
  - `target_limit_gate_ok: true`
  - `controller_tracking_gate.pass: true`
  - `contact_trace_status: PASS_BILATERAL_CONTACT_CANDIDATE` or the task-specific accepted contact status
  - `failure_reasons: []`
- Do not use `--disable-workcell-environment-collisions-for-diagnostic-replay` for final replay/contact validation. Phase80/82 on 2026-07-18 showed that native `/scene/worldBody` collisions are needed for stable HDF5 replay; disabling them can make the object fall and misdiagnose the controller.
- If strict non-target contact checking is needed while keeping normal table/workcell support, use:
  - `--fail-on-non-target-object-contact`
  - `--allowed-non-target-object-contact-category workcell_or_environment`
- `--support-plane-mode object_bottom` is diagnostic only. Phase81 on 2026-07-18 showed it can make contact look stable while corrupting post-step arm tracking, so it is not a final validation substitute.
- Current geometry-isolation reference run:
  - Report: `reports/aloha1_isaac_adaptation/phase83_scene_proxy_hdf5_replay_native_workcell_allowed_support_20260718/gripper_passive_contact_metrics.json`
  - Artifact: `.codex/artifacts/20260718-232540_aloha-phase83-native-workcell-allowed-support-strict-gate`
- Current drive-target reference run:
  - Report: `reports/aloha1_isaac_adaptation/phase97_scene_proxy_hdf5_replay_drive_target_arm1600_kd100_finger200_native_workcell_20260718/gripper_passive_contact_metrics.json`
  - Artifact: `.codex/artifacts/20260718-234743_aloha-phase97-native-workcell-drive-target-arm1600-kd100-finger200`
  - Required runtime tuning for this run: `--arm-kp 1600 --arm-kd 100 --finger-kp 200 --finger-kd 50`.
  - This run keeps `--hdf5-replay-target-hold-steps 1`, so it preserves the 50 Hz HDF5 replay target cadence.
  - PASS metrics: `target_limit_gate_ok: true`, `controller_tracking_gate.pass: true`, `max_controlled_error: 0.012857437133789062`, `contact_trace_status: PASS_BILATERAL_CONTACT_CANDIDATE`, `failure_reasons: []`.
- Do not treat higher arm stiffness as automatically better. Phase94 (`--arm-kp 2400 --arm-kd 200 --finger-kp 200 --finger-kd 50`) passed tracking but failed the strict non-target contact gate because the object touched the same-side gripper base.

## USD Reference Safety
- When referencing a USD asset into another Isaac stage, do not assume the source file's `defaultPrim` is the desired asset root.
- If the desired asset root is known, pass the explicit prim path in the reference, for example `AddReference(asset_path, "/Bottle500")` or USDA syntax `@asset.usd@</Bottle500>`.
- After authoring a reference, validate in Isaac runtime that expected child prims actually compose into the destination stage. For Bottle500, `/World/Bottle500/Visuals/VIS_Bottle` and `/World/Bottle500/Collisions` must exist; an empty `/World/Bottle500` with only debug axes is a failed reference, not a valid bottle.
- Debug axes or Grasp Editor frames must not be used as evidence that the referenced object loaded. Confirm at least one real Mesh under the referenced object root and check its world bounding box.

## NVIDIA MCP Requirement
- For read-only Isaac investigation, first read this document and inspect local project state. Use the NVIDIA official Isaac MCP only when the answer depends on official Isaac API behavior, USD semantics, physics behavior, GUI behavior, or Isaac runtime state.
- Before modifying Isaac Sim code, USD stages, scene-generation scripts, physics setup, GUI controls, or Isaac runtime behavior, first use the NVIDIA official Isaac MCP: `isaac-sim-mcp` / `mcp__isaac_sim_mcp`.
- This modification-time requirement is mandatory, not a preference. Do not make Isaac implementation changes based only on shell inspection, community MCPs, web/forum searches, or local guesses.
- If the NVIDIA official Isaac MCP is unavailable, fails, or cannot answer the required Isaac-specific point for a planned modification, stop and report that the hard prerequisite is unavailable before changing Isaac implementation code. Do not silently substitute another MCP or community source for that modification.
- After the NVIDIA official Isaac MCP has been used for the modification, `isaaclab`, `isaacsim-control`, and `isaacsim-python` may be used only as secondary implementation or inspection tools.
- `isaacsim-control` and `isaacsim-python` remain local simulation tools only; never use them to affect the real ALOHA robot.

## Isaac Code Change Expert Gate
- Before each Isaac code, USD, physics, contact policy, replay validation, or simulation behavior change, first consult both standing expert threads when available:
  - Isaac/physics expert: must provide official Isaac documentation, USD/PhysX semantics, math/physics rationale, and concrete acceptance criteria.
  - Robotics examples expert: must provide relevant Isaac examples or robot manipulation patterns, and concrete regression criteria.
- Do not proceed with blind parameter sweeps as the basis for code changes. Experiments are allowed only after a documented hypothesis and acceptance criteria are established from official docs/examples or measured ALOHA facts.
- Do not relax contact policy gates to make a run pass unless the experts and the evidence show that the collider is semantically correct to allow. Prefer fixing geometry, placement, phase classification, or collision proxy semantics first.
- Record the expert-backed rationale in the worklog or report for the change. Keep bounded evidence paths instead of copying long logs.

## Secondary MCP Order
- After the mandatory NVIDIA official MCP step, use `isaaclab` read-only probes when Isaac Lab state or APIs are needed.
- Use `isaacsim-control` only after confirming the Isaac Sim extension socket is local and the task is simulation-only.
- Use `isaacsim-python` only when direct Python execution inside Isaac Sim is necessary.
