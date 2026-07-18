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

## Confirmed ALOHA Isaac Startup Stage
- As of 2026-07-15, the only user-confirmed ALOHA Isaac startup stage is:
  - Directory: `/home/eii/project/openpi0.5-rtc-reward-learning/local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/`
  - USD: `/home/eii/project/openpi0.5-rtc-reward-learning/local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/aloha2_menagerie_scene_deep_black_real_start_pose.usd`
- `examples/aloha_isaac/scripts/open_workcell_gui.py` defaults to this USD when launched without `--usd`; that default is intentional for the confirmed ALOHA scene.
- Normal startup should not pass `--usd`; use the script default so stale scratch stages cannot be loaded by accident.
- `open_workcell_gui.py` rejects noncanonical USD paths by default. Use `--allow-noncanonical-usd` only for an explicit experiment or conversion debug session.
- On this desktop, normal non-headless Isaac startup must move the main Isaac Sim window to user-facing workspace 2. This keeps the current workspace usable for the user while agents inspect or screenshot Isaac on the second workspace. The launcher does this by default with `xdotool`; use `--no-move-to-startup-workspace` only when the user explicitly wants Isaac on the current workspace.
- When taking Isaac screenshots or doing visual inspection, do not move Isaac back to the user's active workspace. Prefer window-id based screenshots or operate on workspace 2 so the user's desktop remains usable.
- Do not substitute `examples/aloha_isaac/config/workcell_user_measured.yaml` for normal ALOHA Isaac startup. That measured workcell path was rejected by the user because it loads the wrong ALOHA configuration.
- The directory also contains `config.yaml`, but that file records MJCF conversion metadata. For GUI startup, the effective startup target is the USD above, not the YAML file name.
- Other `local_eval_assets/aloha_isaac*` directories are intermediate or scratch assets, not startup targets. Do not clean them by deletion unless the generator chain has been checked, because some are source layers for the confirmed stage.

## ALOHA1 Replay Validation Gate
- For Menagerie/Trossen `/scene` HDF5 left-arm replay, use the controller-ready mapping:
  - `configs/aloha/trossen_scene_base_link_aloha1_left_controller_mapping.yaml`
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
- Current reference run:
  - Report: `reports/aloha1_isaac_adaptation/phase83_scene_proxy_hdf5_replay_native_workcell_allowed_support_20260718/gripper_passive_contact_metrics.json`
  - Artifact: `.codex/artifacts/20260718-232540_aloha-phase83-native-workcell-allowed-support-strict-gate`

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

## Secondary MCP Order
- After the mandatory NVIDIA official MCP step, use `isaaclab` read-only probes when Isaac Lab state or APIs are needed.
- Use `isaacsim-control` only after confirming the Isaac Sim extension socket is local and the task is simulation-only.
- Use `isaacsim-python` only when direct Python execution inside Isaac Sim is necessary.
