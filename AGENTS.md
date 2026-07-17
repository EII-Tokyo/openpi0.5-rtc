# AGENTS

This file is the lightweight entry point for agents working in this repository. Keep detailed machine facts and long operational constraints in `docs/agents/`, not in this root file.

## Always Read First
- For any task touching the real robot, remote machine `192.168.1.103`, Docker services, checkpoints, replay data, Isaac Sim, or local annotation data, first read the relevant file listed below.
- If multiple topics apply, read all matching files before acting. The most specific safety constraint wins.
- Do not print, commit, or copy secrets such as `NVIDIA_API_KEY`, `NGC_API_KEY`, service tokens, or private credentials.

## Knowledge Base Rules
- The user's Obsidian knowledge base has its own global rules and math verifier. On this machine it is at `/home/eii/Documents/Notes/openpi0.5-rtc-reward-learning/AGENTS.md`; on another machine, first locate the vault root and read that vault-local `AGENTS.md`.
- Do not duplicate knowledge-base formula, notation, or figure constraints in this repository `AGENTS.md`; keep those rules in the vault so they work across machines and do not drift from project runtime rules.
- When editing knowledge-base notes with formulas, run the vault verifier from the vault root, for example:

```bash
cd /home/eii/Documents/Notes/openpi0.5-rtc-reward-learning && npm run check:math -- "<note path>"
```

## Topic Routing
- Remote `192.168.1.103`, robot containers, SSH command hygiene, remote project paths, `uv` locations, 103 startup/stop commands, or actor checkpoint paths:
  - Read `docs/agents/remote_103_operations.md`.
- Real ALOHA hardware diagnostics, DYNAMIXEL Wizard, serial aliases, or any action that can touch physical motors:
  - Read `docs/agents/aloha_hardware_debug.md`.
  - Also read `docs/agents/remote_103_operations.md` if the action happens on `192.168.1.103`.
- Isaac Sim, Isaac Lab, Claude Code MCP, Codex MCP, or the local MCP installation on machine `101`:
  - Read `docs/agents/isaac_mcp_toolchain.md`.
  - NVIDIA official Isaac MCP is mandatory before modifying Isaac Sim code, USD stages, scene-generation scripts, physics setup, GUI controls, or Isaac runtime behavior. Read-only investigation does not require starting Isaac MCP unless the answer depends on official Isaac API behavior.
- Photo-to-Isaac scene reconstruction, camera rack CAD, pipe/table spatial models, generated USD layers, or `scene_reconstruction/` assets:
  - Read `docs/agents/scene_reconstruction.md`.
  - Use the local `photo-to-isaac-cad` skill when available.
- FreeCAD CAD review, blank FreeCAD viewports, STEP/FCStd visual checks, or CAD display/debug work:
  - Read `docs/agents/scene_reconstruction.md`.
- Visual Tutor, visible GUI teaching, FreeCAD/Isaac step-by-step lessons, `my-gui-teacher`, `my-visual-tutor`, or `visual_tutor/`:
  - Read `docs/agents/visual_tutor.md`.
  - Use the local `my-visual-tutor` skill when available.
- ROS MCP, `robotmcp/ros-mcp-server`, rosbridge, rosapi, or MCP-based ROS inspection/control:
  - Read `docs/agents/ros_mcp.md`.
  - Also read `docs/agents/remote_103_operations.md` for any `192.168.1.103` work and `docs/agents/aloha_hardware_debug.md` before any action that can affect the real ALOHA robot.
- Backend pytest, local test environment variables, segment DB test paths, or container-vs-host path assumptions:
  - Read `docs/agents/backend_test_environment.md`.
- VLA/RLToken checkpoint selection, `z_rl` dimensions, same-forward runtime, sidecar RLToken bans, actor/critic startup, or re-encoding decisions:
  - Read `docs/agents/rlt_checkpoints.md`.
- Local offline key-region annotation, local data mounts, cleaned 2026-06-22 data, or syncing annotation data:
  - Read `docs/agents/local_key_region_annotation.md`.
- Canonical replay, manifests, formal-vs-ablation replay, `/rlt_policy_forward_events`, human expert data conversion, or 2048/512 replay mixing:
  - Read `docs/agents/canonical_rlt_replay_data.md`.
- Image channel order, HDF5/LeRobot conversion, runtime camera path, gripper normalization, or gripper actuation:
  - Read `docs/agents/image_and_gripper_flow.md`.
- Historical training benchmark facts:
  - Read `docs/agents/training_notes.md`.

## Hard Safety Defaults
- Do not control the real robot unless the user explicitly asks for a real-hardware action and the relevant safety docs above have been read.
- Do not start broad robot compose profiles when an explicit service list is safer. For robot starts on `192.168.1.103`, follow `docs/agents/remote_103_operations.md`.
- Do not mix legacy 512-dim replay with active 2048 lower+right RLToken training data unless the user explicitly requests a controlled ablation.
- Do not use cam3-derived VLA/RLToken checkpoints for rinse or bottle-mouth insertion tasks that require `cam_low`.
- Do not treat saved-video/mp4 re-encoded `z_rl` as formal same-forward training replay unless the user explicitly requests an ablation.
- Do not enable RLToken sidecar fallback for normal robot actor tests; same-forward `z_rl` is required.
- ROS MCP is configured locally through `uvx`, but it requires robot-side rosbridge/rosapi before it can inspect `192.168.1.103`. Use it read-only by default unless the user explicitly asks for a real robot action.

## Maintenance Rule
- When adding new long-lived operational facts, put them in the appropriate `docs/agents/*.md` file and add only a short routing bullet here.

<!-- codex-optimization-project-evidence -->
## Evidence And Task State
- High-output diagnostics for training logs, Docker logs, Isaac/ROS logs, tests, MCP, or extensions should use the `my-evidence-first-debugging` skill or `codex-evidence`; keep full output in `.codex/artifacts/` and summarize only bounded evidence in conversation.
- Use `.codex/TASK_STATE.md` for dynamic long-task handoff state. Do not put temporary task state into this `AGENTS.md` or `docs/agents/*.md`.
