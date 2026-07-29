# ALOHA1 Visual Tutor Gateway diagnosis

Status: **HARD_BLOCKER**

Classification:
`DRY_RUN_STDIO_PROTOTYPE_NO_LIVE_ISAAC_ACTION_BRIDGE`

## Verified runtime boundary

- Codex has exactly one MCP connection: `mcpjungle_lab`.
- The active endpoint is
  `http://127.0.0.1:18080/v0/groups/codex-research/mcp`.
- MCPJungle is reachable and NVIDIA Isaac documentation calls work.
- The live MCPJungle registry and `codex-research` group do not contain
  `my-gui-teacher`, a Visual Tutor application probe, or Isaac GUI teaching
  actions.
- Chrome liveview is not an approved Isaac Visual Tutor bridge and cannot
  substitute for it.

## Visual Tutor implementation audit

The current project server
`visual_tutor/my_gui_teacher/server.py` is a stdio JSON-RPC server. It has no
Streamable HTTP endpoint.

Its Isaac application probe checks project paths and declares safety
properties, but it does not verify a live Isaac process, extension heartbeat,
current Stage, timeline state, or action channel.

The Isaac adapter accepts `simulation_only` lesson steps and returns dry-run
success. It does not execute a widget query, menu action, Grasp Editor action,
or screenshot capture. The Isaac extension is a passive status panel that can
write a JSON snapshot when its local buttons are pressed; it is not connected
to the MCP server.

The focused Visual Tutor suite passed `7/7`, but those tests cover dry-run and
static contracts only. They do not verify HTTP, MCPJungle, live Isaac, or a
Grasp Editor round trip.

## Additional safety gaps

- Timeline pause is declared but not enforced at extension startup.
- `action_kind` has no explicit allowlist.
- Lesson/checkpoint paths have no approved-root containment gate.
- Retry count has no server-side cap and lesson timeout is not enforced.
- The extension snapshot has no freshness or ownership validation.

These gaps must be closed before exposing the server through MCPJungle.

## Grasp Editor 2.0.20 native schema conflict

The approved diagnostic control is Variant B:

- active: `left_finger`;
- fixed/observer: `right_finger`;
- right finger is not directly commanded.

Isaac Sim 5.1 Grasp Editor 2.0.20 natively exports:

- grasp name `grasp_0`;
- `cspace_position` containing only `left_finger`;
- `pregrasp_cspace_position` containing only `left_finger`.

The current project canonical loader requires:

- grasp name `horizontal_body_grasp`;
- exact `left_finger` and `right_finger` mappings.

Therefore canonical promotion is independently blocked as
`HARD_BLOCKER_CANONICAL_SCHEMA_MISMATCH`. Raw GUI export may be retained as
evidence, but it cannot be silently renamed or promoted.

Forbidden workarounds include guessing `right_finger=-left_finger`, switching
to dual-active Variant A just to satisfy the old parser, or overwriting the
canonical config before an approved contract exists.

## Minimum safe architecture

The recommended direction, pending explicit design approval, is:

1. add a restricted live Isaac bridge with a real heartbeat and exact Stage,
   timeline, extension, and action-channel readback;
2. expose only named Grasp Editor operations with strict schemas and no
   arbitrary coordinates, shell, Python, ROS, deletion, Stage overwrite, or
   real-robot control;
3. publish it as Streamable HTTP through an independent MCPJungle group such
   as `codex-visual-tutor`, not by expanding `codex-research`;
4. use a fresh `codex-full --mcp-group codex-visual-tutor` session and verify
   the exact live tool allowlist before GUI use;
5. separately approve either loader-native left-active/right-observer support
   or a deterministic raw-to-canonical promotion schema with preserved raw
   lineage.

Until both design choices are approved and verified:

```text
ACTUAL_GRASP_EDITOR_GUI = NOT_RUN
NATIVE_RAW_EXPORT = NOT_RUN
CANONICAL_PROMOTION = BLOCKED_SCHEMA_MISMATCH
PRE_IK_GEOMETRY = NOT_RUN
IK = NOT_RUN
DYNAMIC_GRASP_VIDEO = NOT_RUN
TASK_PASS = NOT_ESTABLISHED
TASK8 = NOT_RUN
```
