# Visual Tutor Final Report

## Environment

Audit report:

```text
visual_tutor/reports/environment_audit.md
```

Key facts:

- Ubuntu host: `ubuntu`
- Current session: X11, `DISPLAY=:1`
- GPU: NVIDIA RTX 3090 Ti, driver 575.57.08
- Noninteractive sudo: unavailable
- `xdotool`: available
- `wmctrl`, `scrot`, `dogtail`: missing
- FreeCAD and FreeCADCmd: missing
- Isaac `.venv_issac`: available
- NVIDIA official Isaac MCP was used before Isaac implementation work.

## Implementation Path

Chosen path:

```text
generic Visual Tutor core
+ FreeCAD probe-only adapter
+ Isaac Sim native Kit Extension skeleton
+ high-level stdio MCP server
```

No system packages were installed. No sudo was used.

## Created Files

Core:

- `visual_tutor/my_visual_tutor/models.py`
- `visual_tutor/my_visual_tutor/engine.py`
- `visual_tutor/my_visual_tutor/adapters.py`
- `visual_tutor/my_visual_tutor/lesson_io.py`

MCP:

- `visual_tutor/my_gui_teacher/server.py`
- `visual_tutor/my_gui_teacher/README.md`

Isaac Extension:

- `visual_tutor/isaac_extensions/my.isaac.visual_tutor/config/extension.toml`
- `visual_tutor/isaac_extensions/my.isaac.visual_tutor/my/isaac/visual_tutor/extension.py`

Lessons:

- `visual_tutor/lessons/freecad_minimal_probe.yaml`
- `visual_tutor/lessons/isaac_cube_dry_run.yaml`

Tests:

- `visual_tutor/tests/`

User Skill:

- `/home/eii/.codex/skills/my-visual-tutor`

## MCP

Registered MCP server:

```text
my-gui-teacher
```

Command:

```text
python3 /home/eii/project/openpi0.5-rtc-reward-learning/visual_tutor/my_gui_teacher/server.py
```

Config backup:

```text
/home/eii/.codex/config.toml.bak_my_gui_teacher_20260715_145927
```

## Supported Versions

This commit records profiles through audit, but does not yet calibrate a real GUI profile.

- FreeCAD: not installed, unsupported beyond `probe_app`.
- Isaac Sim: local `.venv_issac` exists; extension skeleton created for Kit/Isaac extension loading.

## Tests

Command:

```bash
PYTHONPATH=visual_tutor pytest -q visual_tutor/tests
```

Result:

```text
7 passed
```

Validated:

- Lesson schema loading.
- State machine completion and pause behavior.
- FreeCAD missing state fails safely.
- MCP exposes only high-level tools.
- MCP can run the Isaac dry-run lesson.
- Isaac extension manifest and simulation-only flags exist.

## User Controls

Supported through MCP:

```text
pause_lesson
resume_lesson
next_step
repeat_step
step_back
abort_lesson
save_checkpoint
restore_checkpoint
finish_lesson
```

Keyboard shortcuts such as F8/F9/F10/Esc were not globally installed in this phase because that would require a desktop-level hotkey integration and conflict check. The current safe control surface is MCP plus future local control panel.

## Streaming

No WebRTC Streaming server or client was installed or reconfigured in this phase.

Current status:

```text
streaming.configured = false
```

Reason: the local machine already has a usable X11 desktop and the minimal implementation can be validated through stdio MCP and offline tests. Remote/103 streaming should be added only after selecting the actual target host and confirming the matching Isaac Sim streaming workflow.

## How To Start Teaching

After restarting Codex or refreshing MCP availability, use MCP tool calls:

```text
probe_app(app="Isaac Sim")
start_lesson(lesson_path="visual_tutor/lessons/isaac_cube_dry_run.yaml")
next_step()
pause_lesson()
resume_lesson()
abort_lesson()
```

For FreeCAD:

```text
probe_app(app="FreeCAD")
start_lesson(lesson_path="visual_tutor/lessons/freecad_minimal_probe.yaml")
```

Because FreeCAD is not installed, the FreeCAD lesson should report the missing app and stop safely.

## How To Enable Isaac Extension

In Isaac Sim Extension Manager, add this search path:

```text
/home/eii/project/openpi0.5-rtc-reward-learning/visual_tutor/isaac_extensions
```

Then enable:

```text
my.isaac.visual_tutor
```

The extension creates a passive panel and does not control real hardware.

## How To Add A Lesson

Create a YAML file with:

```text
schema_version: visual-tutor-lesson/v1
mode: demonstrate | guided-practice | build | hybrid
steps:
  - id: ...
    app: FreeCAD | Isaac Sim
    action_kind: ...
    semantic_target: ...
    expected_state: ...
    retry_limit: 1
    safety_class: simulation_only
```

Keep each lesson to one small goal.

## How To Add A Third Adapter

1. Implement `TutorAdapter` methods in `visual_tutor/my_visual_tutor/adapters.py`.
2. Add the adapter to `adapter_for_app`.
3. Add a probe-only lesson.
4. Add tests for safe failure, checkpointing, and MCP dispatch.

## Rollback

Remove MCP:

```bash
codex mcp remove my-gui-teacher
```

Restore Codex config if needed:

```bash
cp /home/eii/.codex/config.toml.bak_my_gui_teacher_20260715_145927 /home/eii/.codex/config.toml
```

Remove Skill:

```bash
rm -rf /home/eii/.codex/skills/my-visual-tutor
```

Remove project files:

```bash
rm -rf visual_tutor
```

If the changes are committed, prefer reverting the commit.
