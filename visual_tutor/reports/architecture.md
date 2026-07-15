# Visual Tutor Architecture

## Goal

Build a visible, step-by-step teaching system for FreeCAD and Isaac Sim where Codex acts like a teacher: observe, point, pause, act, verify, and checkpoint one small step at a time.

## Components

```text
my-visual-tutor Skill
        |
        v
my-gui-teacher MCP server
        |
        v
LessonEngine state machine
        |
        +--> FreeCADAdapter
        |
        +--> IsaacSimAdapter
                 |
                 +--> my.isaac.visual_tutor Kit Extension
```

## MCP Boundary

The MCP server exposes only high-level tools:

```text
probe_app
start_lesson
get_lesson_status
next_step
repeat_step
step_back
pause_lesson
resume_lesson
abort_lesson
capture_current_state
save_checkpoint
restore_checkpoint
finish_lesson
```

It does not expose arbitrary coordinate clicking, shell commands, sudo, Python execution, file deletion, or real robot control.

## State Machine

The engine implements:

```text
IDLE
PREFLIGHT
OBSERVING
POINTING
WAITING_BEFORE_ACTION
ACTING
VERIFYING
CHECKPOINTED
PAUSED
RECOVERING
COMPLETED
ABORTED
FAILED
```

Each step follows:

```text
observe
locate
point
wait
act
wait
verify
checkpoint
```

Retries are bounded by the lesson step's `retry_limit`; the default lesson schema uses one retry.

## FreeCAD Route

Current environment does not have FreeCAD or FreeCADCmd installed, so the FreeCAD adapter is probe-only in this commit. The intended route is:

```text
Dogtail / AT-SPI semantic controls
-> image/template fallback
-> FreeCAD window-relative coordinates
```

Verification will use FreeCAD-native state when available:

- `ActiveDocument`
- object `TypeId`
- shape validity
- placement
- selection
- active workbench
- undo/transaction state

## Isaac Route

Isaac Sim uses a native Kit Extension skeleton instead of Xephyr. The intended route is:

```text
omni.kit.ui_test widget/menu query
-> application UI action
-> screenshot visual fallback
-> viewport-relative fallback
```

Current extension:

```text
visual_tutor/isaac_extensions/my.isaac.visual_tutor
```

It is passive and simulation-only. It can capture basic stage, selection, and timeline status into:

```text
visual_tutor/checkpoints/isaac_extension/latest_state.json
```

## Lesson Format

Lessons are YAML or JSON with schema:

```text
visual-tutor-lesson/v1
```

Each step includes:

- `id`
- `app`
- `description`
- `action_kind`
- `semantic_target`
- `visual_fallback`
- `relative_coordinate_fallback`
- `expected_state`
- `timeout_seconds`
- `retry_limit`
- `checkpoint`
- `undo_strategy`
- `pause_duration_seconds`
- `safety_class`

Examples:

- `visual_tutor/lessons/freecad_minimal_probe.yaml`
- `visual_tutor/lessons/isaac_cube_dry_run.yaml`
