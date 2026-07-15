# Visual Tutor

`visual_tutor/` contains the project-local Visual Tutor system for teaching FreeCAD and Isaac Sim through small, verified steps.

## Current Implementation

- Core state machine: `my_visual_tutor/`
- Stdio MCP server: `my_gui_teacher/server.py`
- Example lessons: `lessons/`
- Isaac extension skeleton: `isaac_extensions/my.isaac.visual_tutor/`
- Tests: `tests/`
- Audit and reports: `audit/`, `reports/`

## Safety Scope

This first version is simulation-only and dry-run oriented. It does not control the real robot, publish ROS messages, expose arbitrary shell/Python execution, or expose arbitrary coordinate clicking through MCP.

## Test

```bash
PYTHONPATH=visual_tutor pytest -q visual_tutor/tests
```

## MCP

Registered name:

```text
my-gui-teacher
```

Command:

```text
python3 /home/eii/project/openpi0.5-rtc-reward-learning/visual_tutor/my_gui_teacher/server.py
```

Rollback:

```bash
codex mcp remove my-gui-teacher
```
