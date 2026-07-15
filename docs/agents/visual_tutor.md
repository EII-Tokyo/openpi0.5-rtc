# Visual Tutor Operations

Read this before changing `visual_tutor/`, `my-gui-teacher`, `my-visual-tutor`, or Isaac Visual Tutor extension files.

## Current Scope

- Project-local implementation root: `visual_tutor/`.
- MCP server name: `my-gui-teacher`.
- User Skill: `/home/eii/.codex/skills/my-visual-tutor`.
- Isaac Extension: `visual_tutor/isaac_extensions/my.isaac.visual_tutor`.

## Safety

- Do not expose arbitrary coordinates, shell execution, sudo, Python execution, file deletion, or real robot control through the MCP server.
- Do not start or control the real robot.
- Isaac work is simulation-only by default.
- Keep timeline paused unless a lesson explicitly and safely requires otherwise.
- FreeCAD original CAD files must remain read-only; tests should use temporary copies or checkpoints.

## Validation

Run:

```bash
PYTHONPATH=visual_tutor pytest -q visual_tutor/tests
python3 /home/eii/.codex/skills/.system/skill-creator/scripts/quick_validate.py /home/eii/.codex/skills/my-visual-tutor
```

## MCP Registration

Registered with:

```text
codex mcp add my-gui-teacher -- python3 /home/eii/project/openpi0.5-rtc-reward-learning/visual_tutor/my_gui_teacher/server.py
```

Rollback:

```text
codex mcp remove my-gui-teacher
```
