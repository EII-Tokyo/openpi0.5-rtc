# Visual Tutor Environment Audit

## Localization Decision

- `my-core-adaptive` selected `/home/eii/isaac_mcp_setup/repos/isaacsim-mcp`, but project AGENTS and the current git root identify this repository as the source of truth.
- Implementation workspace: `/home/eii/project/openpi0.5-rtc-reward-learning/visual_tutor`.
- Existing unrelated untracked paths are preserved: `scene_reconstruction/cad/.tmp/`, `scene_reconstruction/cad/aloha_incremental/`.

## System

- Hostname: `ubuntu`
- User: `eii`
- Git root: `/home/eii/project/openpi0.5-rtc-reward-learning`
- Python: `Python 3.12.3`
- DISPLAY: `:1`
- WAYLAND_DISPLAY: `None`
- XDG_SESSION_TYPE: `x11`
- Noninteractive sudo exit: `1`

## Capability Matrix

| Area | Capability | Current status | Need | Handling |
| --- | --- | --- | --- | --- |
| Display | DISPLAY | :1 | required | Use current X11 display if present; Wayland fallback needs non-global approach. |
| Display | WAYLAND_DISPLAY | unset | context | Do not default to whole-desktop ydotool. |
| GUI automation | xdotool | available | recommended | Visible mouse movement on X11/Xephyr. |
| GUI automation | wmctrl | missing | recommended | Window discovery/move/activate. |
| GUI automation | scrot | missing | recommended | Screenshot capture. |
| GUI automation | dogtail | Traceback (most recent call last):   File "<string>", line 1, in <module> ModuleNotFoundError: No module named 'dogtail' | recommended | AT-SPI semantic UI control for FreeCAD when available. |
| FreeCAD | FreeCAD | missing | required for FreeCAD adapter | If missing, keep adapter probe-only and do not install without approval. |
| FreeCAD | FreeCADCmd | missing | recommended | Checkpoint/verification via FreeCAD Python. |
| Isaac | .venv_issac | True | required for Isaac adapter | Use existing Isaac environment. |
| Isaac MCP | nvidia-isaac-docs | documented in docs/agents | required before Isaac modifications | Already used before implementation. |
| MCP | Codex mcp list | 0 | required | Use high-level server only; no arbitrary shell tools. |

## my- Skills

- `/home/eii/.codex/skills/my-core-adaptive/SKILL.md`
- `/home/eii/.codex/skills/my-core-research/SKILL.md`
- `/home/eii/.codex/skills/my-core-safe-ops/SKILL.md`
- `/home/eii/.codex/skills/my-kb-learn/SKILL.md`
- `/home/eii/.codex/skills/my-kb-maintain/SKILL.md`
- `/home/eii/.codex/skills/my-robot-photo-to-isaac-cad/SKILL.md`
- `/home/eii/.codex/skills/my-visual-tutor/SKILL.md`

## Bounded Search Summary

- USD/USDAs found in controlled roots: `59`
- Extension manifests found in controlled roots: `13`

## Route Decision

- Implement a project-local minimal Visual Tutor core and high-level MCP server first.
- FreeCAD adapter starts as probe/checkpoint skeleton unless FreeCAD is available.
- Isaac adapter uses project-local extension skeleton and OpenUSD/Kit-compatible APIs; no robot or ROS control.
- No package installation in this phase because noninteractive sudo is unavailable and a minimal implementation can be tested without it.
