# Environment Audit

## Workspace Decision

- Selected output workspace: `/home/eii/project/openpi0.5-rtc-reward-learning/scene_reconstruction`
- The adaptive-localization helper suggested `/home/eii/isaac_mcp_setup/aloha_project`, but that is only an Isaac MCP scratch workspace. The user explicitly required the current repository `scene_reconstruction/` directory, so outputs stay here.

## System

- Hostname: `ubuntu`
- User: `eii`
- Git root: `/home/eii/project/openpi0.5-rtc-reward-learning`
- CPU architecture: `x86_64`
- Python: `Python 3.12.3`
- GPU: `NVIDIA GeForce RTX 3090 Ti, 575.57.08, 24564 MiB`
- Noninteractive sudo exit: `1`

## Capability Matrix

| Capability | Current status | Version/evidence | Needed | Plan |
| --- | --- | --- | --- | --- |
| Isaac Sim | available | from .venv_issac / project docs | required | use existing .venv_issac |
| ALOHA USD | available | bounded search | required | reuse read-only base USD |
| OpenUSD Python | missing | .venv_issac | required | use Isaac Python |
| FreeCAD | missing | PATH/module probe | recommended | fallback to proxy CAD/USDA |
| FreeCADCmd | missing | PATH probe | recommended | not required for first pass |
| HEIC conversion | available | PATH probe | conditional | use JPEG/MOV extracts if needed |
| EXIF read | missing | PATH probe | recommended | fallback to PIL metadata |
| COLMAP | missing | PATH probe | optional | not selected unless photos support SfM |
| Context7 MCP | check codex mcp list | codex mcp list | installed per user | do not reinstall |
| USD Code MCP | not installed by default | not required | recommended | skip; use official Isaac MCP and OpenUSD Python |
| FreeCAD MCP | not installed by default | not required | optional | skip; use reproducible scripts |

## Route Decision

- Selected route: Route C first, OpenUSD/Isaac proxy geometry.
- Reason: the task needs a credible spatial model now, while FreeCAD is not guaranteed and first-pass dimensions include estimates. Proxy geometry keeps all parameters centralized and can later be exported or migrated to FreeCAD/OpenSCAD.
- COLMAP is not selected for this pass. Thin metal frame, repeated profiles, reflective surfaces, and limited scale constraints make SfM unreliable as a source of truth.

## Raw Audit Files

- `scene_reconstruction/audit/environment_audit.json`
