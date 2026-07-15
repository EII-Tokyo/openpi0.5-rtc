# Photo-to-Isaac CAD Scene Reconstruction Final Report

## Scope

This run built a first-pass parameterized spatial model for the ALOHA workcell from local photos and the current Isaac Sim ALOHA scene. Raw photos and original ALOHA USD assets were kept read-only. All generated outputs live under `scene_reconstruction/`.

## Environment Investigation

| Item | Result |
| --- | --- |
| OS | Ubuntu 24.04.2 LTS |
| Architecture | x86_64 |
| User | `eii` |
| Git repo | `/home/eii/project/openpi0.5-rtc-reward-learning` |
| Python | system `Python 3.12.3`; Isaac env uses Python 3.11 |
| GPU | NVIDIA GeForce RTX 3090 Ti |
| Driver | 575.57.08 |
| Noninteractive sudo | unavailable, exit `1` |
| Isaac Sim | available through `.venv_issac`, observed Isaac Sim 5.1 stack during audit |
| Isaac Lab | previously present in local Isaac toolchain notes; not required for this asset build |
| ALOHA asset | `local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/aloha2_menagerie_scene_deep_black_real_start_pose.usd` |
| Image tools | `convert` and `ffmpeg` available; no HEIC files found in selected photo directory |
| CAD tools | FreeCAD, FreeCADCmd, OpenSCAD, Blender and COLMAP were not available |
| MCP | NVIDIA official Isaac MCP was used before Isaac/USD code changes |
| Skill | `photo-to-isaac-cad` installed and validated at `/home/eii/.codex/skills/photo-to-isaac-cad` |

Full audit: `scene_reconstruction/reports/environment_audit.md`.

## Route Choice

Chosen route: OpenUSD / Isaac proxy geometry.

Reasons:

- FreeCAD/OpenSCAD/COLMAP were not locally available and noninteractive sudo was unavailable.
- Current rack and exact camera bracket dimensions are still estimated, so manufacturing-grade CAD would be misleading.
- The current need is spatial validation inside Isaac Sim, so layered USDA proxy geometry is the safest first version.
- USD layering keeps the original ALOHA asset read-only.

COLMAP was not used because the scene has thin black rack profiles, reflective/transparent objects, limited explicit scale targets, and the task can proceed with measured table/pipe parameters plus estimated rack/camera placeholders.

## Generated Outputs

| Type | Path |
| --- | --- |
| Parameters | `scene_reconstruction/config/scene_parameters.yaml` |
| Evidence report | `scene_reconstruction/reports/scene_evidence.md` |
| Photo inventory | `scene_reconstruction/reports/photo_inventory.md` |
| Contact sheet | `scene_reconstruction/photos/contact_sheet.jpg` |
| CAD top/front/side/isometric drawings | `scene_reconstruction/cad/drawings/` |
| Rack parameters | `scene_reconstruction/cad/rack_parameters.yaml` |
| Camera rack USD | `scene_reconstruction/usd/camera_rack.usda` |
| Pipe/table USD | `scene_reconstruction/usd/pipe.usda` |
| Camera proxy USD | `scene_reconstruction/usd/real_layout_override.usda` |
| Combined Isaac scene | `scene_reconstruction/usd/aloha_real_scene.usda` |
| Generated stage audit | `scene_reconstruction/audit/generated_scene_stage_audit.json` |
| Visual comparison | `scene_reconstruction/renders/` |

## Parameter Status

Trusted or measured:

- Table size: 1.10 m by 0.60 m.
- Pipe diameter: 0.005 m.
- Pipe length: 0.225 m.
- Pipe mount height: 0.07 m.
- Pipe side tilt: 44 degrees.
- Pipe base offset outside table edge: 0.095 m.
- Pipe edge projection position: 0.58 m from the measured table side, converted into current table-centered coordinates.

Read from USD:

- ALOHA root stage, stage units, up axis, and existing robot prim transforms.
- Existing proxy camera positions from the prior overlay scene.

Estimated:

- Camera rack width, depth, height, profile section.
- Camera mount plate dimensions.
- Exact camera optical centers and intrinsics.
- Exact rack-to-table alignment.

Unknown:

- Real camera intrinsics, lens distortion, and calibrated extrinsics.
- Exact rack extrusion dimensions and bracket geometry.
- Exact transform from physical table frame to robot base frame.

## USD Validation

The generated scene was validated with Isaac/OpenUSD libraries without launching the GUI:

```bash
.venv_issac/bin/python scene_reconstruction/scripts/audit_openusd_stage.py \
  --stage scene_reconstruction/usd/aloha_real_scene.usda \
  --prefix generated_scene
```

Validation result:

- Stage opened successfully.
- Stage units: meters, Z-up.
- Sublayers:
  - original ALOHA USD;
  - `camera_rack.usda`;
  - `pipe.usda`;
  - `real_layout_override.usda`.
- Prim count: 344.
- Camera prims:
  - `/World/ReconstructionCameras/cam_low_proxy`
  - `/World/ReconstructionCameras/cam_right_wrist_hint_proxy`
- Expected rack, pipe and camera proxy prims were present.

## Visual Comparison

Generated views:

- `scene_reconstruction/renders/front_oblique/`
- `scene_reconstruction/renders/rack_oblique/`

Each contains:

- `real.jpg`
- `simulated.png`
- `overlay.png`
- `edges.png`

Current confidence: low-to-medium. The overlay proves that the proxy model captures the coarse workcell concept, but it is not photogrammetrically aligned. Exact camera pose fitting remains future work.

## Usage

Regenerate photo inventory:

```bash
python3 scene_reconstruction/scripts/inventory_photos.py
```

Regenerate CAD-style drawings and USD layers:

```bash
python3 scene_reconstruction/scripts/build_parametric_assets.py
```

Validate the generated USD:

```bash
.venv_issac/bin/python scene_reconstruction/scripts/audit_openusd_stage.py \
  --stage scene_reconstruction/usd/aloha_real_scene.usda \
  --prefix generated_scene
```

Regenerate visual comparison assets:

```bash
python3 scene_reconstruction/scripts/render_visual_comparison.py
```

Open in Isaac Sim with the existing project launcher only as an explicit experiment:

```bash
OMNI_KIT_ACCEPT_EULA=YES .venv_issac/bin/python \
  examples/aloha_isaac/scripts/open_workcell_gui.py \
  --usd scene_reconstruction/usd/aloha_real_scene.usda \
  --allow-noncanonical-usd
```

## Rollback

No system packages were installed. No sudo commands were run. No original photos or original ALOHA USD files were modified.

No real robot commands were sent. Existing Isaac/MCP processes that were already running on the machine were left untouched because they may belong to the user's active desktop workflow.

Rollback steps:

```bash
rm -rf scene_reconstruction
rm -rf /home/eii/.codex/skills/photo-to-isaac-cad
git checkout -- AGENTS.md docs/agents/scene_reconstruction.md
```

If committed, revert the commit instead of deleting files manually.

## Remaining Work

1. Measure rack width, depth, height and extrusion section.
2. Measure or calibrate camera optical centers and camera pitch/yaw.
3. Add AprilTag or ChArUco board photos for camera pose fitting.
4. Replace estimated camera/rack values in YAML with measured values.
5. If FreeCAD becomes safely available, add a manufacturing-oriented CAD route while keeping this proxy USD route reproducible.
