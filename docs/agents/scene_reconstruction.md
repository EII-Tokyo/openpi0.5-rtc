# Scene Reconstruction Operations

Read this before tasks that build or modify `scene_reconstruction/`, camera rack CAD, pipe/table spatial models, photo-to-USD reconstruction assets, or visual comparison reports.

## Workspace

- Use `/home/eii/project/openpi0.5-rtc-reward-learning/scene_reconstruction/` for generated assets in this repository.
- Keep raw photos under `/home/eii/Downloads/iphone` read-only.
- Keep original ALOHA USD assets under `local_eval_assets/` read-only.
- Do not write generated CAD/USD files into the original Isaac asset directory.

## Current Route

- Current first-pass route: OpenUSD / Isaac proxy geometry.
- Reason: FreeCAD/OpenSCAD/COLMAP are not installed, noninteractive sudo is unavailable, and rack/camera dimensions are still partly estimated.
- Generated assets are layered USDA files, not flattened replacements.

## Required Evidence Style

- Store parameters in `scene_reconstruction/config/scene_parameters.yaml`.
- Use only these status values: `measured`, `read_from_usd`, `derived`, `estimated`, `unknown`.
- Low-confidence rack and camera dimensions must remain marked as `estimated`.
- Distinguish GUI viewport, USD `Camera` prims, Isaac sensor cameras, and ROS camera frames.

## FreeCAD Launch Rule

- On this machine, FreeCAD is the snap build at `/snap/bin/freecad` / `/snap/bin/freecad.cmd`.
- For ALOHA incremental CAD iterations, open models through the checked-in opener scripts under `scene_reconstruction/cad/aloha_incremental/scripts/open_iter*.py`.
- Do not rely on direct `.FCStd` opening or `--safe-mode` for normal review; snap FreeCAD has shown cases where direct file opening starts but the expected model view is not reliably visible.
- Example:

```bash
/snap/bin/freecad scene_reconstruction/cad/aloha_incremental/scripts/open_iter003_lower_camera_top_position.py
```

- For command-line generation or verification, put helper scripts inside the repository, then run them through `freecad.cmd -c`. Snap FreeCAD may not be able to read temporary scripts from `/tmp`.
- Known issue: imported ALOHA mesh payloads can emit `The mesh data structure has some defects` during FreeCAD load. Treat this as a mesh-asset warning unless a visual or geometry check fails. Do not repair original imported assets in place; create a new iteration and preserve before/after diagnostics.

## Validation Commands

From repository root:

```bash
python3 scene_reconstruction/scripts/inventory_photos.py
python3 scene_reconstruction/scripts/build_parametric_assets.py
.venv_issac/bin/python scene_reconstruction/scripts/audit_openusd_stage.py --stage scene_reconstruction/usd/aloha_real_scene.usda --prefix generated_scene
python3 scene_reconstruction/scripts/render_visual_comparison.py
```

## Skill

- A local Codex skill was installed at `/home/eii/.codex/skills/photo-to-isaac-cad`.
- Use it for future photo-to-Isaac CAD/USD reconstruction tasks.
- Rollback: delete `/home/eii/.codex/skills/photo-to-isaac-cad`.
