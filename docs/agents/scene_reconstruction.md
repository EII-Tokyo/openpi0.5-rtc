# Scene Reconstruction Operations

**Status: LEGACY_PHOTO_PROXY_WORKFLOW_ONLY.**

Read this only for tasks that build or modify `scene_reconstruction/`,
camera-rack/table/pipe proxies derived from photos, or the historical visual
comparison workflow. It is not the CAD-to-Isaac ingestion standard.

For supplier/measured CAD, STEP/IGES/FCStd review, assembly interpretation,
tessellation, robot-link mapping, or CAD-derived USD, use
`docs/agents/cad_to_isaac_asset_mapping.md`. That document supersedes this one
whenever the two overlap.

## Workspace

- Use `/home/eii/project/openpi0.5-rtc-reward-learning/scene_reconstruction/` for generated assets in this repository.
- Keep raw photos under `/home/eii/Downloads/iphone` read-only.
- Keep original ALOHA USD assets under `local_eval_assets/` read-only.
- Do not write generated CAD/USD files into the original Isaac asset directory.

## Historical Route

- The 2026-07 photo reconstruction used OpenUSD / Isaac proxy geometry.
- The original reason included missing CAD/photogrammetry capabilities and incomplete rack/camera measurements. That capability statement is historical and must not be reused as a current host fact.
- Probe FreeCAD, OpenSCAD, COLMAP, Isaac, and Python capabilities at the start of every new task.
- Generated assets are layered USDA files, not flattened replacements.

## Required Evidence Style

- Store parameters in `scene_reconstruction/config/scene_parameters.yaml`.
- Use only these status values: `measured`, `read_from_usd`, `derived`, `estimated`, `unknown`.
- Low-confidence rack and camera dimensions must remain marked as `estimated`.
- Distinguish GUI viewport, USD `Camera` prims, Isaac sensor cameras, and ROS camera frames.

## FreeCAD Launch Rule

- On this machine, FreeCAD is the snap build at `/snap/bin/freecad` / `/snap/bin/freecad.cmd`.
- Known local issue: normal FreeCAD startup can load the `AICopilot` addon and emit PySide/shiboken errors such as `Unable to import shiboken2 ... AICopilot`. When this happens, a valid CAD file may open to a blank viewport even though the geometry exists.
- Do not conclude a CAD file is empty or broken from a blank FreeCAD viewport alone. First inspect the file contents and shape bounds with command-line FreeCAD or archive checks.
- For snap FreeCAD command-line checks, put helper scripts inside the repository and run them through `freecad.cmd -c "exec(open('/abs/path/to/script.py').read())"`. Snap FreeCAD may not be able to read temporary scripts from `/tmp`.
- If a normal FreeCAD GUI view is blank, retry visual inspection with safe mode and the exported STEP:

```bash
/snap/bin/freecad --safe-mode <model.step>
```

- The 2026-07-17 bottle asset check confirmed this pattern: `assets/bottle_500ml/cad/bottle_500ml.FCStd` contained valid `OuterRevolution`, `InnerRevolution`, and `BottleMaster` shapes with a `68 mm x 68 mm x 206 mm` bounding box, but normal GUI opening showed a blank viewport; `--safe-mode` opening of `assets/bottle_500ml/cad/bottle_500ml.step` displayed correctly.
- Prefer opening the exported STEP for quick visual review when an FCStd view provider does not restore correctly. Do not regenerate or overwrite CAD geometry merely because the normal FreeCAD GUI opened to a blank viewport.
- For ALOHA incremental CAD iterations, open models through the checked-in opener scripts under `scene_reconstruction/cad/aloha_incremental/scripts/open_iter*.py`.
- Do not rely on direct `.FCStd` opening or `--safe-mode` for normal review; snap FreeCAD has shown cases where direct file opening starts but the expected model view is not reliably visible.
- Example:

```bash
/snap/bin/freecad scene_reconstruction/cad/aloha_incremental/scripts/open_iter003_lower_camera_top_position.py
```

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

- Use `/home/eii/.codex/skills/my-robot-photo-to-isaac-cad` only for photo-derived reconstruction.
- Use `/home/eii/.codex/skills/my-robot-cad-to-isaac` for supplier or measured CAD asset mapping.
