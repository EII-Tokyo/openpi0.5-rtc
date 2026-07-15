# Scene Reconstruction

This directory contains generated, reproducible scene reconstruction assets for the local ALOHA workcell. Raw photos and original Isaac/USD assets are not stored or modified here.

## Regenerate

```bash
python3 scene_reconstruction/scripts/inventory_photos.py
python3 scene_reconstruction/scripts/build_parametric_assets.py
.venv_issac/bin/python scene_reconstruction/scripts/audit_openusd_stage.py --stage scene_reconstruction/usd/aloha_real_scene.usda --prefix generated_scene
python3 scene_reconstruction/scripts/render_visual_comparison.py
```

## Key Files

- Parameters: `config/scene_parameters.yaml`
- Combined Isaac scene: `usd/aloha_real_scene.usda`
- Drawings: `cad/drawings/`
- Final report: `reports/final_report.md`

## Status

This is a first-pass OpenUSD proxy model. Measured values and estimated values are separated in `config/scene_parameters.yaml`.
