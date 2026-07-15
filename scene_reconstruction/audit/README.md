# Audit Outputs

This directory contains read-only scene and environment audits generated during photo-to-Isaac scene reconstruction.

## Prefixes

- `original_aloha_*`: audit of the confirmed original ALOHA USD.
- `overlay_*`: audit of the pre-existing table/pipe overlay scene found locally.
- `generated_scene_*`: audit of the newly generated scene at `scene_reconstruction/usd/aloha_real_scene.usda`.
- Unprefixed files are the last generic audit output and should not be used as provenance when a prefixed file exists.

## Current Generated Scene Check

Use:

```bash
.venv_issac/bin/python scene_reconstruction/scripts/audit_openusd_stage.py \
  --stage scene_reconstruction/usd/aloha_real_scene.usda \
  --prefix generated_scene
```

Expected:

- meters per unit: `1.0`;
- up axis: `Z`;
- prim count: `344`;
- camera prims:
  - `/World/ReconstructionCameras/cam_low_proxy`
  - `/World/ReconstructionCameras/cam_right_wrist_hint_proxy`
