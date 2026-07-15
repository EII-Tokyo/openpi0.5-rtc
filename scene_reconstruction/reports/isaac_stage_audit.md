# Isaac Stage Audit

- Stage: `/home/eii/project/openpi0.5-rtc-reward-learning/local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/aloha2_menagerie_scene_deep_black_real_start_pose.usd`
- Original stage modified: `no`
- Meters per unit: `1.0`
- Up axis: `Z`
- Prim count: `324`

## Camera Prim Audit

- No `UsdGeom.Camera` prim found; GUI viewport camera must not be treated as a real sensor camera.

## Important Distinction

- GUI viewport: an editor view used by the human. It is not automatically a simulated sensor.
- `UsdGeom.Camera`: a stage prim with transform and optical parameters. Only these are recorded in `cameras.json`.
- Isaac sensor camera / render product: runtime sensor pipeline. None was assumed unless found in the stage.

## Candidate Prim Paths

- Robot candidates: `80` sample paths in JSON
- Table candidates: `6` sample paths in JSON
- Pipe candidates: `0` sample paths in JSON
