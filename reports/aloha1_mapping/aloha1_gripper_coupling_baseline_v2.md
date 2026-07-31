# ALOHA 1 Gripper Coupling Baseline V2

- Status: `PASS`
- Classification: `RESET_DEPENDENT`
- Frozen Stage hash match: `True`
- Native Grasp Editor runs: `5`
- Native Grasp Editor mean residual: `0.0017794594168663025 m`
- Native Grasp Editor mimic gate: `FAIL`
- Fresh World.reset runs: `5`
- Fresh World.reset mean residual: `0.0008201375603675842 m`
- Fresh World.reset mimic gate: `PASS`
- Next gate: `ISOLATED_COUPLING_AB`
- Task 8: `NOT_RUN`

## Interpretation

The residual depends on the runtime initialization/reset boundary. This does not prove a PhysX internal defect or authorize a final coupling change.

The five native GUI runs and five fresh-reset runs use the same frozen Stage, bottle asset, contact target, 60 Hz frequency, unchanged PhysX mimic, and left-finger-only command. The measured difference is assigned to the runtime initialization/reset boundary. This does not prove an internal PhysX defect and does not authorize changing the final asset.

## Evidence scope

- Runtime readback: local Isaac/Kit/PhysX/Grasp Editor reports.
- Official API: direct NVIDIA Isaac MCP query made before the diagnostic.
- Engineering inference: `RESET_DEPENDENT` is a boundary classification.
- Not proven: an internal solver defect or final control mapping.

Task 8 remains `NOT_RUN`.
