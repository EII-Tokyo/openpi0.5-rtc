# ALOHA bottle CAD comparison

- Status: `PASS`
- Primary: project-authored Bottle500
- Reference: downloaded `500mlbottle.step`
- Scope: CAD geometry and diagnostic visual mesh only
- Physics: `NOT_RUN_THIS_AUDIT`

| Item | Project primary | Downloaded reference |
|---|---|---|
| Source | parametric FCStd | user-provided AP214 STEP |
| CAD long axis | `+Z` | `+Y` |
| Display transform | identity | `+90°` about X |
| Optimal dimensions | `68 × 68 × 206 mm` | `60.0549 × 60.0549 × 192.7344 mm` after display-axis mapping |
| Solids / faces | `1 / 27` | `2 / 1,599` |
| Visual vertices / triangles | `1,418 / 2,832` | `27,090 / 54,260` |
| Future grasp role | **primary** | geometry reference only |

The downloaded reference is much more detailed: ribs, wave features and
bottom/neck detail are directly visible. The project primary is a simplified
rotational bottle shell. The user explicitly selected the project bottle as
the future-grasp primary; visual-detail superiority does not override that
selection.

The reference STEP's ordinary `Shape.BoundBox` overestimates its extents.
`Part.Shape.optimalBoundingBox()` agrees with the `0.20 mm / 20°` surface mesh,
so the optimal dimensions are used in the comparison. Both fresh
tessellation runs produced byte-identical OBJ files and identical canonical
geometry signatures.

The final fresh verification manifests are:

- `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-bottle-cad-comparison/final_determinism/run_a.SIiFoN/manifest.json`;
- `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-bottle-cad-comparison/final_determinism/run_b.sVLAES/manifest.json`.

No visual mesh is promoted as a collision mesh. Existing Bottle500 collision,
mass, material and static-hold behavior must be revalidated separately with
the current supplier-CAD gripper.
