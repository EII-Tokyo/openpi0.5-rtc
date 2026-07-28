# ALOHA Viper Finger Tessellation Determinism

- Overall status: `PARTIAL`
- Two-run determinism gate: `PASS`
- Production angular-deflection gate: `HARD_BLOCKER`
- Final/default visual and collision assets modified: `false`

The installed FreeCAD snap cannot load MeshPart, and Part.Shape.tessellate does not accept an angular-deflection parameter. These runs prove linear-only reproducibility, not the requested production tessellation parameter closure.

| Finger | Byte hash | Canonical geometry | Vertices | Triangles | Components | Degenerate |
|---|---|---|---:|---:|---:|---:|
| left_finger | MATCH | MATCH | 1808 | 3616 | 1 | 0 |
| right_finger | MATCH | MATCH | 1808 | 3616 | 1 | 0 |
