# Project Bottle500 CAD audit

- Status: `PASS`
- Selection: `ACTIVE_PRIMARY_FOR_FUTURE_DIGITAL_GRASP_TESTS`
- FCStd:
  `/home/eii/project/openpi0.5-rtc-reward-learning/assets/bottle_500ml/cad/bottle_500ml.FCStd`
- FCStd SHA-256:
  `3594f60200e54181bc8480a229484293a0d386c146d3f235b32e31a0c16bbf8a`
- Exported STEP SHA-256:
  `863001b4d939d7d8c879497b5054fe93f426662761e6fb7a80550096fd9bc780`

The project-authored source chain contains the generating Python script,
parameter spreadsheet, `OuterRevolution`, `InnerRevolution`, `BottleMaster`,
FCStd, exported STEP, visual OBJ, Blender file and an existing Isaac USD.
`BottleMaster` is a valid one-solid bottle shell.

The project-pinned FreeCAD 1.1.1 / OCCT 7.8.1 audit confirms:

- optimal and ordinary B-Rep dimensions: `68 × 68 × 206 mm`;
- CAD long axis: `+Z`;
- one solid, one shell, 27 faces, 49 edges and 26 vertices;
- B-Rep volume: `62,849.88581362739 mm³`;
- FCStd and exported STEP have matching bounds, area, volume and topology.

Different FCStd/STEP B-Rep byte hashes are not a failure: the FCStd object is
a compound wrapper while the exported STEP is a solid, but their geometry
metrics agree within the recorded tolerances.

Fresh tessellation runs with the mandated `0.20 mm` linear and `20°` angular
deflection produced identical OBJ hashes and canonical geometry signatures:
`1,418` vertices, `2,832` triangles, no degenerate triangles.

This is a CAD and visual-geometry PASS, not a new physics PASS. The existing
USD and 41-piece collision design remain available, but collision, mass,
material and static hold have not been revalidated in this audit against the
current supplier-CAD gripper. The FCStd `25 g` parameter remains
`TEMPORARY_REQUIRES_MEASUREMENT`; it does not silently replace the current
Task 5 `20 g` diagnostic baseline.

Machine evidence:

- `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-500ml-bottle-source/project_bottle_fcstd_audit.json`
- `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-132415_20260729-aloha-project-bottle-optimal-bbox-audit-v3`
- `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-bottle-cad-comparison`
