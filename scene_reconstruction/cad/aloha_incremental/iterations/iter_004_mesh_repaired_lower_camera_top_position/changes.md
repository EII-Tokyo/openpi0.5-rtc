# Iteration 004: mesh-repaired lower camera top-position model

This iteration is a sanitized copy of `iter_003_lower_camera_top_position`.

Purpose:

- remove the repeated FreeCAD startup warning `The mesh data structure has some defects`;
- keep the lower-camera placement and scene geometry from iter_003;
- avoid modifying original imported mesh assets or older iterations.

Repair policy:

- copy each `Mesh::Feature` mesh payload;
- apply conservative FreeCAD mesh cleanup methods;
- reassign the mesh so FreeCAD serializes a fresh mesh data structure;
- save as a new FCStd file.

The detailed before/after mesh metrics are in `mesh_repair_report.json`.
