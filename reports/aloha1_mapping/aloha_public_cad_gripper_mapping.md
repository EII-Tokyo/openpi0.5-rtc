# ALOHA Public CAD Gripper Mapping

- Status: `PASS`
- Installation/orientation mapping: `PASS`
- Source connection geometry audit: `PASS` (`SOURCE_CAD_SLIDING_CARRIAGE_COMMON_VOLUME_RECORDED`)
- Production angular-controlled tessellation: `PASS`
- Supplier-CAD mounting datum registration: `PASS` (`CONTROLLED_ORTHOGONAL_PLANAR_DATUM_REGISTRATION`)
- Isaac isolated installation screenshots: `PASS` (8 raw/annotated pairs)
- Isaac/default asset mutation: `false`
- CAD +X opening side → URDF `left_finger` (+Y)
- CAD -X opening side → URDF `right_finger` (-Y via mimic)
- Frame evidence: URDF gripper-bar visual uses `Rz(+90 deg)`, mapping CAD/STL +X to URDF +Y.

## Primary purchase-confirmed follower assembly

- Source: `Simple Aloha Viper 2024-5-13.step` (`337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571`)
- Root / finger group: `Dummy_Aloha_VX_v3` / `Aloha_VX_Fingers_2024_4_21_v2`
- Blue / CAD +X / URDF left: `Part__Feature007` (`Aloha VX Fingers 2024-4-21 v2`)
- Orange / CAD -X / URDF right: `Part__Feature008` (`Aloha VX Fingers 2024-4-21 v001`)
- Shared source placement determinant: `1.0`
- CAD unit → Isaac unit: `0.001 m/mm`
- Supplier static state: `CLOSED_REFERENCE`
- Derived open state: left `+36 mm` CAD X; right `-36 mm` CAD X
- Visual evidence: `PASS` (8 raw/annotated pairs)

The supplier shell/sliding-carriage Boolean common volumes are recorded as source connection geometry. They are not silently relabeled as an unexpected simulated collision.

## Toolchain

- FreeCAD: `1 / 1 / 1 / 44227 +647 (Git) / Unknown / 2026/04/14 22:09:59 / tag: 1.1.1 / 0108fd4b4850cc46e625b60e53cea7a7bbe69f8d`
- OpenCascade: `7.8.0`
- Blender: `5.2.0 LTS` / `BLENDER_WORKBENCH`
- Angular deflection control: `EXPLICIT_MESHPART_LINEAR_AND_ANGULAR`
- CAD local axes → finger link: `+X→+Y`, `+Y→+Z`, `+Z→+X`; determinant `1.0`; mirror `false`.
- Mounting datum threshold: `0.0002 m`; full-surface ICP was not used for the decision.

The standalone 2025 finger STEP is not silently substituted for the embedded 2024 instances because their labels, bounds, and volumes identify different revisions. Purchase-confirmed Simple Viper is the follower-primary assembly; Widow and Stationary are cross-checks.

The Isaac screenshot PASS is a CAD-installation visual gate only. It does not claim collider, contact, dynamics, or bottle-grasp acceptance.

## Stationary follower instances

| Follower CAD object | Position (mm) | CAD closed inner gap | +X finger | -X finger |
|---|---|---:|---|---|
| Dummy_Aloha_VX_SV2_v001 | `[24.4999999769513, -469.350288299484, 20.0000000005095]` | 4.488278416 mm | `Part__Feature650` → left_finger | `Part__Feature651` → right_finger |
| Dummy_Aloha_VX_SV2_v3 | `[24.4999999770112, 469.350288299459, 20.0000001192651]` | 4.488278416 mm | `Part__Feature640` → left_finger | `Part__Feature641` → right_finger |
