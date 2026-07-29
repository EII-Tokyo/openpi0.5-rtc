# ALOHA Viper follower left/right CAD identity

- Status: `PARTIAL`
- Classification: `VERIFIED_SINGLE_REUSABLE_ROBOT_PRODUCT`
- Robot-local identity: `VERIFIED`
- Workcell placement: `NOT_VERIFIED`
- Task 8: `NOT_RUN`

## Result

The supplier STEP contains one complete ViperX product `Dummy_Aloha_VX_v3` and no second workcell instance. The pinned Xacro configuration identifies both followers as `aloha_vx300s`; after removing only the left/right instance prefix, their generated URDFs have the same canonical SHA-256. The first-hand purchase chain identifies a pair of ViperX 300 6DOF follower arms. Therefore the right follower is a new robot-local instance of the same product, not missing CAD and not mirrored geometry.

## Boundary

- A robot-local `follower_right` diagnostic Stage may be generated at the local origin.
- No complete supplier-CAD or calibrated workcell transform is available here; `HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM` remains.
- A robot-local PASS must not be described as dual-arm workcell placement PASS.

## Supplier handed fingers

- Blue: `left_finger`, embedded v2, CAD `+X`.
- Orange: `right_finger`, embedded v2, CAD `-X`.
- No mirroring, standalone-v3 substitution, generic 856-face mesh, or historical gym-aloha mesh is permitted.

## B-Rep validity

- Status: `PARTIAL`
- Invalid source objects retained without healing: `Dummy_Aloha_VX_v3, Part__Feature005`
- This is a source-geometry validity limitation, recorded separately from the product-identity conclusion.

## Toolchain

- FreeCAD: `1.1.1`
- OpenCascade: `7.8.1`
- Tessellation contract: `0.20 mm`, `20 deg`, `Relative=False`.

## License

- `UNKNOWN_HARD_BLOCKER`: public download access is not explicit redistribution permission. Original STEP/PDF files remain local read-only artifacts and are not committed.
