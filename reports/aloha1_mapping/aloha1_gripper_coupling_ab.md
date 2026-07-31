# ALOHA 1 Official Gripper Coupling A/B

- Status: `PASS`
- Classification: `PHYSX_MIMIC_PRIMARY`
- Passing diagnostic path: `official_symmetric_adapter`
- Promotion authorized: `False`
- Next gate: `GRASP_EDITOR_DIAGNOSTIC_ON_PASSING_PATH`
- Task 8: `NOT_RUN`

## Variant A — unchanged PhysX mimic

- Fresh runs: `5`
- Mean residual: `0.0017794594168663025 m`
- Mimic gate: `FAIL`
- Maximum impulse: `0.0005472996575105919 N s`
- Minimum separation: `-0.00012263594544492662 m`

## Variant B — official symmetric diagnostic adapter

- Classification: `DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING`
- Fresh runs: `6`
- Mean residual: `2.390146255493164e-05 m`
- Mimic gate: `PASS`
- Maximum impulse: `0.0009838331445415989 N s`
- Minimum separation: `-0.00019903041538782418 m`

## Contact-equivalence validity gate

- Status: `PASS`
- Measured impulse ratio B/A: `1.7976133020374836`
- Measured additional penetration: `7.639446994289756e-05 m`

Variant B removes only the right-finger PhysX mimic in an isolated layer, copies the unchanged left drive values, and distributes one official actuation coordinate as +q/-q targets. It is evidence that the current mimic representation is primary at this runtime boundary; it is not a final asset promotion.

A rejected state-projection probe reached zero algebraic residual but caused about 9.4 mm penetration and a 0.645 N s impulse. It is retained only as rejected diagnostic evidence.

Task 8 remains `NOT_RUN`.
