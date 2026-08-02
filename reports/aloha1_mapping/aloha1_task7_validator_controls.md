# ALOHA1 Task 7 validator controls

- Overall: `PARTIAL`
- Released Isaac 5.1 UR10 positive control: `FAIL`
- UR10 RobotRules: `['FAIL', 'FAIL']` / [10, 10] blockers
- UR10 PhysicsRules: `['FAIL', 'FAIL']` / [2, 2] blockers
- All negative controls: `PASS`, two fresh-process signatures identical
- Bottle500 candidate: `PARTIAL` with 0 blockers; static-environment candidate: `PASS`
- Both physical candidates require user review before promotion
- Task 8: `NOT_RUN`

The released Isaac 5.1 UR10 is not clean under local Asset Validation 1.1.0: the literal results are deterministic but include blocking findings. It is therefore not misreported as a passing positive control.

| Negative control | Expected rule | Fresh consistency | Added defect |
|---|---|---:|---:|
| negative_robot_api | RobotSchema | True | 1 |
| negative_mass_api | RigidBodyHasMassAPI | True | 1 |
| negative_collider | RigidBodyHasCollider | True | 1 |
