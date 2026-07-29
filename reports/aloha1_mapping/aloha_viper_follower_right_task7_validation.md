# follower_right robot-local Task 7 validation

- Status: `FAIL`
- Scope: `ROBOT_LOCAL_DIAGNOSTIC_NOT_WORKCELL_PLACEMENT`
- Stage: `/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/supplier_cad_follower_right/1.0/supplier_cad_follower_right.usda`
- Stage immutable: `True`
- Arm one-joint: `PASS`
- Gripper direction / aperture / mimic: `PASS` / `PASS` / `FAIL`
- Screenshot visual gate: `PASS` (auxiliary only)
- Dual-arm workcell placement: `PARTIAL` / unverified
- Task 8: `NOT_RUN`

| Official category | Status | Blocking | Warnings |
|---|---|---:|---:|
| IsaacSim.PhysicsRules | FAIL | 5 | 0 |
| IsaacSim.RobotRules | FAIL | 4 | 7 |
| IsaacSim.SimReadyAssetRules | PASS | 0 | 0 |

## HARD_BLOCKER

- `HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM`

This report validates only the isolated follower_right robot product in its local frame. It does not validate a dual-arm workcell placement. Visual screenshots are auxiliary; numeric runtime and official-rule results are authoritative.
