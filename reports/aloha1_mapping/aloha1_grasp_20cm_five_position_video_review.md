# ALOHA 20 cm five-position video review

- Visual evidence status: **PASS**
- Five-position acceptance: **FAIL**
- Machine passes: **4/5**
- Task 8: **NOT_RUN**

## Per-position result

| Position | Machine | Reason | Visual evidence |
|---|---:|---|---:|
| position_01 | PASS | stable_20cm_hold | PASS |
| position_02 | FAIL | height_target_not_reached | PASS |
| position_03 | PASS | stable_20cm_hold | PASS |
| position_04 | PASS | stable_20cm_hold | PASS |
| position_05 | PASS | stable_20cm_hold | PASS |

## Root-cause boundary

- Position 2 retained bilateral solver contact but reached only 0.198400335 m against the unchanged 0.200 m gate.
- Its measured relative vertical slip change was 0.011552822 m.
- A diagnostic-only +2 mm lift reached the height gate but failed hold after 0.083333 s with drop 0.010775575 m.
- Classification: **POSITION_DEPENDENT_CONTINUOUS_SLIP_OR_ROTATIONAL_INSTABILITY_NOT_RESOLVED**. The extra lift is not promoted.

The raw and annotated videos were reviewed through complete frame contact-sheet montages plus annotated phase keyframes. Every view shows the full arm and the gripper/bottle inset; position 2 is correctly labeled as machine FAIL. Visual review validates evidence quality, not physical acceptance.
