# ALOHA ViperX supplier-CAD finger Task 5 dynamic structure

- Overall status: `PASS`
- Numeric no-bottle structure gate: `PASS`
- Runtime readback visual gate: `PASS_AUXILIARY_RUNTIME_READBACK_REPLAY`
- Bottle/contact/grasp: `NOT_RUN`
- Task 7 / Task 8: `NOT_RUN` / `NOT_RUN`

| Profile | Status | max base drift m | max arm drift | max intended finger error m |
|---|---|---:|---:|---:|
| baseline | FAIL | 0.0762137854 | 3.14161468 | 0.0360000003 |
| finger_max_force_only | FAIL | 0.0758782677 | 3.14162493 | 0.00195637718 |
| root_frame_only | FAIL | 0.000189307056 | 2.91157389 | 0.0360000003 |
| finger_max_force_plus_root_frame | FAIL | 0.000160097923 | 2.58399582 | 0.000603297725 |
| arm_max_force_over_combined | PASS | 2.87047775e-05 | 0.000118136406 | 4.65661287e-08 |

## Causal result

- Correcting the computed root-joint frame removes the approximately 76 mm base snap.
- Setting only the two finger maxForce values to the 5 N URDF effort limits restores intended finger tracking.
- Setting the six arm maxForce values to their generated URDF effort limits reduces arm drift below the numeric gate.
- These are isolated diagnostic settings, not promotion of the final/default asset.

The final isolated profile passes every machine-readable numeric no-bottle gate. Three fixed-camera Isaac viewport replays of exact runtime readbacks also pass visual review. They are auxiliary evidence, not same-frame physics proof.
