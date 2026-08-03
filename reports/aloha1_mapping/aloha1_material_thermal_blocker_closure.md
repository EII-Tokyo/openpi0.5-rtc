# ALOHA1 material and continuous-duty closure

- Status: **HARD_BLOCKER**
- Runtime material binding verified: `True`
- Physical friction calibrated: `False`
- Continuous force envelope verified: `False`
- Diagnostic friction scan: **NOT_RUN_NO_CALIBRATED_PAIR_PROPERTY**
- Final/default asset modified: `False`

## Result

The existing Isaac runtime audit verifies that the temporary physics materials are bound and combine as authored. It does not identify the real finger-pad/bottle materials or calibrate their friction. The exact manufacturer motor tables likewise do not provide the measured loaded continuous gripper force/thermal envelope needed for a final maxForce.

## Missing exact evidence

- `EXACT_FINGER_PAD_MATERIAL_AND_SURFACE_FINISH`
- `EXACT_BOTTLE_MATERIAL_AND_SURFACE_FINISH`
- `EXACT_PAIR_STATIC_DYNAMIC_FRICTION_AND_RESTITUTION`
- `MEASURED_CONTINUOUS_TORQUE_SPEED_CURRENT_THERMAL_CURVE`

The authored value `0.7` remains `TEMPORARY_UNCALIBRATED`. A successful bottle
hold or a generic plastic table is not accepted as calibration. The published
stall and 20%-of-stall estimates are not treated as a measured loaded thermal
envelope.
