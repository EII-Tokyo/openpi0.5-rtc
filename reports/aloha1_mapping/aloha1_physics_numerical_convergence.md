# ALOHA1 Isaac 5.1 numerical convergence

- Status: `PARTIAL`
- Frequency: `None`
- Position iterations: `None`
- Velocity iterations: `None`
- Final repeat deterministic: `None`
- Physical model parameters were frozen; only dt/solver iterations changed.
- Grasp PASS/FAIL was not used to create a tolerance.

## Free-motion baseline

| Hz | position error (m) | COM velocity error (m/s) |
|---:|---:|---:|
| 60 | 2.48323023e-07 | 5.0038075e-07 |
| 120 | 6.188213e-07 | 8.76489172e-07 |
| 240 | 1.32420869e-06 | 2.0562098e-06 |
| 480 | 5.15430904e-06 | 4.2856218e-06 |
| 960 | 4.80976553e-06 | 8.84369516e-06 |

## Frequency-pair convergence

| pair (Hz) | joint position max (rad) | bottle position max (m) | contact onset delta (s) | all gates |
|---:|---:|---:|---:|:---:|
| 60→120 | 0.0447011374 | 0.0107638725 | 0.6000000312924385 | False |
| 120→240 | 0.0968990505 | 0.00966357373 | 0.10416667209938169 | False |
| 240→480 | 0.0970179465 | 0.02098129 | 0.08958333800546825 | False |
| 480→960 | 0.0756346652 | 0.0105089169 | 0.2541666799224913 | False |

## Boundary

This validates numerical sensitivity of the current isolated diagnostic model. It does not promote the rejected contact-band collider, calibrate friction, or establish continuous-duty actuator limits.
A machine `PARTIAL` caused only by cross-step numerical disagreement is not a visible grasp failure. When every cell still physically passes, a failure video is `NOT_REQUIRED`; the signed telemetry and pairwise metrics are the authoritative evidence.
