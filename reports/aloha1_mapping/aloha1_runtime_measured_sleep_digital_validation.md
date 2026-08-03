# ALOHA1 runtime-measured Sleep digital validation

- Status: `PASS_DIAGNOSTIC_DIGITAL_ONLY`
- Classification: `RUNTIME_MEASURED_SLEEP_ALIGNED_IN_ISAAC`
- Sequence: `SLEEP_HOME_SLEEP` × `3`
- Fresh Isaac processes: `2`
- Numeric signatures match: `true`
- Real motion commands: `0`
- Final/default asset modified: `false`

## Result

The digital follower_left starts at the median of 9000 stationary real JointState samples, moves Sleep → Home → Sleep for three cycles, and ends at the same runtime Sleep reference. Two fresh Isaac Sim 5.1 processes produced the same normalized numeric signature.

The elbow and wrist_angle limit differences are accepted only through an anonymous session layer classified `DIAGNOSTIC_ONLY_RUNTIME_ALIGNMENT`. No source USD, final asset, or default joint mapping was changed.

This validates the isolated digital initialization and trajectory. It does not authorize or claim a synchronized real-hardware run.

## Runtime limit readback

| Joint | Bound | Authored rad | USD degrees readback |
|---|---|---:|---:|
| `elbow` | `upper` | 1.622951746 | 92.988288879 |
| `wrist_angle` | `lower` | -1.883728504 | -107.929695129 |
