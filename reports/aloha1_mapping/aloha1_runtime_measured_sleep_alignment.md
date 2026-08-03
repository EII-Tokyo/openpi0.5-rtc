# ALOHA1 runtime-measured Sleep alignment

- Status: `READY_FOR_ISOLATED_DIGITAL_VALIDATION`
- Classification: `RUNTIME_MEASURED_SLEEP_DIAGNOSTIC_ALIGNMENT`
- Samples: `9000`
- Sequence: `SLEEP_HOME_SLEEP` × `3`
- Diagnostic limit policy: `DIAGNOSTIC_ONLY_RUNTIME_ALIGNMENT_NOT_FINAL_ASSET`
- Real motion commands: `0`
- Final/default asset modified: `false`

## Runtime Sleep reference

`[0.0, -1.8453789949417114, 1.6229517459869385, -0.006135923322290182, -1.8837285041809082, -0.006135923322290182]` rad

## Diagnostic-only limit changes

| Joint | Bound | Source (rad) | Diagnostic (rad) | Delta (rad) |
|---|---|---:|---:|---:|
| `elbow` | `upper` | 1.605702758 | 1.622951746 | 0.017248988 |
| `wrist_angle` | `lower` | -1.867502093 | -1.883728504 | -0.016226411 |

These changes exist only in the isolated runtime session layer so the digital arm can start from the frozen real readback. They are not hardware calibration and are not eligible for final/default asset promotion.
