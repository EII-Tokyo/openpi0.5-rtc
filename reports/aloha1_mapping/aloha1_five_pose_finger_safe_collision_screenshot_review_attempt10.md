# ALOHA1 attempt10 finger-safe collision screenshot review

- Status: `PASS`
- Machine status: `PASS`
- Visual-model review: `PASS`
- Capture records: `120`
- Raw + annotated images: `240`
- Existing five user-confirmed MP4s rerun: `false`
- Candidate promotion: `NOT_PROMOTED`
- Task 8: `NOT_RUN`

| Sample | Status | Captures | Images | Runtime signature |
|---|---:|---:|---:|---|
| `sample_01` | `PASS` | 24 | 48 | `11838209dc4ca742dcdc5480f00466f35907316545cdbcbed7171bb83ef02cf6` |
| `sample_02` | `PASS` | 24 | 48 | `a3316fb6bbbe2140f68ce0a0b47e2acb826e873ac049a117e19e8ddcb52d4863` |
| `sample_03` | `PASS` | 24 | 48 | `8b7a32db54c0bf79d6059919f241ca1b2b215d63557627606ff90d08d7050ef1` |
| `sample_04` | `PASS` | 24 | 48 | `8b05a06978c4e7b22220284f982c358c4190e360eddbc84e920d64d150969a5a` |
| `sample_05` | `PASS` | 24 | 48 | `ee728e36f5bca1a90b6f4a5f8058f812cad083595ea22e669f0b8442a000f7ec` |

## Retake history

- `sample_01_rejected_origin_only_attempt2`: `REJECTED_SUBJECT_FRAMING` — Origin-only closeup did not contain the complete supplier-CAD finger geometry.
- `sample_01_rejected_render_latency_attempt3`: `REJECTED_STALE_RENDER` — One paused app update did not converge the RGB render to the new camera pose.
- `sample_01_rejected_three_updates_attempt4`: `REJECTED_STALE_RENDER` — Three paused app updates still showed stale framing.
- `sample_02_invalid_wrong_initial_qpos`: `INVALID_OPERATOR_INPUT_EXCLUDED` — Wrong initial qpos was detected and interrupted before inclusion; it is not evidence for any sample.
- `sample_02_original`: `REJECTED_CLOSEUP_OCCLUSION` — The vertical frame occluded the supplier-CAD fingers after release.
- `final_selected_retakes`: `PASS` — Subject-bounds camera fit plus 20 paused render updates; runtime signatures match attempt10 and no physics parameters or steps changed.

## Evidence boundary

This PASS is an auxiliary visual-evidence gate over exact frozen PNG files and manifests. Runtime contact, pose, velocity, drop, finger-limit and overlap telemetry remains authoritative. It does not promote the diagnostic session layer or final/default collider.
