# Phase 42 Gripper Proxy Gap Gate

## Question

After the gripper-only bbox proxy stage passed arm and gripper DOF smoke tests, the next question was narrower:

Can the gripper collision proxies actually move apart or together with the simulated gripper control signal?

This matters before any bottle/contact test. If the finger collision proxies are visually present but do not move with the controlled finger links, then a bottle contact test would be ambiguous.

## Inputs

- Stage:
  `local_eval_assets/aloha1_clean_runtime_20260718/aloha1_dual_bbox_proxy_runtime.usda`
- Validator:
  `aloha_isaac_replay/scripts/validate_aloha1_gripper_proxy_gap.py`
- Full Isaac logs:
  - `.codex/artifacts/20260718-035412_phase42-gripper-proxy-gap`
  - `.codex/artifacts/20260718-035613_phase42-opposed-fingers-gap`
  - `.codex/artifacts/20260718-035614_phase42-same-fingers-gap`

## Results

The bbox distances below are reported in USD stage units from `UsdGeom.BBoxCache`. The runtime stage declares `metersPerUnit = 0.01`, so these values should be interpreted as stage-unit deltas unless explicitly converted.

| control mode | status | maximum DOF delta | maximum proxy center-distance delta | interpretation |
| --- | --- | ---: | ---: | --- |
| `gripper` | `FAILED_GATE` | 0.0060 | 0.000253 stage units | The aggregate gripper DOF tracks, but does not create enough proxy gap motion for a contact gate. |
| `opposed_fingers` | `PASS` | 0.0221 | 0.001828 stage units | The finger proxies are kinematically attached to moving finger links. |
| `same_fingers` | `PASS` | 0.0221 | 0.001828 stage units | Direct finger DOF control also produces measurable proxy motion. |

## Interpretation

The gripper-only bbox proxies are not inert or detached. They can move when the actual finger DOFs are driven.

The failed `gripper` mode is important: it means the aggregate `gripper` DOF should not be used as the only evidence that the collision geometry is opening or closing. For the next contact smoke test, the control signal should explicitly command the finger DOFs or first establish a verified mimic/gear relation from the aggregate gripper command to the finger joints.

## Next Gate

The next safe gate is a passive object contact smoke test:

1. Place a small passive cylinder or box between the left finger proxies.
2. Drive `left_finger/right_finger` with the verified direct-finger control mode.
3. Check for bounded object motion, no explosions, no NaN, no large sustained jitter, and plausible contact with the finger proxies.

This is still not a full grasp success test. It only validates that the local gripper collision layer can participate in simple contact without destabilizing the articulation.
