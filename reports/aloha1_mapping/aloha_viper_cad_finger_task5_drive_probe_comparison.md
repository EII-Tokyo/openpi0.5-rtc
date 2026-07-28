# ALOHA ViperX CAD finger Task 5 drive-probe comparison

- Overall dynamic structure gate: `FAIL`
- Classification: `MAX_FORCE_IMPROVES_FINGER_TRACKING_BUT_DISJOINT_ROOT_JOINT_BLOCKS_DYNAMIC_VALIDATION`
- Bottle/contact/grasp: `NOT_RUN`
- Task 8: `NOT_RUN`

| Metric | Baseline | Max-force-only |
|---|---:|---:|
| maxForce left/right N | `{'left': 0.0, 'right': 0.0}` | `{'left': 5.0, 'right': 5.0}` |
| all intended directions correct | `False` | `True` |
| mean intended final error m | `0.0184030728` | `0.000854833052` |
| max base translation drift m | `0.0762137854` | `0.0758782677` |
| max arm DOF drift | `3.14161468` | `3.14162493` |

The 5 N profile changes only `drive:linear:physics:maxForce`. It improves finger motion but does not make the approved review Stage dynamically valid. The next isolated variable is the computed `rootJoint_vx300s_left` frame relation. No bottle test is allowed until that dynamic structure gate passes.
