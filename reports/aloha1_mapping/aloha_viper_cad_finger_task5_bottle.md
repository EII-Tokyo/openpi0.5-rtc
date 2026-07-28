# ALOHA ViperX supplier-CAD finger Task 5 bottle result

## Result

- Machine status: `PASS`
- Scope: `follower_left` isolated supplier-CAD diagnostic only
- Trials: `20/20 PASS`, each from a fresh World reset
- Determinism: `PASS`, one deterministic signature
- Bottle: `0.020 kg`, `0.065 m` diameter, upright cylinder proxy
- Hold interval: `2.0 s` at `60 Hz`
- Maximum drop over the complete 120-frame hold:
  `0.0004539191722869873 m`
- Drop gate: `0.010 m`
- Final-position drop: `0.00016772747039794922 m`
- Maximum penetration: `0.00016659701941534877 m`
- Persistent penetration: `false`
- Left/right physical surface contact before release: `true/true`
- Contact at hold end: `true/true`
- Unexpected finger/bar/internal collision gate: `false`
- Fixed joint, Surface Gripper, parent attachment: `false/false/false`

This is a digital static-suspension hold PASS under the serialized diagnostic
gate. It is not a calibrated sim-to-real grasp claim, final collider
promotion, or bottle-lift trajectory PASS.

## Contact semantics

The contact reporter first emitted envelope events near positive
`10 mm` separation. Those events were not accepted as physical bilateral
contact. Release was permitted only after each finger produced a contact
sample with `separation <= 0`:

- left first physical separation: `-0.000005884788606635993 m`
- right first physical separation: `-0.0000010477378964424133 m`

Both contact paths resolve to the supplier-CAD v2 convex-hull finger
colliders and `/workcell/Task5BottleSession/BottleProxy`. Contact positions,
normals, impulses, materials and the complete per-frame event stream are in
`aloha_viper_cad_finger_task5_bottle.json`.

The fixed/kinematic bottle phase is used only to establish bilateral contact
and normal direction. It is explicitly excluded from the static-hold PASS.
The hold starts only after the bottle is made dynamic with gravity enabled.

## Runtime readback caveat

`SingleRigidPrim.get_linear_velocity()` and finite-difference velocity from
the bottle pose disagree at hold end:

- API vertical velocity: `+0.06703243404626846 m/s`
- pose-derived final vertical velocity: `+0.000050067901611328125 m/s`
- maximum absolute API-versus-pose difference:
  `0.14052210748195648 m/s`

The report records this as
`RUNTIME_READBACK_DISAGREEMENT_RECORDED_NOT_USED_TO_OVERRIDE_POSITION_DROP_GATE`.
Neither signal is hidden. The PASS uses the serialized full-interval
position-drop gate; the velocity discrepancy remains unresolved runtime
readback evidence.

## Frozen diagnostic inputs

- Source review Stage:
  `/home/eii/project/openpi0.5-rtc-reward-learning/local_eval_assets/aloha_isaac_assets/aloha_viperx.usd`
- Source Stage SHA-256:
  `b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e`
- Bottle diagnostic Stage:
  `/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/cad_finger_task5_bottle/aloha_viperx_supplier_cad_bottle_task5.usda`
- Diagnostic Stage SHA-256:
  `62697e4b25a7ec82234cc9ebd79d4a6d530a6ead0165519cbd275c0fa3f32178`
- Collider: supplier assembly embedded v2 finger `convexHull`
- Finger max force: `5 N` per side from the generated URDF effort limit
- Static/dynamic friction: `0.7`, `TEMPORARY_UNCALIBRATED`
- Restitution: `0`
- Self collision: disabled
- `solve_articulation_contact_last`: `true`
- Contact/rest offsets: not authored

The source Stage, default configuration and final/default collider hashes were
unchanged before and after the run.

## Screenshot evidence

Four raw and four annotated images cover:

- open
- bilateral physical contact
- release
- hold end

All eight were opened and inspected individually with the vision model. They
use one fixed camera. The contact, release and hold images contain camera-
projected runtime contact points and normals; the open image labels only
CAD-derived inward-surface samples.

- Raw root:
  `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/task5_bottle_acceptance_v3/screenshots_raw`
- Annotated root:
  `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/task5_bottle_acceptance_v3_annotation_attempt2/screenshots_annotated`
- Visual review:
  `reports/aloha1_mapping/aloha_viper_cad_finger_task5_bottle_screenshot_review.json`

Screenshots are auxiliary evidence. Runtime contact, pose, velocity, drop,
penetration and deterministic signatures are authoritative.

## Remaining boundaries

- `follower_right`: `NOT_RUN`; the approved Stage contains follower_left only.
- lift trajectory:
  `HARD_BLOCKER_NO_USER_APPROVED_SUPPLIER_STAGE_LIFT_TRAJECTORY`
- physical fingertip/bottle friction: not measured
- bottle shape and inertia: incomplete/uncalibrated
- production angular-controlled CAD tessellation: `HARD_BLOCKER`
- final/default collider: unchanged
- Task 8: `NOT_RUN`
