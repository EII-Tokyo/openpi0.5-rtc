# Phase104 Grasp-YAML BottleUSD Contact Failure

## Purpose

Phase104 tested whether the existing Bottle500 GraspSpec can place the real Bottle500 USD relative to the current ALOHA1 left gripper frame and then pass the active-contact gate.

This is the first gate that stops using `gap_center` for a bottle-shaped object. The bottle pose is computed from:

```text
T_world_object = T_world_gripper * inverse(T_object_gripper)
```

where `T_object_gripper` comes from `assets/bottle_500ml/grasp/bottle_aloha_left_grasps.yaml`.

## Command

```bash
codex-evidence --name aloha-phase104-grasp-yaml-bottleusd-active-probe -- \
  .venv_issac/bin/python aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py \
  --stage-usd local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/aloha2_menagerie_scene_deep_black_real_start_pose_proxy_runtime.usda \
  --stage-units-in-meters 1.0 \
  --contact-proxy-profile scene_base_link \
  --output-dir reports/aloha1_isaac_adaptation/phase104_grasp_yaml_bottleusd_active_probe_20260719 \
  --side left \
  --settle-steps 20 \
  --physics-dt 0.02 \
  --gravity 0.0 \
  --finger-kp 200 \
  --finger-kd 50 \
  --open-offset 0.014 \
  --close-offset -0.014 \
  --right-finger-close-sign 1.0 \
  --limit-margin 0.001 \
  --object-placement grasp_yaml \
  --object-creation raw_usd \
  --object-shape bottle_usd \
  --object-usd assets/bottle_500ml/isaac/bottle_500ml_sim.usd \
  --object-usd-prim-path /Bottle500 \
  --object-grasp-yaml assets/bottle_500ml/grasp/bottle_aloha_left_grasps.yaml \
  --object-grasp-name grasp_mid \
  --object-gripper-frame /scene/left_base_link/left_gripper_link \
  --object-contact-offset 0.001 \
  --object-rest-offset 0.0 \
  --proxy-contact-offset 0.001 \
  --proxy-rest-offset 0.0 \
  --support-plane-mode none \
  --closure-profile linear \
  --moving-fingers both \
  --trace-contact-pairs \
  --require-active-target-contact \
  --min-contact-motion 1e-05 \
  --max-object-displacement 1.0
```

## Verified Run

Artifact:

`.codex/artifacts/20260719-003953_aloha-phase104-grasp-yaml-bottleusd-active-probe`

Report:

`reports/aloha1_isaac_adaptation/phase104_grasp_yaml_bottleusd_active_probe_20260719/gripper_passive_contact_metrics.json`

Observed result:

| Field | Observed value |
| --- | --- |
| validator status | `FAILED_GATE` |
| failure reasons | `contact_trace_gate_failed`, `active_target_contact_gate_failed` |
| active target contact | `FAIL_NO_ACTIVE_TARGET_CONTACT_DURING_CLOSE` |
| left finger target contact | `false` |
| right finger target contact | `false` |
| object displacement | `2.018797213858771e-05 m` |
| non-target categories | `same_side_robot_non_target` |

## Interpretation

The failure is useful and should not be bypassed.

The computed Bottle500 pose caused the bottle collision bodies to contact the gripper base/bar/prop, not the finger proxies. The contact trace contains many pairs like:

```text
/World/phase43_passive_contact_cube/Collisions/COL_Body_*
with
/scene/left_base_link/left_gripper_base/collisions/vx300s_7_gripper_bar/...
```

No target finger proxy contact was reported.

This means the existing GraspSpec is not yet a verified physical grasp for the current ALOHA1 proxy-runtime scene. It is a frame-transform candidate only.

## Why This Matters

Phase103 proved the bilateral finger contact gate is reachable with a controlled proxy cube. Phase104 shows that the real Bottle500 placement problem is now the blocker.

The current bottleneck is not "PhysX cannot report contacts." The current bottleneck is:

1. the gripper frame used by the GraspSpec may not represent the actual finger closing frame;
2. the Bottle500 CAD collision frame may not match the intended grasp frame;
3. the current side-grasp offset may put the bottle body into the gripper base rather than between fingers;
4. the finger qpos convention used by the GraspSpec does not yet match the scene-base proxy gate.

## Next Gate

Do not tune around this by disabling gripper-base collisions.

The next gate should generate grasp candidates from measured or simulated finger surfaces:

1. read the left/right finger proxy bboxes at open pose;
2. compute a bottle axis and radius placement that puts the bottle body between the two finger surfaces;
3. verify no gripper-base/wrist contact during settle;
4. verify first target finger contact appears during close;
5. only then move back to full Bottle500 under gravity and support-plane constraints.
