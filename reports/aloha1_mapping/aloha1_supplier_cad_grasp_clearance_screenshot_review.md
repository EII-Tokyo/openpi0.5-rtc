# ALOHA1 Supplier-CAD Complete-Gripper Screenshot Review

Status: **PASS** for screenshot evidence quality and the static CAD geometry
gate. This is **not** an Isaac collision, IK, bilateral-contact, lift, or
static-hold PASS. Task 8 remains **NOT_RUN**.

User review: **PASS**, confirmed on 2026-07-30 with “确认，通过。继续。”
The confirmation applies only to this static CAD screenshot/grasp-frame gate.

## What the images verify

The final evidence uses the complete supplier gripper assembly, including the
embedded v2 handed pair:

- blue: `left_finger`, supplier CAD `+X` side;
- orange: `right_finger`, supplier CAD `-X` side;
- cyan: project `Bottle500`;
- dark transparent geometry: complete supplier gripper shell;
- red wireframe: conservative runtime URDF `gripper_bar` AABB.

The true-world-top camera looks along gripper `+X`, because the validated
vertical approach maps gripper `+X` to world `-Z`. The world-side camera looks
along gripper `-Y`, with image-up equal to gripper `-X` / world `+Z`. Each
rejected/corrected pair uses exactly the same orthographic camera.

The images make the key correction visible:

- the official EE helper is not the grasp center;
- run13 used bottle-axis center `x = 111.272 mm` and left/right q of
  `54.985/-63.742 mm`;
- its conservative hard clearance to the runtime bar envelope was only
  `2.547 mm`, and runtime data showed no bilateral physical finger contact;
- the static CAD candidate uses bottle-axis center `x = 132.155 mm`;
- the CAD pad-contact frame is `x = 135.521 mm`, accounting for the tilted
  supplier pad normal;
- left/right q are symmetric at `+48.317/-48.317 mm`;
- its max-min hard margin is `23.430 mm`.

These values come from the deterministic FreeCAD/OCCT geometry report
[`aloha1_supplier_cad_grasp_clearance.json`](./aloha1_supplier_cad_grasp_clearance.json).
They are not guessed visual offsets.

## Final screenshot locations

Final raw images:

- `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260730-aloha1-grasp-editor-five-position/full_gripper_cad_clearance/static_visual_evidence_attempt4/screenshots_raw/rejected_run13_true_world_top_raw.png`
- `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260730-aloha1-grasp-editor-five-position/full_gripper_cad_clearance/static_visual_evidence_attempt4/screenshots_raw/corrected_cad_true_world_top_raw.png`
- `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260730-aloha1-grasp-editor-five-position/full_gripper_cad_clearance/static_visual_evidence_attempt4/screenshots_raw/rejected_run13_world_side_raw.png`
- `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260730-aloha1-grasp-editor-five-position/full_gripper_cad_clearance/static_visual_evidence_attempt4/screenshots_raw/corrected_cad_world_side_raw.png`

Final annotated images:

- `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260730-aloha1-grasp-editor-five-position/full_gripper_cad_clearance/static_visual_evidence_attempt4/screenshots_annotated/rejected_run13_true_world_top_annotated.png`
- `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260730-aloha1-grasp-editor-five-position/full_gripper_cad_clearance/static_visual_evidence_attempt4/screenshots_annotated/corrected_cad_true_world_top_annotated.png`
- `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260730-aloha1-grasp-editor-five-position/full_gripper_cad_clearance/static_visual_evidence_attempt4/screenshots_annotated/rejected_run13_world_side_annotated.png`
- `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260730-aloha1-grasp-editor-five-position/full_gripper_cad_clearance/static_visual_evidence_attempt4/screenshots_annotated/corrected_cad_world_side_annotated.png`

Each raw image is 1600×1000. Each annotated image is 2260×1000. Absolute
paths, SHA-256 values, camera parameters, geometry signatures, and individual
vision-model conclusions are in the JSON report.

## Visual-model review

All eight final files were reviewed individually at original resolution:

- complete bottle and relevant gripper geometry are uncropped;
- the blue/orange handed mapping is readable in the true-world-top view;
- the supplier shell does not hide the critical relationship;
- the red bar envelope is an edge frame, not a solid overlay;
- rejected and corrected images are visibly different;
- labels and arrows do not obscure the inward finger regions;
- side-view left/right contacts honestly share one projected point (`LR`);
- green/red borders mean static geometry candidate PASS/rejected geometry
  FAIL, not physical grasp success.

## Retake history

1. The first Blender engine probe was rejected after the log showed a Python
   exception despite process exit code 0.
2. The first rendered batch was rejected for overexposure and insufficient
   finger visibility.
3. The next batch was rejected for bottle/proximal geometry cropping.
4. The third batch was rejected because the side views still clipped the
   proximal shell.
5. Attempt 4 raw images passed.
6. Attempt 4's first annotated set was rejected for long-label overlap.
7. The final attempt 4 annotations use short `EE/GF/RF/BC/LR` local labels and
   passed individual visual review.

The next allowed gate is user review, followed by direct NVIDIA official MCP
verification and isolated Isaac Sim 5.1 runtime integration. No Isaac Stage,
Grasp Editor configuration, IK target, collider, or final asset was modified
by this static screenshot step.
