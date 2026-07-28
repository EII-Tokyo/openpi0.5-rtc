# ALOHA Viper Gripper Screenshot Review

- Status: `PASS`
- Gate: `CAD_INSTALLATION_VISUAL_EVIDENCE_ONLY`
- User confirmation: `AWAITING_USER_VISUAL_CONFIRMATION`
- Isaac runtime/contact/hold: `NOT_RUN`
- Final/default asset modified: `false`
- Source: `Simple Aloha Viper 2024-5-13.step`
- Source SHA-256: `337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571`

Every accepted raw and annotated image below was inspected individually with the vision model. File existence, color bounds, or hashes alone were not accepted.

| State | View | Vision review | Raw | Annotated |
|---|---|---:|---|---|
| closed | true_top | PASS | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt5_candidate/screenshots_raw/closed_true_top_raw.png` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt5_candidate/screenshots_annotated_v3/closed_true_top_annotated.png` |
| open | true_top | PASS | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt5_candidate/screenshots_raw/open_true_top_raw.png` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt5_candidate/screenshots_annotated_v3/open_true_top_annotated.png` |
| closed | true_bottom | PASS | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt5_candidate/screenshots_raw/closed_true_bottom_raw.png` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt5_candidate/screenshots_annotated_v3/closed_true_bottom_annotated.png` |
| open | true_bottom | PASS | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt5_candidate/screenshots_raw/open_true_bottom_raw.png` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt5_candidate/screenshots_annotated_v3/open_true_bottom_annotated.png` |
| closed | tip_end | PASS | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt5_candidate/screenshots_raw/closed_tip_end_raw.png` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt5_candidate/screenshots_annotated_v3/closed_tip_end_annotated.png` |
| open | tip_end | PASS | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt5_candidate/screenshots_raw/open_tip_end_raw.png` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt5_candidate/screenshots_annotated_v3/open_tip_end_annotated.png` |
| closed | base_oblique | PASS | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt5_candidate/screenshots_raw/closed_base_oblique_raw.png` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt5_candidate/screenshots_annotated_v3/closed_base_oblique_annotated.png` |
| open | base_oblique | PASS | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt5_candidate/screenshots_raw/open_base_oblique_raw.png` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt5_candidate/screenshots_annotated_v3/open_base_oblique_annotated.png` |

## Interpretation

- Blue is the embedded CAD +X handed B-Rep mapped to `left_finger`.
- Orange is the embedded CAD -X handed B-Rep mapped to `right_finger`.
- The magenta samples are CAD-derived annotation points, not physical contact points.
- `tip_end` is the strongest evidence that both recessed inner surfaces face the gripper center.
- Open/closed pairs share identical orthographic camera metadata and differ in image content and B-Rep minimum distance.
- A fresh repeat render reproduced all 8 raw pixel hashes.

## Retakes

- `attempt_2`: `REJECTED_INSUFFICIENT_ILLUMINATION` — Eevee output was too dark to inspect the inner surfaces; file existence and nonzero pixels were not accepted. Path: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/screenshots_raw`
- `attempt_3`: `REJECTED_INSUFFICIENT_SURFACE_VISIBILITY` — Correct mm-to-m rendering units did not make Eevee surface details sufficiently visible for the CAD orientation gate. Path: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt3_metric_scene/screenshots_raw`
- `attempt_4`: `REJECTED_CROPPING_AND_BASE_END_OCCLUSION` — Workbench fixed visibility, but finger tips were cropped, open tip-end fingers touched the frame edge, and the pure base-end view hid both fingers behind the gripper shell. Path: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt4_workbench/screenshots_raw`
- `attempt_5_raw`: `ACCEPTED_AFTER_PER_IMAGE_VISION_REVIEW` — Eight raw images passed individual visual inspection after increasing the paired frame margin and replacing the occluded base-end view with a proven base oblique. Path: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt5_candidate/screenshots_raw`
- `attempt_5_annotated_v1`: `REJECTED_LABEL_OVERLAP` — Long local left/right labels overlapped in the closed true-top and closed tip-end evidence. Path: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt5_candidate/screenshots_annotated`
- `attempt_5_annotated_v2`: `REJECTED_LABEL_OVERLAP` — Short L/R tags fixed the first overlap, but the closed base-oblique distance label still covered the R tag. Path: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt5_candidate/screenshots_annotated_v2`
- `attempt_5_annotated_v3`: `ACCEPTED_AFTER_PER_IMAGE_VISION_REVIEW` — All eight annotations passed individual visual inspection; local labels are short L/R tags and edge-near measurement labels use collision-free placement. Path: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/attempt5_candidate/screenshots_annotated_v3`

This PASS does not claim collider, contact, grasp, hold, or Isaac runtime correctness.
