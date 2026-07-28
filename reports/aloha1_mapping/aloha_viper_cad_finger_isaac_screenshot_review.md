# ALOHA ViperX supplier-CAD finger Isaac screenshot review

- Status: `PASS`
- Gate: `ISAAC_CAD_INSTALLATION_VISUAL_EVIDENCE_ONLY`
- Scope: CAD installation visual evidence only.
- Boundary: **NO collision/contact/grasp acceptance**.
- Approved source Stage: `/home/eii/project/openpi0.5-rtc-reward-learning/local_eval_assets/aloha_isaac_assets/aloha_viperx.usd`
- Approved source SHA-256: `b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e`
- Diagnostic Stage: `/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/cad_finger_installation_v2/aloha_viperx_cad_finger_diagnostic.usda`
- Diagnostic Stage SHA-256: `9f64f2ef6e280d3505c900a7b13e649331cf8bb227d910928647762ef4a5edc3`

## Capture review

| Capture | Verdict | Raw | Annotated |
|---|---:|---|---|
| `closed_base_oblique` | `PASS` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/v2_attempt5/screenshots_raw/closed_base_oblique_raw.png` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/v2_attempt5/annotation_v2/screenshots_annotated/closed_base_oblique_annotated.png` |
| `closed_tip_end` | `PASS` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/v2_attempt5/screenshots_raw/closed_tip_end_raw.png` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/v2_attempt5/annotation_v2/screenshots_annotated/closed_tip_end_annotated.png` |
| `closed_true_bottom` | `PASS` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/v2_attempt5/screenshots_raw/closed_true_bottom_raw.png` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/v2_attempt5/annotation_v2/screenshots_annotated/closed_true_bottom_annotated.png` |
| `closed_true_top` | `PASS` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/v2_attempt5/screenshots_raw/closed_true_top_raw.png` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/v2_attempt5/annotation_v2/screenshots_annotated/closed_true_top_annotated.png` |
| `open_base_oblique` | `PASS` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/v2_attempt5/screenshots_raw/open_base_oblique_raw.png` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/v2_attempt5/annotation_v2/screenshots_annotated/open_base_oblique_annotated.png` |
| `open_tip_end` | `PASS` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/v2_attempt5/screenshots_raw/open_tip_end_raw.png` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/v2_attempt5/annotation_v2/screenshots_annotated/open_tip_end_annotated.png` |
| `open_true_bottom` | `PASS` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/v2_attempt5/screenshots_raw/open_true_bottom_raw.png` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/v2_attempt5/annotation_v2/screenshots_annotated/open_true_bottom_annotated.png` |
| `open_true_top` | `PASS` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/v2_attempt5/screenshots_raw/open_true_top_raw.png` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/v2_attempt5/annotation_v2/screenshots_annotated/open_true_top_annotated.png` |

## Open/closed paired-camera gates

| View | Camera exact | Open gap > closed gap |
|---|---:|---:|
| `true_top` | `True` | `True` |
| `true_bottom` | `True` | `True` |
| `tip_end` | `True` | `True` |
| `base_oblique` | `True` | `True` |

## Retake history

- `historical_v1`: `REJECTED_WRONG_180_DEG_PALM_FLIP` — The prior CAD-to-link rotation inverted the palm-facing orientation.
- `v2_attempt1`: `REJECTED_CAMERA_PHYSX_USD_POSE_MISMATCH` — Camera targeting used runtime PhysX body pose while rendering disjoint authored USD body transforms, yielding blank evidence.
- `v2_attempt2`: `REJECTED_SIMPLE_Q_TRANSLATION_WITH_DISJOINT_LINK_FRAME` — Simple q translation did not compensate the approved Stage's authored link-frame offset; closed surface gap was about 92.5 mm.
- `v2_attempt3`: `REJECTED_CROPPING_AND_ARM_OCCLUSION` — Top/open geometry was cropped and the base-side view was obstructed by arm/shell visuals.
- `v2_attempt4`: `REJECTED_BASE_OBLIQUE_GRIPPER_SHELL_OCCLUSION` — Six views passed, but the gripper-shell visual still hid the base-side finger details.
- `v2_attempt5_raw`: `PASS` — All eight raw captures were reviewed individually; base-oblique hides only the shell visual in the anonymous capture session.
- `annotation_v1`: `REJECTED_PANEL_HASH_TRUNCATION` — The diagnostic Stage hash was truncated in the annotation panel.
- `annotation_v2`: `PASS` — All eight annotated captures were reviewed individually with full hashes, non-overlapping labels, and explicit visual-only scope.

## Acceptance boundary

This PASS proves only that the isolated supplier-CAD visual installation is consistently presented from four paired views. It does not validate collider geometry, contact, dynamics, or bottle grasping. Task 8 remains `NOT_RUN`.
