# ALOHA ViperX supplier-CAD bottle screenshot review

- Status: `PASS`
- Scope: `follower_left`, supplier assembly embedded v2 fingers
- Runtime hold: `20/20 PASS`, deterministic=`true`
- Maximum full-interval drop: `0.000453919172 m`
- Screenshot role: auxiliary visual evidence only
- Task 8: `NOT_RUN`

| Phase | Frame | Time s | Projected contacts | Raw | Annotated | Vision |
|---|---:|---:|---:|---|---|---|
| open | `209` | `3.483333` | `0` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/task5_bottle_acceptance_v3/screenshots_raw/open_raw.png` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/task5_bottle_acceptance_v3_annotation_attempt2/screenshots_annotated/open_annotated.png` | PASS |
| bilateral_contact | `389` | `6.483333` | `2` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/task5_bottle_acceptance_v3/screenshots_raw/bilateral_contact_raw.png` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/task5_bottle_acceptance_v3_annotation_attempt2/screenshots_annotated/bilateral_contact_annotated.png` | PASS |
| release | `389` | `6.483333` | `2` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/task5_bottle_acceptance_v3/screenshots_raw/release_raw.png` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/task5_bottle_acceptance_v3_annotation_attempt2/screenshots_annotated/release_annotated.png` | PASS |
| hold_end | `509` | `8.483333` | `2` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/task5_bottle_acceptance_v3/screenshots_raw/hold_end_raw.png` | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/isaac_cad_finger/task5_bottle_acceptance_v3_annotation_attempt2/screenshots_annotated/hold_end_annotated.png` | PASS |

The open image contains CAD-derived inward-surface samples, not physical contacts. Bilateral-contact, release, and hold-end images each contain two runtime-projected physical contact points and normals. Release and hold-end raw geometry is nearly unchanged because the bottle remained held; their distinct frame/time and machine trajectory are recorded in the annotated images and runtime report.

The screenshot PASS does not replace the contact, pose, velocity, drop, penetration, or deterministic runtime gates.
