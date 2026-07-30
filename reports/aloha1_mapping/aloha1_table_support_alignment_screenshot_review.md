# ALOHA1 Table/Support Alignment Screenshot Review

- Status: `PASS`
- Diagnostic Stage: `/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/table_support_alignment/1.0/aloha1_table_support_aligned_workcell.usda`
- SHA-256: `2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c`
- Scope: `DIAGNOSTIC_ONLY_NOT_FINAL_ASSET`
- Isaac GUI workspace: `2` (X11 index `1`)
- User review: `PASS`

The images are supporting visual evidence. Runtime AABB stack measurements in the JSON validation report are authoritative.

## Captures

### aligned_overview

- Attempt: `ACCEPTED_FINAL_VISUAL_PASS`
- Vision review: `PASS`
- Raw: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260730-aloha-support-table-alignment/screenshots_raw/aligned_overview_raw.png`
- Annotated: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260730-aloha-support-table-alignment/screenshots_annotated_v2/aligned_overview_annotated.png`
- Goal: Both follower bases, lower support frame and tabletop visible in one view; old air gap absent.

### aligned_support_side_attempt1

- Attempt: `REJECTED_SUPPORT_INTERFACE_OCCLUDED`
- Vision review: `REJECTED`
- Raw: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260730-aloha-support-table-alignment/screenshots_raw/aligned_support_side_raw.png`
- Annotated: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260730-aloha-support-table-alignment/screenshots_annotated_v2/aligned_support_side_annotated.png`
- Goal: Strict side view of table/support plane; rejected because the front extrusion hides the critical interface.

### aligned_left_base_side

- Attempt: `ACCEPTED_FINAL_VISUAL_PASS`
- Vision review: `PASS`
- Raw: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260730-aloha-support-table-alignment/screenshots_raw/aligned_left_base_side_raw.png`
- Annotated: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260730-aloha-support-table-alignment/screenshots_annotated_v2/aligned_left_base_side_annotated.png`
- Goal: Left robot base, supporting extrusions and tabletop are all visible at the vertical stack interface.

### aligned_right_base_side

- Attempt: `ACCEPTED_FINAL_VISUAL_PASS`
- Vision review: `PASS`
- Raw: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260730-aloha-support-table-alignment/screenshots_raw/aligned_right_base_side_raw.png`
- Annotated: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260730-aloha-support-table-alignment/screenshots_annotated_v2/aligned_right_base_side_annotated.png`
- Goal: Right robot base, supporting extrusions and tabletop are all visible at the vertical stack interface.

## Retake history

- `REJECTED_OVERLAPPING_REGION_BOXES`: The first annotated batch placed large overlapping region boxes over the left/right base interfaces.

- `REJECTED_SUPPORT_INTERFACE_OCCLUDED`: The strict side camera put the front support extrusion between the camera and the table/support interface.
