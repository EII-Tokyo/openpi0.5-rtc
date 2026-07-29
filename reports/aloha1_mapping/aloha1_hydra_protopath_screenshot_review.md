# ALOHA 1 Hydra protoPath screenshot review

- Status: **PASS**
- Review method: `VISION_MODEL_INDIVIDUAL_IMAGE_REVIEW`
- Accepted screenshots reviewed individually: `49/49`
- Boundary: screenshot review is auxiliary; runtime error counts and mesh inventories are authoritative.

## Accepted evidence

| Variant | Accepted images | Vision result |
|---|---:|---|
| A | 6 | PASS |
| B | 6 | PASS |
| C1 | 6 | PASS |
| C2 | 6 | PASS |
| C3_RESUME1 | 6 | PASS |
| C4 | 6 | PASS |
| B_REPEAT | 6 | PASS |
| RESTORE | 6 | PASS |
| D_RETAKE8 | 1 | PASS |

## Variant D retake history

| Attempt | Status | Reason |
|---|---|---|
| D | REJECTED_VIEW_TOO_DISTANT | both followers were too small to inspect link-mesh completeness |
| D_RETAKE1 | REJECTED_CAMERA_NOT_ACTIVE | camera change did not affect the active viewport |
| D_RETAKE2 | REJECTED_CAMERA_NOT_ACTIVE | focal-length change did not affect the active viewport |
| D_RETAKE3 | REJECTED_CAPTURE_NOT_CREATED | local Sdf import error prevented capture |
| D_RETAKE4 | REJECTED_CAMERA_NOT_ACTIVE | viewport remained on the default distant camera |
| D_RETAKE5 | REJECTED_WRONG_TARGET | active camera exposed only a close workcell-frame segment |
| D_RETAKE6 | REJECTED_CROPPED | follower base and distal links were cropped |
| D_RETAKE7 | REJECTED_OCCLUDED_AND_CROPPED | rack occluded the gripper and the base was cropped |
| D_RETAKE8 | PASS | session-only environment visibility isolation exposes both complete materialized followers |

Machine report: `/home/eii/project/openpi0.5-rtc-reward-learning/reports/aloha1_mapping/aloha1_hydra_protopath_screenshot_review.json`
Accepted D close view: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha1-signal-correspondence/hydra_protopath_diagnosis/D_RETAKE8/native_raw.png`
