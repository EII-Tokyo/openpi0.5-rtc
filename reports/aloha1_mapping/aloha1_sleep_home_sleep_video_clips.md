# ALOHA1 Sleep → Home → Sleep review clips

- Status: `PASS_VISUAL_EVIDENCE_CLIP`
- Sequence: first Sleep hold → return to Home → second Sleep hold.
- Source capture frames: `91–286` inclusive.
- Source physics interval: `6.0166666667–18.95 s`.
- Output: `196` frames, `15 fps`, `13.066667 s`, `960×540`.
- Isaac was not rerun and the two source videos were not modified.

## Clips

- Normal:
  `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260803-aloha1-home-sleep-digital-twin/sleep_home_sleep_review_clips/aloha1_sleep_home_sleep_normal.mp4`
- Red full-arm collision overlay:
  `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260803-aloha1-home-sleep-digital-twin/sleep_home_sleep_review_clips/aloha1_sleep_home_sleep_collision_overlay.mp4`

The visual model reviewed initial Sleep, intermediate Home and final Sleep
frames from both clips. The complete arm is visible, the three states are
distinguishable and the red collision overlay is readable.

This is a visual-evidence PASS only. The official Sleep target remains outside
three frozen USD joint limits, so the digital motion gate remains `FAIL` and no
real-robot motion is authorized.
