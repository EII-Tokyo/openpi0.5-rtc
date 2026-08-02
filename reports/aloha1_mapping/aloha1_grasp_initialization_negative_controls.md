# ALOHA1 grasp initialization negative controls

- Status: `PASS`
- Fresh processes: `True`
- Task 8: `NOT_RUN`

| Scenario | Result | Expected | Observed | Annotated evidence |
|---|---|---|---|---|
| STATIC_LOAD_WITHOUT_RESET | EXPECTED_FAIL_OBSERVED | FAIL_INITIALIZATION_CONTRACT | FAIL_INITIALIZATION_CONTRACT,FINGER_LIMIT_VIOLATION,FINGER_PAIR_OVERLAP | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260802-aloha1-five-pose-finger-safety/negative_controls_attempt3/visual_retake_003/static_load_without_reset/screenshots_annotated/failure_or_control_annotated.png` |
| ILLEGAL_Q_ZERO | EXPECTED_FAIL_OBSERVED | FINGER_PAIR_OVERLAP | FINGER_LIMIT_VIOLATION,FINGER_PAIR_OVERLAP | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260802-aloha1-five-pose-finger-safety/negative_controls_attempt3/visual_retake_003/illegal_q_zero/screenshots_annotated/failure_or_control_annotated.png` |
| LEGAL_OPEN_CLOSE_SWEEP | PASS | None |  | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260802-aloha1-five-pose-finger-safety/negative_controls_attempt3/visual_retake_003/legal_open_close_sweep/screenshots_annotated/failure_or_control_annotated.png` |
| SAMPLE_02_ENVIRONMENT_INTERFERENCE | EXPECTED_FAIL_OBSERVED | FINGER_LIMIT_VIOLATION | FINGER_LIMIT_VIOLATION | `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260802-aloha1-five-pose-finger-safety/negative_controls_attempt3/visual_retake_003/sample_02_environment_interference/screenshots_annotated/failure_or_control_annotated.png` |

Expected failures count as control PASS only when the exact machine failure code is observed. Screenshots are auxiliary evidence; qpos, overlap, contacts, Stage hash, and runtime telemetry are authoritative.
