# RL Training Foundation Worklog - 2026-07-19

## Context
- User clarified replay reproduction is only a test method; the real target is future Isaac Sim RL training.
- Main direction: extract RL-ready reset/action-step/metric/reward gates from existing drive-target validation path.

## Issues observed
- Prior summarized worklog path aloha_isaac_replay/adaptive_tasks/replay_quality_phase_20260719_062354/WORKLOG.md is not present in the current filesystem.
- A broad rg source search produced excessive output before switching to codex-evidence.

## Changes
- Added aloha_isaac_replay/rl/drive_target_env.py for drive-target target conversion and reward-readiness metrics.
- Added aloha_isaac_replay/scripts/run_rl_drive_target_smoke.py and scripts/run_aloha_isaac_rl_drive_target_smoke.py.

## Verified
- python -m py_compile passed for new modules.
- pytest -q aloha_isaac_replay/tests/test_drive_target_env.py passed.

## Isaac smoke
- Command artifact: .codex/artifacts/20260719-062945_aloha-rl-drive-target-smoke-8frames
- Output JSON: reports/aloha_isaac_replay/rl_drive_target_smoke/smoke_8frames_20260719.json
- Status: PASS for 8-frame drive-target RL step smoke.
- 40-frame drive-target RL smoke PASS.
  - Artifact: .codex/artifacts/20260719-063013_aloha-rl-drive-target-smoke-40frames
  - JSON: reports/aloha_isaac_replay/rl_drive_target_smoke/smoke_40frames_20260719.json
- Added aloha_isaac_replay/rl/readiness.py so drive-target replay PASS cannot be mistaken for full RL training readiness.
- Added readiness report to prevent drive-target replay PASS from being treated as full RL-training readiness.
  - Artifact: .codex/artifacts/20260719-063227_aloha-rl-drive-target-smoke-readiness
  - JSON: reports/aloha_isaac_replay/rl_drive_target_smoke/smoke_readiness_8frames_20260719.json
- Added optional causality probe: same reset plus two different drive targets must produce measurably different next states.
- Causality probe smoke PASS.
  - Artifact: .codex/artifacts/20260719-063324_aloha-rl-drive-target-causality-smoke
  - JSON: reports/aloha_isaac_replay/rl_drive_target_smoke/smoke_causality_8frames_20260719.json
- Cleanup note: rm -rf of aloha_isaac_replay/rl/__pycache__ was rejected by local safety policy; cleaned with bounded find -delete for '*.pyc' followed by rmdir.

## 2026-07-19 Phase135 link

The bottle-grasp replay line now has a stronger pre-RL gate:

```text
reports/aloha1_isaac_adaptation/phase135_active_tabletop_policy_bottle_visual_cylinder_proxy_20260719/gripper_passive_contact_metrics.json
```

This validates:

- visible BottleUSD body and semantic frames;
- single enabled cylinder `physics_proxy` for target contact;
- left gripper closes from non-contact to bilateral target contact;
- table support is explicitly mapped through `phase132_active_tabletop_contact_policy.yaml`;
- rail/unknown workcell contact remains denied by policy default.

This is still not `READY_FOR_RL_TRAINING`. It is the accepted prerequisite for the future environment reset/contact/reward implementation. The next RL work should connect this validated object/table/gripper setup to reset, step, action, observation, reward, termination, and no-future-label leakage gates.
