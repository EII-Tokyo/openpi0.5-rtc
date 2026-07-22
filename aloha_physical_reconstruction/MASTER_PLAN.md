# ALOHA Physical Reconstruction Master Plan

This workspace reconstructs the real ALOHA bottle-to-pipe task in Isaac Sim
one confirmed step at a time.

## Current Gate

- Current step: Step 0 - baseline audit.
- Current status: waiting for human confirmation.
- Next step is blocked until the user replies: `确认第 0 步通过`.

## Rules

- Do not modify original ALOHA USD assets.
- Do not modify original bottle assets.
- Do not start RL, RLT, PPO, VLA inference, sim-to-real training, or real robot control.
- Record physical parameters in `configs/physical_reconstruction/parameter_registry.yaml`.
- Only proceed to the next step after explicit user confirmation.

## Step Outputs

- Step 0 report: `reports/step_00_baseline_audit.md`
- Step 0 stage overview image: `artifacts/screenshots/step_00_stage_overview.png`
- Step 0 bottle overview image: `artifacts/screenshots/step_00_bottle_current.png`
