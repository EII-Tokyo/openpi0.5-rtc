# ALOHA1 Task 8 Lightweight Optimization Plan

> **Resumed by explicit user authorization on 2026-08-03:** the intervening
> model-first reports remain valid diagnostic evidence, but incomplete physical
> calibration and the rejected finite contact-patch candidate are non-blocking
> reminders rather than Task 8 entry gates. Approximate simulation is allowed
> with explicit provenance and limitations. Final/default promotion remains a
> separate review decision.

**Goal:** produce a measured, isolated optimization candidate without
rewriting Task 7 history or changing final/default assets.

## 1. Record and freeze

- Add the user authorization/status boundary to README and task state.
- Recompute the baseline Stage and finger-limit-layer hashes.
- Record defaultPrim, root, sublayers, references, payloads and required prims.
- Fail closed if a frozen input changed.

## 2. Verify local APIs

- Record direct NVIDIA MCP queries used for optimization guidance.
- Treat local Isaac Sim 5.1 / Kit 107.3.3 source as version authority.
- Confirm `isaacsim.benchmark.services` API and the exact runtime settings read
  back by each benchmark process.

## 3. Baseline inventory and benchmark

- Add a tested static USD inventory/diff tool.
- Count geometry, materials, composition arcs, physics/collision schemas and
  dependencies.
- Run a fresh-process baseline load/fixed-frame benchmark and save bounded
  stdout/stderr plus JSON metrics.

## 4. Select and build one candidate

- Rank only inventory-backed opportunities.
- Start with a visual/composition-only candidate.
- Keep collision, mass, inertia, joints, drives, mimic, limits, timestep,
  solver and control parameters unchanged.
- Save source/candidate hashes and a structured USD diff.

## 5. Compare and smoke-test

- Re-run the same benchmark for baseline and candidate.
- Verify articulation/DOF/control/physics signatures.
- Run one open/close and, if composition changed, one short Bottle500
  grasp/lift/hold smoke.
- Do not rerun the five accepted MP4s unless the candidate changes physical
  behavior or the lightweight smoke exposes a regression.

## 6. Failure evidence gate

- For every reproducible failure, capture before/first-anomaly/final-failure
  raw and annotated screenshots.
- Record a full-arm collision-enabled video for every reproducible failure,
  including render-only failures.
- Visually review every accepted failure artifact and record absolute paths,
  hashes, camera and telemetry.
- Explain the mathematical/physical cause when proven; otherwise use
  `INCONCLUSIVE` and record the next discriminating experiment.

## 7. Close and commit

- Generate JSON and Markdown Task 8 comparison reports.
- Update README and task state without overwriting historical sections.
- Run focused pytest, task-owned Ruff and py_compile.
- Inspect diffs and commit logical batches without pushing.
- Leave candidate promotion as an explicit later user decision.
