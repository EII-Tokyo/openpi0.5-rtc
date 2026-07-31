# ALOHA1 Post-Grasp Task 7 Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. This
> repository run is explicitly inline-only; do not dispatch subagents.

**Goal:** Integrate the user-confirmed five-pose 20 cm grasp into Task 7
without suppressing NVIDIA rule failures or entering Task 8.

**Architecture:** Add a report-only acceptance layer that consumes frozen
machine-readable evidence. It verifies hashes and Stage composition, reports
runtime control, workcell physics, static hold and dynamic pickup separately,
and keeps asset-promotion readiness independent. It does not open or modify an
Isaac Stage.

**Tech Stack:** Python 3.11 project `.venv`, JSON, pytest, Ruff, existing
Isaac Sim 5.1 reports.

---

### Task 1: Add post-grasp classification tests

**Files:**

- Create: `tests/aloha1_mapping/test_task7_post_grasp_acceptance.py`
- Create: `tools/aloha1_mapping/task7_post_grasp_acceptance.py`

- [ ] Write a failing test asserting that runtime control, workcell physics,
  table alignment, static hold, five-pose dynamic pickup, visual review and
  user confirmation must all be `PASS`.
- [ ] Write a failing test asserting that asset-promotion `PARTIAL` makes the
  aggregate `PARTIAL`, while preserving `runtime_grasp_acceptance=PASS`.
- [ ] Write a failing test asserting that any dynamic-grasp `FAIL` makes the
  aggregate `FAIL`.
- [ ] Write a failing test asserting that `Task 8` remains `NOT_RUN`.
- [ ] Run:
  `.venv/bin/python -m pytest -q tests/aloha1_mapping/test_task7_post_grasp_acceptance.py`
  and confirm failure because the classifier does not yet exist.
- [ ] Implement pure classification functions with only
  `PASS/FAIL/PARTIAL/NOT_RUN` outputs.
- [ ] Re-run the test and require all cases to pass.

### Task 2: Build the frozen evidence report

**Files:**

- Create: `tools/build_aloha1_task7_post_grasp_acceptance.py`
- Modify: `tests/aloha1_mapping/test_task7_post_grasp_acceptance.py`
- Generate:
  `reports/aloha1_mapping/aloha1_task7_post_grasp_acceptance.json`
- Generate:
  `reports/aloha1_mapping/aloha1_task7_post_grasp_acceptance.md`

- [ ] Add tests requiring exact hashes for the Task 7A report, table-alignment
  report, static-hold report, five-pose grasp report, promotion-readiness
  report and official-rule applicability report.
- [ ] Require the aligned Stage SHA-256
  `2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c`
  and its source Stage SHA-256
  `d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf`.
- [ ] Require the aligned Stage to contain the source signal Stage as a
  sublayer and to declare table-translation-only with no robot, collider or
  physics-parameter modification.
- [ ] Require the five-pose report to contain five machine passes, five
  evidence passes, visual-model `PASS`, user `PASS`, and Task 8 `NOT_RUN`.
- [ ] Generate JSON and Markdown reports without changing any input report or
  USD.
- [ ] Recompute every input hash after generation and require equality.

### Task 3: Run Task 7 regression and document the result

**Files:**

- Modify: `README_ALOHA1_ISAACSIM_5_1.md`
- Modify: `.codex/TASK_STATE.md`

- [ ] Run focused post-grasp and existing Task 7 acceptance tests.
- [ ] Run all `tests/aloha1_mapping` with `.venv/bin/python -m pytest`.
- [ ] Run Ruff and `py_compile` for the new builder/classifier/tests.
- [ ] Verify the aligned Stage and source Stage hashes are unchanged.
- [ ] Record:
  `runtime_grasp_acceptance=PASS`,
  `asset_promotion_readiness=PARTIAL`,
  `official_rules_literal_status=FAIL`,
  `task7_aggregate=PARTIAL`, and `task8=NOT_RUN`.
- [ ] Inspect the bounded diff and preserve all unrelated dirty files.
