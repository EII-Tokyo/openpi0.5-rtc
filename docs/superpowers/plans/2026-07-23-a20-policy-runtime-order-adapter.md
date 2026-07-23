# A20 Policy-to-Runtime Order Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make deterministic left/right-interleaved PhysX 16-DOF order compatible with the established left-block/right-block ALOHA/OpenPI 14D contract through an explicit, fail-closed adapter.

**Architecture:** A pure mapping module owns the 14D/16D contract and affine gripper expansion. Layer 1 embeds the trusted policy contract derived from the hash-bound A17 mapping; Layer 2 preserves raw runtime records, joins them by unique path, validates deterministic order and semantic metadata, and emits a versioned adapter object. No Isaac action, physics step, USD write, or real-robot operation is added.

**Tech Stack:** Python 3.11, pytest, YAML/JSON mapping artifacts, existing Isaac Sim 5.1 no-step probe, Ruff.

---

### Task 1: Commit the recovered determinism-provenance fix

**Files:**
- Modify: `aloha_isaac_rebuild/scripts/run_a20_runtime_articulation_discovery.py:497`
- Test: `aloha_isaac_rebuild/tests/test_a20_runtime_discovery_aggregation.py:1325`

- [ ] **Step 1: Verify the recovered diff is limited to the four reviewed fields**

Confirm the runner adds `probe_returncode`, `git_head`, and `git_dirty`, while the test also mutates `safety_checker_sha256`.

Run:

```bash
git diff -- aloha_isaac_rebuild/scripts/run_a20_runtime_articulation_discovery.py aloha_isaac_rebuild/tests/test_a20_runtime_discovery_aggregation.py
```

Expected: one runtime tuple change, one provenance tuple change, and one parameterized regression test.

- [ ] **Step 2: Run the recovered focused tests**

Run:

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q aloha_isaac_rebuild/tests/test_a20_runtime_discovery_aggregation.py
```

Expected: `199 passed`.

- [ ] **Step 3: Commit only the recovered files**

```bash
git add -- aloha_isaac_rebuild/scripts/run_a20_runtime_articulation_discovery.py aloha_isaac_rebuild/tests/test_a20_runtime_discovery_aggregation.py
git commit -m "fix: bind A20 determinism to probe provenance"
```

### Task 2: Add the pure 14D/16D mapping contract

**Files:**
- Create: `aloha_isaac_rebuild/scripts/a20_policy_runtime_order_adapter.py`
- Create: `aloha_isaac_rebuild/tests/test_a20_policy_runtime_order_adapter.py`

- [ ] **Step 1: Write failing tests for contract extraction and interleaved indices**

Create tests that load `a17_clean_articulation_mapping_plan.json`, build a raw runtime record list in this path order:

```python
INTERLEAVED_PATHS = [
    "/aloha/joints/left_waist", "/aloha/joints/right_waist",
    "/aloha/joints/left_shoulder", "/aloha/joints/right_shoulder",
    "/aloha/joints/left_elbow", "/aloha/joints/right_elbow",
    "/aloha/joints/left_forearm_roll", "/aloha/joints/right_forearm_roll",
    "/aloha/joints/left_wrist_angle", "/aloha/joints/right_wrist_angle",
    "/aloha/joints/left_wrist_rotate", "/aloha/joints/right_wrist_rotate",
    "/aloha/joints/left_left_finger", "/aloha/joints/right_left_finger",
    "/aloha/joints/left_right_finger", "/aloha/joints/right_right_finger",
]
```

Assert that `build_order_adapter(build_policy_contract(mapping), runtime_records)` returns:

```python
assert adapter["schema_version"] == "a20-policy-runtime-order-v1"
assert adapter["runtime_order"] == INTERLEAVED_PATHS
assert adapter["canonical_to_runtime_indices"] == [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15]
assert len(adapter["policy_to_runtime"]) == 14
assert adapter["policy_to_runtime"][6]["runtime_indices"] == [12, 14]
assert adapter["policy_to_runtime"][13]["runtime_indices"] == [13, 15]
```

- [ ] **Step 2: Run the new tests and verify RED**

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q aloha_isaac_rebuild/tests/test_a20_policy_runtime_order_adapter.py
```

Expected: collection fails because `a20_policy_runtime_order_adapter` does not exist.

- [ ] **Step 3: Implement minimal contract construction**

Implement these public functions:

```python
SCHEMA_VERSION = "a20-policy-runtime-order-v1"

def build_policy_contract(mapping: dict[str, object]) -> dict[str, object]: ...

def build_order_adapter(
    policy_contract: dict[str, object], runtime_records: list[dict[str, object]]
) -> dict[str, object]: ...
```

`build_policy_contract` must extract the 16 canonical DOFs and their complete `canonical_mapping` records from `joint_records`, require OpenPI indices exactly `0..13`, require cardinality 1 for arm indices and 2 for indices 6 and 13, and reject missing/duplicate/non-finite `sign`, `offset`, or `scale` values. `build_order_adapter` must consume the already validated policy contract and join canonical and runtime records by unique clean path without sorting the runtime records.

- [ ] **Step 4: Run the contract tests and verify GREEN**

Run the Task 2 test command. Expected: contract/index tests pass.

- [ ] **Step 5: Add fail-closed inventory tests**

Add parameterized tests for duplicate runtime path, missing path, unexpected path, duplicate runtime index, missing OpenPI index, wrong arm/gripper cardinality, and non-finite affine value. Each must assert `ValueError` with a stable diagnostic substring.

- [ ] **Step 6: Run the adapter tests**

Expected: all adapter tests pass with no warnings.

- [ ] **Step 7: Commit the pure contract**

```bash
git add -- aloha_isaac_rebuild/scripts/a20_policy_runtime_order_adapter.py aloha_isaac_rebuild/tests/test_a20_policy_runtime_order_adapter.py
git commit -m "feat: add ALOHA policy runtime order adapter"
```

### Task 3: Add pure policy/runtime value conversion and round-trip checks

**Files:**
- Modify: `aloha_isaac_rebuild/scripts/a20_policy_runtime_order_adapter.py`
- Modify: `aloha_isaac_rebuild/tests/test_a20_policy_runtime_order_adapter.py`

- [ ] **Step 1: Write failing 14D-to-16D tests**

Add tests for:

```python
runtime = policy_to_runtime([0.0] * 14, adapter)
assert len(runtime) == 16
assert runtime[adapter["policy_to_runtime"][6]["runtime_indices"][0]] == pytest.approx(0.021)
assert runtime[adapter["policy_to_runtime"][6]["runtime_indices"][1]] == pytest.approx(-0.021)
```

Also cover normalized gripper values `0.0`, `0.5`, and `1.0`, both hands, non-finite policy values, and wrong vector length.

- [ ] **Step 2: Verify RED**

Run the adapter test file. Expected: failure because `policy_to_runtime` is missing.

- [ ] **Step 3: Implement policy expansion**

Implement:

```python
def policy_to_runtime(
    policy_values: list[float], adapter: dict[str, object]
) -> list[float]: ...
```

Use `runtime_value = offset + scale * policy_value` for every mapped runtime DOF and write results at the recorded raw runtime indices.

- [ ] **Step 4: Verify GREEN, then write failing inverse tests**

Add tests that call `runtime_to_policy`, verify exact arm values, and verify both finger-derived values agree for each gripper. Deliberately perturb one finger and require a stable `inconsistent gripper readback` failure.

- [ ] **Step 5: Implement inverse conversion and self-check**

Implement:

```python
def runtime_to_policy(
    runtime_values: list[float], adapter: dict[str, object], *, tolerance: float = 1e-6
) -> list[float]: ...

def round_trip_check(adapter: dict[str, object]) -> dict[str, object]: ...
```

Invert each affine transform with `(runtime_value - offset) / scale`; require all values for one policy index to agree within tolerance. `round_trip_check` must test finite 14D samples with gripper values `0.0`, `0.5`, and `1.0` and return structured PASS/FAIL evidence.

- [ ] **Step 6: Run tests and commit**

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q aloha_isaac_rebuild/tests/test_a20_policy_runtime_order_adapter.py
git add -- aloha_isaac_rebuild/scripts/a20_policy_runtime_order_adapter.py aloha_isaac_rebuild/tests/test_a20_policy_runtime_order_adapter.py
git commit -m "feat: validate ALOHA policy runtime round trips"
```

### Task 4: Bind the trusted policy contract into Layer 1

**Files:**
- Modify: `aloha_isaac_rebuild/scripts/audit_a20_usd_dof_metadata.py`
- Modify: `aloha_isaac_rebuild/tests/test_a20_usd_dof_metadata.py`

- [ ] **Step 1: Write a failing real-artifact Layer 1 assertion**

Extend `test_collect_real_a17_a19_metadata_matches_exactly`:

```python
contract = result["policy_contract"]
assert contract["schema_version"] == "a20-policy-runtime-order-v1"
assert contract["policy_dimension"] == 14
assert contract["runtime_dimension"] == 16
assert [entry["openpi_index"] for entry in contract["policy_entries"]] == list(range(14))
```

- [ ] **Step 2: Verify RED**

Run `test_collect_real_a17_a19_metadata_matches_exactly`. Expected: missing `policy_contract`.

- [ ] **Step 3: Add policy contract collection**

Call `build_policy_contract(mapping)` inside `_collect` and include the result in Layer 1 evidence. Include an empty or invalid contract in structured collection failures so downstream validation fails closed.

- [ ] **Step 4: Extend `_layer1_errors` and tests**

Require the exact schema, dimensions, indices, cardinalities, finite transforms, and consistency with the 16 expected joint paths. Add mutation tests for missing and malformed policy contracts.

- [ ] **Step 5: Run Layer 1 and coordinator tests, then commit**

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q aloha_isaac_rebuild/tests/test_a20_usd_dof_metadata.py aloha_isaac_rebuild/tests/test_a20_runtime_discovery_aggregation.py
git add -- aloha_isaac_rebuild/scripts/audit_a20_usd_dof_metadata.py aloha_isaac_rebuild/tests/test_a20_usd_dof_metadata.py aloha_isaac_rebuild/scripts/run_a20_runtime_articulation_discovery.py aloha_isaac_rebuild/tests/test_a20_runtime_discovery_aggregation.py
git commit -m "feat: bind A20 Layer 1 to the ALOHA policy contract"
```

### Task 5: Replace raw-order equality with deterministic semantic mapping

**Files:**
- Modify: `aloha_isaac_rebuild/scripts/run_a20_runtime_articulation_discovery.py`
- Modify: `aloha_isaac_rebuild/tests/test_a20_runtime_discovery_aggregation.py`

- [ ] **Step 1: Replace the old reverse-order expectation with a failing interleaved PASS test**

Construct three identical interleaved runs from the canonical fixture and assert:

```python
result = aggregate_runtime_runs(layer1, runs)
assert result["status"] == "PASS_A20_RUNTIME_ARTICULATION_DISCOVERY_NO_STEP"
assert result["order_adapter"]["mapping_complete"] is True
assert result["order_adapter"]["round_trip_check"]["status"] == "PASS"
assert result["raw_order_matches_canonical"] is False
```

- [ ] **Step 2: Verify RED for the intended reason**

Expected: current aggregation returns `runtime_records_mismatch` because it compares by array position.

- [ ] **Step 3: Implement path-aligned semantic comparison**

Keep the raw records untouched. Build the adapter from Layer 1's trusted policy contract and the first run. For each run:

- require its raw runtime fingerprint to equal the other runs;
- resolve records by unique path;
- compare name/type/axis/limits/body metadata against the corresponding canonical record;
- preserve raw `index` as runtime evidence rather than comparing it with canonical index;
- emit the versioned `order_adapter` and `raw_order_matches_canonical` information.

Replace positional `runtime_records_mismatch` with stable error codes for inventory, semantic metadata, runtime determinism, and adapter validation.

- [ ] **Step 4: Add fail-closed regression tests**

Cover one-run order changes, same paths with wrong type/limit/body/source metadata, duplicate paths, adapter corruption, unsafe flags, and stale provenance. All must remain FAIL.

- [ ] **Step 5: Update `is_exact_runtime_pass`**

Require a complete adapter, PASS round trip, deterministic raw facts, reaggregation to the same adapter, and live trusted Layer 1 inputs. Do not require raw order equality with canonical order.

- [ ] **Step 6: Run coordinator tests and commit**

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q aloha_isaac_rebuild/tests/test_a20_runtime_discovery_aggregation.py
git add -- aloha_isaac_rebuild/scripts/run_a20_runtime_articulation_discovery.py aloha_isaac_rebuild/tests/test_a20_runtime_discovery_aggregation.py
git commit -m "fix: validate A20 runtime order through explicit mapping"
```

### Task 6: Update bounded report semantics and design documentation

**Files:**
- Modify: `aloha_isaac_rebuild/scripts/run_a20_runtime_articulation_discovery.py`
- Modify: `aloha_isaac_rebuild/tests/test_a20_runtime_discovery_aggregation.py`
- Modify: `docs/superpowers/specs/2026-07-23-a20-two-layer-articulation-discovery-design.md`

- [ ] **Step 1: Write failing report assertions**

For deterministic interleaved runs, require:

```text
Three-run raw runtime determinism: PASS
Runtime joint semantic match: PASS
Policy-to-runtime mapping: PASS
Policy/runtime round trip: PASS
Raw order equals canonical order: no (informational)
```

Keep `Overall: NOT_READY` when Asset Validator is not clean.

- [ ] **Step 2: Verify RED and update report rendering**

Replace `Canonical ordered-record match` with the five explicit lines above. Keep report size, escaping, issue limits, digest, generation ID, and all safety/readiness lines unchanged.

- [ ] **Step 3: Update the original A20 design contract**

Amend Layer 2 to require deterministic raw order plus explicit semantic mapping instead of raw positional equality. Link the approved adapter design and preserve all no-step/no-action boundaries.

- [ ] **Step 4: Run report tests and commit**

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q aloha_isaac_rebuild/tests/test_a20_runtime_discovery_aggregation.py
git add -- aloha_isaac_rebuild/scripts/run_a20_runtime_articulation_discovery.py aloha_isaac_rebuild/tests/test_a20_runtime_discovery_aggregation.py docs/superpowers/specs/2026-07-23-a20-two-layer-articulation-discovery-design.md
git commit -m "docs: define mapped A20 runtime order gate"
```

### Task 7: Full offline verification and current evidence regeneration

**Files:**
- Generated: `aloha_isaac_rebuild/artifacts/validation/a20_usd_dof_metadata_gate.json`
- Generated: `aloha_isaac_rebuild/artifacts/validation/a20_runtime_articulation_discovery_gate.json`
- Generated: `aloha_isaac_rebuild/reports/a20_two_layer_articulation_discovery.md`

- [ ] **Step 1: Run all focused unit tests through bounded evidence**

```bash
codex-evidence --name a20-order-adapter-focused-tests -- env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q aloha_isaac_rebuild/tests/test_a20_policy_runtime_order_adapter.py aloha_isaac_rebuild/tests/test_a20_usd_dof_metadata.py aloha_isaac_rebuild/tests/test_a20_articulation_gate_common.py aloha_isaac_rebuild/tests/test_a20_runtime_discovery_aggregation.py
```

Expected: all tests pass, zero warnings/errors.

- [ ] **Step 2: Run Ruff and diff checks**

```bash
.venv/bin/ruff check aloha_isaac_rebuild/scripts/a20_policy_runtime_order_adapter.py aloha_isaac_rebuild/scripts/audit_a20_usd_dof_metadata.py aloha_isaac_rebuild/scripts/run_a20_runtime_articulation_discovery.py aloha_isaac_rebuild/tests/test_a20_policy_runtime_order_adapter.py aloha_isaac_rebuild/tests/test_a20_usd_dof_metadata.py aloha_isaac_rebuild/tests/test_a20_runtime_discovery_aggregation.py
git diff --check
```

- [ ] **Step 3: Regenerate fresh Layer 1**

Run the existing A20 Layer 1 command through `codex-evidence`. Require current input hashes, 16 authored DOFs, zero mismatches, and all safety flags false.

- [ ] **Step 4: Run three fresh no-step Layer 2 probes**

Run the existing coordinator through `codex-evidence`. Require three successful runtime probes, deterministic raw interleaved order, a complete adapter, PASS round trip, and no prohibited operation. Do not run if Layer 1 fails or live inputs changed.

- [ ] **Step 5: Regenerate the bounded offline report**

Require Layer 2 mapping PASS while retaining `Overall: NOT_READY` if the independent Asset Validator blocker remains.

- [ ] **Step 6: Verify generated evidence and final diff**

Check JSON parseability, provenance hashes, report digest/generation ID, exact safety flags, report size below 32 KiB, and that no user-owned A19/config/audit file was accidentally staged.
