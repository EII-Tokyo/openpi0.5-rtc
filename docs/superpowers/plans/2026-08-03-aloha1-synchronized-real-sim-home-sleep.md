# ALOHA1 Synchronized Real–Simulation Home/Sleep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed, sample-index-aligned experiment that plays the same frozen Home/Sleep manifest on Isaac Sim 5.1 and the real `follower_left`, records `cam_high` and both telemetry streams, and produces an auditable comparison without changing robot or simulation parameters.

**Architecture:** A pure Python protocol package owns run identity, readiness, scheduling, safety gates, and comparison. An Isaac worker and a ROS1 real worker each play the stored manifest locally after a coordinated future start; the LAN is used for prepare/start control and evidence transfer, not for 50 Hz command streaming. Fake transports verify all control logic before a separately authorized read-only 103 preflight or live publisher can be instantiated.

**Tech Stack:** Python 3.11 project `.venv`, Python 3/ROS1 Noetic on machine 103, Isaac Sim 5.1.0.0 / Kit 107.3.3 / PhysX 107.3.26, pytest, YAML/JSON/CSV, ROS `sensor_msgs/JointState`, project `aloha.msg/RGBGrayscaleImage`, OpenCV/FFmpeg.

---

### Task 1: Freeze synchronized experiment configuration and protocol types

**Files:**
- Create: `configs/aloha1_home_sleep_synchronized_real_sim.yaml`
- Create: `tools/aloha1_mapping/home_sleep_sync.py`
- Create: `tests/aloha1_mapping/test_home_sleep_sync.py`

- [ ] **Step 1: Write failing tests for immutable run identity and timing**

```python
def test_build_run_identity_binds_manifest_and_workers():
    identity = build_run_identity(
        run_id="run-001",
        manifest_sha256="a" * 64,
        command_signature="b" * 64,
        command_rate_hz=50,
    )
    assert identity["sample_period_ns"] == 20_000_000
    assert identity["workers"] == ["isaac", "real", "cam_high"]


def test_start_classification_uses_one_command_period():
    assert classify_start_skew(20_000_000, sample_period_ns=20_000_000) == "SYNCHRONIZED_START_PASS"
    assert classify_start_skew(20_000_001, sample_period_ns=20_000_000) == "POST_ALIGNED_ONLY"
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv/bin/python -m pytest -q tests/aloha1_mapping/test_home_sleep_sync.py`

Expected: import/attribute failure because `home_sleep_sync.py` does not exist.

- [ ] **Step 3: Implement immutable protocol helpers**

Implement typed JSON-compatible helpers for:

```python
WORKERS = ("isaac", "real", "cam_high")
STATES = ("CREATED", "PREPARED", "READY", "ARMED", "RUNNING", "COMPLETE", "ABORTED")

def build_run_identity(*, run_id: str, manifest_sha256: str,
                       command_signature: str, command_rate_hz: int) -> dict[str, object]:
    return {
        "run_id": run_id,
        "manifest_sha256": manifest_sha256,
        "command_signature": command_signature,
        "command_rate_hz": command_rate_hz,
        "sample_period_ns": 1_000_000_000 // command_rate_hz,
        "workers": list(WORKERS),
    }

def deadline_ns(start_monotonic_ns: int, sample_index: int,
                sample_period_ns: int) -> int:
    return start_monotonic_ns + sample_index * sample_period_ns

def classify_start_skew(skew_ns: int, sample_period_ns: int) -> str:
    return "SYNCHRONIZED_START_PASS" if abs(skew_ns) <= sample_period_ns else "POST_ALIGNED_ONLY"

def validate_ready_record(record: Mapping[str, object],
                          identity: Mapping[str, object]) -> list[str]:
    return [
        field for field in ("run_id", "manifest_sha256", "command_signature")
        if record.get(field) != identity.get(field)
    ]
```

The config binds the already tracked manifest, digital validation report,
frozen Stage, finger-limit layer, expected ROS topics, expected camera type,
workspace 2, and explicit `real_access_authorized: false` /
`real_motion_authorized: false` defaults.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/aloha1_mapping/test_home_sleep_sync.py`

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit**

```bash
git add -f configs/aloha1_home_sleep_synchronized_real_sim.yaml \
  tools/aloha1_mapping/home_sleep_sync.py \
  tests/aloha1_mapping/test_home_sleep_sync.py
git commit -m "aloha1: add synchronized replay protocol"
```

### Task 2: Add fake workers and fail-closed coordinator

**Files:**
- Create: `tools/run_aloha1_home_sleep_sync.py`
- Modify: `tools/aloha1_mapping/home_sleep_sync.py`
- Modify: `tests/aloha1_mapping/test_home_sleep_sync.py`

- [ ] **Step 1: Write failing coordinator tests**

Cover:

```python
def test_coordinator_never_arms_before_all_workers_ready():
    workers = {
        "isaac": FakeWorker("isaac", ready=True),
        "real": FakeWorker("real", ready=False),
        "cam_high": FakeWorker("cam_high", ready=True),
    }
    report = run_coordinator(identity=_identity(), workers=workers, samples=_samples(3))
    assert report["status"] == "BLOCKED_NOT_ALL_READY"
    assert all(worker.arm_calls == 0 for worker in workers.values())

def test_manifest_mismatch_aborts_without_transport_publish():
    real = FakeWorker("real", ready=True, manifest_sha256="wrong")
    report = run_coordinator(
        identity=_identity(),
        workers={"isaac": FakeWorker("isaac"), "real": real,
                 "cam_high": FakeWorker("cam_high")},
        samples=_samples(3),
    )
    assert report["status"] == "BLOCKED_IDENTITY_MISMATCH"
    assert real.publish_count == 0

def test_fake_workers_execute_all_1850_indices_once():
    report = run_coordinator(identity=_identity(), workers=_ready_workers(),
                             samples=_samples(1850))
    assert report["status"] == "PASS_FAKE_TRANSPORT"
    assert report["workers"]["real"]["sample_indices"] == list(range(1850))

def test_late_real_worker_never_bursts_missed_commands():
    real = FakeWorker("real", late_at_index=2)
    report = run_coordinator(identity=_identity(), workers=_ready_workers(real=real),
                             samples=_samples(5))
    assert report["status"] == "ABORTED_DEADLINE_MISS"
    assert real.sample_indices == [0, 1]

def test_operator_stop_aborts_both_workers():
    workers = _ready_workers(real=FakeWorker("real", operator_stop_at_index=2))
    report = run_coordinator(identity=_identity(), workers=workers, samples=_samples(5))
    assert report["status"] == "REAL_EXECUTION_ABORTED"
    assert workers["isaac"].abort_calls == 1
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest -q tests/aloha1_mapping/test_home_sleep_sync.py`

Expected: missing coordinator/fake-worker symbols.

- [ ] **Step 3: Implement minimal coordinator and fake transport**

Define a transport-neutral worker contract:

```python
class Worker(Protocol):
    def prepare(self, identity: Mapping[str, object]) -> Mapping[str, object]:
        raise NotImplementedError
    def arm(self, start_wall_time_ns: int) -> Mapping[str, object]:
        raise NotImplementedError
    def run(self) -> Mapping[str, object]:
        raise NotImplementedError
    def abort(self, reason: str) -> Mapping[str, object]:
        raise NotImplementedError
```

The offline CLI defaults to `--transport fake`, writes JSON/CSV evidence, and
contains no SSH, ROS, serial, or Isaac imports on the fake path.

- [ ] **Step 4: Verify GREEN and dry-run contract**

Run:

```bash
.venv/bin/python -m pytest -q tests/aloha1_mapping/test_home_sleep_sync.py
.venv/bin/python tools/run_aloha1_home_sleep_sync.py \
  --config configs/aloha1_home_sleep_synchronized_real_sim.yaml \
  --transport fake \
  --output reports/aloha1_mapping/aloha1_home_sleep_sync_fake_run.json
```

Expected: 1850 samples per command worker, status `PASS_FAKE_TRANSPORT`, zero
network/ROS/serial/torque actions.

- [ ] **Step 5: Commit**

```bash
git add -f tools/run_aloha1_home_sleep_sync.py \
  tools/aloha1_mapping/home_sleep_sync.py \
  tests/aloha1_mapping/test_home_sleep_sync.py \
  reports/aloha1_mapping/aloha1_home_sleep_sync_fake_run.json
git commit -m "aloha1: add fail closed replay coordinator"
```

### Task 3: Implement alignment and comparison metrics

**Files:**
- Modify: `tools/compare_aloha1_home_sleep_real_sim.py`
- Create: `tools/aloha1_mapping/home_sleep_alignment.py`
- Create: `tests/aloha1_mapping/test_home_sleep_alignment.py`

- [ ] **Step 1: Write failing alignment tests**

Test exact sample-key matching, preserved source order, signed joint error,
start skew, dropped samples, duplicate samples, endpoint metrics, and
`POST_ALIGNED_ONLY` classification.

```python
def test_alignment_uses_cycle_segment_and_sample_index_not_row_number():
    real = [{"cycle": 1, "segment": "move", "sample_index": 1, "q": [2.0]}]
    isaac = [{"cycle": 1, "segment": "move", "sample_index": 1, "q": [1.5]}]
    report = align_rows(real, isaac, joint_names=("waist",))
    assert report["matched_keys"] == [[1, "move", 1]]

def test_alignment_preserves_signed_error():
    report = align_rows(_rows(q=2.0), _rows(q=1.5), joint_names=("waist",))
    assert report["per_joint"]["waist"]["signed_real_minus_isaac_mean_rad"] == 0.5

def test_missing_real_sample_is_reported_not_interpolated_away():
    report = align_rows(_rows(indices=(0, 2)), _rows(indices=(0, 1, 2)),
                        joint_names=("waist",))
    assert report["missing_real_keys"] == [[1, "move", 1]]

def test_dynamic_pass_is_not_claimed_without_frozen_thresholds():
    report = classify_correspondence(_matching_metrics(), thresholds=None)
    assert report["DYNAMIC_TRAJECTORY_CORRESPONDENCE"] == "CALIBRATION_PENDING"
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest -q tests/aloha1_mapping/test_home_sleep_alignment.py`

- [ ] **Step 3: Implement pure comparison logic**

Return machine-readable per-joint/per-cycle metrics and independent layers:

```text
COMMAND_IDENTITY
JOINT_SEMANTICS
KINEMATIC_ENDPOINT_CORRESPONDENCE
DYNAMIC_TRAJECTORY_CORRESPONDENCE
START_SYNCHRONIZATION
```

No missing sample is silently synthesized in the raw comparison. Optional
resampling is stored as a separate derived table.

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m pytest -q tests/aloha1_mapping/test_home_sleep_alignment.py`

- [ ] **Step 5: Commit**

```bash
git add -f tools/compare_aloha1_home_sleep_real_sim.py \
  tools/aloha1_mapping/home_sleep_alignment.py \
  tests/aloha1_mapping/test_home_sleep_alignment.py
git commit -m "aloha1: align real and digital replay signals"
```

### Task 4: Add a transport-independent real worker and camera recorder

**Files:**
- Create: `tools/aloha1_mapping/home_sleep_real_worker.py`
- Create: `tools/aloha1_mapping/cam_high_recorder.py`
- Create: `tools/run_aloha1_home_sleep_real_worker.py`
- Create: `tests/aloha1_mapping/test_home_sleep_real_worker.py`

- [ ] **Step 1: Write failing fake-adapter tests**

Cover exact joint-name mapping, stale readback, opposite motion, command
rejection, operator stop, camera loss, frame timestamp preservation, and
absence of `Present_Current`.

```python
def test_real_worker_rejects_reordered_joint_state():
    state = FakeJointState(names=("shoulder", "waist"), positions=(0.0, 0.0))
    report = RealWorkerCore.expected_six_dof().preflight(state)
    assert report["status"] == "BLOCKED_JOINT_ORDER"

def test_real_worker_aborts_before_publish_on_stale_readback():
    sink = FakeCommandSink()
    worker = _worker(sink=sink, readback_age_ns=100_000_001,
                     maximum_readback_age_ns=100_000_000)
    assert worker.step(_sample(0))["status"] == "ABORTED_STALE_READBACK"
    assert sink.publish_count == 0

def test_real_worker_stops_without_command_burst_after_deadline_miss():
    sink = FakeCommandSink()
    worker = _worker(sink=sink, clock=FakeClock(late_at_index=2))
    report = worker.run(_samples(5))
    assert report["published_indices"] == [0, 1]

def test_cam_high_preserves_source_and_receive_timestamps():
    record = frame_record(_camera_message(source_stamp_ns=10), receive_monotonic_ns=20,
                          receive_wall_time_ns=30)
    assert record["source_stamp_ns"] == 10
    assert record["receive_monotonic_ns"] == 20

def test_missing_present_current_is_not_a_failure():
    report = _worker(hardware_status={}).preflight(_valid_joint_state())
    assert report["present_current"] == "NOT_AVAILABLE"
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest -q tests/aloha1_mapping/test_home_sleep_real_worker.py`

- [ ] **Step 3: Implement pure worker core and fake adapters**

The production core accepts injected `JointStateSource`, `CommandSink`,
`CameraSource`, `Clock`, and `StopController` protocols. No `rospy` import is
allowed in the core module. The CLI remains `DRY_RUN` unless both a signed
authorization record and `--execute-real` are supplied.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
.venv/bin/python -m pytest -q tests/aloha1_mapping/test_home_sleep_real_worker.py
.venv/bin/python tools/run_aloha1_home_sleep_real_worker.py --transport fake --dry-run
```

Expected: all tests pass; CLI reports zero commands published.

- [ ] **Step 5: Commit**

```bash
git add -f tools/aloha1_mapping/home_sleep_real_worker.py \
  tools/aloha1_mapping/cam_high_recorder.py \
  tools/run_aloha1_home_sleep_real_worker.py \
  tests/aloha1_mapping/test_home_sleep_real_worker.py
git commit -m "aloha1: add testable real replay worker"
```

### Task 5: Add the ROS1 Noetic adapter and read-only preflight

**Files:**
- Create: `tools/aloha1_mapping/home_sleep_ros1_adapter.py`
- Create: `tools/preflight_aloha1_home_sleep_sync_real.py`
- Create: `tests/aloha1_mapping/test_home_sleep_ros1_adapter.py`
- Modify: `docs/agents/remote_103_operations.md`

- [ ] **Step 1: Verify official and live API evidence before code**

Read pinned Interbotix ROS1 source for `JointGroupCommand`, group-name
semantics, publish path, and stop/hold behavior. Record repository, commit,
license, source path, and SHA-256. Before any Stage/runtime changes, verify
Isaac APIs through direct NVIDIA MCP. Before any 103 access, require explicit
read-only authorization.

- [ ] **Step 2: Write failing serialization and gate tests**

```python
def test_ros_adapter_serializes_exact_six_joint_group_command():
    message = serialize_joint_group_command(_fake_message_type(), "arm", [0, 1, 2, 3, 4, 5])
    assert message.name == "arm"
    assert message.cmd == [0, 1, 2, 3, 4, 5]

def test_ros_import_is_deferred_until_live_gate_passes(monkeypatch):
    monkeypatch.setattr(builtins, "__import__", _reject_rospy_import)
    report = build_ros_adapter(authorization={"real_motion_authorized": False})
    assert report["status"] == "NOT_RUN_AUTHORIZATION_REQUIRED"

def test_read_only_preflight_never_constructs_publisher():
    factory = FakeRosFactory()
    run_read_only_preflight(factory)
    assert factory.publisher_count == 0

def test_unverified_stop_path_blocks_live_status():
    report = live_adapter_gate(_passing_live_gates() | {"stop_path_verified": False})
    assert report["status"] == "BLOCKED"
    assert "stop_path_verified" in report["failed_gates"]
```

- [ ] **Step 3: Verify RED**

Run: `.venv/bin/python -m pytest -q tests/aloha1_mapping/test_home_sleep_ros1_adapter.py`

- [ ] **Step 4: Implement deferred ROS adapter**

The adapter imports `rospy` and project messages only inside the authorized
factory. Read-only preflight subscribes/introspects but never constructs the
command publisher. All remote scripts execute only from
`/home/eii/openpi0.5-rtc-reward-learning`.

- [ ] **Step 5: Verify offline GREEN**

Run focused tests and the preflight with no authorization; expect
`NOT_RUN_AUTHORIZATION_REQUIRED`, zero network access, zero publisher objects.

- [ ] **Step 6: Commit**

```bash
git add -f tools/aloha1_mapping/home_sleep_ros1_adapter.py \
  tools/preflight_aloha1_home_sleep_sync_real.py \
  tests/aloha1_mapping/test_home_sleep_ros1_adapter.py \
  docs/agents/remote_103_operations.md
git commit -m "aloha1: add gated ROS1 replay adapter"
```

### Task 6: Add the synchronized Isaac worker

**Files:**
- Create: `tools/run_aloha1_home_sleep_isaac_worker.py`
- Modify: `tools/validate_aloha1_home_sleep_digital.py`
- Modify: `tests/aloha1_mapping/test_home_sleep_sync.py`

- [ ] **Step 1: Query direct NVIDIA official Isaac MCP**

Confirm local Isaac Sim 5.1 APIs for `SimulationApp`, Stage opening,
`SingleArticulation`, `ArticulationAction`, timeline/play state, render capture,
and shutdown. Save the bounded evidence log. Do not use latest/6.0 APIs.

- [ ] **Step 2: Write failing worker-contract tests**

Test CLI argument validation, hash pinning, READY record construction,
sample-index scheduling, GUI workspace metadata, and immutable Stage behavior
without importing Isaac in unit tests.

- [ ] **Step 3: Verify RED**

Run the focused sync tests.

- [ ] **Step 4: Implement worker wrapper around validated digital core**

Reuse `validate_aloha1_home_sleep_digital.py` runtime logic rather than
duplicating physics setup. Add external start-deadline and run-ID support while
preserving the already qualified numeric path. The GUI launches on workspace 2
and captures the complete arm.

- [ ] **Step 5: Run two fresh Isaac processes**

Verify manifest/Stage hashes, 1850 command indices, numeric signature, GUI
video metadata, and no source/final asset modification.

- [ ] **Step 6: Commit**

```bash
git add -f tools/run_aloha1_home_sleep_isaac_worker.py \
  tools/validate_aloha1_home_sleep_digital.py \
  tests/aloha1_mapping/test_home_sleep_sync.py
git commit -m "aloha1: add synchronized Isaac replay worker"
```

### Task 7: Produce the ready-for-live integration package

**Files:**
- Modify: `tools/run_aloha1_home_sleep_sync.py`
- Create: `tools/build_aloha1_home_sleep_sync_preflight_report.py`
- Create: `reports/aloha1_mapping/aloha1_home_sleep_sync_offline_preflight.json`
- Create: `reports/aloha1_mapping/aloha1_home_sleep_sync_offline_preflight.md`
- Modify: `README_ALOHA1_ISAACSIM_5_1.md`
- Modify: `.codex/TASK_STATE.md`

- [ ] **Step 1: Run full offline/fake integration**

Require all fake workers, scheduler, abort, alignment, camera, ROS deferral,
and report-contract tests to pass. Confirm no ROS/SSH/network/serial/torque
actions occurred.

- [ ] **Step 2: Run fresh Isaac integration**

Verify the synchronized Isaac worker independently. Do not start the real
adapter.

- [ ] **Step 3: Build readiness report**

Status is `READY_FOR_SUPERVISED_REAL_EXECUTION` only when offline and Isaac
gates pass. The report lists the exact remaining live gates and never equates
readiness with real correspondence.

- [ ] **Step 4: Run regression**

```bash
.venv/bin/python -m pytest -q tests/aloha1_mapping
.venv/bin/python -m ruff check tools/aloha1_mapping/home_sleep_sync.py \
  tools/aloha1_mapping/home_sleep_alignment.py \
  tools/aloha1_mapping/home_sleep_real_worker.py \
  tools/aloha1_mapping/cam_high_recorder.py \
  tools/aloha1_mapping/home_sleep_ros1_adapter.py \
  tools/run_aloha1_home_sleep_sync.py \
  tools/run_aloha1_home_sleep_real_worker.py \
  tools/preflight_aloha1_home_sleep_sync_real.py \
  tools/run_aloha1_home_sleep_isaac_worker.py
.venv/bin/python -m py_compile tools/aloha1_mapping/home_sleep_sync.py \
  tools/aloha1_mapping/home_sleep_alignment.py \
  tools/aloha1_mapping/home_sleep_real_worker.py \
  tools/aloha1_mapping/cam_high_recorder.py \
  tools/aloha1_mapping/home_sleep_ros1_adapter.py \
  tools/run_aloha1_home_sleep_sync.py \
  tools/run_aloha1_home_sleep_real_worker.py \
  tools/preflight_aloha1_home_sleep_sync_real.py \
  tools/run_aloha1_home_sleep_isaac_worker.py
git diff --check
```

- [ ] **Step 5: Commit**

```bash
git add -f tools/run_aloha1_home_sleep_sync.py \
  tools/build_aloha1_home_sleep_sync_preflight_report.py \
  reports/aloha1_mapping/aloha1_home_sleep_sync_offline_preflight.json \
  reports/aloha1_mapping/aloha1_home_sleep_sync_offline_preflight.md \
  README_ALOHA1_ISAACSIM_5_1.md .codex/TASK_STATE.md
git commit -m "aloha1: prepare supervised real sim replay"
```

### Task 8: Execute the separately authorized supervised live run

**Files:**
- Create after execution: `reports/aloha1_mapping/aloha1_home_sleep_sync_real_execution.json`
- Create after execution: `reports/aloha1_mapping/aloha1_home_sleep_sync_real_telemetry.csv`
- Create after execution: `reports/aloha1_mapping/aloha1_home_sleep_sync_camera_manifest.json`
- Create after execution: `reports/aloha1_mapping/aloha1_home_sleep_sync_comparison.json`
- Create after execution: `reports/aloha1_mapping/aloha1_home_sleep_sync_comparison.md`

- [ ] **Step 1: Obtain same-session authorization**

Require explicit authorization for read-only 103 access, then a separate live
motion confirmation after the read-only preflight passes and the operator
confirms clear workspace and stop readiness.

- [ ] **Step 2: Run read-only preflight**

Confirm ROS types, exact joint order, current Home entry, camera coverage,
driver state, and verified stop path. Any mismatch blocks publishing.

- [ ] **Step 3: Run one coordinated three-cycle experiment**

Start `cam_high`, real worker, and Isaac worker through prepare/ready/start.
Do not automatically retry an aborted run.

- [ ] **Step 4: Align and review evidence**

Generate real/Isaac/side-by-side videos, paired key frames, metrics, and literal
classification. Visually review every retained image and all videos.

- [ ] **Step 5: Verify and commit evidence**

Commit code and legal machine-readable reports only. Keep raw media/logs in
`.codex/artifacts/`. Do not push.
