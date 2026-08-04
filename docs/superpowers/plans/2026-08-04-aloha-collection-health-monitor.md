# ALOHA Collection Health Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a read-only monitor on machine 103 that distinguishes ROS publisher gaps, recorder-local callback stalls, system pressure, and serial faults during repeated collection-script exits.

**Architecture:** One standard-library Python tool contains a pure analysis core, an ROS 2 probe mode executed inside `aloha2-collect`, and a host-supervisor mode executed from the approved project directory. The supervisor streams the tool to the container over stdin, writes bounded evidence beneath `.codex/artifacts/collect-health-monitor/`, and observes recorder lifecycle and system state without controlling the robot.

**Tech Stack:** Python 3.11 standard library, ROS 2 Humble `rclpy` and `sensor_msgs` in the existing container, Docker CLI, pytest.

---

## File Structure

- Create `tools/collect_health_monitor.py`: gap tracking, classification, ROS probe, and host supervisor.
- Create `tests/monitoring/test_collect_health_monitor.py`: unit tests for thresholds, classification, ring bounds, paths, and safe commands.
- Create `docs/operations/aloha_collection_health_monitor.md`: launch, artifact, and stop instructions.

### Task 1: Pure monitoring model

**Files:**
- Create: `tests/monitoring/test_collect_health_monitor.py`
- Create: `tools/collect_health_monitor.py`

- [ ] **Step 1: Write the failing gap and ring tests**

```python
def test_topic_tracker_marks_warning_and_fault():
    tracker = monitor.TopicTracker("/leader_left/joint_states", 0.05, 0.10)
    assert tracker.observe(1.000, 1) is None
    assert tracker.observe(1.060, 2)["severity"] == "warning"
    assert tracker.observe(1.180, 3)["severity"] == "fault"


def test_time_ring_is_bounded():
    ring = monitor.TimeRing(retention_seconds=2.0)
    for second in range(5):
        ring.append(float(second), {"second": second})
    assert [item["second"] for item in ring.items()] == [2, 3, 4]
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/monitoring/test_collect_health_monitor.py -q`

Expected: FAIL because the monitor module does not exist.

- [ ] **Step 3: Implement the minimal model**

```python
@dataclass
class TopicTracker:
    topic: str
    warning_seconds: float
    fault_seconds: float
    last_receive: float | None = None

    def observe(self, received: float, sequence: int) -> dict[str, object] | None:
        previous, self.last_receive = self.last_receive, received
        if previous is None:
            return None
        gap = received - previous
        severity = "fault" if gap > self.fault_seconds else "warning" if gap > self.warning_seconds else None
        if severity is None:
            return None
        return {"kind": "topic_gap", "topic": self.topic, "severity": severity,
                "gap_seconds": gap, "sequence": sequence}
```

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m pytest tests/monitoring/test_collect_health_monitor.py -q`

Expected: PASS.

### Task 2: Classification and safety contract

**Files:**
- Modify: `tests/monitoring/test_collect_health_monitor.py`
- Modify: `tools/collect_health_monitor.py`

- [ ] **Step 1: Add failing classification and command tests**

```python
def test_healthy_probe_at_stale_exit_is_callback_stall():
    result = monitor.classify_incident(
        recorder_reported_stale=True, topic_faults=[], serial_errors=[], pressure=False
    )
    assert result == "recorder_callback_stall"


def test_probe_command_is_subscription_only(tmp_path):
    joined = " ".join(monitor.build_probe_command(tmp_path / "monitor.py"))
    assert "docker exec -i aloha2-collect" in joined
    assert "ros2 topic pub" not in joined
    assert "ros2 service call" not in joined
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/monitoring/test_collect_health_monitor.py -q`

Expected: FAIL because classification and command construction are absent.

- [ ] **Step 3: Implement classification and command construction**

```python
def classify_incident(*, recorder_reported_stale, topic_faults, serial_errors, pressure):
    if serial_errors:
        return "serial_fault"
    if topic_faults and pressure and len({event["topic"] for event in topic_faults}) > 1:
        return "system_pressure"
    if topic_faults:
        return "publisher_gap"
    if recorder_reported_stale:
        return "recorder_callback_stall"
    return "insufficient_evidence"


def build_probe_command(script_path: pathlib.Path) -> list[str]:
    return ["docker", "exec", "-i", "aloha2-collect", "bash", "-lc",
            "source /opt/ros/humble/setup.bash; "
            "source /root/interbotix_ws/install/setup.bash; "
            "exec python3 - --mode ros-probe"]
```

- [ ] **Step 4: Verify GREEN**

Run: `.venv/bin/python -m pytest tests/monitoring/test_collect_health_monitor.py -q`

Expected: PASS.

### Task 3: ROS probe and host supervisor

**Files:**
- Modify: `tests/monitoring/test_collect_health_monitor.py`
- Modify: `tools/collect_health_monitor.py`

- [ ] **Step 1: Add failing topic and artifact-boundary tests**

```python
def test_artifact_path_is_below_project_root(tmp_path):
    run_dir = monitor.create_run_directory(tmp_path, "20260804T160000")
    assert run_dir.parent == tmp_path / ".codex/artifacts/collect-health-monitor"


def test_default_topics_cover_all_four_arms():
    assert monitor.DEFAULT_TOPICS == (
        "/leader_left/joint_states", "/leader_right/joint_states",
        "/follower_left/joint_states", "/follower_right/joint_states",
    )
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/python -m pytest tests/monitoring/test_collect_health_monitor.py -q`

Expected: FAIL because the artifact helper and topic constants are absent.

- [ ] **Step 3: Implement both runtime modes**

`run_ros_probe()` imports ROS packages lazily, subscribes to exactly the four
topics, emits one-second summaries and immediate gap events as JSONL, and
creates no publisher or client. `run_host_supervisor()` asserts the project
root, creates a timestamped run directory, streams this script to `docker
exec -i`, samples `docker stats` and recorder presence once per second, records
Docker events/logs, and writes a bounded incident snapshot on topic faults or a
recorder present-to-absent transition. All subprocesses use argument lists and
bounded timeouts.

- [ ] **Step 4: Verify tests and syntax**

Run:

```bash
.venv/bin/python -m pytest tests/monitoring/test_collect_health_monitor.py -q
.venv/bin/python -m py_compile tools/collect_health_monitor.py
```

Expected: all tests pass and compilation exits 0.

### Task 4: Documentation, commit, deployment, and live verification

**Files:**
- Create: `docs/operations/aloha_collection_health_monitor.md`

- [ ] **Step 1: Document launch and evidence paths**

```bash
cd /home/eii/openpi0.5-rtc-reward-learning
nohup .venv/bin/python tools/collect_health_monitor.py --mode host \
  > .codex/artifacts/collect-health-monitor/launcher.log 2>&1 &
```

Document that stopping the monitor affects only its own PID and never the
recorder or container.

- [ ] **Step 2: Run bounded verification**

```bash
.venv/bin/python -m pytest tests/monitoring/test_collect_health_monitor.py -q
git diff --check
```

Expected: PASS with no whitespace errors.

- [ ] **Step 3: Commit the implementation batch**

```bash
git add tools/collect_health_monitor.py tests/monitoring/test_collect_health_monitor.py \
  docs/operations/aloha_collection_health_monitor.md \
  docs/superpowers/plans/2026-08-04-aloha-collection-health-monitor.md
git commit -m "feat: monitor ALOHA collection health"
```

- [ ] **Step 4: Sync only committed monitor files to machine 103**

Preview and copy the monitor, test, and operations files only into
`/home/eii/openpi0.5-rtc-reward-learning`. Do not modify
`/home/eii/aloha-2.0`.

- [ ] **Step 5: Start and verify the live monitor**

Within ten seconds, verify four topics near 100 Hz, a live monitor PID,
artifact paths beneath the approved project root, and no monitor-created ROS
publishers or service clients. Preserve the PID and run directory for incident
review.
