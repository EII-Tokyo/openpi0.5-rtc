# ALOHA Continuous Collection Health Monitor Design

## Objective

Determine why `record_episodes_copy.py` repeatedly exits during continuous
collection without controlling, stopping, or reconfiguring the real robot.
The monitor must preserve time-aligned evidence from before and after a
failure, especially around `joint_state_stale` health-check failures.

## Safety Boundary

The monitor is read-only. It creates ROS subscriptions but no publishers or
service clients. It never changes torque, starts or stops robot processes,
restarts containers, or modifies collected episodes. All artifacts stay under
the project-local `.codex/artifacts/collect-health-monitor/` directory on
machine 103.

## Architecture

The monitoring session has two components:

1. A ROS 2 probe in the existing `aloha2-collect` container subscribes to the
   four leader/follower `joint_states` topics. It records monotonic receive
   times, ROS header times, per-topic message counts, frequency, and maximum
   receive gaps.
2. A host-side supervisor records the recorder process lifecycle, bounded
   Docker log excerpts, process and system pressure, recent project-file
   activity, and kernel USB/FTDI/tty events using a common wall-clock
   timeline.

The supervisor retains a bounded rolling history. A topic gap greater than 50
ms is a warning; a gap greater than 100 ms is a fault because it matches the
recorder's current freshness threshold. A recorder exit is also a fault. When
a fault occurs, the session preserves the preceding 30 seconds and collects
15 seconds of post-fault evidence before writing a summary.

## Classification

The summary reports one of these evidence-based classifications:

- `publisher_gap`: the independent ROS probe also observed a topic gap.
- `recorder_callback_stall`: the independent probe remained healthy while the
  recorder reported stale joint state.
- `system_pressure`: multiple topics stalled together with CPU, memory, or I/O
  pressure.
- `serial_fault`: the gap aligns with DYNAMIXEL or USB/serial errors.
- `insufficient_evidence`: the available evidence cannot distinguish causes.

The monitor does not infer a hardware fault merely because `leader_left` is
the first name reported by the recorder.

## Artifacts and Bounds

Each run writes to a timestamped directory containing newline-delimited probe
events, one-second system samples, bounded Docker/kernel excerpts, metadata,
and a final machine-readable classification summary. Logs are size-bounded
and rotated. Camera images, credentials, environment dumps, and episode
contents are excluded.

## Verification

Automated tests cover receive-gap classification, simultaneous-topic stall
classification, healthy-topic versus recorder-exit classification, bounded
ring-buffer behavior, and summary serialization. Deployment verification
confirms all four subscriptions receive near 100 Hz, the monitor creates no
ROS publishers or service clients, artifacts are written only inside the
approved project directory, and collection/robot processes remain unchanged.
