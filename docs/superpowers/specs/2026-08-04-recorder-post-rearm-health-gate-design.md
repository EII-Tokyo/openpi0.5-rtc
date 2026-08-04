# Recorder Post-Rearm Health Gate Design

## Problem

Continuous collection currently checks that all robot joint-state streams are
fresh before the current-pose rearm gesture. Once the gesture is accepted,
`restore_teleop_modes()` changes operating modes and torque state. Those service
calls block each `xs_sdk` publisher for roughly 260--310 ms, but the recorder
enters `teleop_wait` immediately afterward with a 100 ms leader freshness
limit. The recorder can therefore reject its own cached leader state even
though all publishers recover to 100 Hz.

The independent monitor established the causal sequence:

1. all four streams are fresh before rearm;
2. mode and torque calls create a bounded publishing gap;
3. publishers recover;
4. the recorder observes the pre-gap cached state and fails closed;
5. safe sleep completes before the process exits.

## Scope and Source Ownership

The live ROS2 collection source was copied without modification from
`/home/eii/aloha-2.0` into the project-owned
`third_party/aloha_collection/`. The original directory remains unchanged.
The copied `scripts/collect.sh` resolves its repository root from its own
location, so launching that script mounts this project-owned copy into the
`aloha2-collect` container.

This change does not modify the ROS1 inference `aloha_ros_nodes` service or the
root `docker-compose.yml`.

## Design

Keep the existing pre-rearm health gate and the health checks used while the
operator performs the open-close gesture. Add a required post-restore callback
to `wait_for_safe_current_pose_rearm()` and invoke it only after
`restore_teleop()` returns successfully.

The recorder supplies that callback as `_wait_for_health_gate(...)` for all
initialized robot interfaces, with phase `current_pose_rearm_post_restore`, the
existing 100 ms freshness limit, the existing three-consecutive-sample
requirement, and the existing two-second timeout.

`RobotHealthMonitor.wait_for_fresh()` records the sequence of every robot when
the gate begins. It only passes after each robot advances by at least three
valid messages. Consequently, samples cached before or during mode/torque
reconfiguration cannot release the gate.

The order is:

```text
pre-rearm fresh gate
  -> operator open-close gesture with freshness checks
  -> final pre-restore freshness check
  -> restore_teleop_modes (watchdog intentionally not armed)
  -> post-restore gate requiring three new samples from all four robots
  -> teleop_wait with the unchanged 100 ms watchdog
```

## Failure Semantics

- If `restore_teleop()` fails, the post-restore gate is not called and the
  existing fail-safe cleanup owns recovery.
- If any publisher does not recover, the post-restore gate times out and the
  attempt exits through the existing safe cleanup path.
- If a stop is requested during the gate, the existing health-gate stop check
  aborts without entering teleoperation.
- The 100 ms runtime threshold remains unchanged, so genuine communication
  loss during `teleop_wait` or collection still fails fast.
- No health fault is globally cleared or ignored.

## Verification

Pure Python tests must prove:

1. the post-restore callback runs after restore and before success;
2. it is not called if restore fails;
3. its failure propagates and prevents successful rearm;
4. `wait_for_fresh()` rejects pre-gate cached samples and requires the configured
   number of new samples from every robot;
5. recorder wiring uses all initialized robots and the post-restore phase;
6. existing robot-health, current-pose rearm, episode-attempt, and launcher
   tests remain green.

No automated verification may start ROS nodes, publish commands, change torque,
or access a real motor bus.
