# ALOHA1 follower_left no-motion interlock

Status: `PARTIAL_SLEEP_REFERENCE_REPLAN_REQUIRED`.

`puppet_left` is the legacy ROS1 name for the current `follower_left` role.
This phase started no leader/master or follower-right driver and published no
robot command.

## Controlled GUI/headless result

The frozen Stage, command manifest, finger-limit layer, physics configuration
and trajectory were identical. Only rendering changed.

- GUI on workspace 2: `ABORTED_DEADLINE_MISS`. Frame 0 started with only
  `76 ns` skew, but GUI rendering made frame 1 `27.696179 ms` late, above the
  strict 60 Hz one-period gate of about `16.667 ms`. Only 1/2220 physics frames
  ran.
- Headless: `PASS`, 2220/2220 frames, start skew `81 ns`, maximum lateness
  `79.913 us`, no burst catch-up, and normalized signature
  `d93ae226dcb2a11a728f4abda1dc821867d1eae0893c3cc01fdb4e8113696562`.

The GUI error was therefore a synchronization-deadline failure caused by
rendering cost, not a Stage, articulation, collision or physics-model failure.
The authoritative synchronized Isaac worker must remain headless. A future GUI
observer must be decoupled from that safety-critical clock.

## Real read-only result

The 103 runtime saved 9000 `JointState` rows over `89.989079347 s`. Maximum
observed position span was `0.0030679703 rad` (or metres for prismatic DOFs),
and first and last vectors are identical. Both arm and gripper command topics
had no publisher; diagnostic command count is zero.

The exact cam_high probe saved 5400/5400 frames at 640x480/60 fps with zero
hardware resets. The scene is visible; table clutter is retained only as a
non-blocking observation because the user explicitly waived workspace
clearance as a gate.

## Newly identified initialization mismatch

The real follower is not at the frozen manifest Home state. Its maximum
absolute arm error from Home is `1.5837285042 rad`; its maximum error from the
selected historical Sleep is `0.3137285042 rad`. The digital Stage, however,
is initialized at Home.

Publishing the first Home sample directly would hide a large real preposition
move inside the first synchronized frame. That is rejected. The user selected
the corrected sequence: both sides initialize at the same Sleep state and then
execute three formal `Sleep -> Home -> Sleep` cycles.

The runtime-measured Sleep reference is the per-joint median of the 9000
stationary real samples:
`[0, -1.845378995, 1.622951746, -0.006135923, -1.883728504,
-0.006135923] rad`. It is a runtime reference, not the historical official
Sleep vector. The digital follower must be initialized from this frozen value,
and the real readback must still be within a predeclared tolerance before a
publisher may be created. Real motion requires a fresh explicit authorization.

The 90-second real capture overlapped only the first approximately `4.69 s` of
the controlled headless run, so this report does not claim a complete live
three-cycle comparison. No final/default asset was changed. Task 8 remains
`COMPLETE_WITH_NO_PROMOTION`.
