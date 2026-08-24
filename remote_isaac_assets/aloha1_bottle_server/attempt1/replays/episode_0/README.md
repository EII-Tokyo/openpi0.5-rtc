# Episode 0 labeled replay

This bundle replays all 918 frames of `2026-05-11_twist/episode_0` at 50 Hz.
It teleports the recorded ALOHA arm/finger state and moves Bottle500/BottleCap
kinematically using the four authoritative manual-label intervals.

The result is classified as `KINEMATIC_VISUAL_REPLAY_NOT_PHYSICS_ACCEPTANCE`.
It is intended to verify data/label/scene alignment; it does not prove a stable
dynamic grasp, threaded contact, or successful physics-based uncapping.

## In the streaming Isaac Sim session

Open `remote_stream_cap_stage.usda`, then run this file in Script Editor:

`isaac_script/replay_episode0_labeled.py`

The script creates an anonymous session layer, does not save the Stage, does not
use ROS, and pauses the timeline after frame 917. Running it again cancels an
older replay task, reuses the existing replay layer, rebuilds the PhysX view at
a stopped-timeline boundary, and restarts from frame 0. The Script Editor entry resolves both the `hxz` source
path and the `aloha` runtime-bundle path when Isaac executes a temporary copy.

If the USD Stage changes or a PhysX/articulation view is invalidated during the
run, replay stops at the first affected frame, pauses the timeline, restores the
previous USD edit target, and prints one `[Episode0Replay] FAIL` summary rather
than continuing through the remaining frames with a stale tensor view.
Bottle500 and BottleCap are configured Kinematic in that anonymous layer before
the first PhysX Play, so their rigid-body views are not initialized as Dynamic
and then changed underneath the tensor backend.

## Headless validation

```bash
/home/eii/Applications/isaacsim-5.1.0/python.sh \
  remote_isaac_assets/aloha1_bottle_server/attempt1/replays/episode_0/validate_episode0_kinematic_replay.py
```
