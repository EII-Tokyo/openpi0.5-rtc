# Thread/Release v1

This is a versioned Isaac Sim 5.1 checkpoint of the BottleCap right-hand
thread and explicit `THREADED -> RELEASED` design.

- Entry Stage: `remote_stream_threaded_release_v1.usda`
- Bottle500 startup mode: Dynamic (`physics:kinematicEnabled = false`)
- BottleCap startup mode: Dynamic (`physics:kinematicEnabled = false`)
- Initial state: `THREADED`
- `THREADED`: Prismatic locked at `0 m`, Revolute locked at `0 deg`, and
  Rack-and-Pinion coupling disabled. Ordinary Play therefore cannot start an
  unthreading motion, and no angular Drive is authored.
- `UNTHREADING`: the explicit transition script opens Prismatic to
  `0–0.012 m`, removes the Revolute limits, and enables the coupling.
- `RELEASED`: Prismatic, Revolute, and coupling are all disabled.
- Thread pitch: `0.003 m/turn` (`TEMPORARY_UNCALIBRATED`)
- Axial travel: `0.012 m` (`TEMPORARY_UNCALIBRATED`)
- Release threshold: `0.01175 m` (`TEMPORARY_UNCALIBRATED`)
- Gripper-pad effective friction: static `2.0`, dynamic `1.5`, combine mode
  `maximum` (`USER_ACCEPTED_SIMULATION_EFFECTIVE_FRICTION`; Bottle lift succeeded).
- Experimental segmented-pad entry Stage:
  `remote_stream_threaded_release_segmented_pad_test_v1.usda`. It adds three
  symmetric axial collision segments per left-arm finger, keeps the accepted
  compliant-contact material, and does not replace the visible robot mesh.
  Within this test overlay only, the two supplier CAD finger collision meshes
  are disabled so the symmetric segments, recessed by `3 mm` from the measured
  supplier contact planes, are the exclusive Bottle contact surfaces. The base
  and compliant-pad Stages remain unchanged. On the reproduced difficult random
  pose `(x=0.005534 m, y=-0.191095 m, yaw=114.185 deg)`, the complete lift
  workflow passed with bilateral finger contact and no non-finger contact, but
  Bottle-axis rotation only changed from about `30.58 deg` to `30.40 deg`.
  Therefore this is retained as an isolated negative/diagnostic experiment and
  is not promoted as the Streaming Server startup Stage.
- Positive cap-local `+Z` rotation removes the cap along Bottle-local `+Z`.
- The released test state is evidence only and is not authored as startup state.
- This version is not connected to ROS or real ALOHA hardware.

This entry Stage is the current Streaming Server startup Stage. The service
loader must verify the persisted Slider, Prismatic, Revolute, right-hand
Rack-and-Pinion coupling, and `THREADED` state on every clean startup. The
loader must preserve and verify Dynamic mode for both Bottle500 and BottleCap;
it must not author a runtime Kinematic override. The unthreaded
`../../remote_stream_cap_stage.usda` remains the rollback baseline.
