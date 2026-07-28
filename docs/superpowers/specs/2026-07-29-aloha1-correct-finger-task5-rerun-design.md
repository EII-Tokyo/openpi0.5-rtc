# ALOHA1 Correct-Finger Task 5 Rerun Design

Date: 2026-07-29

## Goal

Restart the follower-gripper validation at
`TASK5_PREFLIGHT_CORRECT_FINGER_ASSET_IDENTITY_AND_INSTALL_TRANSFORM` using the
user-confirmed Stationary ALOHA 1 custom fingers. Every critical test phase
must emit machine-audited screenshots whose absolute paths are recorded in
the reports for later user review.

This work remains limited to the follower gripper and the 20 g bottle proxy.
It does not continue workcell, camera-calibration, ROS, insertion, or Task 8
optimization.

## Frozen Version Boundary

- Isaac Sim: `5.1.0.0`
- Kit: `107.3.3`
- PhysX: `107.3.26`
- Python: `3.11.13`
- Camera API:
  `isaacsim.sensors.camera.Camera` from the installed Isaac Sim 5.1 tree

The official NVIDIA Isaac documentation MCP was queried before this design.
It confirms that standalone Python controls physics/render steps explicitly
and that the supported camera flow is `Camera.initialize()`,
`world.step(render=True)`, and `Camera.get_rgba()`. The installed 5.1 source
confirms those exact methods and signatures.

## Correct Finger Provenance

The custom-finger source is fixed to:

- repository: `https://github.com/huggingface/gym-aloha.git`
- release branch:
  `user/aliberts/2024_05_07_remove_upper_bounds`
- version-introducing commit:
  `51837ba5f7d5b96255f01c3d39d53dea473b4829`
- local path:
  `/home/eii/project/openpi0.5-rtc-reward-learning/external/gym-aloha`
- license: Apache-2.0
- left STL SHA-256:
  `df73ae5b9058e5d50a6409ac2ab687dade75053a86591bb5e23ab051dbf2d659`
- right STL SHA-256:
  `56fb3cc1236d4193106038adf8e457c7252ae9e86c7cee6dabf0578c53666358`

The repository is checked out detached at the fixed commit. The installed
`gym-aloha==0.1.1` copies have the same two mesh hashes. The rejected generic
856-triangle finger
`a4baacd9a64df1be60ea5e98f50f3c660e1b7a1fe9684aace6004c5058c09483`
must not appear in the new diagnostic asset or runtime contact paths.

The fixed MJCF provides the local geometry evidence:

- mesh scale: `[0.001, 0.001, 0.001]`;
- left geometry: `pos=[0.005, -0.052, 0]`,
  `euler=[3.14, 1.57, 0]`;
- right geometry: `pos=[0.005, +0.052, 0]`,
  `euler=[3.14, 1.57, 0]`;
- left slide range: `[+0.021, +0.057] m`;
- right slide range: `[-0.057, -0.021] m`;
- both slide axes: `[0, 1, 0]`.

These are source-confirmed MJCF values, not measurements inferred from
screenshots.

## Selected Architecture

### Isolated replacement layer

Create a new diagnostic subtree:

```text
assets/Trossen/ALOHA1/1.0/diagnostics/gripper_correct_finger/
  source/
  convex_hull/
  convex_decomposition/
  screenshots/
```

The diagnostic asset references the existing follower debug configuration.
It deactivates only the existing generic left/right finger visual and
collision meshes, then authors the correct left/right custom meshes using the
fixed source transforms. It does not edit the original URDF, imported source
USD, drive configuration, or final/default collider.

The implementation must read back the final runtime prim paths, mesh hashes,
transforms, collision approximation tokens, DOF names/order, joint limits,
and legal initial q. A rejected generic mesh reference or illegal `q=0`
causes preflight `FAIL`.

### Why the alternatives are rejected

Using the historical converted USD directly would bypass the current follower
articulation/control mapping. Re-importing a complete derived URDF would mix
the finger correction with a full importer rerun and change more than one
experimental input. Both remain useful comparison evidence but are not the
Task 5 baseline.

## Screenshot Contract

Screenshots are required outputs, not optional debug artifacts.

Every capture record must contain:

```json
{
  "capture_name": "closed_isometric",
  "phase": "asset_preflight",
  "status": "PASS",
  "absolute_path": "/absolute/path/image.png",
  "sha256": "64-hex-digits",
  "pixel_sha256": "64-hex-digits",
  "width": 1280,
  "height": 900,
  "camera_pose": {
    "position": [0.0, 0.0, 0.0],
    "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]
  },
  "simulation_state": {
    "physics_step": 0,
    "left_finger_q_m": 0.021,
    "right_finger_q_m": -0.021
  }
}
```

The required phases and views are:

| Phase | Required captures |
| --- | --- |
| `asset_preflight` | legal open and legal closed; closing-axis and isometric |
| `collider_geometry` | Hull and Decomposition; overview and inner-surface close-up |
| `bilateral_contact` | first verified bilateral contact; closing-axis and isometric |
| `release_hold` | release frame and hold-end/drop frame; closing-axis and isometric |

The runtime captures use the installed Isaac Sim 5.1 `Camera` API. Static
cooked-piece close-ups may use the existing deterministic Blender renderer,
but are labeled `STATIC_GEOMETRY_SUPPLEMENT`.

For every required screenshot:

- the path must be absolute and inside the diagnostic artifact root;
- the file must exist and be a non-empty PNG;
- the decoded image must be exactly `1280 × 900`;
- RGB variance must exceed a minimum blank-image threshold;
- file and pixel hashes must be present;
- the capture must be linked to an exact test phase and state;
- a missing or invalid capture makes that phase `FAIL`.

All records are aggregated into:

```text
reports/aloha1_mapping/gripper_correct_finger_screenshot_manifest.json
```

Every phase report also embeds its own screenshot records so the user does not
need to infer paths from filenames.

## Test Sequence

1. Provenance and asset-identity preflight.
2. Legal open/closed articulation-state readback.
3. Visual/collision path and transform audit.
4. Initial-overlap audit.
5. Correct-finger Convex Hull cooking and geometry screenshots.
6. Correct-finger Convex Decomposition cooking and geometry screenshots.
7. Frozen A/B geometry and runtime comparison.
8. Task 5 motion, mimic, aperture, penetration, bilateral contact, and hold.
9. Only if Task 5 still fails: repeat the v2 force diagnosis.
10. Rerun Task 7 after corrected Task 5. Keep Task 8 `NOT_RUN`.

No friction, drive, mimic, bottle mass, timestep, or decomposition parameter
is changed while comparing Hull with Decomposition.

## Reports

The rerun produces new reports and never overwrites the historical generic
finger reports:

```text
reports/aloha1_mapping/gripper_correct_finger_preflight.json
reports/aloha1_mapping/gripper_correct_finger_screenshot_manifest.json
reports/aloha1_mapping/gripper_correct_finger_collider_comparison.json
reports/aloha1_mapping/gripper_correct_finger_ab_results.json
reports/aloha1_mapping/gripper_correct_finger_task5.json
reports/aloha1_mapping/gripper_correct_finger_restart_summary.json
```

Each report uses literal `PASS`, `FAIL`, `PARTIAL`, or `NOT_RUN`. It records
input hashes, absolute screenshot paths, and the exact artifact directory.

## Error Handling

- Source hash, commit, or license mismatch: `FAIL` before USD authoring.
- Existing target diagnostic directory with a different manifest:
  stop without overwriting.
- Generic finger reference in the new composed stage: `FAIL`.
- Illegal initial finger q or stale authored `q=0`: `FAIL`.
- Missing runtime Camera frame after a bounded render warm-up: `FAIL`.
- Empty/blank/invalid screenshot: `FAIL`.
- Source/final asset hash change: `FAIL`.
- Contact or hold failure: record the physical result; do not tune another
  variable in the same run.

## Acceptance Boundary

This design is accepted for implementation by the user's instruction to add
mandatory screenshots, report their locations, and continue from the recorded
restart boundary. The user-confirmed visual orientation gate remains `PASS`,
but Task 5 and all downstream hold conclusions remain `RE-RUN REQUIRED` until
the correct-finger reports complete.
