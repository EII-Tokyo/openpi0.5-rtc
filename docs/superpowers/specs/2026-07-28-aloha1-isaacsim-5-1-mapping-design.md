# Stationary ALOHA 1 to Isaac Sim 5.1 Mapping Design

## Status and Authority

This design implements the user's 2026-07-28 specification. That specification
is the approved functional design and explicitly authorizes ordinary
non-hardware implementation steps without additional confirmation. Project
`AGENTS.md` and `docs/agents/isaac_mcp_toolchain.md` remain the controlling
safety rules.

## Scope

The system maps Stationary ALOHA 1 into a reproducible Isaac Sim 5.1.0 asset
family. The minimum runnable scene contains two independent
`aloha_vx300s` follower articulations. `aloha_wx250s` leaders are generated and
importable but excluded unless `ENABLE_LEADERS` is enabled. The workcell,
physics, cameras, observations, and validation data are layered so that source
assets remain immutable and missing measurements cannot be confused with
calibrated values.

## Evidence Model

Every datum uses exactly one provenance class:

- `official_source`: directly read from Isaac Sim 5.1, Trossen/Interbotix
  source, or another pinned upstream source.
- `project_reuse`: read from an existing repository report or asset and
  validated for compatibility with this mapping.
- `measured`: supplied by or measured from the real Stationary ALOHA 1.
- `derived`: deterministically calculated from cited inputs.
- `engineering_inference`: a reversible design decision backed by evidence but
  not a measurement.
- `temporary_placeholder`: an interface-enabling value that cannot be used for
  calibration or sim-to-real claims.
- `hard_blocker`: required real or proprietary information that is unavailable
  and cannot be derived.

Machine reports carry source path, repository URL, branch or tag, commit,
license, content hash, access date, and provenance class where applicable.

## Asset Architecture

Each robot family uses an immutable source/import layer and separate feature
layers:

1. `source/` contains pinned URDF and resolved source resources.
2. the imported base USD preserves the importer result without manual physics
   retuning;
3. configuration/physics layers hold drive, collision, material, and variant
   changes;
4. a final robot USD composes the base and selected feature layers;
5. `aloha1_workcell.usd` references two follower assets and optionally the
   leaders, while table, frame, camera mounts, pipe, and bottle remain separate
   workcell prims/assets.

No joint, visual mesh, or collision mesh merging occurs before baseline
validation. Each robot remains one independent articulation.

## Reproducible URDF Pipeline

A single checked-in shell entry point reads
`configs/aloha1_xacro_args.yaml`, verifies all pinned repositories and commits,
constructs the ROS package path, and expands four Xacros using the same
ALOHA bringup argument names. It then resolves all `package://` and Xacro
dependencies to accessible local resources and refuses to publish a URDF with
an unresolved mesh.

The URDF auditor parses XML without implicit alphabetical sorting. It reports
tree structure, explicit source joint order, limits, dynamics, mimic
relationships, gripper definitions, and mesh hashes into JSON and CSV.

## Isaac Sim 5.1 Import Pipeline

The standalone importer starts Isaac Sim headless, enables the exact installed
5.1 extension IDs, and uses only APIs confirmed in the installed
`isaacsim.asset.importer.urdf` source/examples. Initial settings are static
base, mimic preserved, collision-from-visuals disabled when collision geometry
exists, self-collision disabled, and no aggressive fixed-joint or mesh merge.
The importer records every configuration property and a verbose log.

The import is idempotent: the input hashes and importer configuration determine
the output identity, staging uses a temporary destination, validation precedes
replacement, and reruns must yield the same semantic inventory even if binary
USD serialization metadata differs.

## Mapping and Control

Joint order is independently extracted from generated URDF, Isaac runtime
articulation DOFs, and the active ALOHA control/dataset code. The joint map
stores all three explicit orders plus prefixes, sign, offset, limits,
effort/max-force, mimic data, and gripper raw/normalized transforms. Any
unresolved sign or offset is a failing or partial field, never silently zero.

The one-joint-at-a-time harness drives only one active DOF per case, records all
readbacks and curve samples, and verifies moved index, direction, range, and
return-to-initial state.

## Physics and Gripper

The source dynamics audit never treats PhysX default density as calibrated
dynamics. Missing mass, center of mass, or inertia is recorded in
`missing_dynamics.json`. The `debug_acceleration_drive` variant validates
interfaces while dynamics remain uncalibrated. The
`sim2real_force_drive` variant is present but cannot receive a sim-to-real PASS
until measured dynamics and controller mapping exist.

Mimic joints have no competing active drive. Finger-tip material values must
come from cited measurements/configuration or be marked
`temporary_placeholder`. Gripper contact tests distinguish geometry/contact
functionality from real friction fidelity.

## Workcell and Cameras

The workcell is reference-composed. Measurements already recorded in project
evidence may be reused only after compatibility checks. Missing base/table,
pipe, bottle, or camera calibration becomes a structured `hard_blocker`; it
does not stop articulation, schema, or interface work.

Four logical camera names always exist. Each camera config independently tracks
resolution, intrinsics, distortion/cropping, frame rate, and mounting
calibration status. Uncalibrated cameras use `calibration_pending` and cannot
be labeled final. Observation keys and ordering are explicitly matched to the
training/runtime camera schema.

## Validation and Status

`tools/validate_aloha1_asset.py` emits deterministic machine reports and a
top-level status of `PASS`, `FAIL`, or `PARTIAL`. It checks references,
articulations, roots, DOF order, drives/mimic, finite limits/dynamics, initial
targets, first-frame motion, static hold, joint direction/range, gripper,
collisions, reference poses, and the three requested Isaac validation rule
families.

`PASS` means every required non-blocked baseline check passed. `PARTIAL` is used
only when executable checks pass but explicitly enumerated real-measurement or
proprietary blockers prevent the requested fidelity claim. `FAIL` identifies
one or more reproducible failed checks. Viewport appearance is never an
acceptance signal.

## Optimization Gate

Optimization work begins only after the unoptimized baseline validation passes.
The same regression suite compares joint tree, DOF order, control interface,
collision metrics, and measured performance before and after optimization.
