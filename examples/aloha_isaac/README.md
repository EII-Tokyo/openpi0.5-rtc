# ALOHA Isaac Sim / Isaac Lab Workcell

This folder is the first Isaac route scaffold for the bottle-mouth insertion
simulation work. It is intentionally small: first create a visible workcell with
table frame, pipe axis, cameras, and an optional ALOHA USD reference; then add
real calibration and Isaac Lab task logic step by step.

## Confirmed GUI Startup

For normal GUI startup, use the confirmed ALOHA scene through the default entry
point. Do not pass `--usd`:

```bash
OMNI_KIT_ACCEPT_EULA=YES .venv_issac/bin/python \
  examples/aloha_isaac/scripts/open_workcell_gui.py
```

This opens:

```text
local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/aloha2_menagerie_scene_deep_black_real_start_pose.usd
```

Other `local_eval_assets/aloha_isaac*` stages are intermediate or scratch
assets. The GUI launcher rejects them by default; pass
`--allow-noncanonical-usd` only for an explicit experiment.

On this workstation the launcher also moves the Isaac Sim window to workspace 2
by default so it does not cover the user's active desktop. Override only when
needed:

```bash
OMNI_KIT_ACCEPT_EULA=YES .venv_issac/bin/python \
  examples/aloha_isaac/scripts/open_workcell_gui.py \
  --no-move-to-startup-workspace
```

## Current Scope

- Check whether the active Python environment can import Isaac Sim / Isaac Lab.
- Create a minimal USD workcell stage from `config/workcell_minimal.yaml`.
- Keep the user-measured workcell explicit about calibration status: table,
  pipe, measurement markers, and camera hints come from the user's measurements;
  ALOHA is loaded as visual context, but its base placement is still approximate.
- Optionally convert the local Gym ALOHA MJCF model to USD when Isaac Lab is
  installed for legacy experiments.
- Keep all measured real-world parameters in one YAML file so table origin,
  robot base pose, pipe pose, and camera hints can be iterated safely.
- Derive the user-measured pipe placeholder from explicit table-edge
  measurements instead of opaque `start` / `end` coordinates.

This is not yet a trusted physics training environment. It is the stage-0
visual and asset-integration skeleton.

## Install Check

Run:

```bash
python3 examples/aloha_isaac/scripts/check_isaac_install.py
```

The current repository Python environment is allowed to fail this check. Isaac
Sim is a separate heavy runtime and should normally live in its own Python 3.11
environment.

## Build Scratch Stages

The following commands are for scratch stage generation and inspection. They are
not the normal GUI startup path.

After Isaac Sim is installed:

```bash
python3 examples/aloha_isaac/scripts/create_basic_workcell_stage.py \
  --config examples/aloha_isaac/config/workcell_minimal.yaml
```

Expected output:

```text
usd=/abs/path/local_eval_assets/aloha_isaac_minimal/aloha_workcell.usd
```

To build the current user-measured first-pass table/pipe workcell, use:

```bash
python3 examples/aloha_isaac/scripts/create_basic_workcell_stage.py \
  --config examples/aloha_isaac/config/workcell_user_measured.yaml
```

Expected output:

```text
usd=/abs/path/local_eval_assets/aloha_isaac_user_measured/aloha_workcell_user_measured.usda
```

Open that USD in Isaac Sim to inspect:

- `/World/Table`
- `/World/TableFrameT`
- `/World/PipePlaceholder`
- `/World/Cameras/cam_low`
- `/World/Cameras/cam_right_wrist_hint`
- `/World/Aloha/LeftFollowerVx300s`
- `/World/Aloha/RightFollowerVx300s`

The user-measured pipe in `workcell_user_measured.yaml` is intentionally
defined from measurement facts:

- table size `1.22 m x 0.625 m`
- current water-pipe component is on the `w1` side, at `y = 0.3125 m` when
  the table center is the world origin
- A point `0.58 m` from the left table edge, so `A = (-0.03, 0.3125, 0.0)`
- pipe base center `0.095 m` outside the `w1` edge
- pipe length `0.225 m`, diameter `0.005 m`, mount height `0.07 m`, side tilt `44 deg`
- in top view, the pipe axis is parallel to the table edge and points toward
  the left arm, so its x coordinate decreases while y stays fixed

The stage generator derives the visible pipe axis and also adds:

- `/World/PipePlaceholder/measurement_A_on_w1_edge`
- `/World/PipePlaceholder/measurement_9p5cm_base_offset`

Use these blue markers in Isaac Sim to verify that the 9.5 cm table-edge
constraint is visually correct before using the pipe for any reward logic.

The user-measured config loads ALOHA arms as visual context. The current trusted
measurements define the table and pipe; the real left/right ALOHA base poses
still need measurement before this scene can be used for reward or collision
judgment.

## Convert the Local ALOHA MJCF to USD

After Isaac Lab is installed:

```bash
python3 examples/aloha_isaac/scripts/convert_mjcf_to_usd.py
```

Default input:

```text
local_eval_assets/aloha_workcell_minimal/model/workcell.xml
```

Default output:

```text
local_eval_assets/aloha_isaac_assets/aloha_viperx.usd
```

The generated USD path is the same path referenced by
`config/workcell_minimal.yaml`. Do not use this converted single-arm asset for
the user-measured Stationary AI scene; use Trossen's official
`assets/robots/stationary_ai/stationary_ai.usd` instead.

## First Validation Checklist

1. The USD opens in Isaac Sim without missing references.
2. The table frame `T` is visible and not hidden under the table.
3. `cam_low` points at the bottle-mouth / pipe placeholder region.
4. `cam_right_wrist_hint` points at the same region.
5. ALOHA appears as visual context, but its base pose is treated as approximate.
6. Joint names, joint order, and limits are checked before any replay or RL.

Do not start reward training from this skeleton until real table, base, camera,
pipe, bottle, and gripper parameters are measured and replay validation passes.

## Deprecated 103 Site Parameters

A previous attempt built an Isaac stage by copying read-only parameters from a
separate 103 `aloha_rally` site model.  That stage was not reliable enough for
this user's real workcell and has been removed.

Do not regenerate an Isaac training model from those 103 site parameters.  The
next usable model should be built from fresh measurements taken on the user's
actual table, pipe, bottle, cameras, and ALOHA base positions.
