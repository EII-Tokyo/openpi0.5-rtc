# ALOHA Isaac Sim / Isaac Lab Minimal Workcell

This folder is the first Isaac route scaffold for the bottle-mouth insertion
simulation work. It is intentionally small: first create a visible workcell with
table frame, pipe axis, cameras, and an optional ALOHA USD reference; then add
real calibration and Isaac Lab task logic step by step.

## Current Scope

- Check whether the active Python environment can import Isaac Sim / Isaac Lab.
- Create a minimal USD workcell stage from `config/workcell_minimal.yaml`.
- Use Trossen's official Stationary AI dual-arm USD for the user-measured
  workcell when `external/trossen_ai_isaac` is available.
- Optionally convert the local Gym ALOHA MJCF model to USD when Isaac Lab is
  installed for legacy experiments.
- Keep all measured real-world parameters in one YAML file so table origin,
  robot base pose, pipe pose, and camera hints can be iterated safely.

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

## Build the Minimal Stage

After Isaac Sim is installed:

```bash
python3 examples/aloha_isaac/scripts/create_basic_workcell_stage.py \
  --config examples/aloha_isaac/config/workcell_minimal.yaml
```

Expected output:

```text
usd=/abs/path/local_eval_assets/aloha_isaac_minimal/aloha_workcell.usd
```

To build the current user-measured first-pass workcell, use:

```bash
git clone --depth 1 https://github.com/TrossenRobotics/trossen_ai_isaac.git external/trossen_ai_isaac
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
- `/World/Aloha/StationaryAI` if Trossen's official `stationary_ai.usd` exists

If no ALOHA USD exists yet, the script places simple base-hint boxes only when
the config provides robot layout hints. The user-measured config intentionally
uses the official Stationary AI asset rather than hand-placing left/right arms.

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
5. The official Stationary AI asset loads under `/World/Aloha/StationaryAI`.
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
