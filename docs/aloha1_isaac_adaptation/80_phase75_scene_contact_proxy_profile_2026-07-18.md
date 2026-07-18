# Phase 75 Scene Contact Proxy Profile

## Question

Phase 74 blocked a dangerous path: using a `/scene` calibrated overlay while the passive contact validator still read legacy `/puppet_*` proxy paths.

The next question was whether the current user-confirmed Trossen/Menagerie ALOHA stage can use one coherent `/scene` contact profile:

```text
articulation roots
fingertip bbox proxies
finger DOF names
stage units and up axis
```

## Finding

The confirmed GUI startup stage contains `/scene` articulation roots and finger link rigid bodies:

```text
/scene/left_base_link/left_base_link
/scene/right_base_link/right_base_link
/scene/left_base_link/left_left_finger_link
/scene/left_base_link/left_right_finger_link
/scene/right_base_link/right_left_finger_link
/scene/right_base_link/right_right_finger_link
```

Its finger DOF names are not the legacy names:

```text
legacy clean runtime:
left_finger
right_finger

scene_base_link:
left_left_finger
left_right_finger
right_left_finger
right_right_finger
```

Therefore the contact validator needs a profile, not a single global `FINGER_PROXY_PATHS` constant.

## Implementation

Added a shared contact proxy profile module:

```text
aloha_isaac_replay/validation/contact_proxy_profiles.py
```

It defines:

```text
legacy_puppet
scene_base_link
```

Each profile owns:

```text
robot roots
finger proxy paths
finger DOF aliases
stage units
stage up axis
```

Updated:

```text
aloha_isaac_replay/scripts/build_aloha1_bbox_proxy_runtime_stage.py
aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py
aloha_isaac_replay/scripts/validate_aloha1_gripper_proxy_gap.py
aloha_isaac_replay/scripts/create_calibrated_table_base_overlay.py
```

Key behavior:

```text
--contact-proxy-profile legacy_puppet
```

keeps the older `/puppet_*` clean-runtime path.

```text
--contact-proxy-profile scene_base_link
```

uses the user-confirmed `/scene` stage, Isaac meters, and `Z` up.

The calibrated overlay manifest now records the `/scene` validation command with:

```text
--contact-proxy-profile scene_base_link
```

## Runtime Evidence

The `/scene` bbox proxy builder was run headless against:

```text
local_eval_assets/aloha_isaac_menagerie_deep_black_real_start_pose/aloha2_menagerie_scene_deep_black_real_start_pose.usd
```

Result:

```text
status: PASS
contact_proxy_profile: scene_base_link
stage_units_in_meters: 1.0
stage_up_axis: Z
selected_proxy_count: 4
```

Generated proxy paths:

```text
/scene/left_base_link/left_left_finger_link/bbox_collision_proxy
/scene/left_base_link/left_right_finger_link/bbox_collision_proxy
/scene/right_base_link/right_left_finger_link/bbox_collision_proxy
/scene/right_base_link/right_right_finger_link/bbox_collision_proxy
```

Artifact:

```text
.codex/artifacts/20260718-212613_scene-bbox-proxy-build-metadata
```

The generated temporary USD was inspected and confirmed:

```text
metersPerUnit = 1
upAxis = "Z"
```

## Passive Contact Smoke

A short non-final passive contact smoke was run on the generated `/scene` proxy stage:

```text
--stage-units-in-meters 1.0
--contact-proxy-profile scene_base_link
--settle-steps 2
--close-steps 2
```

Result:

```text
status: PASS
overall_pass: True
contact_proxy_profile: scene_base_link
```

Artifact:

```text
.codex/artifacts/20260718-212946_scene-passive-contact-smoke-dof-profile
```

This PASS means:

```text
the /scene articulation can be loaded
the /scene fingertip proxy paths exist
the scene finger DOF aliases work
short-step physics did not explode
```

It does not mean:

```text
Bottle500 grasp is validated
finger-object contact pairs were found
friction/material realism is validated
full-arm replay is validated
the calibrated table contact gate is complete
```

## Current Blocker

The profile and namespace problem is now addressed for `/scene`.

The next blocker is stricter physics validation:

```text
same /scene profile
calibrated table/base transform
support-plane/table collider
bottle collider
contact-pair tracing enabled
HDF5 replay or controller target path
```

Only after those are in the same profile should the project claim final bottle/table/gripper contact validity.

## Validation

Validated locally:

```text
.venv/bin/python -m pytest -q \
  aloha_isaac_replay/tests/test_contact_proxy_profiles.py \
  aloha_isaac_replay/tests/test_bbox_proxy_runtime_stage_builder.py \
  aloha_isaac_replay/tests/test_passive_contact_csv_writer.py \
  aloha_isaac_replay/tests/test_calibrated_table_base_overlay.py
```

Result:

```text
23 passed
```

No real robot or `192.168.1.103` control command was used.
