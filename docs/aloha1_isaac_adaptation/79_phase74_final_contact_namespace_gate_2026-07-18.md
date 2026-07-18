# Phase 74 Final Contact Namespace Gate

## Question

Isaac expert review found a non-hardware blocker after the table/base guards:

```text
calibrated overlay namespace != contact validator proxy namespace
```

The calibrated overlay generator currently targets the user-confirmed GUI stage:

```text
/scene/left_base_link
/scene/right_base_link
```

The passive contact validator still imports legacy clean-runtime proxy paths:

```text
/puppet_left_vx300s/.../bbox_collision_proxy
/puppet_right_vx300s/.../bbox_collision_proxy
/puppet_left_vx300s/root_joint
/puppet_right_vx300s/root_joint
```

If final replay/contact validation runs with a `/scene` calibrated overlay while using `/puppet_*` finger proxies, it is not validating one coherent stage.

## Implementation

Updated:

```text
aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py
```

The preflight now inspects static stage namespace hints before Isaac startup when:

```text
--require-calibrated-table-frame
```

is enabled.

It records:

```text
stage_namespace_hints
finger_proxy_namespace_roots
```

and blocks the known unsafe combination:

```text
stage uses /scene
validator uses legacy /puppet_* FINGER_PROXY_PATHS
```

The check is intentionally conservative. It does not claim a stage is contact-ready. It only prevents a known false-positive final validation path.

## Validation

Validated locally:

```text
.venv/bin/ruff format aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py aloha_isaac_replay/tests/test_passive_contact_csv_writer.py
.venv/bin/python -m py_compile aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py
.venv/bin/python -m pytest -q aloha_isaac_replay/tests/test_passive_contact_csv_writer.py aloha_isaac_replay/tests/test_calibrated_table_base_overlay.py aloha_isaac_replay/tests/test_table_frame_candidate_audit.py
git diff --check
```

Result:

```text
23 passed
```

No real robot, `192.168.1.103`, or Isaac runtime action was used.

## Current Blocker

Final contact validation still needs a single coherent contact-capable stage/profile where all of these live in the same namespace and unit convention:

```text
articulation roots
fingertip collision proxies
calibrated table collider
bottle/object collider
```

Until then, final replay/contact validation should stop before Isaac runtime instead of producing misleading contact metrics.
