# Phase 60 Measured Table Candidate

## Question

Phase 59 proved that the already-grasped HDF5 replay remains well tracked under gravity with a low local support surface, but the support surface was still a square patch.

Phase 60 asks:

If the support is changed to the user's measured desktop footprint, does the already-grasped gravity replay remain stable, and what contacts does the table candidate create?

## Implementation

The passive-contact validator now supports rectangular support dimensions:

```text
--support-plane-size-x
--support-plane-size-y
```

The old square option remains supported:

```text
--support-plane-size
```

Phase 60 used the user's measured desktop dimensions:

```text
length = 1.220 m
width  = 0.625 m
```

The user-measured Isaac workcell config was also updated from the older `1.10 m x 0.60 m` table to:

```text
examples/aloha_isaac/config/workcell_user_measured.yaml
table.size = [1.22, 0.625, 0.04]
```

The pipe derivation now follows from that updated table:

```text
A = (-0.03, 0.3125, 0.0)
pipe start = (-0.03, 0.4075, 0.07)
pipe end = (-0.1919, 0.4075, 0.2263)
```

## Command Artifact

```text
.codex/artifacts/20260718-144519_phase60-gravity-start143-measured-table-candidate
```

Structured report:

```text
reports/aloha1_isaac_adaptation/phase60_gravity_start143_measured_table_candidate_20260718/gripper_passive_contact_metrics.json
```

## Result

| Check | Result |
| --- | --- |
| status | `PASS` |
| contact trace status | `PASS_BILATERAL_CONTACT_CANDIDATE` |
| object close displacement | `0.0227964` stage units |
| max object displacement | `0.0561948` stage units |
| support/table size | `[1.22, 0.625, 0.04]` |
| wrong unique contact pairs | `12` |

Tracking:

| Group | Max abs error | Mean max abs error | Final max abs error |
| --- | ---: | ---: | ---: |
| left arm | `0.0213716` | `0.0084204` | `0.0170583` |
| gripper | `0.0323760` | `0.0195057` | `0.0162378` |
| controlled | `0.0323760` | `0.0204382` | `0.0170583` |

Diagnostic contact classification:

| Diagnostic prim | Total contact rows | Object rows | Expected-finger rows | Other rows |
| --- | ---: | ---: | ---: | ---: |
| `/World/phase58_static_support_plane` | `800` | `770` | `30` | `0` |

The remaining finger/table contact pair is:

```text
/World/phase58_static_support_plane
/puppet_left_vx300s/puppet_left_right_finger_link/bbox_collision_proxy
```

## Interpretation

The measured table footprint does not break the already-grasped replay tracking. The left-arm tracking error remains near the Phase 57 range.

However, the table candidate still contacts the right fingertip proxy. That means this is not yet a final trusted table pose.

The current Phase 60 table is still placed relative to the bottle's initial bounding box:

```text
--support-plane-mode object_bottom
```

That is useful for a controlled diagnostic, but it is not a true table-to-robot transform.

## Decision

The measured table dimensions are now usable in Isaac diagnostics.

But the table placement semantics are not complete. The final workcell needs a table collider in a fixed measured world pose, not a collider derived from the bottle bounding box.

## Next Gate

Phase 61 should promote the table from a bottle-relative diagnostic support to a fixed workcell object:

1. create a separate table prim with measured size `[1.22, 0.625, 0.04]`;
2. place it using a documented table frame, not object-bottom placement;
3. run the same Phase 57 already-grasped replay;
4. require low tracking error;
5. classify table-object, table-finger, and unexpected contacts separately;
6. reject the table pose if the fingertip proxy collides with the table during the replay.

