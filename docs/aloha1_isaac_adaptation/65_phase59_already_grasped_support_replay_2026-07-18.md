# Phase 59 Already-Grasped Replay With Support

## Question

Phase 58 showed that adding a local support plane under the bottle can prevent frame-0 gravity failure, but it also creates support-related contacts and large early left-arm tracking error.

Phase 59 asks:

If the same low support plane is used only after the replay starts from the already-grasped frame 143, does it still corrupt arm tracking?

## Command Artifact

```text
.codex/artifacts/20260718-143902_phase59-gravity-start143-low-support-bottle-usd-hdf5-replay
```

Structured report:

```text
reports/aloha1_isaac_adaptation/phase59_gravity_start143_low_support_bottle_usd_hdf5_replay_20260718/gripper_passive_contact_metrics.json
```

## Result

| Check | Result |
| --- | --- |
| status | `PASS` |
| contact trace status | `PASS_BILATERAL_CONTACT_CANDIDATE` |
| both expected fingers contacted object | true |
| wrong contact pairs | 12 |
| object close displacement | `0.0226009` stage units |
| max object displacement | `0.0561392` stage units |

Support plane:

| Field | Value |
| --- | --- |
| path | `/World/phase58_static_support_plane` |
| center | `[0.5932279, 0.7853100, -0.2396451]` |
| size xy | `0.24` |
| thickness | `0.005` |

Tracking summary:

| Group | Max abs error | Mean max abs error | Final max abs error |
| --- | ---: | ---: | ---: |
| left arm | `0.0213327` | `0.0084229` | `0.0170798` |
| gripper | `0.0323760` | `0.0194853` | `0.0161136` |
| controlled | `0.0323760` | `0.0204276` | `0.0170798` |

## Comparison With Earlier Gates

| Gate | Start semantics | Support | Left-arm max abs error | Max object displacement |
| --- | --- | --- | ---: | ---: |
| Phase 55 | frame 0, zero gravity | none | `0.02046` | bounded |
| Phase 56 | frame 0, gravity | none | `0.02511` | `72.9293` |
| Phase 57 | frame 143, gravity | none | `0.02252` | `0.4069` |
| Phase 58c | frame 0, gravity | low local support | `0.37961` | bounded |
| Phase 59 | frame 143, gravity | low local support | `0.02133` | `0.0561` |

Phase 59 restores the low tracking error seen in Phase 55 and Phase 57.

This means the large Phase 58c tracking error is not caused by the support plane alone. It is caused by the support plane interacting with the open-gripper, pre-grasp part of the replay.

## Remaining Problem

Phase 59 still records support-related wrong contact pairs.

One pair includes the right fingertip proxy:

```text
/World/phase58_static_support_plane
/puppet_left_vx300s/puppet_left_right_finger_link/bbox_collision_proxy
```

Therefore the support plane still cannot be treated as a final real table model.

## Phase 59b Contact Classification

The validator was then extended to classify diagnostic contacts for support or table prims. This does not change the pass/fail rule; it only makes the extra contacts interpretable.

Command artifact:

```text
.codex/artifacts/20260718-144155_phase59b-contact-classification-start143-low-support
```

Structured report:

```text
reports/aloha1_isaac_adaptation/phase59b_contact_classification_start143_low_support_20260718/gripper_passive_contact_metrics.json
```

Result:

| Field | Value |
| --- | ---: |
| status | `PASS` |
| contact trace status | `PASS_BILATERAL_CONTACT_CANDIDATE` |
| left-arm max abs error | `0.0213327` |
| wrong contact pair count | `12` |
| support contact row count | `800` |
| support-object contact row count | `770` |
| support-expected-finger contact row count | `30` |
| support-other contact row count | `0` |

This confirms that the local support plane is not touching arbitrary unrelated geometry. The remaining artificial contact is specific: support-object contact dominates, with some support-finger contact.

That is better evidence than the old undifferentiated `wrong_contact_pairs` list.

## Conclusion

Phase 59 strengthens the current interpretation:

1. ALOHA1 left-arm and gripper tracking remains valid when the replay starts from the already-grasped HDF5 state.
2. The frame-0 gravity failure is primarily an initialization and support problem, not a joint mapping problem.
3. The object-bottom support shortcut is useful for diagnosis but still creates artificial contacts.
4. A final workcell model needs a real table collider in a measured world pose, not a support patch directly tied to the bottle bounding box.

## Decision

Keep Phase 57 as the clean gravity-on local contact gate.

Use Phase 59 as supporting evidence that already-grasped replay plus gravity can remain well tracked even with a nearby support surface.

Do not promote the local support plane to final workcell semantics.

## Next Gate

The next implementation step should add a separate table collider with measured dimensions and pose:

```text
table length = 1.220 m
table width  = 0.625 m
```

The gate should verify:

1. the table collider exists as a separate static prim;
2. the table does not contact the gripper proxies during Phase 57 replay;
3. the bottle remains bounded under gravity;
4. the left-arm tracking error stays near the Phase 57 range;
5. all extra contact pairs are explicitly classified as table-object, table-gripper, or unexpected.
