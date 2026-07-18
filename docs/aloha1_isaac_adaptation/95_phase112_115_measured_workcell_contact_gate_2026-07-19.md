# Phase 112-115: Measured Workcell Contact Gate

## Question

After adding the user-measured `/World/Table` and pipe overlay, which remaining contacts block a strict BottleUSD HDF5 replay gate, and can we obtain a pass without adding a diagnostic support plane?

## Evidence

- Phase112 runtime build: `.codex/artifacts/20260719-012108_aloha-phase112-rebuild-measured-workcell-runtime-exact-leaves-v2`
- Phase112 prim audit: `.codex/artifacts/20260719-012122_aloha-phase112-measured-workcell-prim-audit-v2`
- Phase112 contact gate: `.codex/artifacts/20260719-012211_aloha-phase112-measured-workcell-contact-gate`
- Phase113 runtime build: `.codex/artifacts/20260719-012244_aloha-phase113-rebuild-measured-workcell-no-gripper-base-bar-runtime`
- Phase113 contact gate: `.codex/artifacts/20260719-012329_aloha-phase113-no-gripper-base-bar-contact-gate`
- Phase114 offset probe summary: `reports/aloha1_isaac_adaptation/phase114_object_offset_probe_20260719/summary.json`
- Phase115 strict measured-workcell gate: `.codex/artifacts/20260719-012958_aloha-phase115-strict-measured-workcell-no-support-plane-offset0`

## Phase112: Explicit leaf collider disabling was required

Disabling a parent prefix was not enough for composed instance-proxy collision leaves. Attempting to author directly on a composed leaf first failed because the target was under an instance proxy.

The runtime builder now de-instances the necessary source ancestors before applying explicit leaf-level `collisionEnabled = false`.

Phase112 confirmed:

| Path | Collision state |
| --- | --- |
| `/World/Table` | enabled |
| `/World/PipePlaceholder/axis` | enabled |
| `/scene/worldBody/table/collisions/table/table/table` | disabled |
| `/scene/worldBody/__22/collisions/__22/__22/extrusion_1220` | disabled |

The contact gate then failed only on:

```text
/scene/left_base_link/left_gripper_base/collisions/vx300s_7_gripper_bar/vx300s_7_gripper_bar
```

That narrowed the problem from "workcell is wrong" to "same-side gripper-base bar collider is interfering with this BottleUSD placement".

## Phase113: Gripper-base bar was isolated

Phase113 disabled the exact gripper-base bar collider in addition to the stale worldBody table/rail leaves.

The gate no longer failed on the old table/rail or gripper-base bar. It failed because the bottle fell and contacted:

```text
/scene/worldBody/floor/collisions/CollisionPlane
```

This changed the diagnosis again: controller tracking was still good, but object placement was outside the stable grasp/contact window.

## Phase114: Object offset probe

The offset probe changed only the object X offset.

| Offset X | Gate result | Contact categories | Interpretation |
| --- | --- | --- | --- |
| `-0.02` | failed | target finger + floor/workcell | too far through the grasp window; object falls |
| `0.00` | pass | target finger only | stable target contact |
| `0.04` | pass | target finger only | stable target contact |
| `0.08` | failed | target finger + floor/workcell | previous default was too far out |
| `0.12` | failed | floor/workcell and no target contact | outside usable contact window |

The important engineering point: the previous failure was not caused by the ALOHA1 controller. It was caused by placing the bottle outside the contact window for the current gripper/object geometry.

## Phase115: Strict measured-workcell pass

Phase115 reran the gate with:

- measured-workcell runtime stage;
- stale `/scene/worldBody/table` and `/scene/worldBody/__22` leaves disabled;
- stale left gripper-base bar leaf disabled;
- object center offset `0.0, 0.0, 0.0`;
- `support_plane_mode = none`;
- workcell contact policy enabled.

Result:

| Metric | Value |
| --- | --- |
| status | `PASS` |
| contact trace status | `PASS_BILATERAL_CONTACT_CANDIDATE` |
| failure reasons | `[]` |
| support plane | `none` |
| object contact categories | `target_finger` only |
| workcell contact policy | `PASS_WORKCELL_CONTACT_POLICY` |
| max controlled error | about `0.01286 rad` |
| total object displacement | about `0.1139 m` |

## Current Conclusion

The current ALOHA1 Isaac adaptation is no longer blocked by a global bbox proxy failure or by the stale workcell collision leaves for this strict contact gate. A narrow, verified pass exists when the BottleUSD object is placed at the Phase114 offset `0.0`.

This is still not the final bottle-in-pipe task. It is the current stable prerequisite gate:

1. Trossen/Menagerie ALOHA scene can load.
2. ALOHA1 left-arm HDF5 drive-target replay tracks within threshold.
3. BottleUSD can contact both target fingers.
4. Stale non-semantic workcell collisions can be rejected.
5. The strict pass does not require diagnostic support-plane geometry.

## Next Work

The next step should move from "already near the gripper contact window" toward a longer replay:

1. keep Phase115 as the regression gate;
2. add pregrasp or bottle approach replay only after Phase115 remains stable;
3. add a calibrated table/base transform before treating table contacts as final task support;
4. avoid broad collision deletion; disable exact stale leaves only when there is audit evidence.

