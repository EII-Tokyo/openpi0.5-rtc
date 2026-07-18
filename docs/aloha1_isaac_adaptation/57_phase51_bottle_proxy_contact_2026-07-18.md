# Phase 51 Bottle-Proxy Contact Gate

## Question

Phase 50 proved that simple cylinders and capsules can create stable fingertip contact. Phase 51 asks a more task-shaped question:

Can a lightweight bottle proxy, built as one rigid body with multiple child colliders, pass a local ALOHA1 gripper contact gate?

The proxy is not a final bottle CAD. It is a minimal physics test object:

- body: cylinder;
- neck: smaller cylinder;
- mouth: sphere;
- rigid body and mass API on the object root;
- collision API on each child shape.

This follows Isaac/PhysX semantics: multiple child colliders under one rigid body can act as one compound rigid body.

## Code Change

Updated:

```text
aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py
```

The validator now supports:

- `--object-shape bottle_proxy`;
- child-collider contact/rest offset authoring for compound objects;
- per-finger target contact coverage;
- explicit bilateral contact gate semantics;
- cross-side proxy overlap diagnostics.

The important gate correction is:

- single-finger push tests still require non-zero object motion;
- bilateral closure tests do not require object translation, because a symmetrically held object can correctly stay centered;
- bilateral closure instead requires both expected fingertip proxies to contact the object and the object motion to remain finite and bounded.

## Evidence

### Single-finger bottle proxy

Command artifact:

```text
.codex/artifacts/20260718-140435_phase51-left-finger-bottle-proxy-contact-regression
```

Result:

| Check | Result |
| --- | --- |
| status | `PASS` |
| contact trace status | `PASS_SINGLE_FINGER_CONTACT_ISOLATION` |
| expected finger contacted object | true |
| object displacement | 0.1476506916 |
| no explosion | true |

Interpretation:

The compound bottle proxy is physically usable for local single-finger contact.

### Dual-stage bilateral test with small object

Command artifact:

```text
.codex/artifacts/20260718-140454_phase51-bilateral-bottle-proxy-contact-overlap-gate
```

Result:

| Check | Result |
| --- | --- |
| status | `FAILED_GATE` |
| contact trace status | `FAIL_NO_TARGET_CONTACT` |
| object fill fraction | 0.1 |
| open finger surface gap | 0.0711998815 |
| object side length | 0.0071199882 |

Interpretation:

This is an invalid bilateral grasp size. The object diameter is only 10% of the open finger surface gap, so it cannot reliably contact both sides.

### Dual-stage bilateral test with larger light object

Command artifact:

```text
.codex/artifacts/20260718-140614_phase51-bilateral-bottle-proxy-fill08-contact
```

Result:

| Check | Result |
| --- | --- |
| status | `FAILED_GATE` |
| contact trace status | `FAIL_OBJECT_EJECTION` |
| both expected fingers contacted object | true |
| object displacement | 0.2839312083 |
| max allowed displacement | 0.25 |

Interpretation:

The contact topology is correct, but the light object is ejected. This is a physics-parameter issue, not a missing collider issue.

### Dual-stage bilateral test with heavier object

Command artifact:

```text
.codex/artifacts/20260718-140740_phase51-bilateral-bottle-proxy-fill08-mass05-contact
```

Result:

| Check | Result |
| --- | --- |
| status | `PASS` |
| contact trace status | `PASS_BILATERAL_CONTACT_CANDIDATE` |
| both expected fingers contacted object | true |
| object displacement | 0.1602754827 |
| no explosion | true |
| non-target contact pairs | 4 |

Interpretation:

Increasing object mass to `0.05` stabilizes the bilateral contact. However, this dual-stage test still reports non-target contacts involving the opposite side's fingertip proxies, so it is only a candidate. It is not a clean local gripper gate.

### Left-only bilateral test

The left-only stage was generated with the same fingertip-pad proxy parameters:

```text
.codex/artifacts/20260718-140832_phase51-left-only-fingertip-offset-stage-build
```

Generated stage:

```text
local_eval_assets/aloha1_clean_runtime_20260718/aloha1_left_fingertip_pad_proxy_offset_runtime.usda
```

Contact artifact:

```text
.codex/artifacts/20260718-140929_phase51-left-only-bilateral-bottle-proxy-final-contact
```

Result:

| Check | Result |
| --- | --- |
| status | `PASS` |
| contact trace status | `PASS_BILATERAL_CONTACT_CANDIDATE` |
| both expected fingers contacted object | true |
| non-target contact pairs | 0 |
| object displacement | 3.115442791e-08 |
| no explosion | true |
| contact motion policy | `not_required_for_bilateral_closure` |

Interpretation:

This is the first clean local bottle-proxy gripper contact gate in this sequence:

1. the compound bottle proxy loads as one rigid body with child colliders;
2. both left fingertip proxies contact the object;
3. there are no non-target contact pairs;
4. the object remains stable and finite;
5. the gate does not incorrectly require translation during symmetric bilateral closure.

## Why The Gate Changed

The old passive-contact gate required:

```text
object displacement >= min_contact_motion
```

That rule is valid for a single-finger push. If one finger pushes a free passive object and the object never moves, contact probably did not transfer force.

It is not valid for a bilateral grasp. If two fingertips close symmetrically on an object centered between them, a good result can be:

```text
left finger contacts object
right finger contacts object
object displacement remains near zero
```

So the new logic is:

| Mode | Required contact | Required object motion |
| --- | --- | --- |
| single finger | object touches the moving expected finger | yes |
| both fingers | object touches both expected fingers | no |

Both modes still require bounded finite object motion. A bilateral object that gets ejected still fails.

## Important Limitation

The current dual-arm clean runtime stage is not a final full-workcell asset. It sublayers left and right generated side wrappers without a validated real dual-arm base transform. Dual-stage contact tests can therefore be polluted by opposite-side proxy contacts.

For local gripper physics gates, use the left-only or right-only stage until the dual-arm workcell transform is explicitly validated.

## Decision

Treat the following as the current clean local bottle-proxy contact gate:

```text
reports/aloha1_isaac_adaptation/phase51_left_only_bilateral_bottle_proxy_fill08_mass05_contact_20260718/gripper_passive_contact_metrics.json
```

Do not use the dual-stage `PASS_BILATERAL_CONTACT_CANDIDATE` alone as final evidence because it still includes non-target contacts.

## Next Gate

The next step should use the clean left-only bottle-proxy gate as the baseline and move toward real task replay:

1. add a table or fixed support only if it does not introduce non-target contact;
2. replace bottle proxy with the current Bottle500 collision asset or a measured bottle proxy;
3. replay a short real left-gripper close segment;
4. require both target contact coverage and bounded object motion;
5. only then reintroduce the full dual-arm scene with measured base transforms.
