# ALOHA1 Gripper Orientation Diagnostic Design

Date: 2026-07-28

## Goal

Determine whether the historical converted ALOHA gripper USD has a real
left/right finger assembly error or whether the earlier offline screenshots
only made the fingers appear reversed.

This diagnostic is limited to visual-mesh identity, installation transforms,
closing direction, and geometric aperture. It does not change collision
geometry, drive parameters, mimic behavior, friction, bottle properties, or
the final asset configuration.

## Frozen Inputs

- Historical converted USD:
  `/home/eii/project/openpi0.5-rtc-reward-learning/local_eval_assets/aloha_isaac_assets/aloha_viperx.usd`
- USD SHA-256:
  `b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e`
- Left custom finger STL SHA-256:
  `df73ae5b9058e5d50a6409ac2ab687dade75053a86591bb5e23ab051dbf2d659`
- Right custom finger STL SHA-256:
  `56fb3cc1236d4193106038adf8e457c7252ae9e86c7cee6dabf0578c53666358`
- Rejected generic ViperX finger SHA-256:
  `a4baacd9a64df1be60ea5e98f50f3c660e1b7a1fe9684aace6004c5058c09483`

The historical USD and both custom finger meshes remain read-only.

## Root-Cause Evidence Before Implementation

The earlier screenshots used one Matplotlib `Poly3DCollection` per mesh.
Collection-level depth sorting is not a reliable representation of
mutually occluding meshes. The resulting blue/orange overlap can therefore
look reversed even when the world-space geometry is not.

Runtime USD readback currently shows:

- the left finger occupies the lower world-Y side;
- the right finger occupies the higher world-Y side;
- the principal left inward surface normal points approximately along `+Y`;
- the principal right inward surface normal points approximately along `-Y`.

A naive cross-assignment of raw left/right mesh point arrays while preserving
the current mesh transforms moves their centers from approximately
`Y=0.490/0.510 m` to `Y=0.406/0.594 m`. That experiment would create a much
larger separation and is rejected as an ungrounded fix.

## Selected Approach

Create a separate, read-only diagnostic render pipeline using a real
depth-buffered renderer. Blender is preferred because it is installed locally
and can render isolated geometry without switching the user's active Isaac Sim
Stage.

The renderer will:

1. Read the five relevant visual meshes and their composed world transforms
   from the frozen USD:
   gripper body, gripper bar, gripper prop, left custom finger, and right
   custom finger.
2. Label the fingers by physical world side:
   `physical_left` and `physical_right`.
3. Preserve the authored closed-state geometry.
4. Reconstruct an open-state diagnostic using only the original MJCF
   prismatic-joint endpoints and closure directions.
5. Render both states from identical:
   closing-axis, top, and isometric views.
6. Keep diagnostic colors out of the source USD.

No screenshot will be accepted solely because it looks plausible.

## Machine Gates

The diagnostic must report all of the following before screenshots are shown
to the user:

- source USD hash is unchanged;
- physical-left center remains on the lower world-Y side;
- physical-right center remains on the higher world-Y side;
- principal inward normals face one another;
- open-to-closed aperture decreases monotonically;
- the open state has positive separation;
- the two finger meshes do not cross to the opposite physical side;
- the closed state presents opposed grasping surfaces;
- all screenshots are non-empty and have deterministic hashes;
- no source USD or active GUI Stage was modified.

Results are classified as:

- `ASSEMBLY_ORIENTATION_CONFIRMED`
- `ASSEMBLY_ORIENTATION_ERROR`
- `RENDERING_ARTIFACT_CONFIRMED`
- `INCONCLUSIVE`

## Conditional Follow-Up

Only if the depth-correct render and numeric gates still demonstrate an
assembly error will a separate diagnostic copy compare one change at a time:

1. raw mesh cross-assignment;
2. local rotation of one or both fingers;
3. exchange of each mesh together with its matching local transform.

The final USD will not be changed until the user confirms the corrected
diagnostic screenshots.

## Deliverables

The diagnostic run will produce an ignored artifact directory containing:

- authored-closed closing-axis, top, and isometric PNGs;
- reconstructed-open closing-axis, top, and isometric PNGs;
- a machine-readable JSON manifest with hashes, transforms, aperture,
  surface-normal evidence, and PASS/FAIL gates;
- a bounded execution log.
