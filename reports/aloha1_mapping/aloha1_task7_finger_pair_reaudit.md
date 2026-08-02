# ALOHA1 Task 7 finger-pair corrective re-audit

- Status: `FAIL` (the screenshot geometry gate, not the final asset)
- Classification: `ILLEGAL_STATIC_Q_ZERO_BYPASSED_RUNTIME_LIMITS`
- Task 7: `PARTIAL`
- Task 8: `NOT_RUN`

## What the disputed screenshot actually shows

The image came from a deliberately rejected helper-body candidate. The capture script loaded the USD and rendered authored transforms without a physics reset, joint-limit solve, or articulation readback. Its finger geometry therefore remained at static `q=(0, 0)`, outside the authored legal intervals.

At static zero, the two independently authored supplier-CAD colliders are `OVERLAP` with `3.18334017203e-05 m^3` overlap and `0.00871138955292 m` signed margin.

At the legal closed limits `(+0.021, -0.021) m`, they are `SEPARATED` with `0 m^3` overlap.

## Corrected interpretation

- The left and right finger collision meshes are separate prims under separate finger links; they were not merged at the base.
- Articulation self-collision is disabled in the frozen diagnostic configuration, so finger-finger contact is not the closing stop.
- The authored prismatic limits are the closing stop. A static viewport capture that bypasses them is invalid finger-installation evidence.
- The previous screenshot `PASS` is revoked. Visual legibility remains PASS, but supplier-CAD orientation, legal runtime qpos, and pair response were NOT_RUN in that capture.
- The image alone cannot distinguish a reversed installation from an illegal unsolved state. The numeric q-state experiment identifies the latter for this capture.
