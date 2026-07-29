# Trossen ViperX-300 6DOF official reference

- Status: `PASS`
- Evidence class: `FIRST_PARTY_OFFICIAL_DOCUMENTATION`
- Publisher: Trossen Robotics
- Source:
  `https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/vx300s.html#viperx-300-6dof`
- Accessed: `2026-07-29T04:07:29Z`

The official page records a 6-DOF ViperX-300 with `750 mm` reach,
`1500 mm` total span, `1 mm` repeatability, `5–8 mm` accuracy, nine servos,
and a `750 g` working payload. The payload note recommends no more than
50 percent extension when carrying 750 g.

It also provides the official default joint-limit table, the nine-servo
ID/model table, Product-of-Exponentials `M` and `Slist`, and links to the
technical drawing and solid STEP files. The gripper range is listed as
`42–116 mm`.

This page is registered as an official cross-check. It supports product
identity, joint-limit and servo mapping audits. It does not replace the pinned
URDF joint order, Isaac runtime DOF readback, supplier-CAD finger installation,
measured workcell placement, mass/inertia, drive gains, or the project's
explicit gripper normalization mapping.
