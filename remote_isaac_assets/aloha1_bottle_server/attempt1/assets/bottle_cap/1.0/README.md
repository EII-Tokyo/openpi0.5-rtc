# BottleCap diagnostic asset v1

This asset is derived from the project Bottle500 FreeCAD neck dimensions. The
source Bottle CAD remains read-only and is not modified by the cap build.

The cap height, top thickness, mass, inertia approximation, radial clearance,
and friction coefficients are `TEMPORARY_UNCALIBRATED`. They are suitable for
simulation integration testing only and must be replaced by measured physical
cap properties before a calibrated manipulation result is claimed.

## Source evidence

- Bottle FCStd: `assets/bottle_500ml/cad/bottle_500ml.FCStd`
- Bottle thread outer diameter: 30 mm
- Bottle support outer diameter: 34 mm

## Diagnostic cap parameters

- Outer diameter: 34 mm
- Inner diameter: 30.8 mm
- Height: 22 mm
- Top thickness: 2 mm
- Nominal mass: 0.004 kg
- Cap static/dynamic friction: 0.90 / 0.75
- Restitution: 0

## Visual identification

- Deep-blue semi-matte plastic body with clearcoat
- 32 bright-blue vertical grip ribs
- Bright-blue top accent disc
- Visual envelope height: 22.25 mm

The ribs and top accent are visual-only geometry and do not carry
`CollisionAPI`. The physical cap envelope and its 17 collision shapes remain
unchanged at 22 mm height.

Run the FreeCAD builder with the pinned project-local FreeCAD 1.1.1 wrapper,
then run the USD builder and validator with the pinned Isaac Sim 5.1 USD Python
environment.
