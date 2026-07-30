# ALOHA1 Complete Gripper CAD Clearance

- Status: `PASS`
- Classification: `COMPLETE_SUPPLIER_GRIPPER_PROJECT_BOTTLE_CLEARANCE`
- Supplier CAD: complete Simple Aloha Viper gripper assembly
- Bottle: project-authored Bottle500 B-Rep
- Task 8: `NOT_RUN`

## Result

- Rejected run13 bottle-axis center: `0.111271885 m`
- Corrected bottle-axis center: `0.132154988 m`
- Corrected pad-contact midpoint: `0.135520804 m`
- Max-min minimum hard margin: `0.023429991 m`
- Contact finger q: `left=0.048316875 m`, `right=-0.048316875 m`
- Bottle-axis center offset from pad frame: `[-0.003365816517218456, 0.0, 0.0]`

## Evidence Boundary

- Supplier shell maximum approach extent: `0.069100000 m`.
- Runtime URDF gripper-bar conservative maximum extent: `0.076524997 m`.
- The runtime bar is the controlling forbidden envelope; the supplier shell and runtime mesh are retained separately.
- The two fresh FreeCAD semantic signatures match: `e3130d9c5646f48e0fc2494bf85dd82dd31c6c1256f807ceee6f94eb71c03c9e`.

This is a static geometry gate. It does not prove contact, hold, IK reachability, or dynamic pickup.
