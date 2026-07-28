# ALOHA ViperX CAD finger Task 5 geometry audit

- Status: `PARTIAL`
- Stage: `/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/cad_finger_task5_convex_hull/aloha_viperx_supplier_cad_task5.usda`
- Stage SHA-256: `8040edd01859af9f8c51285d198d34aae19e66625a2d5f21729879774e1644d9`
- Runtime mutation: none saved; legal poses were session-only.
- Collider approximation audited: `convexHull`.
- Method: world-transformed source points → numerical convex hull → normalized halfspace LP and intersection volume.
- Boundary: attachment-component overlap is reported numerically and is not automatically called an error.

## closed

| Pair | Relation | Margin m | Overlap m³ | Scope |
|---|---:|---:|---:|---|
| `gripper_shell ↔ left_finger` | `SEPARATED` | `-0.0100624928` | `0` | `FINGER_TO_ATTACHMENT_COMPONENT_REQUIRES_ASSEMBLY_SEMANTIC_REVIEW` |
| `gripper_shell ↔ right_finger` | `SEPARATED` | `-0.0100625073` | `0` | `FINGER_TO_ATTACHMENT_COMPONENT_REQUIRES_ASSEMBLY_SEMANTIC_REVIEW` |
| `gripper_bar ↔ left_finger` | `OVERLAP` | `0.00678750807` | `8.31248621e-06` | `FINGER_TO_ATTACHMENT_COMPONENT_REQUIRES_ASSEMBLY_SEMANTIC_REVIEW` |
| `gripper_bar ↔ right_finger` | `OVERLAP` | `0.00678749846` | `8.312468e-06` | `FINGER_TO_ATTACHMENT_COMPONENT_REQUIRES_ASSEMBLY_SEMANTIC_REVIEW` |
| `sliding_carriage ↔ left_finger` | `SEPARATED` | `-0.00597537247` | `0` | `FINGER_TO_ATTACHMENT_COMPONENT_REQUIRES_ASSEMBLY_SEMANTIC_REVIEW` |
| `sliding_carriage ↔ right_finger` | `SEPARATED` | `-0.00597862253` | `0` | `FINGER_TO_ATTACHMENT_COMPONENT_REQUIRES_ASSEMBLY_SEMANTIC_REVIEW` |
| `left_finger ↔ right_finger` | `SEPARATED` | `-0.00205729714` | `0` | `FINGER_TO_FINGER_UNEXPECTED_IF_OVERLAP` |

## partial

| Pair | Relation | Margin m | Overlap m³ | Scope |
|---|---:|---:|---:|---|
| `gripper_shell ↔ left_finger` | `SEPARATED` | `-0.0122724794` | `0` | `FINGER_TO_ATTACHMENT_COMPONENT_REQUIRES_ASSEMBLY_SEMANTIC_REVIEW` |
| `gripper_shell ↔ right_finger` | `SEPARATED` | `-0.0122724798` | `0` | `FINGER_TO_ATTACHMENT_COMPONENT_REQUIRES_ASSEMBLY_SEMANTIC_REVIEW` |
| `gripper_bar ↔ left_finger` | `OVERLAP` | `0.00678751254` | `8.31249445e-06` | `FINGER_TO_ATTACHMENT_COMPONENT_REQUIRES_ASSEMBLY_SEMANTIC_REVIEW` |
| `gripper_bar ↔ right_finger` | `OVERLAP` | `0.00678749538` | `8.31246252e-06` | `FINGER_TO_ATTACHMENT_COMPONENT_REQUIRES_ASSEMBLY_SEMANTIC_REVIEW` |
| `sliding_carriage ↔ left_finger` | `SEPARATED` | `-0.010726981` | `0` | `FINGER_TO_ATTACHMENT_COMPONENT_REQUIRES_ASSEMBLY_SEMANTIC_REVIEW` |
| `sliding_carriage ↔ right_finger` | `SEPARATED` | `-0.0107572359` | `0` | `FINGER_TO_ATTACHMENT_COMPONENT_REQUIRES_ASSEMBLY_SEMANTIC_REVIEW` |
| `left_finger ↔ right_finger` | `SEPARATED` | `-0.0185586222` | `0` | `FINGER_TO_FINGER_UNEXPECTED_IF_OVERLAP` |

## maximum_legal_aperture

| Pair | Relation | Margin m | Overlap m³ | Scope |
|---|---:|---:|---:|---|
| `gripper_shell ↔ left_finger` | `SEPARATED` | `-0.0185839161` | `0` | `FINGER_TO_ATTACHMENT_COMPONENT_REQUIRES_ASSEMBLY_SEMANTIC_REVIEW` |
| `gripper_shell ↔ right_finger` | `SEPARATED` | `-0.0185839148` | `0` | `FINGER_TO_ATTACHMENT_COMPONENT_REQUIRES_ASSEMBLY_SEMANTIC_REVIEW` |
| `gripper_bar ↔ left_finger` | `OVERLAP` | `0.00678750809` | `8.31248746e-06` | `FINGER_TO_ATTACHMENT_COMPONENT_REQUIRES_ASSEMBLY_SEMANTIC_REVIEW` |
| `gripper_bar ↔ right_finger` | `OVERLAP` | `0.00678749074` | `8.31245454e-06` | `FINGER_TO_ATTACHMENT_COMPONENT_REQUIRES_ASSEMBLY_SEMANTIC_REVIEW` |
| `sliding_carriage ↔ left_finger` | `SEPARATED` | `-0.0185273001` | `0` | `FINGER_TO_ATTACHMENT_COMPONENT_REQUIRES_ASSEMBLY_SEMANTIC_REVIEW` |
| `sliding_carriage ↔ right_finger` | `SEPARATED` | `-0.0185282796` | `0` | `FINGER_TO_ATTACHMENT_COMPONENT_REQUIRES_ASSEMBLY_SEMANTIC_REVIEW` |
| `left_finger ↔ right_finger` | `SEPARATED` | `-0.0350599576` | `0` | `FINGER_TO_FINGER_UNEXPECTED_IF_OVERLAP` |

## Interpretation

- Finger-to-finger overlap gate: `True`.
- Finger-to-shell/bar/carriage relations remain assembly evidence. A volumetric common region may be a designed mounting interface and requires CAD assembly semantics.
- This static audit does not prove dynamic collision resolution, drive tracking, contact, or bottle hold.
