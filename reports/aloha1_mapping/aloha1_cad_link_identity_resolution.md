# ALOHA1 CAD/link geometry source resolution

- Status: **PASS**
- Official source commit: `b66d5b905725351dd71d3251a06cd3f4c777940f`
- Supplier CAD repaired: `false`
- Mirror used: `false`

| Link | Supplier CAD boundary | Link geometry authority | Resolution |
|---|---|---|---|
| `gripper_bar_link` | `COMBINED_GRIPPER_SOLID_NO_INDEPENDENT_PRODUCT` | `PINNED_OFFICIAL_URDF_MESH` | `RESOLVED_WITH_EXPLICIT_SOURCE_BOUNDARY` |
| `gripper_prop_link` | `COMBINED_GRIPPER_SOLID_NO_INDEPENDENT_PRODUCT` | `PINNED_OFFICIAL_URDF_MESH` | `RESOLVED_WITH_EXPLICIT_SOURCE_BOUNDARY` |
| `wrist_link` | `EXPOSED_INVALID_BREP` | `PINNED_OFFICIAL_URDF_MESH` | `RESOLVED_WITH_EXPLICIT_SOURCE_BOUNDARY` |

The supplier STEP is not falsely split into URDF products. Its fused gripper solid and invalid wrist B-Rep remain explicit evidence boundaries; the byte-identical pinned Interbotix meshes provide the link-level geometry identities. This report does not promote a collider or modify an asset.
