# ALOHA1 official parameter source audit

- Status: **PASS**
- Product: `Interbotix ViperX-300 6DOF` / `aloha_vx300s`
- Frozen required sources: `16`
- Formal parameter candidate gate: **PASS**
- Deterministic signature: `ed69318a0450fdd57127f8cfecf55563a3e96b9cd04ebd7d1a4340fd8d6a68ed`

## Frozen source chain

| ID | Authority | Evidence | Commit / SHA-256 |
|---|---|---|---|
| `interbotix_aloha_vx300s_motor_config` | Trossen Robotics / Interbotix | `OFFICIAL_PINNED_SOURCE` | `b66d5b905725351dd71d3251a06cd3f4c777940f` |
| `interbotix_aloha_vx300s_xacro` | Trossen Robotics / Interbotix | `OFFICIAL_PINNED_SOURCE` | `b66d5b905725351dd71d3251a06cd3f4c777940f` |
| `interbotix_core_humble` | Trossen Robotics / Interbotix | `OFFICIAL_PINNED_SOURCE` | `af18d4fe24ba08e09a0f1e92afaca1863e3205de` |
| `interbotix_manipulators_humble` | Trossen Robotics / Interbotix | `OFFICIAL_PINNED_SOURCE` | `b66d5b905725351dd71d3251a06cd3f4c777940f` |
| `interbotix_vx300s_motor_config` | Trossen Robotics / Interbotix | `OFFICIAL_PINNED_SOURCE` | `b66d5b905725351dd71d3251a06cd3f4c777940f` |
| `interbotix_vx300s_xacro` | Trossen Robotics / Interbotix | `OFFICIAL_PINNED_SOURCE` | `b66d5b905725351dd71d3251a06cd3f4c777940f` |
| `interbotix_xs_driver` | Trossen Robotics / Interbotix | `OFFICIAL_PINNED_SOURCE` | `da27b8b2b6c7677844f74581b82c01829a834e1c` |
| `interbotix_xsarm_default_modes` | Trossen Robotics / Interbotix | `OFFICIAL_PINNED_SOURCE` | `b66d5b905725351dd71d3251a06cd3f4c777940f` |
| `isaacsim_urdf_importer_2_4_30` | NVIDIA local Isaac Sim installation | `LOCAL_VERSION_PINNED_OFFICIAL_IMPLEMENTATION` | `6ee1a8ab74c492930c792f7cabc657e022fb06dab4198ef413f6e1b359439baf` |
| `physx_schema_107_3` | NVIDIA local Isaac Sim installation | `LOCAL_VERSION_PINNED_OFFICIAL_IMPLEMENTATION` | `fe075bce4bde5ba7db69c6ccef0c4c26909336ab34c619129fc276f7cb4d7abc` |
| `robotis_xm430_w350_manual` | ROBOTIS | `OFFICIAL_DIRECT` | `7944311f7e2670bee18ff5ef023eceb9911f4239e66d66273ea5ee21886a0fd2` |
| `robotis_xm430_w350_product` | ROBOTIS | `OFFICIAL_DIRECT` | `5d404cecc8b304a6aec1b2e8621157a87b11bcd8c7be6d0ed2244a223ad7abed` |
| `robotis_xm540_w270_manual` | ROBOTIS | `OFFICIAL_DIRECT` | `e662ccc500f61f8dc3ca8463555de4389c90dbe8c8139d24a4c968bb2d2f249a` |
| `robotis_xm540_w270_product` | ROBOTIS | `OFFICIAL_DIRECT` | `92668f52a76463de6f6bac289f5b941606c1b030443cfb1b777c07e70982b9ba` |
| `supplier_simple_aloha_viper_step` | ALOHA supplier public CAD package | `SUPPLIER_CAD_USER_CONFIRMED` | `337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571` |
| `trossen_vx300s_spec` | Trossen Robotics | `OFFICIAL_DIRECT` | `93313cf328da611146ea608613dd2550f3c17e7bbb47728b9f06fa297bebdff9` |

## Retained official-source conflict

Trossen's ViperX-300 page contains contradictory ID 6/7 joint-name tables. The conflict is retained. The pinned official motor configurations and Xacro support `ID6=forearm_roll` and `ID7=wrist_angle`; this resolution does not rewrite or hide the contradictory webpage row.

## Local mirror boundary

`external/ros2-essentials` is a third-party aggregate mirror, not the Interbotix authority. Its local `aloha_vx300s.yaml` sleep positions differ from the pinned upstream file, so that local sleep pose is not labeled official.

## License boundary

The supplier STEP is user-confirmed public vendor material, but no formal redistribution license text was found. It remains local read-only evidence and is not committed or redistributed (`UNKNOWN_HARD_BLOCKER` only for redistribution).

## Findings

- None.
