# Authorized ALOHA Viper Review Stage Audit

- Status: `PASS`
- Stage: `/home/eii/project/openpi0.5-rtc-reward-learning/local_eval_assets/aloha_isaac_assets/aloha_viperx.usd`
- Source SHA-256 before: `b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e`
- Source SHA-256 after: `b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e`
- Default prim: `/workcell`
- Stage units: `1.0 m/unit`
- Up axis: `Z`
- Required key prims: `PASS`
- Layer stack: `PASS`
- Instance structure: `PASS`

## Authorization boundary

- Source Stage mutation: `FORBIDDEN`
- Default/final collider mutation: `FORBIDDEN`
- Allowed output: independent diagnostic layer only

## Required prims

- `/workcell`: `PASS` (Xform)
- `/workcell/vx300s_left`: `PASS` (Xform)
- `/workcell/vx300s_left/vx300s_left`: `PASS` (Xform)
- `/workcell/vx300s_left/vx300s_left_gripper_link`: `PASS` (Xform)
- `/workcell/vx300s_left/vx300s_left_left_finger_link`: `PASS` (Xform)
- `/workcell/vx300s_left/vx300s_left_right_finger_link`: `PASS` (Xform)
- `/workcell/joints/vx300s_left_left_finger`: `PASS` (PhysicsPrismaticJoint)
- `/workcell/joints/vx300s_left_right_finger`: `PASS` (PhysicsPrismaticJoint)

## Used file-backed layers

- `/home/eii/project/openpi0.5-rtc-reward-learning/local_eval_assets/aloha_isaac_assets/aloha_viperx.usd` — `b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e`
- `/home/eii/project/openpi0.5-rtc-reward-learning/local_eval_assets/aloha_isaac_assets/configuration/aloha_viperx_base.usd` — `4e73aed0d6a404a016fcb87e4874f75593034334ca23915a55ee8e55d4fecb47`
- `/home/eii/project/openpi0.5-rtc-reward-learning/local_eval_assets/aloha_isaac_assets/configuration/aloha_viperx_physics.usd` — `2adb136e5f01b4185a487893ead580fc1f9196ce401efe8b787bec119c6979c0`
- `/home/eii/project/openpi0.5-rtc-reward-learning/local_eval_assets/aloha_isaac_assets/configuration/aloha_viperx_robot.usd` — `85c711ce7943c037561716b7aa0ebb571d3d75ffd8f368d8a595d8e7b1450467`
- `/home/eii/project/openpi0.5-rtc-reward-learning/local_eval_assets/aloha_isaac_assets/configuration/aloha_viperx_sensor.usd` — `3e9f49229b9592ea8008daf233ada0b765ba61d6820486d926e9b3dba37df808`

## Instance-proxy consequence

Both finger visual and collision branches are instanceable, and their Mesh prims are instance proxies. The permitted strategy is to de-instance only the visual branch in the independent diagnostic layer; collision branches remain unchanged until separately audited.
