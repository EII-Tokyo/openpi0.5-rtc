# ALOHA Hardware Debug

Read this before using hardware diagnostics, serial devices, Dynamixel tools, or anything that can touch the real ALOHA robot on `192.168.1.103`.

For product identity, component models, manufacturer specifications, and any
hardware-to-simulation parameter mapping, also read
`docs/agents/aloha_official_hardware_parameter_sources.md`. A machine-103
configuration or register snapshot is runtime evidence, not a substitute for
the exact-model official manufacturer source.

- On `192.168.1.103`, the real ALOHA / Dynamixel GUI diagnostic tool is **DYNAMIXEL Wizard 2.0**.
- Launch path:
  - `/home/eii/ROBOTIS/DynamixelWizard2/DynamixelWizard2.sh`
- Desktop entry:
  - `/home/eii/.local/share/applications/DynamixelWizard2.desktop`
- Known 103 Dynamixel serial aliases:
  - `/dev/ttyDXL_puppet_left -> ttyUSB4`
  - `/dev/ttyDXL_puppet_right -> ttyUSB0`
  - `/dev/ttyDXL_master_left -> ttyUSB2`
  - `/dev/ttyDXL_master_right -> ttyUSB3`
  - Legacy/follower aliases may mirror these devices, such as `/dev/ttyDXL_follower_left` and `/dev/ttyDXL_follower_right`.
- DYNAMIXEL Wizard directly opens the servo serial bus. Before using it to scan or diagnose motors, stop robot control containers that may hold `/dev/ttyUSB*` / `/dev/ttyDXL*`, otherwise ROS `xs_sdk` and Wizard can conflict, causing scan failures or unsafe control contention.
- Use it for hardware-level Dynamixel checks such as bus visibility, servo IDs, operating mode, profile type, errors, and basic motor diagnostics. Do not treat it as an RLT/VLA software debugger.
