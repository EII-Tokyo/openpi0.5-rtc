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
- Do not hard-code historical `ttyUSB*` numbers. On 2026-08-03, read-only
  udev/runtime inspection on machine 103 verified the current left-side role
  aliases as follows:
  - `/dev/ttyDXL_puppet_left -> ttyUSB0`, FTDI serial `FTAAMM8J`;
  - `/dev/ttyDXL_follower_left -> ttyUSB0`, FTDI serial `FTAAMM8J`;
  - `/dev/ttyDXL_master_left -> ttyUSB4`, FTDI serial `FTAAML38`;
  - `/dev/ttyDXL_leader_left -> ttyUSB4`, FTDI serial `FTAAML38`.
- The older fixed mapping that assigned `puppet_left` to `ttyUSB4` is stale on
  the current machine. Always resolve and record the semantic `/dev/ttyDXL_*`
  alias immediately before access; use that alias in launch files. Do not
  infer the right-side mapping from the left side.
- DYNAMIXEL Wizard directly opens the servo serial bus. Before using it to scan or diagnose motors, stop robot control containers that may hold `/dev/ttyUSB*` / `/dev/ttyDXL*`, otherwise ROS `xs_sdk` and Wizard can conflict, causing scan failures or unsafe control contention.
- Use it for hardware-level Dynamixel checks such as bus visibility, servo IDs, operating mode, profile type, errors, and basic motor diagnostics. Do not treat it as an RLT/VLA software debugger.
