# ROS MCP Tooling

Read this before using, changing, debugging, or reinstalling `robotmcp/ros-mcp-server`, rosbridge, or ROS MCP access for the real ALOHA robot on `192.168.1.103`.

## Current Decision
- The user asked on 2026-07-15 to investigate ROS on `192.168.1.103` and install the matching `robotmcp/ros-mcp-server` locally.
- Local Codex MCP is configured to start `ros-mcp` through `uvx`; the local machine does not need a full ROS install for this MCP server.
- Treat ROS MCP as a powerful robot-inspection/control bridge. Use it read-only by default unless the user explicitly asks for a real robot action.

## Local Codex Configuration
- Local Codex config path: `/home/eii/.codex/config.toml`.
- MCP server entry:

```toml
[mcp_servers.ros-mcp]
command = "/home/eii/.local/bin/uvx"
args = ["--from", "ros-mcp", "ros-mcp", "--transport=stdio"]
```

- Verified local command:

```bash
/home/eii/.local/bin/uvx --from ros-mcp ros-mcp --help
```

- After changing `~/.codex/config.toml`, restart Codex so the new MCP server is loaded.

## 103 ROS Facts From 2026-07-15 Investigation
- The `192.168.1.103` host shell itself did not expose a host ROS environment or host `roscore` / `rostopic` / `rosnode` commands.
- ROS was observed running inside Docker containers with host networking.
- The observed active ROS distro was ROS1 Noetic:
  - `ROS_DISTRO=noetic`
  - `rosversion -d` returned `noetic`
- Observed active ROS nodes included:
  - `/realsense_publisher`
  - `/puppet_left/xs_sdk`
  - `/puppet_right/xs_sdk`
  - `/master_left/xs_sdk`
  - `/master_right/xs_sdk`
- Observed camera and arm topics included:
  - `/cam_high`
  - `/cam_low`
  - `/cam_left_wrist`
  - `/cam_right_wrist`
  - `/puppet_left/joint_states`
  - `/puppet_right/joint_states`
  - `/puppet_left/commands/joint_group`
  - `/puppet_right/commands/joint_group`

## Rosbridge Requirement
- `robotmcp/ros-mcp-server` connects to the robot through rosbridge WebSocket, normally port `9090`.
- Initial 2026-07-15 investigation found no `rosbridge` / `rosapi` / `9090` configuration.
- The project now has a dedicated `rosbridge` compose service backed by `scripts/docker/rosbridge.Dockerfile`.
- The service should be started with the normal robot service set, or directly with:

```bash
cd /home/eii/openpi0.5-rtc-reward-learning && docker compose up -d --no-build rosbridge
```

- The active ROS containers checked before adding this service did not show `rosbridge_server`, `rosapi`, or a `9090` listener.
- For ROS1 Noetic on `192.168.1.103`, the upstream package/launch pattern is:

```bash
sudo apt install ros-noetic-rosbridge-server
roslaunch rosbridge_server rosbridge_websocket.launch
```

- Do not run `sudo apt install` on `192.168.1.103` without explicit user approval because that modifies the host system outside the project directory. For this project, use the project-local `rosbridge` compose service/image layer.
- `rosapi` must be present. Verify from the robot-side ROS environment:

```bash
rosservice list | grep rosapi
curl -I http://localhost:9090
```

- Do not launch only a bare `rosrun` rosbridge process if it does not bring up `rosapi`; MCP introspection tools need rosapi services.

## Resolved 103 Status
- As of 2026-07-15, `192.168.1.103` has the project-local `rosbridge` service built and running in the `openpi_reward_learning_eii` compose project.
- Verified running container:
  - `openpi_reward_learning_eii-rosbridge-1`
- Verified rosbridge log line:
  - `Rosbridge WebSocket server started at ws://0.0.0.0:9090`
- Verified 103 host listener:
  - `0.0.0.0:9090`
- Verified `rosapi` services exist on the same ROS graph:
  - `/rosapi/topics`
  - `/rosapi/services`
  - `/rosapi/nodes`
  - `/rosapi/topic_type`
- Verified from the local machine that `ws://192.168.1.103:9090` accepts WebSocket calls to `/rosapi/topics`.
- Verified topics returned through rosbridge include:
  - `/cam_high`
  - `/cam_low`
  - `/puppet_left/joint_states`
  - `/puppet_right/joint_states`
- Therefore, ROS MCP can now inspect the 103 ROS graph through rosbridge after Codex loads the local `ros-mcp` MCP server.

## Rosbridge Python Pitfall
- The robot base image has two relevant Python installations:
  - `/usr/local/bin/python3` is Python 3.10 and may be first on `PATH`.
  - `/usr/bin/python3` is the ROS Noetic system Python, where apt-installed ROS Python packages live.
- `ros-noetic-rosbridge-server` installs ROS Python modules into the system Python environment.
- If `rosbridge` starts with `/usr/local/bin/python3`, `rosbridge_websocket` can fail with:

```text
ModuleNotFoundError: No module named 'twisted'
```

- The project compose service fixes this by putting `/usr/bin` before `/usr/local/bin` in `PATH` before sourcing ROS and launching rosbridge.
- Do not remove that `PATH` override unless you have verified `rosbridge_websocket` still imports `twisted` and starts listening on `9090`.

## 103 Safety Constraints
- If setting up rosbridge on `192.168.1.103`, first read `docs/agents/remote_103_operations.md`.
- If any MCP action can publish to command topics, call services, torque motors, or otherwise affect the physical ALOHA robot, first read `docs/agents/aloha_hardware_debug.md`.
- Keep all 103 project changes under `/home/eii/openpi0.5-rtc-reward-learning` unless the user explicitly approves another path.
- Do not use or modify `/home/eii/openpi0.5-rlt` for this user's robot project.
- Prefer adding a dedicated compose service or project-local launch wrapper for rosbridge instead of ad hoc long-running shell commands.
- Do not expose rosbridge publicly. Bind or firewall it for the local trusted network only.

## Normal Workflow
1. Start or inspect the user's ROS stack from `/home/eii/openpi0.5-rtc-reward-learning` following `docs/agents/remote_103_operations.md`.
2. Ensure rosbridge and rosapi are running on the same ROS graph as the robot.
3. Restart Codex locally after MCP config changes.
4. Ask ROS MCP to connect to `192.168.1.103` and inspect nodes, topics, and services before attempting any write/control operation.

## Official References
- `robotmcp/ros-mcp-server`: https://github.com/robotmcp/ros-mcp-server
- Installation guide: https://github.com/robotmcp/ros-mcp-server/blob/main/docs/install/installation.md
- Codex CLI setup: https://github.com/robotmcp/ros-mcp-server/blob/main/docs/install/clients/codex-cli.md
- Rosbridge setup: https://github.com/robotmcp/ros-mcp-server/blob/main/docs/install/rosbridge.md
