# Isaac MCP Toolchain

Read this before using, changing, debugging, or reinstalling Isaac Sim / Isaac Lab MCP tooling on machine `101`.

## Setup Paths
- Setup root: `/home/eii/isaac_mcp_setup`
- Environment report: `/home/eii/isaac_mcp_setup/environment_report.md`
- Install report: `/home/eii/isaac_mcp_setup/INSTALL_REPORT.md`
- Security notes: `/home/eii/isaac_mcp_setup/SECURITY.md`
- Server config: `/home/eii/isaac_mcp_setup/config/servers.yaml`

## Management Scripts
- Start persistent MCP services: `/home/eii/isaac_mcp_setup/scripts/start_all.sh`
- Stop persistent MCP services: `/home/eii/isaac_mcp_setup/scripts/stop_all.sh`
- Show status: `/home/eii/isaac_mcp_setup/scripts/status_all.sh`
- Run safe verification: `/home/eii/isaac_mcp_setup/scripts/test_all.sh`

## Installed MCP Servers
- `nvidia-isaac-docs`: official NVIDIA Isaac Sim documentation/search MCP from `NVIDIA-Omniverse/kit-usd-agents`, running in Docker as `isaacsim-mcp`, bound only to `127.0.0.1:9904`.
- `isaacsim-control`: community scene-control MCP from `whats2000/isaacsim-mcp-server`, stdio MCP for Claude/Codex, waits for Isaac Sim extension socket `localhost:8766`.
- `isaacsim-python`: community Isaac Python execution/log MCP from `mochan-b/isaacsim-mcp`, stdio MCP for Claude/Codex, waits for Isaac Sim VS Code executor `127.0.0.1:8226`.
- `isaaclab`: local minimal read-only Isaac Lab MCP at `/home/eii/isaac_mcp_setup/repos/isaaclab-mcp-local`, using `/home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/bin/python`.
- ROS MCP is documented separately in `docs/agents/ros_mcp.md`; do not treat Isaac MCP setup as permission to connect MCP tools to the real ALOHA robot.

## Verified State
- `nvidia-isaac-docs` is healthy and responds to MCP initialize on `http://127.0.0.1:9904/mcp`.
- `isaacsim-control` and `isaacsim-python` are installed and registered, but are only partially testable until the corresponding Isaac Sim internal extensions are enabled.
- `isaaclab` local tests passed and the install probe sees Isaac Sim `5.1.0.0` and Isaac Lab `0.54.4`; task listing currently has a known `pxr` import issue.

## Claude Code And Codex Names
- `isaac-sim-mcp` for official NVIDIA docs/search MCP.
- `isaacsim-control` for Isaac scene manipulation.
- `isaacsim-python` for Isaac Python execution and Kit log reading.
- `isaaclab` for local safe Isaac Lab probes.

## Security Constraints
- Do not expose MCP ports to public interfaces; keep all network services bound to `127.0.0.1` / localhost.
- Do not print or commit `NVIDIA_API_KEY` or `NGC_API_KEY`; API keys are stored outside this repository.
- Treat `isaacsim-python` and `isaacsim-control` as high-privilege because they can execute Python or scene operations inside Isaac Sim.
- Do not connect MCP tools to the real ALOHA robot by default.
- Do not start or restart `aloha_ros_nodes`, `runtime`, `rlt_warmup_runtime`, or other real robot control containers as part of MCP work unless the user explicitly approves.

## ALOHA Isaac Scratch Workspace
- Scratch workspace: `/home/eii/isaac_mcp_setup/aloha_project`
- This is only a scratch workspace. The project source of truth remains `/home/eii/project/openpi0.5-rtc-reward-learning`, especially `examples/aloha_isaac`.

## Hard NVIDIA MCP Requirement
- For any Isaac Sim, Isaac Lab, Isaac MCP, ALOHA simulation, USD conversion, physics setup, or Isaac GUI task, first use the NVIDIA official Isaac MCP: `isaac-sim-mcp` / `mcp__isaac_sim_mcp`.
- This is mandatory, not a preference. Do not start with shell-only inspection, community MCPs, web/forum searches, or local code changes for Isaac-related work.
- If the NVIDIA official Isaac MCP is unavailable, fails, or cannot answer the required Isaac-specific point, stop and report that the hard prerequisite is unavailable before continuing. Do not silently substitute another MCP or community source.
- After the NVIDIA official Isaac MCP has been used for the task, `isaaclab`, `isaacsim-control`, and `isaacsim-python` may be used only as secondary implementation or inspection tools.
- `isaacsim-control` and `isaacsim-python` remain local simulation tools only; never use them to affect the real ALOHA robot.

## Secondary MCP Order
- After the mandatory NVIDIA official MCP step, use `isaaclab` read-only probes when Isaac Lab state or APIs are needed.
- Use `isaacsim-control` only after confirming the Isaac Sim extension socket is local and the task is simulation-only.
- Use `isaacsim-python` only when direct Python execution inside Isaac Sim is necessary.
