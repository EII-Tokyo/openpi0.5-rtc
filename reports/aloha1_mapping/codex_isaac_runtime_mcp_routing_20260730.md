# codex-isaac runtime MCP routing — 2026-07-30

Status: `PARTIAL_RUNTIME_BACKEND_NOT_RUNNING`

MCPJungle implementation commit: `440dafd`.

The routing layer is verified. `codex-isaac` continues to access NVIDIA's
official documentation MCP directly at `127.0.0.1:9904`. MCPJungle exposes
zero NVIDIA tools and now exposes the two non-NVIDIA runtime servers:

- `isaacsim-control`: 42 tools;
- `isaacsim-python`: 3 tools.

Both runtime servers are hosted by transient user services and listen only on
the MCPJungle Docker bridge:

- `mcpjungle-isaacsim-control.service` → `172.20.0.1:18766`;
- `mcpjungle-isaacsim-python.service` → `172.20.0.1:18226`.

MCP initialize, tool listing, Context7 regression, the Python connection
probe, and the control `get_scene_info` route were executed through the
`codex-isaac` Jungle group. The Python probe returned an explicit connection
refusal for `127.0.0.1:8226`; the control probe returned the explicit
“Could not connect to Isaac” backend response for `127.0.0.1:8766`.

Therefore the remaining boundary is not MCPJungle routing. No Isaac Sim GUI
process had enabled the two internal backend sockets during this validation.
Stage readback is `NOT_RUN`, and no Stage was switched or modified.

After Isaac Sim is running with the corresponding extensions enabled, start a
fresh `codex-isaac` session so Codex rebuilds its MCP tool registry. The next
acceptance step is read-only verification of the user-approved Stage path,
root prim, sublayers, references, Timeline state, and one articulation
readback before any Grasp Editor mutation.

Evidence:

- `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260730-codex-isaac-runtime-mcp/gateway_sdk_probe.json`
- `/home/eii/mcpjungle-lab/logs/isaac-runtime/isaacsim-control.log`
- `/home/eii/mcpjungle-lab/logs/isaac-runtime/isaacsim-python.log`
- `/home/eii/mcpjungle-lab/logs/isaac-runtime/sync.log`
