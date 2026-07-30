# `codex-isaac` MCP Configuration — 2026-07-30

Status: `PASS_CURRENT_SESSION_DIRECT_NVIDIA_MCP`

## Result

The old `codex-research` profile has been renamed and replaced by a dedicated
`codex-isaac` process:

- Launcher: `/home/eii/.local/bin/codex-isaac`
- Isolated home:
  `/home/eii/mcpjungle-lab/state/codex-home-codex-isaac`
- NVIDIA official Isaac documentation connects directly as `isaac-sim-mcp` at
  `http://127.0.0.1:9904/mcp`.
- Every other external MCP available to this profile routes through
  `mcpjungle_lab` at
  `http://127.0.0.1:18080/v0/groups/codex-isaac/mcp`.
- The Jungle group exposes zero NVIDIA tools.
- Compatibility launchers do not alias `codex-isaac`; fresh
  `codex-full mcp list` reports no configured MCP servers.

The launcher runs Codex with
`--dangerously-bypass-approvals-and-sandbox`. Runtime configuration readback is
`sandbox_mode=danger-full-access` and `approval_policy=never`. This grants full
local filesystem/process access, but it does not authorize real-robot control,
mutation of `192.168.1.103`, secret disclosure, or unrelated destructive
operations.

## Direct NVIDIA evidence

- Listener: `127.0.0.1:9904`
- Client SDK used for the reproducible probe: `mcp==1.25.0`
- Server readback: `NeMo Agent Toolkit MCP 1.25.0`
- `tools/list`: exactly five Isaac documentation/search tools
- Read-only call: `get_isaac_sim_instructions`
- Instruction set: `robot_setup`
- Current-session namespace: `mcp__isaac_sim_mcp`
- MCPJungle used for this call: `false`
- Result: non-empty content and `isError=false`

This call went directly to the NVIDIA endpoint, not through MCPJungle.

## MCPJungle evidence

The live `codex-isaac` group contains:

- NVIDIA Isaac docs: `0`
- Context7: `2`
- ALOHA 103 read-only tools: `15`
- Chrome DevTools liveview: `3`
- Total: `20`

The exact allowlist validator passed and one Context7 read-only call passed. No
ALOHA 103 tool was called.

## Isolation and persistence

Two consecutive profile rebuilds produced the same configuration SHA-256:

`00993670375f0312de5c4aab198ceb6bc766ada73f45f8c4a31616b5534d26c9`

Codex 0.145.0 read back exactly two server entries:

1. `isaac-sim-mcp`
2. `mcpjungle_lab`

The deployed generator and launcher are byte-identical to their managed
sources. The old `codex-research` runtime group is absent. Its former isolated
home is recoverable at:

`/home/eii/mcpjungle-lab/backups/codex-research-retired-20260730/codex-home-codex-research`

## `bee` session continuation

The new profile now uses the stable session-store bridge:

`/home/eii/mcpjungle-lab/state/codex-session-home`

The new isolated home links `sessions/`, `history.jsonl`,
`session_index.jsonl`, and `state_5.sqlite` to that store. Runtime index
readback found:

- Thread name: `bee`
- Thread ID: `019fa738-940b-7960-b831-f3a07329028f`

After the old process exits normally, resume with:

```bash
codex-isaac resume bee
```

Do not run both processes against the same thread concurrently.

## Verification

- Unit tests: `43 passed`
- Ruff: `PASS`
- `py_compile`: `PASS`
- `bash -n`: `PASS`
- Shellcheck: `SKIP_NOT_INSTALLED`
- JSON/group allowlist validation: `PASS`
- Direct NVIDIA initialize/list/read-only call: `PASS`
- Jungle initialize/list/Context7 call: `PASS`
- Deterministic isolated profile generation: `PASS`
- Live `codex-isaac mcp list`: `PASS`
- Non-`codex-isaac` launcher isolation: `PASS`
- `bee` session bridge/index/dry-run: `PASS`

Full evidence:

`/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260730-direct-nvidia-mcp-config`

The current session discovered the direct five-tool NVIDIA surface and
successfully called `get_isaac_sim_instructions("robot_setup")`. The
current-session direct-tool gate is therefore complete. Future Isaac work
should still start through `codex-isaac` so the same isolated routing policy is
reconstructed.
