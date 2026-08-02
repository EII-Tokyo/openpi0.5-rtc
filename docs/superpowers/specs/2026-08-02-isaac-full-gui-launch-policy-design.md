# Isaac Sim Full GUI Launch Policy Design

## Goal

Replace the user-facing trimmed Isaac Sim Python viewer on workspace 3 with
the full Isaac Sim 5.1 experience, load the exact user-approved CAD-derived
ALOHA Stage, preserve a paused startup state for manual Physics Inspector
testing, and make the full-experience requirement durable in project policy.

## Approved Stage

The only Stage authorized for this operation is:

```text
/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_diagnostic.usda
```

Its frozen SHA-256 before process replacement is:

```text
eb3d2b12bb0903589856607c9f05212bf5c22182f539a413587162f4b1027459
```

The Stage must not be saved, flattened, or modified during launch or manual
Inspector testing.

## Process Replacement

The existing workspace-3 trimmed viewer is identified by both its process
command and X11 window identity. It is closed using a normal window-manager
close request. The process must be observed exiting before the replacement is
started. `SIGTERM` is an allowed fallback only if the normal close request does
not terminate the process; `SIGKILL` is outside this design.

The replacement uses the existing reviewed full-experience chain:

```text
/home/eii/.local/bin/isaac-sim-clean
  -> /home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/bin/isaacsim
  -> isaacsim.exp.full.kit
```

The approved Stage path is passed as the positional USD argument. The new
Isaac window is moved to workspace 3 (X11 desktop index 2). No timeline play,
joint target, or robot-control command is issued by the agent.

## Runtime Acceptance Gates

The replacement is accepted only when all of the following are verified:

1. The old trimmed-viewer PID has exited.
2. The new process command resolves to `isaacsim.exp.full.kit`.
3. Kit reports `Isaac Sim Full` startup completion without a fatal Stage-open
   error.
4. The active Stage is the approved absolute path.
5. The new window is viewable on X11 desktop index 2.
6. `omni.physx.supportui` reaches `startup` or `started`, so the
   `show_physics_inspector` action can be registered.
7. The Stage SHA-256 remains unchanged.

The user performs the actual Physics Inspector interaction and any manual
joint movement. The agent does not move either simulated arm during startup
verification.

## Persistent Project Policy

The root `AGENTS.md` gains a short hard-policy section with these rules:

- Trimmed `isaacsim.exp.base.python.kit` launches are permitted only for
  agent-run automation, bounded tests, and diagnostics where a full GUI is not
  required.
- Whenever the user explicitly asks an agent to start, launch, or open Isaac
  Sim, the agent must use `isaacsim.exp.full.kit` unless the user explicitly
  requests the trimmed Python experience.
- A user-facing launch is not accepted from its icon or window title alone;
  the process command or Kit log must prove the full experience, and required
  user-facing extensions must be checked when relevant.
- The timeline remains paused and no robot joint is commanded unless the user
  explicitly authorizes simulation control.
- The window goes to the workspace explicitly named by the user; otherwise the
  repository's default Isaac workspace rule applies.
- The GNOME Dock Isaac favorite must resolve to the full-experience chain.
  Inspect it before changing it and verify it after any change.

## GNOME Dock Decision

The current favorite is `isaac-sim-clean.desktop`. Its `Exec` points to
`/home/eii/.local/bin/isaac-sim-clean`, and that wrapper explicitly launches
`isaacsim.exp.full.kit`. It already satisfies the policy, so neither the
`.desktop` file nor its wrapper is changed. The new full GUI is launched
through this same chain to verify the Dock configuration operationally.

## Evidence And Rollback

Full startup output is saved under a new `.codex/artifacts/` directory and
only bounded evidence is reported in conversation. The `AGENTS.md` edit is
reviewed with a focused diff. If full startup fails, the failure is reported
without silently falling back to the trimmed experience. The closed trimmed
process is not automatically recreated.
