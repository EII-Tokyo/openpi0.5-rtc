# Isaac Sim 5.1 Content Browser Switch Design

## Goal

Stop Isaac Sim 5.1 from loading the defective legacy `isaacsim.asset.browser`
extension, retain the supported `isaacsim.gui.content_browser`, and verify that
the legacy remote traversal and thumbnail-warning storm no longer occur.

## Scope

- Modify only the active Python environment used by the running Isaac Sim:
  `/home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac`.
- Remove the legacy browser from the runtime dependency graph, including its
  deprecated compatibility alias.
- Preserve the installed `isaacsim-asset` wheel files so package metadata,
  upgrades, and repair installs remain valid.
- Preserve the current Stage and project assets. Do not open, replace, save, or
  modify a Stage.
- Do not change robot, ROS, physics, or rendering behavior.

## Design

The active application inherits its extension dependencies from
`isaacsim/apps/isaacsim.exp.base.kit`. The repair will make a timestamped backup
of that file and remove exactly these two dependency entries:

```toml
"isaacsim.asset.browser" = {}
"omni.isaac.asset_browser" = {}
```

The existing official dependency remains unchanged:

```toml
"isaacsim.gui.content_browser" = {}
```

The extension source directories will not be physically deleted. In this
design, “remove the old browser” means it is absent from the resolved runtime
extension graph: it does not register, start, create menus, traverse S3, or
write Asset Browser cache data.

## Execution Flow

1. Record the running Isaac Sim PID, command, current Kit log, and dependency
   entries.
2. Run a failing pre-repair probe that asserts the legacy dependencies are
   absent; it must fail because both are currently present.
3. Stop only the authorized Isaac Sim GUI process and verify it exits.
4. Create a timestamped backup of `isaacsim.exp.base.kit`.
5. Remove only the two legacy dependency lines with a deterministic patch.
6. Validate TOML syntax and assert that Content Browser remains enabled while
   both legacy dependencies are absent.
7. Start Isaac Sim with the same Full app entry point and verify application
   readiness.
8. Inspect the new Kit log for extension registration/startup and traversal
   behavior.
9. Close and start Isaac Sim a second time to verify persistence across
   launches.

## Acceptance Criteria

- `isaacsim.gui.content_browser` registers and starts.
- Neither `isaacsim.asset.browser` nor `omni.isaac.asset_browser` registers or
  starts.
- No request traverses the legacy hard-coded
  `.../Assets/Isaac/5.1/Isaac/Robots` or `Environments` roots on behalf of
  `isaacsim.asset.browser`.
- The new log contains zero
  `Thumbnail ... does not belong to file ...` warnings.
- The new log contains no reference to
  `isaacsim.asset.browser.cache.json`.
- Isaac Sim reaches its normal ready signal on both the first and second
  post-repair launches.
- The installed `isaacsim-asset` distribution remains present and its extension
  source files remain intact.

## Failure Handling and Rollback

- If the edited TOML fails validation, restore the backup before starting
  Isaac Sim.
- If Isaac Sim does not reach readiness or Content Browser fails to start,
  stop the failed process, restore the backup, restart the original
  configuration, and report the failed acceptance signal.
- Never delete the backup during this task.
- Do not clear global Kit or shader caches; they are unrelated to the confirmed
  root cause and clearing them would make startup slower.

## Verification Evidence

Full startup logs remain under
`~/.nvidia-omniverse/logs/Kit/Isaac-Sim Full/5.1/`. Bounded diagnostic summaries
will be stored under the repository `.codex/artifacts/` directory. Verification
will report extension registration, ready time, legacy traversal count,
thumbnail-warning count, cache-error count, and process state for both
post-repair launches.
