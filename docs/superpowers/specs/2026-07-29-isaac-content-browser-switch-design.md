# Isaac Sim 5.1 Content Browser Switch Design

## Goal

Stop Isaac Sim 5.1 from loading either category-based Asset Browser
(`isaacsim.asset.browser` or `omni.kit.browser.asset`), retain the supported
`isaacsim.gui.content_browser`, and verify that remote traversal and the
thumbnail-warning storm no longer occur.

## Scope

- Modify only the active Python environment used by the running Isaac Sim:
  `/home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac`.
- Remove the legacy browser from the runtime dependency graph, including its
  deprecated compatibility alias.
- Remove the NVIDIA Assets category browser from the Full App dependency graph
  and browser menu.
- Preserve the installed legacy extension source directory rather than
  physically deleting files from the `isaacsim-asset` distribution.
- Preserve the current Stage and project assets. Do not open, replace, save, or
  modify a Stage.
- Do not change robot, ROS, physics, or rendering behavior.

## Design

The active application inherits its extension dependencies from
`isaacsim/apps/isaacsim.exp.base.kit`. The repair will make a timestamped backup
of that file and remove exactly these three dependency entries:

```toml
"isaacsim.asset.browser" = {}
"omni.isaac.asset_browser" = {}
"omni.isaac.assets_check" = {}
```

The existing official dependency remains unchanged:

```toml
"isaacsim.gui.content_browser" = {}
```

`omni.isaac.assets_check` is a deprecated compatibility extension whose
manifest depends on `isaacsim.asset.browser`. Removing only the two browser
entries leaves this indirect dependency able to start the old browser, so all
three entries must be removed from the Base App graph.

The legacy extension manifest also declares a lazy-loading `[[trigger]]` for
`Window/Browsers/Isaac`. The repair will back up that manifest and remove only
this trigger block so the old menu cannot re-enable the extension:

```toml
[[trigger]]
menu.name = "Window/Browsers/Isaac"
menu.window = "Isaac Sim Assets"
```

Runtime persistence testing exposed a second category-based browser matching
the reported screenshot: `omni.kit.browser.asset`. The Full App depends on it
directly, its menu is `Window/Browsers/Assets`, and its default settings queue
eight remote S3 roots. When the window opened 96 seconds after startup, it
produced more than 17,000 thumbnail mismatch warnings. The repair therefore
also backs up `isaacsim.exp.full.kit` and
`omni.kit.browser.asset-1.3.12/config/extension.toml`, removes the direct
dependency, and removes this lazy menu trigger:

```toml
[[trigger]]
menu.name = "Window/Browsers/Assets"
menu.window = "NVIDIA Assets"
```

The extension source directories will not be physically deleted. Kit may still
emit discovery or registration lines for installed extension manifests. In
this design, “remove the old browser” applies to both category-based Asset
Browsers: neither is selected, enabled, started, presented as a browser menu,
allowed to traverse S3, or allowed to write Asset Browser cache data.

## Execution Flow

1. Record the running Isaac Sim PID, command, current Kit log, and dependency
   entries.
2. Run a failing pre-repair probe that asserts the legacy dependencies are
   absent; it must fail because both are currently present.
3. Stop only the authorized Isaac Sim GUI process and verify it exits.
4. Create timestamped backups beside all four configuration files.
5. Remove only the four Asset Browser dependency lines and both three-line lazy
   trigger blocks with deterministic patches.
6. Validate all four TOML files and assert that Content Browser remains enabled
   while all four Asset Browser dependencies and both menu triggers are absent.
7. Start Isaac Sim with the same Full app entry point and verify application
   readiness.
8. Inspect the new Kit log for extension registration/startup and traversal
   behavior.
9. Close and start Isaac Sim a second time to verify persistence across
   launches.

## Acceptance Criteria

- `isaacsim.gui.content_browser` registers and starts.
- Neither `isaacsim.asset.browser` nor `omni.isaac.asset_browser` is selected,
  enabled, or started.
- The deprecated `omni.isaac.assets_check` shim is not selected or started.
- `omni.kit.browser.asset` is not selected or started.
- `Window/Browsers/Isaac` and `Window/Browsers/Assets` are absent while the
  official Content Browser remains available.
- No request traverses the legacy hard-coded
  `.../Assets/Isaac/5.1/Isaac/Robots` or `Environments` roots on behalf of
  `isaacsim.asset.browser`.
- The new log contains zero
  `Thumbnail ... does not belong to file ...` warnings.
- The new log contains no reference to
  `isaacsim.asset.browser.cache.json` or
  `omni.kit.browser.asset.cache.json`.
- Isaac Sim reaches its normal ready signal on both the first and second
  post-repair launches.
- The installed `isaacsim-asset` distribution remains present and its legacy
  extension source directory remains available for rollback or package repair.

## Failure Handling and Rollback

- If any edited TOML file fails validation, restore all corresponding backups
  before starting Isaac Sim.
- If Isaac Sim does not reach readiness or Content Browser fails to start,
  stop the failed process, restore both backups, restart the original
  configuration, and report the failed acceptance signal.
- Never delete either backup during this task.
- Do not clear global Kit or shader caches; they are unrelated to the confirmed
  root cause and clearing them would make startup slower.

## Verification Evidence

Full startup logs remain under
`~/.nvidia-omniverse/logs/Kit/Isaac-Sim Full/5.1/`. Bounded diagnostic summaries
will be stored under the repository `.codex/artifacts/` directory. Verification
will report extension registration, ready time, legacy traversal count,
thumbnail-warning count, cache-error count, and process state for both
post-repair launches.
