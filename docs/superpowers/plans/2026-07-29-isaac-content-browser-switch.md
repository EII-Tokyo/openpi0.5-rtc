# Isaac Sim 5.1 Content Browser Switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove both category-based Asset Browsers from the active Isaac Sim 5.1 startup dependency graph while retaining their on-demand menus and verifying the official Content Browser as the default.

**Architecture:** Patch the Base/Full App dependency graphs that cause the Isaac and NVIDIA Assets browsers to load automatically. Preserve timestamped backups, extension source trees, and both lazy menu triggers, then validate the static dependency graph and fresh Kit sessions with bounded log probes.

**Tech Stack:** Isaac Sim 5.1 Kit configuration, TOML, Python 3.11 `tomllib`, Kit startup logs, POSIX process control.

---

### Task 1: Record Preflight Evidence and Prove the Existing Configuration Fails

**Files:**
- Read: `/home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/lib/python3.11/site-packages/isaacsim/apps/isaacsim.exp.base.kit`
- Read: `/home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.browser/config/extension.toml`
- Read: `/home/eii/.nvidia-omniverse/logs/Kit/Isaac-Sim Full/5.1/kit_*.log`
- Evidence: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/isaac_content_browser_switch_preflight.txt`

- [ ] **Step 1: Query the official NVIDIA Isaac documentation through the Gateway**

Use the `mcpjungle_lab` NVIDIA Isaac documentation tool with the `browsers` instruction set. Confirm that Content Browser is the main browser and that Isaac Sim Asset Browser is Beta with an official recommendation to use Content Browser.

- [ ] **Step 2: Resolve the exact running Isaac Sim process**

Run:

```bash
ps -eo pid,lstart,etimes,cmd | rg '/home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/bin/isaacsim .*isaacsim.exp.full.kit'
```

Expected: exactly one GUI process. Record its PID and exact command. Stop if zero or more than one matches.

- [ ] **Step 3: Record the current dependency and trigger lines**

Run:

```bash
rg -n '"isaacsim\.asset\.browser"|"omni\.isaac\.asset_browser"|"isaacsim\.gui\.content_browser"' \
  /home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/lib/python3.11/site-packages/isaacsim/apps/isaacsim.exp.base.kit
rg -n '^\[\[trigger\]\]|^menu\.name = "Window/Browsers/Isaac"|^menu\.window = "Isaac Sim Assets"' \
  /home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.browser/config/extension.toml
```

Expected: all three legacy dependencies, the official Content Browser dependency, and all three trigger lines are present.

- [ ] **Step 4: Run the RED configuration probe**

Run:

```bash
/home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/bin/python - <<'PY'
from pathlib import Path
import tomllib

base = Path("/home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/lib/python3.11/site-packages/isaacsim/apps/isaacsim.exp.base.kit")
legacy = Path("/home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.browser/config/extension.toml")
base_data = tomllib.loads(base.read_text(encoding="utf-8"))
legacy_data = tomllib.loads(legacy.read_text(encoding="utf-8"))
deps = base_data["dependencies"]
assert "isaacsim.gui.content_browser" in deps
assert "isaacsim.asset.browser" not in deps
assert "omni.isaac.asset_browser" not in deps
assert "omni.isaac.assets_check" not in deps
assert "trigger" not in legacy_data
PY
```

Expected: FAIL on `assert "isaacsim.asset.browser" not in deps`. This proves the probe detects the current defect.

### Task 2: Stop Isaac Sim, Back Up Both Configurations, and Apply the Minimal Repair

**Files:**
- Modify: `/home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/lib/python3.11/site-packages/isaacsim/apps/isaacsim.exp.base.kit`
- Modify: `/home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.browser/config/extension.toml`
- Create: timestamped `.bak.20260729_*` file beside each modified file

- [ ] **Step 1: Gracefully stop only the authorized Isaac Sim GUI**

Send `SIGTERM` to the exact PID recorded in Task 1. Poll its existence for up to 30 seconds. Expected: the PID exits and no other matching Full GUI process remains. Do not send `SIGKILL` without a new explicit decision.

- [ ] **Step 2: Create and verify timestamped backups**

Use one timestamp tag for both backups:

```bash
BACKUP_TAG=$(date +%Y%m%d_%H%M%S)
cp -a /home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/lib/python3.11/site-packages/isaacsim/apps/isaacsim.exp.base.kit \
  /home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/lib/python3.11/site-packages/isaacsim/apps/isaacsim.exp.base.kit.bak."$BACKUP_TAG"
cp -a /home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.browser/config/extension.toml \
  /home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.browser/config/extension.toml.bak."$BACKUP_TAG"
```

Assert that both backups exist, are regular files, and are byte-identical to their originals with `cmp`.

- [ ] **Step 3: Remove the three legacy dependencies**

Apply this exact patch:

```diff
 [dependencies]
 # Isaac Sim extensions
 "isaacsim.app.about" = {}
-"isaacsim.asset.browser" = {}
 "isaacsim.core.api" = {}
```

Apply the second exact hunk:

```diff
 # Deprecated extensions for backwards compatibility
-"omni.isaac.asset_browser" = {}
-"omni.isaac.assets_check" = {}
 "omni.isaac.cloner" = {}
```

Do not change `"isaacsim.gui.content_browser" = {}`.

- [ ] **Step 4: Remove the legacy lazy menu trigger**

Apply this exact patch:

```diff
-[[trigger]]
-menu.name = "Window/Browsers/Isaac"
-menu.window = "Isaac Sim Assets"
-
 [[test]]
```

Do not change the extension's Python source or folder URLs.

- [ ] **Step 5: Validate both TOML files**

Run:

```bash
/home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/bin/python - <<'PY'
from pathlib import Path
import tomllib

paths = [
    Path("/home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/lib/python3.11/site-packages/isaacsim/apps/isaacsim.exp.base.kit"),
    Path("/home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.browser/config/extension.toml"),
]
for path in paths:
    with path.open("rb") as stream:
        tomllib.load(stream)
    print(f"valid_toml={path}")
PY
```

Expected: two `valid_toml=` lines and exit code 0. If validation fails, restore both backups immediately.

- [ ] **Step 6: Run the GREEN configuration probe**

Run the exact Python probe from Task 1 Step 4.

Expected: PASS with exit code 0.

- [ ] **Step 7: Verify package preservation**

Run:

```bash
/home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/bin/python -c \
  'from importlib.metadata import version; print(version("isaacsim-asset"))'
test -f /home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/lib/python3.11/site-packages/isaacsim/exts/isaacsim.asset.browser/isaacsim/asset/browser/model.py
```

Expected: version `5.1.0.0` and exit code 0.

### Task 3: First Post-Repair Launch and Runtime Acceptance

**Files:**
- Read: new `/home/eii/.nvidia-omniverse/logs/Kit/Isaac-Sim Full/5.1/kit_*.log`
- Evidence: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/isaac_content_browser_switch_launch1.txt`

- [ ] **Step 1: Start the same Full Isaac Sim application**

From `/home/eii/project/openpi0.5-rtc-reward-learning`, launch:

```bash
env OMNI_KIT_ACCEPT_EULA=YES \
  /home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/bin/python \
  /home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/bin/isaacsim \
  /home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/lib/python3.11/site-packages/isaacsim/apps/isaacsim.exp.full.kit
```

Run it as a managed long-lived process so stdout/stderr are captured without detaching an untracked process.

- [ ] **Step 2: Wait for a fresh ready signal**

Resolve the new log by creation time after the launch. Poll at intervals no longer than 10 seconds for up to 180 seconds.

Acceptance signal:

```text
app ready
Isaac Sim Full App is loaded
```

Failure signal: process exit, fatal startup error, or no ready signal within 180 seconds.

- [ ] **Step 3: Verify the official browser starts and legacy browsers do not**

In the fresh log, assert:

```text
About to startup: [ext: isaacsim.gui.content_browser-
```

is present, while these startup/apply-settings patterns are absent:

```text
About to startup: [ext: isaacsim.asset.browser-
About to startup: [ext: omni.isaac.asset_browser-
About to startup: [ext: omni.isaac.assets_check-
[ext: isaacsim.asset.browser-1.3.23] applying settings
[ext: omni.isaac.asset_browser-1.0.6] applying settings
```

Discovery-only `registered` lines are allowed because the source manifests remain installed.

- [ ] **Step 4: Verify the performance defect is absent**

Assert zero matches in the fresh log for:

```text
Thumbnail .* does not belong to file
isaacsim.asset.browser.cache.json
Start traverse from queue: https://omniverse-content-production.*Assets/Isaac/5.1/Isaac/Robots
Start traverse from queue: https://omniverse-content-production.*Assets/Isaac/5.1/Isaac/Environments
```

Record the ready time, log size, warning count, and process RSS/CPU in the launch-1 evidence file.

- [ ] **Step 5: Roll back on failure**

If Steps 2 through 4 fail, gracefully stop the failed GUI, restore both timestamped backups, validate both TOML files, restart the original Full app, and report the failed acceptance condition. Do not continue to Task 4.

### Task 4: Second Launch Persistence Check and Final Handoff

**Files:**
- Read: second new `/home/eii/.nvidia-omniverse/logs/Kit/Isaac-Sim Full/5.1/kit_*.log`
- Evidence: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/isaac_content_browser_switch_launch2.txt`

- [ ] **Step 1: Gracefully stop the first repaired GUI**

Resolve the exact post-repair PID, send `SIGTERM`, and poll for up to 30 seconds. Expected: process exits cleanly.

- [ ] **Step 2: Start the Full app a second time**

From `/home/eii/project/openpi0.5-rtc-reward-learning`, launch:

```bash
env OMNI_KIT_ACCEPT_EULA=YES \
  /home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/bin/python \
  /home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/bin/isaacsim \
  /home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/lib/python3.11/site-packages/isaacsim/apps/isaacsim.exp.full.kit
```

Resolve a distinct new PID and a log created after this command.

- [ ] **Step 3: Verify the second ready signal**

Poll the second log at intervals no longer than 10 seconds for up to 180 seconds. Require both:

```text
app ready
Isaac Sim Full App is loaded
```

Treat process exit, a fatal startup error, or no ready signal within 180 seconds as failure.

- [ ] **Step 4: Verify second-launch browser selection and performance**

Require this startup pattern:

```text
About to startup: [ext: isaacsim.gui.content_browser-
```

Require zero matches for all of these patterns:

```text
About to startup: [ext: isaacsim.asset.browser-
About to startup: [ext: omni.isaac.asset_browser-
About to startup: [ext: omni.isaac.assets_check-
About to startup: [ext: omni.kit.browser.asset-
[ext: isaacsim.asset.browser-1.3.23] applying settings
[ext: omni.isaac.asset_browser-1.0.6] applying settings
Thumbnail .* does not belong to file
isaacsim.asset.browser.cache.json
omni.kit.browser.asset.cache.json
Add folder to queue: https://omniverse-content-production
Start traverse from queue: https://omniverse-content-production.*Assets/Isaac/5.1/Isaac/Robots
Start traverse from queue: https://omniverse-content-production.*Assets/Isaac/5.1/Isaac/Environments
```

Record the ready time, log size, warning count, and process RSS/CPU in the launch-2 evidence file.

- [ ] **Step 5: Confirm the repaired GUI remains running**

Verify the second PID is alive and corresponds to the Full app command. Leave this repaired Isaac Sim window running for the user.

- [ ] **Step 6: Preserve rollback data and report**

Report both backup paths, both Kit log paths, both evidence artifact paths, startup times, warning counts, and the final PID. Do not delete backups or global Kit/shader caches.

### Task 5: Remove NVIDIA Assets from Automatic Startup

The second repaired GUI was intentionally left running. At 96 seconds, opening
`Window/Browsers/Assets` instantiated `omni.kit.browser.asset`, queued eight
remote S3 roots, and produced 17,892 thumbnail mismatch warnings. This is the
expected cost of explicitly opening that category browser, not an automatic
startup defect. Its direct Full App dependency is still unnecessary because
Content Browser is the selected default.

**Files:**
- Modify: `/home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/lib/python3.11/site-packages/isaacsim/apps/isaacsim.exp.full.kit`
- Modify: `/home/eii/project/openpi0.5-rtc-reward-learning/.venv_issac/lib/python3.11/site-packages/isaacsim/extscache/omni.kit.browser.asset-1.3.12/config/extension.toml`
- Create: `.bak.20260729_114430` beside each modified file
- Read: `/home/eii/.nvidia-omniverse/logs/Kit/Isaac-Sim Full/5.1/kit_20260729_114500.log`

- [ ] **Step 1: Run a RED probe and stop only the authorized GUI**

Assert that `omni.kit.browser.asset` is absent from the Full App dependencies.
Require this probe to fail, then send `SIGTERM` only to the verified Full App
PID.

- [ ] **Step 2: Back up and apply the startup-only patch**

Create and byte-verify both timestamped backups. Remove only
`"omni.kit.browser.asset" = {}` from `isaacsim.exp.full.kit`.

- [ ] **Step 3: Run the GREEN probe**

Parse all four TOML files. Require the official Content Browser dependency,
require all four Asset Browser startup dependencies to be absent, and require
both Asset Browser manifests to retain their original lazy menu triggers.

- [ ] **Step 4: Start the final GUI and verify the pre-interaction state**

Start the Full App and require `app ready` and
`Isaac Sim Full App is loaded`. Before manual browser interaction, require one
`isaacsim.gui.content_browser` startup and zero matches for both Asset Browser
startups, both Asset Browser cache files, and Asset remote-folder queueing.
Leave this final repaired GUI running for the user.

### Task 6: Restore Both On-Demand Menus Per User Direction

The user clarified that the remote traversal observed during verification was
caused by their manual menu action and explicitly requested that no browser
menus be removed.

- [ ] Restore `Window/Browsers/Isaac` and `Window/Browsers/Assets` trigger
  blocks from their verified backups.
- [ ] Require both manifests to be byte-identical to their pre-edit backups.
- [ ] Restart Isaac Sim and verify Content Browser starts once, neither Asset
  Browser starts before manual interaction, and both menu triggers remain in
  the parsed TOML.
