# Machine 103 uv Development Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `.venv-103` in the current project using machine 103's exact dependency lock, uv 0.11.24, and Python 3.12.3 without modifying the existing `.venv` or local dependency files.

**Architecture:** Capture the two remote source-of-truth files in a timestamped project artifact, invoke uv 0.11.24 through an isolated `uvx` tool environment, and point `UV_PROJECT_ENVIRONMENT` at the absolute local `.venv-103` path. Use frozen, dependency-only synchronization so the snapshot is never rewritten and its absent source tree is not installed.

**Tech Stack:** uv/uvx, CPython 3.12.3, SSH, SHA-256, POSIX shell.

---

### Task 1: Capture and validate machine 103 metadata

**Files:**
- Create: `.codex/artifacts/103-uv-environment/<timestamp>/pyproject.toml`
- Create: `.codex/artifacts/103-uv-environment/<timestamp>/uv.lock`
- Create: `.codex/artifacts/103-uv-environment/<timestamp>/metadata.txt`

- [ ] **Step 1: Assert the target environment does not exist**

Run: `test ! -e /home/eii/project/openpi0.5-rtc-reward-learning/.venv-103`

Expected: exit 0. Stop instead of replacing an existing target.

- [ ] **Step 2: Record local pre-operation hashes**

Run:

```bash
sha256sum pyproject.toml uv.lock .venv/bin/python
.venv/bin/python --version
```

Expected: local configuration hashes are recorded and existing Python remains 3.11.13.

- [ ] **Step 3: Copy the remote dependency files into a new artifact directory**

Use an explicit timestamped path beneath `.codex/artifacts/103-uv-environment/` and copy only:

```text
/home/eii/openpi0.5-rtc-reward-learning/pyproject.toml
/home/eii/openpi0.5-rtc-reward-learning/uv.lock
```

- [ ] **Step 4: Verify the captured hashes against a fresh remote readback**

Run SHA-256 locally on the captured files and remotely from the approved project directory. Expected hashes must match pairwise before continuing.

### Task 2: Validate the exact uv and Python toolchain

- [ ] **Step 1: Verify isolated uv 0.11.24 execution**

Run: `uvx --isolated --from 'uv==0.11.24' uv --version`

Expected: `uv 0.11.24`.

- [ ] **Step 2: Verify Python 3.12.3 availability**

Run: `/usr/bin/python3.12 --version`

Expected: `Python 3.12.3`.

- [ ] **Step 3: Dry-run the frozen synchronization**

Run from the captured snapshot directory with:

```bash
UV_PROJECT_ENVIRONMENT=/home/eii/project/openpi0.5-rtc-reward-learning/.venv-103 \
uvx --isolated --from 'uv==0.11.24' uv sync \
  --frozen --no-install-project --no-install-workspace \
  --python /usr/bin/python3.12 --dry-run
```

Expected: the command plans creation of `.venv-103`, does not report lockfile mutation, and exits 0.

### Task 3: Create and verify `.venv-103`

- [ ] **Step 1: Execute the frozen synchronization**

Run the Task 2 command without `--dry-run`. Expected: `.venv-103` is created using CPython 3.12.3 and the remote lock.

- [ ] **Step 2: Verify interpreter and dependency consistency**

Run:

```bash
.venv-103/bin/python --version
uvx --isolated --from 'uv==0.11.24' uv pip check \
  --python .venv-103/bin/python
```

Expected: Python 3.12.3 and no incompatible packages.

- [ ] **Step 3: Verify the environment remains synchronized**

Run from the captured snapshot directory:

```bash
UV_PROJECT_ENVIRONMENT=/home/eii/project/openpi0.5-rtc-reward-learning/.venv-103 \
uvx --isolated --from 'uv==0.11.24' uv sync \
  --frozen --no-install-project --no-install-workspace \
  --python /usr/bin/python3.12 --check
```

Expected: environment is synchronized and the command exits 0.

- [ ] **Step 4: Record package inventory and post-operation hashes**

Save `uv pip freeze --python .venv-103/bin/python` in the same artifact directory. Recompute local `pyproject.toml`, `uv.lock`, `.venv/bin/python`, and captured remote-file hashes; all pre-existing inputs must be unchanged.

- [ ] **Step 5: Verify Git scope**

Run `git status --short` and confirm only the user's pre-existing dirty files remain. `.venv-103` and `.codex/artifacts/` must not appear as tracked changes.
