# Machine 103 uv Development Environment Design

## Objective

Create an isolated development environment in the current project directory
that reproduces machine 103's host-side Python dependency lock without
replacing the existing `.venv` or modifying the local tracked
`pyproject.toml` and `uv.lock`.

## Source of Truth

The source snapshot is read from
`/home/eii/openpi0.5-rtc-reward-learning` on machine 103 and contains its exact
`pyproject.toml` and `uv.lock`. Their SHA-256 hashes are recorded alongside the
snapshot under `.codex/artifacts/103-uv-environment/`.

The target toolchain is:

- uv 0.11.24, matching machine 103;
- uv-managed CPython 3.12.3, matching the machine 103 host Python version;
- the exact machine 103 lockfile, synchronized with `--frozen`.

The ROS container's Python 3.10.12 is not a target because the project declares
`requires-python = ">=3.11"`.

## Isolation

The new environment is `.venv-103` beneath the current project. The existing
`.venv` remains unchanged. The remote configuration snapshot is used as a
temporary uv project with `UV_PROJECT_ENVIRONMENT` set to the absolute
`.venv-103` path. Synchronization uses `--no-install-project` because the
snapshot contains dependency metadata but not a second copy of the project
source tree.

No remote file is modified. No local tracked dependency file is overwritten.
If `.venv-103` already exists, the operation stops rather than replacing it.

## Verification

Before installation, perform a uv dry run against the frozen remote snapshot.
After synchronization, verify:

- `.venv-103/bin/python` reports Python 3.12.3;
- the invoked uv reports 0.11.24;
- `uv sync --check` succeeds against the frozen snapshot;
- `uv pip check --python .venv-103/bin/python` reports compatible packages;
- the local and captured remote configuration hashes are unchanged;
- the pre-existing `.venv` interpreter and tracked Git files are unchanged.

Package inventory is saved as an artifact for later comparison with any
future environment created directly on machine 103.
