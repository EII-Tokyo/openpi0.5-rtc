# Backend Test Environment

Read this before running or changing backend tests, local pytest setup, segment DB paths, or test-only environment defaults.

- Use the uv-managed project environment for all project Python tests:
  `.venv/bin/python -m pytest ...` or `uv run pytest ...`. Do not invoke a
  bare `python`, `python3`, or `pytest` and then interpret system/user-site
  NumPy or SciPy collection failures as project failures.
- The verified 2026-07-29 project environment uses Python `3.11.13`,
  NumPy `2.4.0`, and SciPy `1.15.3`; `uv.lock` is authoritative for the
  resolved versions. The `.venv` does not need to expose `pip`; use `uv` for
  dependency changes instead of installing into system Python.
- Local backend pytest runs are not inside the 103 Docker containers, so they must not rely on container-only paths such as `/app/segment_db/segments.sqlite3`.
- Backend tests under `voice_assistant_web/backend/app` use a test-only `conftest.py` to set `RLT_SEGMENT_DB_PATH`, `RLT_STATE_PATH`, and `EII_PILOT_ENABLE_ROS` before importing backend modules.
- Keep those test defaults isolated and temporary. Do not change production defaults, `.env`, or compose mounts to make local tests pass.
- On `192.168.1.103`, `/app/segment_db/segments.sqlite3` is valid only inside the running containers because compose mounts `/data/openpi0.5-rtc-reward-learning/segment_db` there. The 103 host itself should use host paths under `/data/openpi0.5-rtc-reward-learning` when a host-side command needs the segment DB.
