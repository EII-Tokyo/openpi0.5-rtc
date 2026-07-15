# Run Aloha Sim

## Minimal workcell viewer

This repository includes a small offline MuJoCo viewer for learning the ALOHA
workcell model before building the bottle-mouth insertion environment.

It renders:

- the base `gym_aloha` dual-arm ViperX / ALOHA model,
- the table,
- a visible table frame `T`,
- a red placeholder pipe axis,
- an optional real HDF5 `observations/qpos` replay.

Run it from the repository root:

```bash
MUJOCO_GL=egl .venv/bin/python examples/aloha_sim/workcell_viewer.py
```

Outputs:

```text
local_eval_assets/aloha_workcell_minimal/model/workcell.xml
local_eval_assets/aloha_workcell_minimal/aloha_workcell_replay.mp4
```

The generated `workcell.xml` is the first file to inspect when learning how the
environment changes. It includes the original `gym_aloha` assets through
relative links and adds only the table frame and placeholder pipe.

Useful parameters:

```bash
MUJOCO_GL=egl .venv/bin/python examples/aloha_sim/workcell_viewer.py \
  --hdf5 /path/to/episode.hdf5 \
  --camera angle \
  --pipe-start 0.45,0.58,0.36 \
  --pipe-end 0.25,0.45,0.22 \
  --stride 2
```

Interpretation:

- `--pipe-start` and `--pipe-end` define the current placeholder pipe axis.
- `--hdf5` loads a real rollout and maps its 14D ALOHA qpos into the MuJoCo model.
- The gripper scalar in the HDF5 is mapped to the two MuJoCo slide fingers.
- This is not yet a calibrated bottle insertion simulator. It is the base
  environment used to incrementally add measured table, pipe, bottle, and
  grasp parameters.

## With Docker

```bash
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/aloha_sim/compose.yml up --build
```

## Without Docker

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.10 examples/aloha_sim/.venv
source examples/aloha_sim/.venv/bin/activate
uv pip sync examples/aloha_sim/requirements.txt
uv pip install -e packages/openpi-client

# Run the simulation
MUJOCO_GL=egl python examples/aloha_sim/main.py
```

Note: If you are seeing EGL errors, you may need to install the following dependencies:

```bash
sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env ALOHA_SIM
```
