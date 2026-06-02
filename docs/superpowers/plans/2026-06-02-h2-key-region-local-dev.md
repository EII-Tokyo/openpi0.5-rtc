# H2 Key Region Local Development Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the `reward-learning` branch on H2 under `~/project` so H2 can develop the key-region UI and review, split, clean, and prepare key-region data locally before sending trained actor-critic work back to machine 103 for robot testing.

**Architecture:** Machine 103 remains the only robot-connected runtime host. H2 uses a local repository plus local data mirrors under `~/project`, runs backend/frontend without Docker for UI and data-cleaning development, and only syncs selected data from 103 to H2. Any write-back to 103 must be explicit and one-way after review.

**Tech Stack:** Git, rsync/ssh, uv Python 3.11 environment, FastAPI backend, Vite/React frontend, SQLite segment ledger, NumPy replay shards.

---

### Task 1: Local Repository And Data Roots

**Files:**
- Use: `/home/eii/project/openpi0.5-rtc-reward-learning`
- Use: `/home/eii/project/openpi0.5-rtc-reward-learning-local-data`

- [x] **Step 1: Create H2 project root**

```bash
mkdir -p /home/eii/project
```

Expected: `/home/eii/project` exists.

- [x] **Step 2: Clone reward-learning**

```bash
cd /home/eii/project
git clone --branch reward-learning git@github.com:EII-Tokyo/openpi0.5-rtc.git openpi0.5-rtc-reward-learning
```

Expected: repository exists at `/home/eii/project/openpi0.5-rtc-reward-learning`.

- [x] **Step 3: Pull 103-only commits**

```bash
cd /home/eii/project/openpi0.5-rtc-reward-learning
git remote add machine103 eii@192.168.1.103:/home/eii/openpi0.5-rtc-reward-learning 2>/dev/null || true
git fetch machine103 reward-learning
git reset --hard FETCH_HEAD
```

Expected: `HEAD` is `14df78f Add auto beta control for online RLT`.

- [x] **Step 4: Preserve 103 dirty key-region UI work**

The 103 working tree had these uncommitted files and they are now present on H2:

```text
voice_assistant_web/frontend/src/components/RolloutBrowser.tsx
voice_assistant_web/frontend/src/styles.css
docs/design-previews/key-regions-upgrade-preview.html
```

Verify:

```bash
cd /home/eii/project/openpi0.5-rtc-reward-learning
git status --short
```

Expected: the three paths above are dirty/untracked unless they have since been committed.

- [x] **Step 5: Create H2 local data roots**

```bash
mkdir -p /home/eii/project/openpi0.5-rtc-reward-learning-local-data/{rollouts,replay,replay_clean,rlt_online,segment_db}
```

Expected: all data directories are under `~/project`, not `/data`.

### Task 2: Safe Data Sync From 103 To H2

**Files:**
- Read from 103: `/data/openpi0.5-rtc-reward-learning/rollouts`
- Read from 103: `/data/openpi0.5-rtc-reward-learning/replay`
- Read from 103: `/data/openpi0.5-rtc-reward-learning/segment_db`
- Optional read from 103: `/data/openpi0.5-rtc-reward-learning/rlt_online`
- Write to H2: `/home/eii/project/openpi0.5-rtc-reward-learning-local-data`

- [x] **Step 1: Check 103 data size before syncing**

```bash
ssh eii@192.168.1.103 'du -sh /data/openpi0.5-rtc-reward-learning/rollouts /data/openpi0.5-rtc-reward-learning/replay /data/openpi0.5-rtc-reward-learning/segment_db /data/openpi0.5-rtc-reward-learning/rlt_online 2>/dev/null || true'
```

Observed on 2026-06-02:

```text
2.7G  /data/openpi0.5-rtc-reward-learning/rollouts
35M   /data/openpi0.5-rtc-reward-learning/replay
132K  /data/openpi0.5-rtc-reward-learning/segment_db
2.9G  /data/openpi0.5-rtc-reward-learning/rlt_online
```

- [ ] **Step 2: Sync review data one-way from 103 to H2**

Use trailing slashes to copy contents into the matching local folders. Do not use `--delete` for routine cleaning work, because H2 may have local review notes, cleaned replay, or in-progress splits.

```bash
rsync -aP eii@192.168.1.103:/data/openpi0.5-rtc-reward-learning/rollouts/ \
  /home/eii/project/openpi0.5-rtc-reward-learning-local-data/rollouts/

rsync -aP eii@192.168.1.103:/data/openpi0.5-rtc-reward-learning/replay/ \
  /home/eii/project/openpi0.5-rtc-reward-learning-local-data/replay/

rsync -aP eii@192.168.1.103:/data/openpi0.5-rtc-reward-learning/segment_db/ \
  /home/eii/project/openpi0.5-rtc-reward-learning-local-data/segment_db/
```

Expected: H2 can browse key-region videos, replay shards, and segment ledger state locally.

- [x] **Step 2 verification: H2 mirror populated**

Executed on 2026-06-02 without `--delete`:

```bash
rsync -aP eii@192.168.1.103:/data/openpi0.5-rtc-reward-learning/segment_db/ /home/eii/project/openpi0.5-rtc-reward-learning-local-data/segment_db/
rsync -aP eii@192.168.1.103:/data/openpi0.5-rtc-reward-learning/replay/ /home/eii/project/openpi0.5-rtc-reward-learning-local-data/replay/
rsync -aP eii@192.168.1.103:/data/openpi0.5-rtc-reward-learning/rollouts/ /home/eii/project/openpi0.5-rtc-reward-learning-local-data/rollouts/
```

Expected: these commands only pull from 103 into H2 and do not modify 103.

- [ ] **Step 3: Sync trained actor output only when needed**

```bash
rsync -aP eii@192.168.1.103:/data/openpi0.5-rtc-reward-learning/rlt_online/ \
  /home/eii/project/openpi0.5-rtc-reward-learning-local-data/rlt_online/
```

Expected: H2 can inspect latest published actor metadata without touching 103.

### Task 3: Python Backend And RLT Test Environment

**Files:**
- Use: `/home/eii/project/openpi0.5-rtc-reward-learning/.venv`
- Use: `/home/eii/project/openpi0.5-rtc-reward-learning/voice_assistant_web/backend/requirements.txt`

- [x] **Step 1: Install backend dependencies**

```bash
cd /home/eii/project/openpi0.5-rtc-reward-learning
uv pip install -r voice_assistant_web/backend/requirements.txt
```

Expected: FastAPI, Uvicorn, Redis client, and backend dependencies are installed in the uv environment.

- [x] **Step 2: Run key-region/RLT tests with H2-local paths**

```bash
cd /home/eii/project/openpi0.5-rtc-reward-learning
PYTHONDONTWRITEBYTECODE=1 \
ROLLOUTS_ROOT=/home/eii/project/openpi0.5-rtc-reward-learning-local-data/rollouts \
REPLAY_ROOT=/home/eii/project/openpi0.5-rtc-reward-learning-local-data/replay \
RLT_SEGMENT_DB_PATH=/home/eii/project/openpi0.5-rtc-reward-learning-local-data/segment_db/segments.sqlite3 \
RLT_STATE_PATH=/home/eii/project/openpi0.5-rtc-reward-learning-local-data/segment_db/rlt_control_state.json \
uv run pytest \
  voice_assistant_web/backend/app/rlt_segment_ledger_test.py \
  voice_assistant_web/backend/app/rlt_control_test.py \
  voice_assistant_web/backend/app/rollout_tree_test.py \
  examples/aloha_real/rlt_key_region_recorder_test.py \
  src/openpi/training/rlt_replay_store_test.py \
  scripts/train_rlt_online_test.py \
  -q -p no:cacheprovider
```

Expected: `48 passed`.

- [x] **Step 3: Run expanded runtime/RLT regression tests before code changes**

Use this larger set when changing key-region cleaning, actor runtime, or online RLT behavior:

```bash
cd /home/eii/project/openpi0.5-rtc-reward-learning
PYTHONDONTWRITEBYTECODE=1 \
ROLLOUTS_ROOT=/home/eii/project/openpi0.5-rtc-reward-learning-local-data/rollouts \
REPLAY_ROOT=/home/eii/project/openpi0.5-rtc-reward-learning-local-data/replay \
RLT_SEGMENT_DB_PATH=/home/eii/project/openpi0.5-rtc-reward-learning-local-data/segment_db/segments.sqlite3 \
RLT_STATE_PATH=/home/eii/project/openpi0.5-rtc-reward-learning-local-data/segment_db/rlt_control_state.json \
uv run pytest \
  voice_assistant_web/backend/app/rollout_tree_test.py \
  voice_assistant_web/backend/app/rlt_segment_ledger_test.py \
  voice_assistant_web/backend/app/rlt_control_test.py \
  examples/aloha_real/rlt_key_region_recorder_test.py \
  src/openpi/training/rlt_replay_store_test.py \
  src/openpi/training/rlt_training_test.py \
  scripts/train_rlt_online_test.py \
  packages/openpi-client/src/openpi_client/runtime/runtime_test.py \
  packages/openpi-client/src/openpi_client/action_chunk_broker_test.py \
  packages/openpi-client/src/openpi_client/rlt_actor_runtime_test.py \
  -q -p no:cacheprovider
```

Expected: all selected tests pass. The explicit file list is required because default `pytest` discovery does not include `voice_assistant_web` and `examples`.

Observed on 2026-06-02: `68 passed, 4 warnings`.

### Task 4: Frontend Build Environment

**Files:**
- Use: `/home/eii/project/openpi0.5-rtc-reward-learning/voice_assistant_web/frontend/package.json`

- [x] **Step 1: Install frontend dependencies**

```bash
cd /home/eii/project/openpi0.5-rtc-reward-learning/voice_assistant_web/frontend
/home/eii/.local/bin/npm ci
```

Expected: npm installs dependencies. Current audit shows known vulnerabilities; they do not block local build verification.

- [x] **Step 2: Build frontend**

```bash
cd /home/eii/project/openpi0.5-rtc-reward-learning/voice_assistant_web/frontend
/home/eii/.local/bin/npm run build
```

Expected: `tsc && vite build` exits with code 0.

### Task 5: Local Backend And Frontend Runtime

**Files:**
- Use: `/home/eii/project/openpi0.5-rtc-reward-learning/voice_assistant_web/backend/app/main.py`
- Use: `/home/eii/project/openpi0.5-rtc-reward-learning/voice_assistant_web/frontend/src/services/api.ts`

- [x] **Step 1: Start backend on H2 without Docker**

Port `8011` may already be in use on H2. Use `8012` for local review development if `8011` is occupied.

```bash
cd /home/eii/project/openpi0.5-rtc-reward-learning
PYTHONPATH=/home/eii/project/openpi0.5-rtc-reward-learning \
ROLLOUTS_ROOT=/home/eii/project/openpi0.5-rtc-reward-learning-local-data/rollouts \
REPLAY_ROOT=/home/eii/project/openpi0.5-rtc-reward-learning-local-data/replay \
RLT_SEGMENT_DB_PATH=/home/eii/project/openpi0.5-rtc-reward-learning-local-data/segment_db/segments.sqlite3 \
RLT_STATE_PATH=/home/eii/project/openpi0.5-rtc-reward-learning-local-data/segment_db/rlt_control_state.json \
uv run python -m uvicorn voice_assistant_web.backend.app.main:app --host 127.0.0.1 --port 8012
```

Expected: `http://127.0.0.1:8012/health` responds. On H2, `rospy` import errors are expected and acceptable because H2 is not the robot-connected host.

- [ ] **Step 2: Start frontend on H2**

The Vite config defaults to port 80, so pass `--port 3011` explicitly for this local workflow.

```bash
cd /home/eii/project/openpi0.5-rtc-reward-learning/voice_assistant_web/frontend
VITE_API_BASE=http://127.0.0.1:8012 \
VITE_WS_BASE=ws://127.0.0.1:8012 \
/home/eii/.local/bin/npm run dev -- --host 127.0.0.1 --port 3011
```

Expected: the key-region UI is available at `http://127.0.0.1:3011`.

- [x] **Step 3: Validate backend endpoints**

```bash
curl -s http://127.0.0.1:8012/health
curl -s 'http://127.0.0.1:8012/api/rollouts/tree?path=key_regions' | head
curl -s http://127.0.0.1:8012/api/rlt/key-regions/review | head
```

Expected: health responds and review/tree endpoints return JSON.

Observed on 2026-06-02 after syncing data:

```text
/health -> {"status":"ok"}
/api/rlt/key-regions/review -> 148 items
/api/rollouts/tree?path=key_regions -> key_regions with twist_off_the_bottle_cap child
```

### Task 6: Data Cleaning Capability Boundary

**Files:**
- Inspect: `/home/eii/project/openpi0.5-rtc-reward-learning/voice_assistant_web/backend/app/main.py`
- Inspect: `/home/eii/project/openpi0.5-rtc-reward-learning/voice_assistant_web/frontend/src/components/RolloutBrowser.tsx`
- Inspect: `/home/eii/project/openpi0.5-rtc-reward-learning/docs/design-previews/key-regions-upgrade-preview.html`

- [x] **Step 1: Confirm current implemented review operations**

Current backend implements review/delete/void/restore style operations for key regions. These are enough for local review and cleaning by deletion/voiding.

- [ ] **Step 2: Implement actual crop-to-Q API before claiming full split support**

The static design preview includes a timeline crop concept, but the current backend does not yet expose a confirmed crop-save endpoint that creates a new trainable replay shard from a trimmed time range. To make "裁剪后保存进入 Q 训练" real, implement and test a backend API such as:

```text
POST /api/rlt/key-regions/{key_region_id}/crop
```

The request should include start/end offsets or frame indices, source rollout path, target replay root, reviewer metadata, and an explicit train eligibility flag. The response should return the new clean replay shard path and ledger status.

- [ ] **Step 3: Keep raw and cleaned data separated**

Use raw mirrored data for review:

```text
/home/eii/project/openpi0.5-rtc-reward-learning-local-data/replay
```

Write cropped/cleaned data to a separate root:

```text
/home/eii/project/openpi0.5-rtc-reward-learning-local-data/replay_clean
```

Expected: H2 can clean locally without damaging raw 103 mirrors.

### Task 7: H2 Actor-Critic Training After Cleaning

**Files:**
- Use: `/home/eii/project/openpi0.5-rtc-reward-learning/scripts/train_rlt_online.py`
- Use: `/home/eii/project/openpi0.5-rtc-reward-learning-local-data/replay_clean`
- Use: `/home/eii/project/openpi0.5-rtc-reward-learning-local-data/rlt_online`

- [ ] **Step 1: Train from cleaned replay only**

```bash
cd /home/eii/project/openpi0.5-rtc-reward-learning
PYTHONDONTWRITEBYTECODE=1 \
RLT_SEGMENT_DB_PATH=/home/eii/project/openpi0.5-rtc-reward-learning-local-data/segment_db/segments.sqlite3 \
uv run python scripts/train_rlt_online.py \
  --replay-dir /home/eii/project/openpi0.5-rtc-reward-learning-local-data/replay_clean/rlt_key_regions \
  --output-dir /home/eii/project/openpi0.5-rtc-reward-learning-local-data/rlt_online \
  --recursive-scan \
  --expected-replay-action-horizon 50 \
  --train-action-horizon 10 \
  --min-replay-samples 100 \
  --min-success-episodes 1 \
  --min-failure-episodes 1 \
  --overwrite
```

Expected: trainer uses cleaned H2 replay, not raw mirrored replay.

- [ ] **Step 2: Transfer reviewed actor output to 103 only after manual decision**

```bash
rsync -aP /home/eii/project/openpi0.5-rtc-reward-learning-local-data/rlt_online/ \
  eii@192.168.1.103:/data/openpi0.5-rtc-reward-learning/rlt_online_h2_reviewed/
```

Expected: H2 output lands in a separate reviewed directory on 103 and does not overwrite 103's live online trainer output.

### Task 8: Operating Rules

- [ ] **Rule 1: Never develop H2 data-cleaning code against `http://192.168.1.103` unless actively testing the robot.**

Use `127.0.0.1:8011` and `127.0.0.1:3011` for H2 UI/backend development.

- [ ] **Rule 2: Never use routine `rsync --delete` from 103 into H2 cleaning roots.**

Use `--delete` only for a disposable mirror directory after confirming no local review work is inside it.

- [ ] **Rule 3: Never run H2 local cleaning against 103-mounted live `/data` paths.**

All H2 project and data work stays under `~/project`.

- [ ] **Rule 4: Keep robot runtime on 103.**

H2 can train and clean data, but ALOHA runtime validation remains on machine 103 because the robot is physically connected there.
