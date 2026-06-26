# Key Region Source Filters And Expert Demo Crop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add key-region filters for no-actor data and add an expert-demo browsing/cropping entry for local Hugging Face/LeRobot human demos, without training the D discriminator.

**Architecture:** Existing RLT key region review records will be enriched with action-vs-reference metrics computed from npz shards. Expert demos will use a separate backend service/API so HF/LeRobot data does not pollute the RLT segment ledger; the frontend key region page will switch between RLT records and expert demo records.

**Tech Stack:** FastAPI/Pydantic backend, React/TypeScript frontend, numpy for RLT shard metrics, local Hugging Face/LeRobot cache metadata and video files for expert demos.

---

### Task 1: RLT no-actor classifier

**Files:**
- Modify: `voice_assistant_web/backend/app/schemas.py`
- Modify: `voice_assistant_web/backend/app/main.py`
- Test: `voice_assistant_web/backend/app/rollout_tree_test.py`

- [ ] Write a failing backend test that creates a shard with `action == reference_action`, calls review filtering with `status=noActor`, and expects one record with `actor_inference_kind == "no_actor"`.
- [ ] Write a failing backend test that creates a shard with changed action, calls `status=actorModified`, and expects `actor_inference_kind == "actor_or_modified"`.
- [ ] Implement `_rlt_action_delta_metrics()` and add actor fields to `RLTKeyRegionReviewRecord`.
- [ ] Extend `_filter_key_region_review_records()` for `noActor` and `actorModified`.
- [ ] Run backend tests.

### Task 2: Expert demo service/API

**Files:**
- Create: `voice_assistant_web/backend/app/expert_demo_review.py`
- Modify: `voice_assistant_web/backend/app/schemas.py`
- Modify: `voice_assistant_web/backend/app/main.py`
- Test: `voice_assistant_web/backend/app/expert_demo_review_test.py`

- [ ] Write failing tests against a temporary LeRobot-like metadata/video tree.
- [ ] Implement dataset discovery under `/home/eii/.cache/huggingface/lerobot/lyl472324464`.
- [ ] Return episode records with dataset id, episode index, frame count, video paths, and source paths.
- [ ] Implement crop export endpoint that saves metadata JSON under `/home/eii/data/openpi0.5-rtc-reward-learning/replay/discriminator_expert_crops`.
- [ ] Run expert demo tests.

### Task 3: Frontend filters and expert mode

**Files:**
- Modify: `voice_assistant_web/frontend/src/services/api.ts`
- Modify: `voice_assistant_web/frontend/src/components/KeyRegionsPage.tsx`

- [ ] Add API types for actor metrics and expert demo records.
- [ ] Add RLT status filter options: `No actor / VLA only` and `Actor modified`.
- [ ] Add source selector: `RLT key regions` and `Expert demos for D`.
- [ ] In expert mode, show expert demo cards with dataset, episode, video preview, and crop save button.
- [ ] Run TypeScript build/test command.

### Task 4: Verification

**Files:**
- Verify backend tests.
- Verify frontend build.
- Report current counts for no-actor records and expert demo episodes.
