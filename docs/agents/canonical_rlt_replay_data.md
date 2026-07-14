# Canonical RLT Replay Data

Read this before assembling replay, selecting manifests, training critic/actor, auditing `z_rl`, converting human expert data, or mixing replay directories.

## Formal Replay Requirements
- Current canonical RLT replay data uses the lower+right 4-layer RLToken encoder with `z_rl` / `next_z_rl` dimension `2048`.
- Strong constraint: for the active same-forward runtime, formal critic/actor training replay must use `z_rl` that was recorded at real robot `Policy.infer()` events under HDF5 `/rlt_policy_forward_events`. Do not treat `z_rl` recomputed later from saved mp4/video frames as runtime-equivalent training data.
- Strong constraint: saved mp4/video based re-encoding is ablation/audit data only. Sources such as `rl_token_reencoded`, `precomputed_frame_cache`, `vla_same_forward_low_right_tokens_then_lower_right_rl_token_encoder`, `async_anchor_token_cache_vla_same_forward`, and `dummy_*` must not be used as formal actor/critic training replay unless the user explicitly asks for a controlled ablation.
- Strong constraint: `paper_subsampled_anchor` by itself is not sufficient to be trainable. A formal replay shard must also prove runtime-event lineage, normally with `z_rl_source=vla_same_forward_runtime_output`, `z_alignment=policy_forward_event_*`, and source HDF5 `/rlt_policy_forward_events` metadata. Legacy or missing `z_rl_source` is not trainable by default.
- Do not train by scanning mixed legacy replay directories directly. Train from the canonical manifests unless the user explicitly requests an ablation.

## Fixed Collection Workflow
- `192.168.1.103` is the data-collection machine. During collection, it must save raw rollout material plus real `/rlt_policy_forward_events` from the running same-forward policy.
- The older async-anchor token worker and mp4/video re-encoding workflow is no longer formal training data for the active same-forward runtime. It may be used only for ablation/audit, and its outputs must be clearly labeled non-formal.
- After a collection batch finishes and the user asks to pull data back, copy all required source material from 103 to the local data root `/home/eii/data/openpi0.5-rtc-reward-learning`: raw rollouts, HDF5 `/rlt_policy_forward_events`, runtime cache-block audit data, and any related manifests/audit files.
- Formal trainable replay assembly should consume the recorded policy-forward events, not recompute `z_rl` from mp4. If a rollout lacks `/rlt_policy_forward_events`, classify it as legacy/offline/ablation by default.

## Historical 2026-07-06 Rescue
- 2026-07-06 online RLT replay rescue is now reclassified after the 2026-07-07 runtime-vs-offline z audit:
  - Treat raw 2026-07-06 replay as source material or ablation only; most of it lacks `/rlt_policy_forward_events`.
  - The previously rebuilt `base142` / `actor93` same-forward paper-anchor data was generated from saved video/offline token extraction, so it is not runtime-equivalent formal replay for the active same-forward actor.
  - The 2026-07-06 iterative actor training manifests under `local_rlt_manifests/iterative_same_forward_20260706/` and `local_rlt_manifests/unified_same_forward_20260706/` are historical ablation artifacts unless they are rebuilt from real `/rlt_policy_forward_events`.
  - Do not use those manifests for new formal actor/critic training without an explicit ablation request.

## 2048 Roots And Split
- Legacy 512-dim replay roots such as `rlt_key_regions`, `rlt_key_regions_clean`, and `human_expert_no_actor_q_cam4_provenance_20260629` must not be mixed into a 2048 training run.
- Strong requirement: formal critic/actor training replay for the active same-forward runtime should match the actor's actual state distribution:
  - each replay row should use `z_rl/proprio` from a real policy-forward event recorded during robot control;
  - action chunks should be aligned to that event's true action trunk/window;
  - do not synthesize a new `z_rl` for arbitrary stride-anchor frames by decoding saved mp4 and running VLA/RLToken again;
  - `fixed_segments`, `rl_token_reencoded_aligned_to_proprio_segments`, video-reencoded paper anchors, async-anchor token caches, and precomputed frame caches are audit/ablation artifacts unless explicitly approved for a controlled ablation.
- Local canonical root:
  - Data: `/home/eii/data/openpi0.5-rtc-reward-learning/replay/canonical_2048`
  - Local-only manifest: `/home/eii/data/openpi0.5-rtc-reward-learning/manifests/canonical_2048`
  - 103 mirrored manifest with local filesystem paths: `/home/eii/data/openpi0.5-rtc-reward-learning/manifests/canonical_2048_from_103`
- `192.168.1.103` canonical root:
  - Data: `/data/openpi0.5-rtc-reward-learning/replay/canonical_2048`
  - Manifest: `/data/openpi0.5-rtc-reward-learning/manifests/canonical_2048`
- Historical note: the 2026-07-02 103 canonical manifest contained fixed-segment bootstrap data and is now considered stale for formal training under the paper-aligned replay requirement. Do not reuse it for new formal critic/actor training without rebuilding the bootstrap/clean shards from original rollouts.
- As of 2026-07-02, that stale 103 canonical manifest contained 384 replay shards and 18,721 transitions:
  - `bootstrap`: 146 shards (`train`: 117, `holdout`: 29), source `current109_37_actor6000_20260630`
  - `rlt_raw`: 178 shards, source 103 online/raw actor data
  - `rlt_clean`: 1 shard, source 103 cleaned data currently available under `/data`
  - `expert`: 59 shards, source human expert lower+right 4-layer encoding
- The canonical train and holdout split currently has zero `key_region_id` overlap. Preserve that invariant.
- If a new 512-dim or old cam4-small-query shard is discovered, re-encode it into a separate lower+right 2048 directory first; never overwrite the original shard.

## Human Expert Data
- For human expert / Expert-for-D data, do not re-encode by rewriting old Q replay `z_rl` in place. The correct source-of-truth chain is:
  - Crop JSON: `/home/eii/data/openpi0.5-rtc-reward-learning/replay/discriminator_expert_crops`
  - Original LeRobot cache: `/home/eii/.cache/huggingface/lerobot/lyl472324464/<dataset_id>`
  - Encode frame-level z cache from original LeRobot parquet/video using the lower+right 4-layer RLToken checkpoint.
  - Convert crop JSON + parquet action/state + the new z cache into trainable Q replay.
- For human expert lower+right conversion, prefer the z-only deterministic code path `Policy.infer_rl_token()` over `Policy.infer()`. `Policy.infer()` may go through action sampling before returning `z_rl`, which is unnecessary for replay re-encoding and makes audits harder.
- Current fully regenerated human expert output from original LeRobot data:
  - z cache: `/home/eii/data/openpi0.5-rtc-reward-learning/replay/expert_crop_z_rl_cache_lower_right_4layer_from_raw_zonly_20260703`
  - Q replay: `/home/eii/data/openpi0.5-rtc-reward-learning/replay/human_expert_no_actor_q_lower_right_4layer_from_raw_zonly_20260703`
  - manifest: `/home/eii/project/openpi0.5-rtc-reward-learning/local_rlt_manifests/expert_from_raw_20260703/human_expert_no_actor_q_lower_right_4layer_from_raw_zonly_20260703.jsonl`
  - audit: `/home/eii/project/openpi0.5-rtc-reward-learning/local_rlt_manifests/expert_from_raw_20260703/audit_from_raw_zonly_20260703.json`
  - verified counts: 59 Q replay shards, 58 episode z-cache files, 2,896 transitions, all `z_rl` / `next_z_rl` dimension 2048, all `action == reference_action`.
- Training selection rule for human expert data:
  - Use only the z-only Q replay and manifest above for human expert lower+right 4-layer training.
  - Do not use deleted legacy / non-z-only expert outputs such as `human_expert_no_actor_q_lower_right_4layer_20260701`, `expert_crop_z_rl_cache_lower_right_4layer_20260701`, or `human_expert_no_actor_q_lower_right_4layer_from_raw_20260703`.
  - If the z-only output is missing or suspected stale, regenerate it from the crop JSON and original LeRobot parquet/video via `Policy.infer_rl_token()`; do not restore or train from the old `Policy.infer()`-generated expert replay.
  - Any canonical 2048 manifest that includes human expert data should point to the z-only manifest above, not to an old 20260701 or non-z-only manifest.
