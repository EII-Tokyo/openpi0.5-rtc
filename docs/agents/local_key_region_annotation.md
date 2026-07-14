# Local Key Region Annotation

Read this before starting, debugging, remounting, or syncing local key-region annotation data on machine `101`.

- The local machine `101` is for offline key region data annotation only. Do not treat `http://127.0.0.1:3011/` as a robot-control UI, and do not expect local key presses there to control the robot on `192.168.1.103`.
- For local offline key region annotation, the correct data root is:
  - `/home/eii/data/openpi0.5-rtc-reward-learning`
- The local annotation service should use these mounts:
  - `/home/eii/data/openpi0.5-rtc-reward-learning/rollouts:/app/rollouts`
  - `/home/eii/data/openpi0.5-rtc-reward-learning/replay:/app/replay`
  - `/home/eii/data/openpi0.5-rtc-reward-learning/segment_db:/app/segment_db`
  - `/home/eii/data/openpi0.5-rtc-reward-learning:/home/eii/data/openpi0.5-rtc-reward-learning`
- The extra full-root mount is required because the segment DB may store absolute clean shard paths such as `/home/eii/data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions_clean/...`; without this mount, the backend may show clean/cropped records as missing.
- The 2026-06-22 third-batch cleaned key region data is under:
  - `/home/eii/data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions_clean/twist_off_the_bottle_cap/2026-06-22`
  - `/home/eii/data/openpi0.5-rtc-reward-learning/rollouts/key_regions/twist_off_the_bottle_cap/2026-06-22`
  - `/home/eii/data/openpi0.5-rtc-reward-learning/segment_db/segments.sqlite3`
- Do not assume `/home/eii/project/openpi0.5-rtc-reward-learning-local-data` is the active annotation dataset. It previously contained raw 2026-06-22 files but not the cleaned/cropped 2026-06-22 metadata, which made the UI appear to have no annotations.
- Do not pull, rsync, copy, or overwrite data from `192.168.1.103` unless the user explicitly asks for a data transfer. Before any data movement, first locate and inspect local data under `/home/eii/project/openpi0.5-*` and `/home/eii/data/openpi0.5-rtc-reward-learning`.
- Previous mistake to avoid: the local service was started against `/home/eii/project/openpi0.5-rtc-reward-learning-local-data`, then data was pulled from `192.168.1.103` before confirming the user's already-cleaned local dataset. The correct fix was only to switch the service mounts to `/home/eii/data/openpi0.5-rtc-reward-learning`.
