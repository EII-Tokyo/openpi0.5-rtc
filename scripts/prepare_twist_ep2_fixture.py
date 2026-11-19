"""Rebuild the agreed EP2 report fixture using the same completed-label builder as training."""
from pathlib import Path
import json
import pyarrow.parquet as pq
from openpi.shared.completed_memory_labels import episode_updates,TASK_CATALOG_VERSION
ROOT=Path('/home/eii/long-memory-hierarchy-20261119/artifacts/project-ep2-report');OUT=Path('/home/eii/long-memory-hierarchy-20261119/artifacts/native448-prompts');OUT.mkdir(exist_ok=True)
repo='lyl472324464/2026.03.16_twist_many-without-rinse';revision='6a1b83eda207d7b7bcf9a817191583bc5ad2923d';rows=[]
for f in (ROOT/'datasets'/repo/'data').rglob('*.parquet'):
 rows.extend(r for r in pq.ParquetFile(f).read(columns=['episode_index','frame_index','timestamp','subtask']).to_pylist() if int(r['episode_index'])==1)
rows.sort(key=lambda r:r['frame_index']);assert len(rows)==5134
updates,missing=episode_updates(rows,generic_return_context='bottles_remain');assert not missing
(OUT/'teacher.json').write_text(json.dumps({'schema_version':2,'task_catalog_version':TASK_CATALOG_VERSION,'episodes':{repo:{'1':updates}},'label_source':'Human task boundaries, not independently confirmed visual success. Generic return occupancy was manually reviewed for platformEP2/exportEP1. Training mixture is unchanged; this is the fixed report fixture only.','provenance':{'project_id':'69dc35eb0b9824437fb4f5ad','project_name':'2026.03.16_twist_many','platform_episode':2,'platform_episode_record_id':'69dc3683193d35146f44f413','lerobot_episode':1,'revision':revision,'frames':5134}},ensure_ascii=False));print('FIXTURE_PREPARED',len(updates),flush=True)
