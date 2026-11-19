from pathlib import Path
import json,hashlib,sys
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download,HfApi
r=Path('/home/eii/long-memory-hierarchy-20261119/artifacts/project-ep2-report');m=json.loads((r/'hub-manifest.json').read_text());print('manifestkeys',list(m),flush=True)
repo='lyl472324464/2026.03.16_twist_many-without-rinse';rev='6a1b83eda207d7b7bcf9a817191583bc5ad2923d'
root=r/'datasets'/repo
root.mkdir(parents=True,exist_ok=True)
import shutil
shutil.copytree(r/'dataset',root,dirs_exist_ok=True)
files=HfApi().dataset_info(repo,revision=rev,files_metadata=True).siblings
# Whole repository required by the unmodified MultiLeRobotDataset file checker.
def fetch(f):
 p=Path(hf_hub_download(repo,f.rfilename,repo_type='dataset',revision=rev,local_dir=root))
 expected=f.lfs.sha256 if f.lfs else None
 if expected and hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise RuntimeError('Hash mismatch '+f.rfilename)
 return {'path':f.rfilename,'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
with ThreadPoolExecutor(max_workers=4) as pool:hashes=list(pool.map(fetch,files))
assert HfApi().dataset_info(repo).sha==rev
(r/'verified-files.json').write_text(json.dumps({'revision':rev,'files':hashes},indent=2));print('ALL_VERIFIED',len(hashes),flush=True)
sys.path.insert(0,'/home/eii/long-memory-hierarchy-20261119/completed-code/src')
from openpi.shared.completed_step_memory import completed_record_for_task
from openpi.shared.long_memory import episode_split
rows=sum([pq.read_table(p).to_pylist() for p in root.glob('data/**/*.parquet')],[])
e=[x for x in rows if x['episode_index']==1];updates=[];records=[];lasttask='';description='';seen=set();cycle=0
for row in e:
 ann=json.loads(row['subtask']);task=ann['subtask'];desc=ann.get('bottle_description') or []
 desc=desc[0] if isinstance(desc,list) and desc else desc or ''
 evidence=[]
 new_target=bool(desc and description and desc!=description and task in ('Pick up with left hand','Adjust bottle position'))
 # Object-local report boundary; not a claim of physical home arrival.
 if new_target:
  records=[];seen=set();description='';cycle+=1;evidence.append('New annotated target bottle; separate object-local memory, home arrival is NOT confirmed')
 if desc:description=desc
 # On a human-labeled stage transition, the PREVIOUS stage becomes a completed
 # record. A compound stage remains ONE record for latest-record dropout.
 if lasttask and task!=lasttask and not new_target:
  completed=completed_record_for_task(lasttask)
  if completed and completed not in records:
   records.append(completed);evidence.append('Previous human-labeled subtask ended: '+lasttask)
 update={'observed_through_s':float(row['timestamp']),'source_max_time_s':float(row['timestamp']),
 'source_frame_index':int(row['frame_index']),'completed_records':list(records),'cycle':cycle,
 'state':{'bottle_description':description,'memory':' '.join(records),'current_task':task},'evidence':evidence}
 updates.append(update);lasttask=task
cache={'schema_version':2,'episodes':{repo:{'1':updates}},'coverage':'Only requested platform EP2; not a full training cache',
 'label_source':'Human subtask boundaries at every frame: append the previous stage when the current stage changes. No scene-state fields, GPT teacher or time shift.',
 'limitations':'Completion targets follow human subtask boundaries, not independent visual success verification. Object-local reset at a new annotated target does not prove home arrival. An EP ending alone does not confirm completion.'}
(r/'teacher.json').write_text(json.dumps(cache,ensure_ascii=False));print('SPLIT',episode_split(repo,1),'updates',len(updates),'cycles',cycle+1,flush=True)
(r/'mapping.json').write_text(json.dumps({'project_id':'69dc35eb0b9824437fb4f5ad','project_name':'2026.03.16_twist_many','platform_episode':2,'platform_episode_record_id':'69dc3683193d35146f44f413','lerobot_episode_index':1,'frames':5134,'revision':rev,'mapping_evidence':'EP2 export ordering is platform episode order, zero-based export index; unique exact 5134-frame match. Other episodes can be shorter after without-rinse filtering.'},indent=2))
