import os,sys,json,pathlib,random,hashlib
os.environ['JAX_PLATFORMS']='cpu'
R=pathlib.Path('/home/eii/long-memory-hierarchy-20261119');S=R/'completed-code';ROOT=R/'artifacts/project-ep2-report';OUT=R/'artifacts/gpt62-evaluation';OUT.mkdir(exist_ok=True)
os.environ['HF_LEROBOT_HOME']=str(ROOT/'datasets');os.chdir(S);sys.path[:0]=[str(S/'src'),str(S/'packages/openpi-client/src')]
import numpy as np
from PIL import Image
from openpi.training.config import get_config
from openpi.data.datasets import create_torch_dataset
from openpi.data.dataloaders import TorchDataLoader
from openpi_client.runtime.long_memory_high_level import image_uint8
from openpi.shared.completed_step_memory import CURRENT_TASKS
cfg=get_config('eii_twist_many_platform_ep2_memory_latestdrop20_taskdrop20_report_20261119');ds=create_torch_dataset(cfg.data,cfg.model.action_horizon);base=ds._dataset._dataset._datasets[0]
eps=np.asarray(base.hf_dataset['episode_index']);frames=np.asarray(base.hf_dataset['frame_index']);ids=np.flatnonzero((eps==1)&ds.train_mask);indices=[int(i) for i in ids if int(frames[i])%10==0];rows=[]
random.seed(2026111902);np.random.seed(2026111902)
class Capture:
 def __len__(self):return len(indices)
 def __getitem__(self,j):
  idx=indices[j];raw=ds[idx];f=int(raw['frame_index']);assert int(raw['episode_index'])==1 and f==int(frames[idx])
  folder=OUT/'frames'/str(f);folder.mkdir(parents=True,exist_ok=True);images={};hashes={}
  for camera in ('cam_high','cam_low','cam_left_wrist','cam_right_wrist'):
   a=image_uint8(raw['observation.images.'+camera]);p=folder/(camera+'.png');Image.fromarray(a).save(p);images[camera]=str(p.relative_to(OUT));hashes[camera]=hashlib.sha256(a.tobytes()).hexdigest();assert a.shape==(480,640,3)
  clean=json.loads(raw['long_memory_before_augmentation']);aug=json.loads(raw['long_memory']);record={'frame_index':f,'global_index':idx,'episode_index':1,'time_s':float(raw['timestamp']),'before':clean,'after':aug,'augmentation':raw['memory_augmentation'],'images':images,'pixel_sha256':hashes,'input_text':{'memory':aug['memory'],'current_task':aug['current_task']},'old_reference_task_index':CURRENT_TASKS.index(clean['current_task'])+1,'reference_status':'human_subtask_boundary; disposal and box-departure transition labels require visual adjudication, old generic-return labels not treated as ground truth'};rows.append(record)
  return {'frame_index':np.int64(f)}
loader=TorchDataLoader(Capture(),local_batch_size=1,shuffle=False,num_workers=0,num_batches=len(indices),seed=2026111902)
for batch in loader:pass
assert len(rows)==509 and rows[0]['frame_index']==0 and rows[-1]['frame_index']==5080
(OUT/'samples.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
(OUT/'capture-provenance.json').write_text(json.dumps({'mapping':json.loads((ROOT/'mapping.json').read_text()),'capture_count':len(rows),'stride':10,'excluded_grid_frames':[5090,5100,5110,5120,5130],'exclusion':'is_for_training=false; not redirected to other data','native_loader':'openpi.data.dataloaders.TorchDataLoader','images':'Four unresized lossless RGB PNGs per sample; original640x480','config':cfg.name,'source':str(S),'source_hashes':{str(p.relative_to(S)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [S/'src/openpi/data/long_memory.py',S/'src/openpi/shared/long_memory.py',S/'src/openpi/shared/completed_step_memory.py']},'dropout_counts':{k:sum(x['augmentation'][k] for x in rows) for k in ('memory_dropout_draw','memory_dropout_applied','subtask_dropout_applied')}},indent=2))
pre=next(x for x in rows if x['augmentation']['memory_dropout_applied'] and x['augmentation']['subtask_dropout_applied']);(OUT/'preflight-frame.json').write_text(json.dumps(pre,ensure_ascii=False,indent=2));print('CAPTURE_COMPLETE',len(rows),'PREFLIGHT',pre['frame_index'],flush=True)
