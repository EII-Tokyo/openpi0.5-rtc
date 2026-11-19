"""Fail-closed native launch: immutable sources, datasets, idle GPUs, no config overrides."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,subprocess,time,sys
R=Path('/workspace/long-memory-hierarchy-20261119');C=R/'code'
NAME='eii_twist_cam3_fullft_25repo_h100x2_224_longmem3s_drop_bs16ga16_500step_pilot_20261119'
PY='/workspace/twist-gns-20261117/code/.venv/bin/python'
sys.path.insert(0,str(C/'src'))
os.environ['JAX_PLATFORMS']='cpu'  # Read the native config without touching a GPU.

def main():
 seal=json.loads((R/'artifacts/source.json').read_text())
 for rel,sha in seal['files'].items():
  assert hashlib.sha256((C/rel).read_bytes()).hexdigest()==sha,'Code changed after CPU verification: '+rel
 proof=json.loads((R/'artifacts/datasets-verified.json').read_text())
 from huggingface_hub import HfApi
 api=HfApi()
 def heads(d):
  refs=api.list_repo_refs(d['repo_id'],repo_type='dataset')
  current={x.name:x.target_commit for x in [*refs.branches,*refs.tags] if x.name in ('main','v3.0')}
  assert current==d['refs'],'Published heads changed; prepare again before start'
 with ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(heads,proof['heads']))
 def verify(f):
  p=R/'datasets'/f['repo_id']/f['path'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==f['size']
  before=(p.stat().st_size,p.stat().st_mtime_ns);sha=hashlib.sha256()
  with p.open('rb') as h:
   for b in iter(lambda:h.read(8*1024**2),b''):sha.update(b)
  assert sha.hexdigest()==f['sha256'] and before==(p.stat().st_size,p.stat().st_mtime_ns),'Dataset changed after verification'
 with ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(verify,proof['files']));list(ex.map(heads,proof['heads']))
 used=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 rows=[line.split(',') for line in used.strip().splitlines()]
 assert len(rows)==2 and all('H100' in x[1] and int(x[2])<500 and int(x[3])<5 for x in rows),'GPUs occupied; do not overlap training'
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU compute process still active'
 assert subprocess.check_output(['df','--output=avail','-B1',str(R)],text=True).splitlines()[-1].strip().isdigit()
 free=os.statvfs(R).f_bavail*os.statvfs(R).f_frsize
 assert free>60*1024**3,'Insufficient disk space for the single full first-stage checkpoint'
 # This bounded fresh attempt has no periodic save before its final step499.
 from openpi.training.config import get_config
 config=get_config(NAME)
 assert config.num_train_steps==500 and config.save_interval>=config.num_train_steps
 assert not Path(str(config.checkpoint_dir)).exists(),'Keep any existing checkpoint; never overwrite a prior attempt'
 env={**os.environ,'PYTHONPATH':str(C/'src')+':'+str(C/'packages/openpi-client/src'),'CUDA_VISIBLE_DEVICES':'0,1',
      'JAX_PLATFORMS':'cuda','OPENPI_DATA_HOME':'/workspace/twist-gns-20261117/cache',
      'HF_LEROBOT_HOME':str(R/'datasets'),'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
      'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false',
      'OPENPI_PAUSE_FILE':str(R/'PAUSE'),'OPENPI_METRICS_FILE':str(R/'artifacts/metrics.jsonl'),
      'OPENPI_STATUS_FILE':str(R/'artifacts/training-status.json'),'WANDB_MODE':'online'}
 # Reuse only the existing training service's W&B credential, never its source overrides.
 for line in Path('/workspace/twist-gns-20261117/private.env').read_text().splitlines():
  if '=' in line and not line.lstrip().startswith('#'):
   k,v=line.split('=',1);k=k.strip().removeprefix('export ').strip()
   if k=='WANDB_API_KEY':env[k]=v.strip().strip('\"\x27')
 assert env.get('WANDB_API_KEY'),'W&B credential unavailable; no silent offline fallback'
 assert not (R/'PAUSE').exists(),'Intentional pause is active'
 provenance={**seal,'verified_unix':time.time(),'dataset_receipt':'artifacts/datasets-verified.json',
  'code_directory':str(C),'dataset_directory':str(R/'datasets'),'config':NAME,'command':[PY,'scripts/train.py',NAME],
  'gpu_snapshot':used,'config_loaded_directly':True,'phase':'native_training_start_requested'}
 (R/'artifacts/launch-provenance.json').write_text(json.dumps(provenance,indent=2))
 print('VERIFIED_NATIVE_START',NAME,flush=True)
 os.chdir(C)
 os.execve(PY,[PY,'scripts/train.py',NAME],env)

if __name__=='__main__':main()
