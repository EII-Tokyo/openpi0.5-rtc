"""Select20distinct episodes,10attached/10not attached; lossless current-frame four-camera capture."""
import os,json,pathlib,collections,random,hashlib,ast
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
ROOT=pathlib.Path('/workspace/long-memory-hierarchy-20261119');SOURCE=ROOT/'completed-code';OUT=ROOT/'cap20-fourcam-original';OUT.mkdir(exist_ok=True)
SEED=2026111901;rng=random.Random(SEED)
name='eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo'
tree=ast.parse((SOURCE/'src/openpi/training/config.py').read_text());node=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and any(k.arg=='name' and isinstance(k.value,ast.Constant) and k.value.value==name for k in n.keywords));data=next(k.value for k in node.keywords if k.arg=='data');repos=ast.literal_eval(next(k.value for k in data.keywords if k.arg=='repo_ids'))
candidates=collections.defaultdict(lambda:collections.defaultdict(list));source_files=[]
def cap(raw):
 d=json.loads(raw) if isinstance(raw,str) else raw;s=(d.get('bottle_state') or '').lower()
 if 'bottle in left hand and capped' in s:return 'attached'
 if 'bottle in left hand' in s and ('cap removed' in s or 'uncapped' in s or 'without cap' in s):return 'not_attached'
 return None
for repo in dict.fromkeys(repos):
 grouped=collections.defaultdict(list)
 for f in sorted((ROOT/'datasets'/repo/'data').rglob('*.parquet')):
  pf=pq.ParquetFile(f);cols=['episode_index','frame_index','timestamp','subtask','index'];cols+=['is_for_training'] if 'is_for_training' in pf.schema_arrow.names else []
  for row in pf.read(columns=cols).to_pylist():grouped[int(row['episode_index'])].append(row)
  source_files.append({'path':str(f.relative_to(ROOT/'datasets')),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 for ep,rows in grouped.items():
  rows.sort(key=lambda x:x['frame_index']);labels=[cap(r['subtask']) if r.get('is_for_training',True) else None for r in rows];runs=[];start=0
  for j in range(1,len(rows)+1):
   if j==len(rows) or labels[j]!=labels[start]:
    if labels[start]:
     eligible=[r for r in rows[start:j] if float(r['timestamp'])>=float(rows[start]['timestamp'])+.5 and float(r['timestamp'])<=float(rows[j-1]['timestamp'])-.5]
     if eligible:runs.extend((labels[start],r) for r in eligible)
    start=j
  for label in ('attached','not_attached'):
   valid=[row for kind,row in runs if kind==label]
   if valid:candidates[label][repo].append({'repo_id':repo,'episode_index':ep,'row':rng.choice(valid),'truth':label})
selected=[];seen=set()
for label in ('attached','not_attached'):
 order=list(candidates[label]);rng.shuffle(order)
 for values in candidates[label].values():rng.shuffle(values)
 while sum(x['truth']==label for x in selected)<10:
  advanced=False
  for repo in order:
   while candidates[label][repo] and (repo,candidates[label][repo][-1]['episode_index']) in seen:candidates[label][repo].pop()
   if candidates[label][repo]:
    sample=candidates[label][repo].pop();seen.add((repo,sample['episode_index']));selected.append(sample);advanced=True
    if sum(x['truth']==label for x in selected)==10:break
  if not advanced:raise ValueError('Not enough distinct eligible episodes')
rng.shuffle(selected)
# Use the actual LeRobot decoder, retaining all four current camera streams.
os.chdir(SOURCE)
from openpi.data.episode_metadata import install_episode_metadata_reader
install_episode_metadata_reader()
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from openpi_client.runtime.long_memory_high_level import image_uint8
cameras=('cam_high','cam_low','cam_left_wrist','cam_right_wrist');datasets={};records=[]
for n,s in enumerate(selected,1):
 repo=s['repo_id']
 if repo not in datasets:datasets[repo]=LeRobotDataset(repo,root=ROOT/'datasets'/repo,video_backend='pyav')
 raw=datasets[repo][int(s['row']['index'])]
 assert int(raw['episode_index'])==s['episode_index'] and int(raw['frame_index'])==s['row']['frame_index']
 paths={};sizes={};hashes={}
 for camera in cameras:
  arr=image_uint8(raw['observation.images.'+camera]);im=Image.fromarray(arr);path=OUT/'frames'/f'{n:02d}'/(camera+'.png');path.parent.mkdir(parents=True,exist_ok=True);im.save(path,format='PNG')
  assert np.array_equal(np.asarray(Image.open(path)),arr),'Lossless PNG roundtrip failed'
  paths[camera]=str(path.relative_to(OUT));sizes[camera]=list(im.size);hashes[camera]=hashlib.sha256(arr.tobytes()).hexdigest()
 record={'number':n,'repo_id':repo,'episode_index':s['episode_index'],'frame_index':s['row']['frame_index'],'time_s':s['row']['timestamp'],'truth':s['truth'],'annotation':json.loads(s['row']['subtask']) if isinstance(s['row']['subtask'],str) else s['row']['subtask'],'images':paths,'image_dimensions':sizes,'decoded_rgb_sha256':hashes}
 records.append(record);print('CAPTURE',n,s['truth'],s['episode_index'],sizes,flush=True)
(OUT/'samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
(OUT/'capture-summary.json').write_text(json.dumps({'seed':SEED,'source_config':name,'sample_count':len(records),'class_counts':dict(collections.Counter(r['truth'] for r in records)),'distinct_episodes':len(seen),'distinct_repos':len({r['repo_id'] for r in records}),'camera_count':4,'history_frames':0,'original_resolution_preserved':True,'format':'Lossless PNG of actual decoded source video frame. No resize, crop, mosaic or JPEG re-encoding.','selection':'Repo-balanced random selection from distinct episodes, one randomly selected frame per EP, at least0.5s inside a consistently annotated class segment. No visibility/model-result selection.','source_files':source_files},indent=2))
