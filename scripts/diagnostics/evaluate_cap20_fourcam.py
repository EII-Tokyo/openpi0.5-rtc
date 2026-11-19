"""Independent current-frame four-camera original-resolution cap-attachment test."""
import pathlib,json,base64,sys,time,concurrent.futures,collections
sys.path.insert(0,'/home/eii/long-memory-hierarchy-20261119/completed-code/scripts')
from regenerate_summary_memory_audit import call,key_from
ROOT=pathlib.Path('/home/eii/long-memory-hierarchy-20261119/artifacts/cap20-fourcam-original')
SYSTEM='''You receive four synchronized camera views of ONE current robot timestep: cam_high, cam_low, cam_left_wrist, cam_right_wrist. Identify the active bottle being manipulated by the robot and decide whether a cap is physically attached to THAT bottle's mouth NOW. A cap elsewhere in the scene or held separately by a gripper does not mean the bottle is capped. Compare the bottle neck and any separate cap across all four views. Answer attached if a cap is on the active bottle's mouth, not_attached if its mouth has no attached cap, and unknown if occlusion, blur or ambiguity prevents deciding. Do not guess from a cap's color or the presence of other bottles. There is no historical context or memory. Output only the specified JSON with your classification and a short explanation of visible evidence.'''
SCHEMA={'type':'object','properties':{'cap_status':{'type':'string','enum':['attached','not_attached','unknown']},'reason':{'type':'string'}},'required':['cap_status','reason'],'additionalProperties':False}
rows=list(map(json.loads,(ROOT/'samples.jsonl').read_text().splitlines()));key=key_from(pathlib.Path('/home/eii/openpi0.5-rtc/.env'))
assert len(rows)==20 and collections.Counter(r['truth'] for r in rows)=={'attached':10,'not_attached':10}
def run(row):
 user=[]
 for camera,path in row['images'].items():
  user.extend([{'type':'input_text','text':'camera='+camera},{'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode((ROOT/path).read_bytes()).decode(),'detail':'high'}])
 record={**row,'attempts':[]}
 for attempt in range(1,4):
  started=time.monotonic()
  try:
   prediction,metrics=call(key,SYSTEM,user,SCHEMA,ROOT/'raw'/f"{row['number']:02d}-{attempt}.json")
   record['attempts'].append({'attempt':attempt,'status':'completed',**metrics})
   record.update(status='completed',prediction=prediction,correct=prediction['cap_status']==row['truth'],latency_seconds=metrics['latency_seconds']);break
  except Exception as exc:
   record['attempts'].append({'attempt':attempt,'status':'error','error_type':type(exc).__name__,'elapsed_seconds':time.monotonic()-started})
 else:record.update(status='error',correct=False)
 print('RESULT',row['number'],row['truth'],record.get('prediction',{}).get('cap_status'),record['status'],flush=True)
 return record
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for r in pool.map(run,rows):
  results.append(r)
  (ROOT/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False))
# Derive recorded costs from ALL archived returned responses, even a malformed JSON attempt.
cost=0;usage_responses=0
for file in (ROOT/'raw').glob('*.json'):
 raw=json.loads(file.read_text());u=raw.get('usage')
 if not u:continue
 detail=u.get('input_tokens_details',{});cached=detail.get('cached_tokens',0);write=detail.get('cache_write_tokens',0)
 cost+=((u['input_tokens']-cached-write)*.1+cached*.01+write*.125+u['output_tokens']*.5)/1e6;usage_responses+=1
summary={'model':'gpt-6-luna','reasoning_effort':'none','image_detail':'high','sample_count':20,'images_per_request':4,'history_frames':0,'memory_supplied':False,'truth_labels_sent_to_model':False,'classes':{},'correct':sum(r['correct'] for r in results),'accuracy':sum(r['correct'] for r in results)/20,'unknown':sum(r.get('prediction',{}).get('cap_status')=='unknown' for r in results),'failed_samples':sum(r['status']!='completed' for r in results),'confusion':{},'recorded_usage_cost_usd':cost,'returned_responses_with_usage':usage_responses,'system_prompt':SYSTEM,'output_schema':SCHEMA,'limitations':'20selected frames from20distinct episodes in the requested dataset mixture;10per class. Ground truth comes from explicit human state labels at least0.5s inside a stable segment. Small diagnostic, not a population accuracy guarantee. Unknown/failed samples count as incorrect; all attempts and original replies are retained.'}
for truth in ('attached','not_attached'):
 subset=[r for r in results if r['truth']==truth]
 summary['classes'][truth]={'correct':sum(r['correct'] for r in subset),'total':10,'accuracy':sum(r['correct'] for r in subset)/10}
 summary['confusion'][truth]=dict(collections.Counter(r.get('prediction',{}).get('cap_status','request_failed') for r in subset))
lat=sorted(r['latency_seconds'] for r in results if r['status']=='completed');summary['latency_seconds']={'p50':lat[len(lat)//2],'max':max(lat)}
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False));print('SUMMARY',json.dumps({k:v for k,v in summary.items() if k not in ('system_prompt','output_schema')},ensure_ascii=False),flush=True)
