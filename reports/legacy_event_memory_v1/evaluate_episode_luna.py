"""Non-actuating, sequential GPT-6 Luna evaluation on 100 native-loader samples."""
import argparse,base64,collections,copy,json,math,pathlib,time,urllib.error,urllib.request
from PIL import Image
from openpi.shared.long_memory import decode_memory,encode_memory
from openpi_client.runtime.long_memory_high_level import MEMORY_SCHEMA,SYSTEM_PROMPT

SCHEMA=copy.deepcopy(MEMORY_SCHEMA)
SCHEMA['properties'].update(current_phase={'type':'string','enum':['pickup','unscrew','dispose','return_home','unknown']},
    cap_status={'type':'string','enum':['capped','uncapped','unknown','no_active_bottle']},
    bottle_held_by={'type':'string','enum':['left','right','both','none','unknown']})
SCHEMA['required']=list(SCHEMA['properties'])
INSTRUCTIONS='''For evaluation also report the CURRENT phase: pickup (adjust/reach/grasp), unscrew (cap twisting), dispose (bottle/cap handling into bins), return_home (moving to OR having just reached the initial parked pose with empty grippers), or unknown. Report cap_status for the ACTIVE bottle, and which hand holds the bottle (holding only the cap or neck, even while attached, does not count: identify the gripper directly clamping the BOTTLE BODY, using robot anatomical left/right and the named wrist cameras, not screen-left/right). Never use future frames. Classify current evidence even when old memory is wrong. The evaluation phase is an action-stage label, not a command. At confirmed home completion cycle_complete=true and events=[]; on subsequent calls with no target and empty parked hands retain an empty memory. Do not claim home completion just because the return-home motion started. Keep descriptions concise and use observable attributes; avoid guessing brands.'''


def phase(label):
    """Map the human annotation to an explicit coarse action stage."""
    s=label.lower()
    if 'return' in s and ('home' in s or 'initial' in s):return 'return_home'
    if 'unscrew' in s:return 'unscrew'
    if any(x in s for x in ('trash','dispos','release','cap cleanup')):return 'dispose'
    if 'pick' in s or 'adjust' in s:return 'pickup'
    return None


def expected_state(text):
    """Score only what the human state explicitly says, never infer missing labels."""
    s=text.lower();cap=held=None
    if 'cap removed' in s or 'uncapped' in s or 'without cap' in s:cap='uncapped'
    elif 'capped' in s or 'with cap' in s:cap='capped'
    if 'bottle in left hand' in s:held='left'
    elif 'bottle in right hand' in s:held='right'
    elif 'bottle on table before grasping' in s:held='none'
    return cap,held


def request(key,row,memory,root,cutoff):
    """Return a fully parsed JSON response; latency includes payload preparation."""
    started=time.monotonic();now=math.floor(row['time_s']*1000)/1000
    content=[{'type':'input_text','text':json.dumps({'task':row['task'],'now_s':now,'previous_memory':json.loads(encode_memory(memory))})}]
    selected=[x for x in row['video_frames'] if x['time_s']>cutoff]
    if not selected:selected=[row['video_frames'][-1]]
    for frame in selected:
        for camera,rel in frame['images'].items():
            data=base64.b64encode((root/rel).read_bytes()).decode()
            content.extend([{'type':'input_text','text':f"time_s={math.floor(frame['time_s']*1000)/1000:.3f}; camera={camera}"},
                            {'type':'input_image','image_url':'data:image/jpeg;base64,'+data,'detail':'auto'}])
    body={'model':'gpt-6-luna','store':False,'reasoning':{'effort':'none'},'max_output_tokens':1600,
          'input':[{'role':'system','content':SYSTEM_PROMPT+'\n'+INSTRUCTIONS},{'role':'user','content':content}],
          'text':{'format':{'type':'json_schema','name':'memory_audit','schema':SCHEMA,'strict':True}}}
    req=urllib.request.Request('https://api.openai.com/v1/responses',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=60) as response:
        raw=json.load(response)
        rate={k:v for k,v in response.headers.items() if k.lower().startswith('x-ratelimit-')}
    latency=time.monotonic()-started
    archive=root/'raw-luna-v2';archive.mkdir(exist_ok=True)
    (archive/(str(row['number'])+'.json')).write_text(json.dumps({'response':raw,'rate_limits':rate,'latency_seconds':latency},ensure_ascii=False))
    assert raw.get('status')=='completed','Incomplete structured response'
    text=''.join(b['text'] for x in raw['output'] for b in x.get('content',[]) if b.get('type')=='output_text')
    output=json.loads(text)
    usage=raw['usage'];detail=usage.get('input_tokens_details',{});cached=detail.get('cached_tokens',0);write=detail.get('cache_write_tokens',0)
    cost=((usage['input_tokens']-cached-write)*.1+cached*.01+write*.125+usage['output_tokens']*.5)/1e6
    return output,{'latency_seconds':latency,'usage':usage,'cost_usd':cost,'actual_model':raw['model'],
                         'response_id':raw['id'],'service_tier':raw.get('service_tier'),'rate_limits':rate,
                         'video_timestamps':[f['time_s'] for f in selected],'image_count':sum(len(f['images']) for f in selected)}


def main(a):
    """Carry only predicted memory forward; ground truth never enters API inputs."""
    env={}
    for line in a.credentials.read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k,v=line.split('=',1);env[k.strip().removeprefix('export ').strip()]=v.strip().strip('\"\x27')
    rows=[json.loads(line) for line in (a.root/'samples.jsonl').read_text().splitlines()]
    output=a.root/'luna-results-v2.jsonl';assert not output.exists(),'Preserve previous measured responses'
    memory=();cutoff=-1;reports=[]
    with output.open('x') as f:
        for row in rows:
            record={'number':row['number'],'frame_index':row['frame_index'],'time_s':row['time_s'],'ground_truth':row['annotation'],
                    'memory_input':json.loads(encode_memory(memory)),'expected_phase':phase(row['annotation']['intent'])}
            try:
                result,metrics=request(env['OPENAI_API_KEY'],row,memory,a.root,cutoff)
                record.update(status='completed',prediction=result,**metrics)
                try:
                    next_memory=decode_memory(json.dumps({'events':result['events']}),now_s=row['time_s'])
                    assert not result['cycle_complete'] or not next_memory,'A completed bottle cannot retain old memory'
                    memory=next_memory
                    if result['cycle_complete']:cutoff=row['time_s'];memory=()
                except Exception as exc:
                    record.update(status='invalid_memory',memory_error=type(exc).__name__)
                    # Keep the previous accepted memory; retain usage and latency for ALL returned responses.
                cap,held=expected_state(row['annotation']['observation']);record.update(expected_cap_status=cap,expected_held_by=held)
                record['phase_correct']=result['current_phase']==record['expected_phase'] if record['expected_phase'] else None
                record['cap_correct']=result['cap_status']==cap if cap else None
                record['held_correct']=result['bottle_held_by']==held if held else None
            except urllib.error.HTTPError as exc:record.update(status='api_error',http_status=exc.code)
            except Exception as exc:record.update(status='error',error_type=type(exc).__name__)
            f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();reports.append(record)
            print('LUNA',row['number'],record['status'],record.get('latency_seconds'),record.get('phase_correct'),flush=True)
    good=[r for r in reports if r['status'] in {'completed','invalid_memory'}];latencies=sorted(r['latency_seconds'] for r in good)
    def accuracy(field):
        scored=[r[field] for r in good if r[field] is not None]
        return {'correct':sum(scored),'count':len(scored),'accuracy':sum(scored)/len(scored) if scored else None}
    summary={'model':'gpt-6-luna','reasoning_effort':'none','requests':len(rows),'completed':len(good),'errors':len(rows)-len(good),'fully_valid_memory':sum(r['status']=='completed' for r in good),
             'phase':accuracy('phase_correct'),'cap':accuracy('cap_correct'),'held_by':accuracy('held_correct'),
             'latency_seconds':{'p50':latencies[len(latencies)//2],'p95':latencies[min(len(latencies)-1,math.ceil(.95*len(latencies))-1)],'max':max(latencies),'under_one_second':sum(t<=1 for t in latencies)},
             'total_cost_usd':sum(r['cost_usd'] for r in good),'mean_cost_usd':sum(r['cost_usd'] for r in good)/len(good),
             'ground_truth_phases':dict(collections.Counter(r['expected_phase'] for r in reports)),
             'limitations':'One randomly chosen EP,100 correlated frames. Coarse phase/explicit state accuracy, not overall free-text memory factual accuracy. No ground truth sent to model.',
             'system_prompt':SYSTEM_PROMPT+'\n'+INSTRUCTIONS,'schema':SCHEMA,'prices_usd_per_million':{'input':.1,'cached_input':.01,'cache_write':.125,'output':.5}}
    (a.root/'luna-summary-v2.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print('SUMMARY',json.dumps({k:v for k,v in summary.items() if k not in ('system_prompt','schema')},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--credentials',type=pathlib.Path,default=pathlib.Path('/home/eii/openpi0.5-rtc/.env'));main(p.parse_args())
