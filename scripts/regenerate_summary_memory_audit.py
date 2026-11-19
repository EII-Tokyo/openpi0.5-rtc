"""Paper-inspired compact-summary labels and independent visual high-level audit.

Visual predictions use only
previous predicted state and native-loader images. These sources never mix.
"""
import argparse,base64,copy,dataclasses,json,math,pathlib,time,urllib.request
from openpi.shared.long_memory import MemoryState,encode_memory,decode_memory,parse_annotation
from openpi.shared.completed_step_memory import validate_completed_progress
from openpi_client.runtime.long_memory_high_level import SYSTEM_PROMPT,MEMORY_SCHEMA,parse_update

def key_from(path):
    for line in path.read_text().splitlines():
        if '=' in line:
            k,v=line.split('=',1)
            if k.strip().removeprefix('export ').strip()=='OPENAI_API_KEY':return v.strip().strip('\"\x27')
    raise ValueError('Authorized existing key not found')


def call(key,system,user,schema,raw_path):
    started=time.monotonic()
    body={'model':'gpt-6-luna','store':False,'reasoning':{'effort':'none'},'max_output_tokens':1000,
          'input':[{'role':'system','content':system},{'role':'user','content':user}],
          'text':{'format':{'type':'json_schema','name':'summary_memory','schema':schema,'strict':True}}}
    req=urllib.request.Request('https://api.openai.com/v1/responses',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=90) as response:raw=json.load(response)
    elapsed=time.monotonic()-started
    raw_path.parent.mkdir(exist_ok=True,parents=True);raw_path.write_text(json.dumps(raw,ensure_ascii=False))
    if raw.get('status')!='completed':raise ValueError('Incomplete response, retained raw usage')
    payload=json.loads(''.join(b['text'] for item in raw['output'] for b in item.get('content',[]) if b.get('type')=='output_text'))
    usage=raw['usage'];details=usage.get('input_tokens_details',{});cached=details.get('cached_tokens',0);write=details.get('cache_write_tokens',0)
    cost=((usage['input_tokens']-cached-write)*.1+cached*.01+write*.125+usage['output_tokens']*.5)/1e6
    return payload,{'latency_seconds':elapsed,'usage':usage,'cost_usd':cost,'response_id':raw['id'],'actual_model':raw['model']}


def visual(args):
    key=key_from(args.credentials);rows=[json.loads(x) for x in (args.samples/'samples.jsonl').read_text().splitlines()]
    dest=args.output/'predictions.jsonl';assert not dest.exists(),'Keep previous measurements intact'
    state=MemoryState();cutoff=-math.inf;waiting=True;reports=[]
    for row in rows:
        frames=[f for f in row['video_frames'] if f['time_s']>cutoff] or [row['video_frames'][-1]]
        user=[{'type':'input_text','text':json.dumps({'overall_goal':row['task'],'previous_state':dataclasses.asdict(state)},ensure_ascii=False)}]
        for n,f in enumerate(frames,1):
            for camera,path in f['images'].items():
                encoded=base64.b64encode((args.samples/path).read_bytes()).decode()
                user.extend([{'type':'input_text','text':f'Frame {n}/{len(frames)} (oldest to newest); camera={camera}'},
                             {'type':'input_image','image_url':'data:image/jpeg;base64,'+encoded,'detail':'auto'}])
        record={'number':row['number'],'frame_index':row['frame_index'],'time_s':row['time_s'],
                'input_state':dataclasses.asdict(state),'ground_truth_annotation':row['annotation'],
                'image_count':sum(len(f['images']) for f in frames),'video_timestamps':[f['time_s'] for f in frames]}
        try:
            payload,metrics=call(key,SYSTEM_PROMPT,user,MEMORY_SCHEMA,args.output/'visual-raw'/f"{row['number']}.json")
            record.update(status='returned',prediction=payload,**metrics)
            update=parse_update(payload)
            validate_completed_progress(state.memory,update.state.memory,cycle_complete=update.cycle_complete)
            record['status']='completed'
            if update.cycle_complete:
                state=MemoryState();waiting=True;cutoff=row['time_s']
            elif waiting and (not update.state.bottle_description or any(x in update.state.current_task.lower() for x in ('return home','return to initial','wait for','remain idle'))):state=MemoryState()
            else:
                state=update.state
                if state.bottle_description:waiting=False
            record['accepted_state']=dataclasses.asdict(state)
        except Exception as exc:
            record.update(status='invalid_memory' if 'prediction' in record else 'error',error_type=type(exc).__name__)
            record['accepted_state']=dataclasses.asdict(state)
        with dest.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        reports.append(record);print('VISUAL',row['number'],record['status'],record.get('prediction',{}).get('current_task'),flush=True)
    good=[r for r in reports if r['status']=='completed'];returned=[r for r in reports if 'usage' in r];lat=sorted(r['latency_seconds'] for r in returned)
    result={'schema_version':3,'model':'gpt-6-luna','system_prompt':SYSTEM_PROMPT,'output_schema':MEMORY_SCHEMA,
            'samples':len(rows),'completed':len(good),'responses_returned':len(returned),'errors':len(rows)-len(good),'cost_usd':sum(r['cost_usd'] for r in returned),
            'latency_seconds':{'p50':lat[len(lat)//2],'p95':lat[math.ceil(.95*len(lat))-1]},
            'limitations':'Same single EP and same 100 frames as v1. This is an independent visual prediction audit, not a general accuracy estimate or fixed-3-second closed-loop evaluation. Annotation teacher labels are NOT supplied to visual API calls.'}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['visual']);p.add_argument('--labels',type=pathlib.Path);p.add_argument('--samples',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--credentials',type=pathlib.Path,default=pathlib.Path('/home/eii/openpi0.5-rtc/.env'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);visual(a)
