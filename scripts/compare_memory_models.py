"""Bounded non-actuating same-input model/latency/actual usage comparison."""
import argparse,json,pathlib,time,urllib.error
import numpy as np
from PIL import Image
from openpi.shared.long_memory import MemoryEntry,encode_memory
from openpi_client.runtime.long_memory_high_level import GptMemoryBackend,SYSTEM_PROMPT

# USD per million billed tokens; official model pages checked 2026-11-19.
PRICES={'gpt-5.4':(2.5,.25,15.),'gpt-5.4-mini':(.75,.075,4.5)}

def main(a):
    env={}
    for line in a.credentials.read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k,v=line.split('=',1);env[k.strip()]=v.strip().strip('\"\x27')
    old=json.loads((a.probes/'probes.json').read_text());reports=[]
    for model in a.models:
        backend=GptMemoryBackend(model=model,api_key=env.get('OPENAI_API_KEY'),timeout=60)
        for case,source,stale in [('initial-empty',old[0],False),('held-empty',old[1],False),('held-stale',old[1],True)]:
            sample=source['sample'];folder=a.probes/source['case'];frames=[]
            for i,ts in enumerate(sample['video']['timestamps']):
                frames.append((ts,{c:np.asarray(Image.open(folder/f'{i:02d}-{c}.jpg').convert('RGB')) for c in sample['video']['cameras']}))
            memory=(MemoryEntry(0,'target bottle','on table, grasp not yet confirmed','Adjust grasping pose and pick up target bottle'),) if stale else ()
            record={'model':model,'case':case,'sample':sample,'memory_input':json.loads(encode_memory(memory)),
                    'image_count':sum(len(images) for _,images in frames),'system_prompt':SYSTEM_PROMPT,'prices_per_million':PRICES[model]}
            start=time.monotonic()
            try:
                result=backend(task='Twist off bottle caps and dispose of bottles/caps into their designated bins.',memory=memory,frames=frames,now_s=sample['time_s'])
                usage=backend.last_usage;i=usage['input_tokens'];o=usage['output_tokens'];cached=usage.get('input_tokens_details',{}).get('cached_tokens',0)
                pi,pc,po=PRICES[model];cost=((i-cached)*pi+cached*pc+o*po)/1e6
                record.update(status='completed',latency_seconds=time.monotonic()-start,usage=usage,actual_model=backend.last_model,
                    response_id=backend.last_response_id,memory_output=json.loads(encode_memory(result)),cost_usd=cost,
                    hypothetical_1200_calls_per_hour_usd=1200*cost)
            except urllib.error.HTTPError as exc:record.update(status='api_error',http_status=exc.code,latency_seconds=time.monotonic()-start)
            except Exception as exc:record.update(status='error',error_type=type(exc).__name__,latency_seconds=time.monotonic()-start)
            reports.append(record);a.output.write_text(json.dumps(reports,ensure_ascii=False,indent=2))
            print(model,case,record['status'],round(record['latency_seconds'],3),record.get('cost_usd'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--probes',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--credentials',type=pathlib.Path,default=pathlib.Path('/home/eii/openpi0.5-rtc/.env'));p.add_argument('--models',nargs='+',default=list(PRICES));main(p.parse_args())
