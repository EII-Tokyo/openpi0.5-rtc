"""Independent 1-Hz API dispatch test; never connects responses to robot control."""
import argparse,concurrent.futures,importlib.util,json,math,pathlib,time
from openpi.shared.long_memory import decode_memory


def main(root):
    """Replay twenty evenly spread measured inputs, with bounded asynchronous requests."""
    spec=importlib.util.spec_from_file_location('evaluation',pathlib.Path(__file__).with_name('evaluate_episode_luna.py'))
    e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
    rows=[json.loads(x) for x in (root/'samples.jsonl').read_text().splitlines()]
    results=[json.loads(x) for x in (root/'luna-results-v2.jsonl').read_text().splitlines()]
    summary=json.loads((root/'luna-summary-v2.json').read_text());e.SCHEMA=summary['schema']
    assert summary['system_prompt'].endswith('\n'+e.INSTRUCTIONS)
    e.SYSTEM_PROMPT=summary['system_prompt'][:-len('\n'+e.INSTRUCTIONS)]  # Match the measured prompt exactly.
    out=root.parent/'luna-throughput-1hz';out.mkdir(exist_ok=True)
    if not (out/'frames').exists():(out/'frames').symlink_to(root/'frames',target_is_directory=True)
    env={}
    for line in pathlib.Path('/home/eii/openpi0.5-rtc/.env').read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k,v=line.split('=',1);env[k.strip()]=v.strip().strip('\"\x27')
    beginning=time.monotonic();completed=[]
    def call(index,scheduled):
        started=time.monotonic();row=rows[index]
        memory=decode_memory(json.dumps(results[index]['memory_input']),now_s=row['time_s'])
        prediction,metrics=e.request(env['OPENAI_API_KEY'],row,memory,out,-1)
        return {'number':row['number'],'scheduled_offset_s':scheduled,'actual_start_offset_s':started-beginning,'finished_offset_s':time.monotonic()-beginning,'prediction':prediction,**metrics}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        pending=[]
        for n,index in enumerate(range(0,100,5)):
            delay=beginning+n-time.monotonic()
            if delay>0:time.sleep(delay)
            pending.append(pool.submit(call,index,n));print('DISPATCH',n+1,flush=True)
        for future in concurrent.futures.as_completed(pending):
            try:completed.append(future.result())
            except Exception as exc:completed.append({'error_type':type(exc).__name__})
    (out/'results.json').write_text(json.dumps(completed,indent=2,ensure_ascii=False))
    good=sorted([x for x in completed if 'latency_seconds' in x],key=lambda x:x['actual_start_offset_s']);lat=sorted(x['latency_seconds'] for x in good)
    intervals=[b['actual_start_offset_s']-a['actual_start_offset_s'] for a,b in zip(good,good[1:])]
    report={'requested_hz':1,'requests':20,'completed':len(good),'errors':20-len(good),'max_workers':8,
            'dispatch_interval_mean_s':sum(intervals)/len(intervals),'dispatch_interval_min_s':min(intervals),'dispatch_interval_max_s':max(intervals),
            'full_response_latency_p50_s':lat[len(lat)//2],'full_response_latency_p95_s':lat[math.ceil(.95*len(lat))-1],
            'responses_within_one_second':sum(x<=1 for x in lat),'cost_usd':sum(x['cost_usd'] for x in good),
            'note':'Independent replay load test, not a closed-loop memory controller. A1-Hz request rate does not mean1-second-fresh decisions.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);main(p.parse_args().root)
