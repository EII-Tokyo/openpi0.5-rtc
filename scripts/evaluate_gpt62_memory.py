"""Budget-bounded, paired GPT-6.2 comparison on frozen native-loader samples."""
import argparse,base64,concurrent.futures,fcntl,hashlib,json,pathlib,threading,time,urllib.request
from decimal import Decimal

PRICES={'gpt-6.2-astra':{'input':8,'output':40,'cached':.4,'write':10},'gpt-6.2-sol':{'input':2,'output':10,'cached':.1,'write':2.5}}
MODELS=tuple(PRICES)


def usage_cost(usage,model):
    p=PRICES[model];details=usage.get('input_tokens_details',{});cached=details.get('cached_tokens',0);write=details.get('cache_write_tokens',0)
    return (max(0,usage['input_tokens']-cached-write)*p['input']+cached*p['cached']+write*p['write']+usage['output_tokens']*p['output'])/1e6


class Ledger:
    def __init__(self,path,limit=10):
        self.path=path;self.limit=Decimal(str(limit));self.lock=threading.Lock()
        self.rows=json.loads(path.read_text())['requests'] if path.exists() else []
    def total(self):return sum((Decimal(str(x.get('cost_usd',x['reserved_usd']))) for x in self.rows),Decimal(0))
    def save(self):
        temp=self.path.with_suffix('.tmp');temp.write_text(json.dumps({'budget_usd':float(self.limit),'accounted_cost_including_unresolved_reservations':float(self.total()),'known_billed_usd':sum(x.get('cost_usd',0) for x in self.rows),'requests':self.rows},indent=2));temp.replace(self.path)
    def reserve(self,key,value):
        with self.lock:
            if any(x['key']==key for x in self.rows):return False
            if self.total()+Decimal(str(value))>self.limit:return False
            self.rows.append({'key':key,'reserved_usd':value,'status':'reserved'});self.save();return True
    def finish(self,key,cost=None,**meta):
        with self.lock:
            row=next(x for x in self.rows if x['key']==key);row.update(meta)
            if cost is not None:
                if cost>row['reserved_usd']+1e-9:raise RuntimeError('Usage exceeded conservative reservation; stop')
                row['cost_usd']=cost
            self.save()


def read_key(path):
    for line in path.read_text().splitlines():
        if '=' in line:
            k,v=line.split('=',1)
            if k.strip().removeprefix('export ').strip()=='OPENAI_API_KEY':return v.strip().strip('\"\x27')
    raise ValueError('Authorized key unavailable')


def request_json(key,path,body,timeout=120):
    req=urllib.request.Request('https://api.openai.com/v1/'+path,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as f:return json.load(f)


def main(a):
    if not 0<a.budget<=10:raise ValueError("Total comparison budget must be at most10 USD")
    from openpi.shared.high_level_index_protocol import decode_index_prediction
    a.root.mkdir(parents=True,exist_ok=True);(a.root/'raw').mkdir(exist_ok=True)
    runlock=(a.root/'runner.lock').open('a');fcntl.flock(runlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prompt=(a.plan/'prompt-en.txt').read_text();schema=json.loads((a.plan/'output-schema.json').read_text());key=read_key(a.credentials)
    assert 'dataloader' not in prompt.lower() and 'dropout' not in prompt.lower()
    rows=[json.loads(s) for s in (a.root/'samples.jsonl').read_text().splitlines()]
    first=json.loads((a.root/'preflight-frame.json').read_text())
    if a.mode=='preflight':rows=[first]
    else:
        # Cover the whole EP early if the budget cannot cover the whole stride grid.
        # Every selected frame still belongs to the exact0,10,20,... native grid.
        order=[]
        def spread(lo,hi):
            if lo>hi:return
            mid=(lo+hi)//2;order.append(mid);spread(lo,mid-1);spread(mid+1,hi)
        # Breadth-first interval subdivision, not a contiguous early-EP truncation.
        queue=[(0,len(rows)-1)];order=[]
        while queue:
            lo,hi=queue.pop(0)
            if lo>hi:continue
            mid=(lo+hi)//2;order.append(mid);queue.extend([(lo,mid-1),(mid+1,hi)])
        rows=[rows[i] for i in order if rows[i]['frame_index']!=first['frame_index']]
    ledger=Ledger(a.root/'budget-ledger.json',a.budget);output=a.root/'predictions.jsonl';outlock=threading.Lock()
    def process(row,model):
        frame=row['frame_index'];rid=f'{model}:{frame}';content=[{'type':'input_text','text':json.dumps(row['input_text'],ensure_ascii=False)}]
        for camera,path in row['images'].items():
            encoded=base64.b64encode((a.root/path).read_bytes()).decode();content.extend([{'type':'input_text','text':camera},{'type':'input_image','image_url':'data:image/png;base64,'+encoded,'detail':'high'}])
        inputs=[{'role':'system','content':prompt},{'role':'user','content':content}]
        try:count=request_json(key,'responses/input_tokens',{'model':model,'input':inputs})['input_tokens']
        except Exception as exc:
            return {'key':rid,'status':'count_failed','error_type':type(exc).__name__,'http_code':getattr(exc,'code',None)}
        reserve=((count+2048)*PRICES[model]['write']+a.max_output_tokens*PRICES[model]['output'])/1e6
        if not ledger.reserve(rid,reserve):return {'key':rid,'status':'skipped_budget_or_existing'}
        body={'model':model,'store':False,'service_tier':'default','reasoning':{'effort':'low'},'max_output_tokens':a.max_output_tokens,
              'input':inputs,'text':{'format':{'type':'json_schema','name':'bottle_memory','strict':True,'schema':schema}}}
        record={'key':rid,'model':model,'frame_index':frame,'input_token_count_before_schema':count,'reserved_usd':reserve,'phase':a.mode,'image_detail':'high','reasoning_effort':'low','service_tier':'default'}
        start=time.monotonic();raw=None
        try:
            raw=request_json(key,'responses',body);record['latency_seconds']=time.monotonic()-start
            (a.root/'raw'/f'{model}-{frame}.json').write_text(json.dumps(raw,ensure_ascii=False))
            usage=raw.get('usage');cost=usage_cost(usage,model) if usage else None
            record.update(status=raw.get('status','unknown'),usage=usage,cost_usd=cost,response_id=raw.get('id'),returned_model=raw.get('model'),returned_service_tier=raw.get('service_tier'))
            ledger.finish(rid,cost,status=record['status'],response_id=record['response_id'])
            text=''.join(block['text'] for item in raw.get('output',[]) for block in item.get('content',[]) if block.get('type')=='output_text')
            if text:record['prediction']=json.loads(text);record['mapped_low_level_state']=json.loads(__import__('openpi.shared.long_memory',fromlist=['encode_memory']).encode_memory(decode_index_prediction(record['prediction'])))
            if raw.get('status')!='completed' or 'prediction' not in record:record['status']='incomplete'
        except Exception as exc:
            record.update(status='error',error_type=type(exc).__name__,http_code=getattr(exc,'code',None),latency_seconds=time.monotonic()-start)
            if raw is None:ledger.finish(rid,status='unresolved_charge_reserved',error_type=type(exc).__name__)
        with outlock:
            with output.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(json.dumps({k:record.get(k) for k in ('model','frame_index','status','cost_usd','latency_seconds')},ensure_ascii=False),flush=True)
        return record
    workers=1 if a.mode=='preflight' else a.workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        chunk=1 if a.mode=='preflight' else max(1,workers//2)
        for start in range(0,len(rows),chunk):
            pending=[pool.submit(process,row,model) for row in rows[start:start+chunk] for model in MODELS]
            results=[x.result() for x in pending]
            if any(x['status']=='count_failed' for x in results):break
            if all(x['status']=='skipped_budget_or_existing' for x in results):
                # Existing requests near restart must not prevent the next fresh batch.
                if float(ledger.total())>a.budget-.15:break
            if any(x.get('http_code') in (401,403,429) for x in results):break
    print('FINISHED',a.mode,'budget_accounted',float(ledger.total()),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','sweep']);p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--plan',type=pathlib.Path,required=True);p.add_argument('--credentials',type=pathlib.Path,default=pathlib.Path('/home/eii/openpi0.5-rtc/.env'));p.add_argument('--budget',type=float,default=10);p.add_argument('--max-output-tokens',type=int,default=1536);p.add_argument('--workers',type=int,default=2);main(p.parse_args())
