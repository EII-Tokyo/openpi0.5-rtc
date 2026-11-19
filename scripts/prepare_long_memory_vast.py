"""Prepare independent, latest-main, hash-verified copies; no active dataset writes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,os,subprocess,time,urllib.request,urllib.parse
from huggingface_hub import HfApi
R=Path('/workspace/long-memory-hierarchy-20261119');S=Path('/workspace/twist-gns-20261117/datasets');D=R/'datasets'
api=HfApi();repos=list(dict.fromkeys(json.loads((R/'weighted-repos.json').read_text())))

def heads(repo):
 refs=api.list_repo_refs(repo,repo_type='dataset')
 found={x.name:x.target_commit for x in [*refs.branches,*refs.tags] if x.name in ('main','v3.0')}
 if 'main' not in found:raise ValueError('Missing published main: '+repo)
 return {'repo_id':repo,'revision':found['main'],'refs':found,'resolved_unix':time.time()}

def manifest(d):
 files=[]
 for f in api.list_repo_tree(d['repo_id'],revision=d['revision'],repo_type='dataset',recursive=True):
  if not hasattr(f,'size') or not f.path.startswith(('meta/','data/','videos/')):continue
  files.append({'path':f.path,'size':f.size,'blob_id':f.blob_id,'lfs_sha256':f.lfs.sha256 if f.lfs else None})
 assert files and any(f['path']=='meta/info.json' for f in files)
 return {**d,'files':files}

def digest(p,f):
 sha=hashlib.sha256();git=hashlib.sha1(('blob '+str(f['size'])+'\0').encode());count=0
 with p.open('rb') as h:
  for b in iter(lambda:h.read(8*1024**2),b''):sha.update(b);git.update(b);count+=len(b)
 ok=count==f['size'] and (sha.hexdigest()==f['lfs_sha256'] if f['lfs_sha256'] else git.hexdigest()==f['blob_id'])
 return ok,sha.hexdigest()

def prepare(pair):
 d,f=pair;target=D/d['repo_id']/f['path'];source=S/d['repo_id']/f['path'];target.parent.mkdir(parents=True,exist_ok=True)
 if target.is_symlink():raise ValueError('Independent dataset files required')
 method='existing_verified'
 if not target.exists():
  if source.is_file():
   subprocess.run(['cp','--reflink=auto','--',str(source),str(target)],check=True);method='independent_copy'
  else:method='download'
 ok,sha=digest(target,f) if target.exists() else (False,None)
 if not ok:
  url='https://huggingface.co/datasets/'+d['repo_id']+'/resolve/'+d['revision']+'/'+urllib.parse.quote(f['path'])
  temp=target.with_suffix(target.suffix+'.partial')
  with urllib.request.urlopen(url,timeout=180) as resp,temp.open('wb') as out:
   while b:=resp.read(8*1024**2):out.write(b)
  ok,sha=digest(temp,f)
  if not ok:raise ValueError('Published file hash mismatch: '+str(target))
  temp.replace(target);method='download'
 if source.exists():assert target.stat().st_ino!=source.stat().st_ino
 return {'repo_id':d['repo_id'],**f,'sha256':sha,'method':method,'verified_unix':time.time()}

if __name__=='__main__':
 R.mkdir(exist_ok=True);(R/'artifacts').mkdir(exist_ok=True)
 started=time.time()
 with ThreadPoolExecutor(max_workers=4) as ex:first=list(ex.map(heads,repos));datasets=list(ex.map(manifest,first))
 (R/'artifacts/latest-dataset-manifest.json').write_text(json.dumps(datasets,indent=2))
 pairs=[(d,f) for d in datasets for f in d['files']]
 receipts=[]
 with ThreadPoolExecutor(max_workers=4) as ex:
  for x in ex.map(prepare,pairs):
   receipts.append(x)
   if len(receipts)%25==0:print('VERIFIED',len(receipts),'/',len(pairs),flush=True)
 for d in datasets:
  info=json.loads((D/d['repo_id']/'meta/info.json').read_text());assert info['codebase_version']=='v3.0',(d['repo_id'],'Unsupported published version; startup blocked')
 with ThreadPoolExecutor(max_workers=4) as ex:second=list(ex.map(heads,repos))
 assert [(d['repo_id'],d['refs']) for d in first]==[(d['repo_id'],d['refs']) for d in second],'Published heads changed during preparation'
 proof={'started_unix':started,'completed_unix':time.time(),'heads':second,'files':receipts,'independent_account_copy':True,'root':str(D)}
 (R/'artifacts/datasets-verified.json').write_text(json.dumps(proof,indent=2));print('ALL_DATASETS_VERIFIED',len(repos),len(receipts),flush=True)
