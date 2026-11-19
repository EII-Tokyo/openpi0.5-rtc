"""Prepare every EP from pinned, independently verified LeRobot data; no GPU use."""
import argparse,collections,json,pathlib,hashlib
import pyarrow.parquet as pq
from openpi.shared.completed_memory_labels import TASK_CATALOG_VERSION,episode_updates

def main(a):
 proof=json.loads(a.dataset_receipt.read_text());revisions={x['repo_id']:x['revision'] for x in proof['heads']};contexts=json.loads(a.return_contexts.read_text());repos=list(dict.fromkeys(json.loads(a.repos.read_text())));cache={};missing=[];summary=[]
 for repo in repos:
  grouped=collections.defaultdict(list)
  for file in sorted((a.datasets/repo/'data').rglob('*.parquet')):
   for row in pq.ParquetFile(file).read(columns=['subtask','episode_index','frame_index','timestamp']).to_pylist():grouped[int(row['episode_index'])].append(row)
  context=contexts.get(repo)
  if context and context['revision']!=revisions[repo]:raise ValueError('Return-context evidence is from another revision')
  cache[repo]={}
  for ep,rows in grouped.items():
   if context and str(ep) not in context['mapping']:raise ValueError('Missing exact source episode correspondence')
   rows.sort(key=lambda x:x['frame_index']);updates,invalid=episode_updates(rows,generic_return_context=context['return_context'] if context else None);cache[repo][str(ep)]=updates
   missing.extend({'repo_id':repo,'episode_index':ep,'frame_index':f} for f in invalid)
  summary.append({'repo':repo,'revision':revisions[repo],'episodes':len(grouped),'updates':sum(len(v) for v in cache[repo].values()),'frames':sum(len(v) for v in grouped.values())});print('MEMORY_PREPARED',summary[-1],flush=True)
 out={'schema_version':2,'task_catalog_version':TASK_CATALOG_VERSION,'episodes':cache,'missing_task_frames':missing,'dataset_revisions':revisions,'label_source':'Human subtask transitions define previous-stage completion targets; not independent visual success verification. New human pickup/identity labels begin a target cycle, not confirmed home arrival. Return occupancy from reviewed explicit human labels; no inferred completion at EP end.','return_context_evidence_file':str(a.return_contexts)}
 a.output.write_text(json.dumps(out,ensure_ascii=False,separators=(',',':')));a.output.with_suffix('.summary.json').write_text(json.dumps({'repos':summary,'missing_task_frames':missing,'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()},ensure_ascii=False,indent=2));print('ALL_MEMORY_PREPARED',sum(len(x) for x in cache.values()),'missing_task_frames',len(missing),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--datasets',type=pathlib.Path,required=True);p.add_argument('--repos',type=pathlib.Path,required=True);p.add_argument('--dataset-receipt',type=pathlib.Path,required=True);p.add_argument('--return-contexts',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);main(p.parse_args())
