"""Audit the exact weighted config and build causal high-level supervision (no video decode)."""
import argparse, collections, hashlib, json, pathlib, random
import numpy as np
import pyarrow.parquet as pq
from openpi.shared.long_memory import LongMemoryConfig, MemoryTimeline, condition_at, encode_memory, episode_split, video_indices, parse_annotation


def build(repo_list, root, output):
    output.mkdir(parents=True, exist_ok=True)
    weights=collections.Counter(repo_list);settings=LongMemoryConfig();report={'weights':dict(weights),'unique_repos':len(weights),'weighted_entries':len(repo_list),'repos':[],'errors':[],'examples':collections.Counter()}
    handles={s:(output/f'{s}.jsonl').open('w') for s in ['train','validation','test']}
    try:
        for repo_id, weight in weights.items():
            repo=root/repo_id
            frames=list(repo.glob('data/**/*.parquet'))
            if not frames:raise FileNotFoundError(repo)
            chunks=[]
            for file in sorted(frames):
                pf=pq.ParquetFile(file);cols=['episode_index','frame_index','timestamp','subtask','index']
                if 'is_for_training' in pf.schema_arrow.names:cols+=['is_for_training']
                chunks+=pf.read(columns=cols).to_pylist()
            episodes=collections.defaultdict(list)
            for row in chunks:episodes[int(row['episode_index'])].append(row)
            rr={'repo_id':repo_id,'weight':weight,'frames':len(chunks),'episodes':len(episodes),'descriptions_missing':0,'events':0,'splits':collections.Counter(),'label_counts':collections.Counter(),'files':[], 'unlabeled_frames':0, 'state_unannotated_frames':0}
            for f in [repo/'meta/info.json',*frames,*repo.glob('meta/episodes/**/*.parquet')]:
                rr['files'].append({'path':str(f.relative_to(repo)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
            for ep, rows in episodes.items():
                rows.sort(key=lambda r:r['frame_index']);ts=[float(r['timestamp']) for r in rows]
                if any(r['frame_index']!=i for i,r in enumerate(rows)):raise ValueError(f'{repo_id}/{ep}: noncontiguous frame indices')
                eligible=[bool(r.get('is_for_training',True)) for r in rows]
                parsed=[parse_annotation(r['subtask']) if ok else None for r,ok in zip(rows,eligible,strict=True)]
                rr['unlabeled_frames']+=sum(ok and v is None for ok,v in zip(eligible,parsed,strict=True))
                rr['state_unannotated_frames']+=sum(v is not None and v[1]=='Unknown (not annotated)' for v in parsed)
                timeline=MemoryTimeline(ts,[r['subtask'] for r in rows],eligible,reset_on_home=settings.reset_on_home)
                rr['events']+=len(timeline.events);rr['descriptions_missing']+=sum(not e.bottle for e in timeline.events)
                for e in timeline.events:rr['label_counts'][e.intent]+=1
                split=episode_split(repo_id,ep,settings.split_seed);rr['splits'][split]+=1
                rng=random.Random(f'{repo_id}/{ep}/memory')
                # Whole segments sampled at 1 Hz, not merely their transition frames.
                candidates=set(np.searchsorted(ts,np.arange(0,ts[-1]+1e-6,1)).clip(0,len(ts)-1).tolist())
                # Add delayed observations inside phases, maintaining t>=transition time.
                for event in timeline.events:
                    for delay in [0,.5,1.5,3,6]:
                        t=event.time_s+delay
                        if t<=ts[-1]:candidates.add(min(len(ts)-1,int(np.searchsorted(ts,t))))
                for index in sorted(candidates):
                    if not eligible[index] or parsed[index] is None:continue
                    now=ts[index];target=timeline.at(now,settings.max_entries)
                    # Initial memory is empty; later inputs include full, missing and stale memory.
                    input_memory=() if now==ts[0] else condition_at(timeline,max(0,now-settings.update_seconds),settings,training=True,rng=rng)
                    vi=video_indices(ts,now,seconds=settings.video_seconds,hz=settings.video_hz)
                    sample={'repo_id':repo_id,'episode_index':ep,'frame_index':rows[index]['frame_index'],'time_s':now,'split':split,'repo_weight':weight,'memory_input':json.loads(encode_memory(input_memory)),'memory_target':json.loads(encode_memory(target)),'video':{'seconds':settings.video_seconds,'indices':[rows[i]['index'] for i in vi],'frame_indices':[rows[i]['frame_index'] for i in vi],'timestamps':[ts[i] for i in vi],'cameras':['cam_high','cam_left_wrist','cam_right_wrist']}}
                    # High-level lifecycle fields are supervised separately from low-level event text.
                    label=parsed[index][2].lower()
                    phase='return_home' if 'return' in label and ('home' in label or 'initial' in label) else 'unscrew' if 'unscrew' in label else 'dispose' if any(x in label for x in ('trash','dispos','release','cap cleanup')) else 'pickup' if any(x in label for x in ('pick','adjust')) else 'unknown'
                    sample['high_level_target']={'events':sample['memory_target']['events'],'cycle_complete':bool(settings.reset_on_home and not target and phase=='return_home'),'current_phase':phase}
                    # Repeat by the exact source multiplicity; no class balancing/oversampling.
                    for _ in range(weight):handles[split].write(json.dumps(sample,ensure_ascii=False)+'\n');report['examples'][split]+=1
            report['repos'].append(rr)
            print(repo_id,'frames',rr['frames'],'events',rr['events'],'missing descriptions',rr['descriptions_missing'],flush=True)
    finally:
        for h in handles.values():h.close()
    (output/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    return report

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--repos',type=pathlib.Path,required=True);a.add_argument('--root',type=pathlib.Path,required=True);a.add_argument('--output',type=pathlib.Path,required=True);args=a.parse_args()
    build(json.loads(args.repos.read_text()),args.root,args.output)
