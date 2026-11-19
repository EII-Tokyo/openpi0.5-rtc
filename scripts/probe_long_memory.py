"""Offline, non-actuating GPT probe on real causal multi-camera video windows."""
import argparse,json,pathlib,time,urllib.error
import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from openpi.shared.long_memory import decode_memory,encode_memory
from openpi_client.runtime.long_memory_high_level import GptMemoryBackend,SYSTEM_PROMPT


def materialize(sample, root, output):
    repo=root/sample['repo_id'];info=json.loads((repo/'meta/info.json').read_text())
    episode=next(r for p in repo.glob('meta/episodes/**/*.parquet') for r in pq.read_table(p).to_pylist() if int(r['episode_index'])==sample['episode_index'])
    images={}
    for camera in sample['video']['cameras']:
        col='observation.images.'+camera
        path=repo/info['video_path'].format(video_key=col,chunk_index=episode[f'videos/{col}/chunk_index'],file_index=episode[f'videos/{col}/file_index'])
        offset=episode[f'videos/{col}/from_timestamp']
        requested=[offset+t for t in sample['video']['timestamps']]
        with av.open(str(path)) as container:
            stream=container.streams.video[0];stream.thread_type='AUTO'
            container.seek(max(0,int(requested[0]/float(stream.time_base))-1),stream=stream,backward=True)
            decoded=[]
            for frame in container.decode(stream):
                ft=float(frame.pts*stream.time_base)
                decoded.append((ft,frame.to_image()))
                if ft>=requested[-1]+.1:break
        if not decoded:raise ValueError('No video frames decoded')
        for i,target in enumerate(requested):
            # Nearest available past image, never a later camera observation.
            candidates=[r for r in decoded if r[0]<=target+1e-6]
            if not candidates:raise ValueError('Video seek did not yield a causal frame')
            ft,image=candidates[-1]
            if abs(ft-target)>.1:raise ValueError('Video metadata/PTS does not match requested timestamp')
            image.thumbnail((384,384))
            imagepath=output/f'{i:02d}-{camera}.jpg';image.save(imagepath,quality=88)
            images.setdefault(sample['video']['timestamps'][i],{})[camera]=np.asarray(image)
    return sorted(images.items())


def main(a):
    output=a.output;output.mkdir(parents=True,exist_ok=True)
    initial=held=None
    with a.manifest.open() as f:
        for line in f:
            row=json.loads(line)
            if not row['memory_target']['events']:continue
            last=row['memory_target']['events'][-1]
            if row['time_s']==0 and initial is None:initial=row
            if 'left hand and capped' in last['observation'].lower() and row['time_s']>=6 and held is None:held=row
            if initial and held:break
    backend=None
    if a.live:
        # User authorized reusing host103's existing key. Never copy it to output/source.
        env={}
        for line in a.credentials.read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                k,v=line.split('=',1);env[k.strip()]=v.strip().strip('\"\x27')
        backend=GptMemoryBackend(api_key=env.get('OPENAI_API_KEY'))
    reports=[]
    for label,sample in [('initial-empty',initial),('already-held-empty-memory',held)]:
        if sample is None:raise ValueError('Required real annotation case not found')
        dest=output/label;dest.mkdir(exist_ok=True)
        frames=materialize(sample,a.root,dest)
        record={'case':label,'sample':sample,'memory_input':{'events':[]},'frame_count':len(frames),'camera_count':len(frames[0][1]),'system_prompt':SYSTEM_PROMPT}
        if backend:
            started=time.monotonic()
            try:
                memory=backend(task='Twist off bottle caps and dispose of bottles/caps into their designated bins.',memory=(),frames=frames,now_s=sample['time_s'])
                record.update(status='completed',memory_output=json.loads(encode_memory(getattr(memory,"events",memory))),latency_seconds=time.monotonic()-started)
            except urllib.error.HTTPError as exc:
                record.update(status='api_error',http_status=exc.code)
            except Exception as exc:
                record.update(status='error',error_type=type(exc).__name__)
        (dest/'probe.json').write_text(json.dumps(record,indent=2,ensure_ascii=False));reports.append(record)
        print(label,record.get('status','rendered'),record.get('latency_seconds'),flush=True)
    (output/'probes.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=pathlib.Path,required=True);p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--live',action='store_true');p.add_argument('--credentials',type=pathlib.Path,default=pathlib.Path('/home/eii/openpi0.5-rtc/.env'));main(p.parse_args())
