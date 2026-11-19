"""Capture 100 exact native-loader inputs and causal high-level video from one random EP.

Run on CPU against the immutable training source, not a reconstructed DataLoader.
Image fetching for each history timestamp also uses the real LeRobot __getitem__.
"""
import argparse,collections,hashlib,json,os,pathlib,random,secrets,sys
import numpy as np
from PIL import Image


def main(a):
    """Select reproducibly after drawing a random seed, then record native collation."""
    os.chdir(a.source)  # Config objects load normalization assets during module import.
    sys.path[:0]=[str(a.source/'src'),str(a.source/'packages/openpi-client/src')]
    from openpi.training.config import get_config
    from openpi.data.datasets import create_torch_dataset
    from openpi.data.transforms import compose
    from openpi.data.dataloaders import TorchDataLoader
    from openpi.shared.long_memory import encode_memory,parse_annotation,video_indices,memory_prompt
    from openpi_client.runtime.long_memory_high_level import image_uint8
    prior=a.chosen if a.chosen is not None else a.output/'chosen-episode.json'
    seed=a.seed if a.seed is not None else (json.loads(prior.read_text())['seed'] if prior.exists() else secrets.randbits(48))
    rng=random.Random(seed);random.seed(seed);np.random.seed(seed%2**32)
    a.output.mkdir(parents=True,exist_ok=True)
    config=get_config(a.config);native=create_torch_dataset(config.data,config.model.action_horizon)
    bases=native._dataset._dataset._datasets
    choices=[];seen=set();offset=0
    for dataset_index,base in enumerate(bases):
        if base.repo_id not in seen:
            seen.add(base.repo_id);ep=np.asarray(base.hf_dataset['episode_index'],dtype=int)
            for episode in np.unique(ep):
                valid=np.flatnonzero((ep==episode)&native.train_mask[offset:offset+len(base)])
                if len(valid)>=100:choices.append((dataset_index,int(episode),offset,valid))
        offset+=len(base)
    dataset_index,episode,offset,eligible=rng.choice(choices);base=bases[dataset_index]
    selected=sorted(rng.sample(eligible.tolist(),100));global_indices=[offset+i for i in selected]
    ep_ids=np.asarray(base.hf_dataset['episode_index'],dtype=int);local_ep=np.flatnonzero(ep_ids==episode)
    timestamps=np.asarray(base.hf_dataset['timestamp'],dtype=float)[local_ep]
    cameras=('cam_high','cam_left_wrist','cam_right_wrist')
    assert not prior.exists() or (json.loads(prior.read_text())['repo_id']==base.repo_id and json.loads(prior.read_text())['frame_indices']==[int(base.hf_dataset[i]['frame_index']) for i in selected]),'The review must use exactly the original selected EP and frames'
    chosen={'seed':seed,'repo_id':base.repo_id,'episode_index':episode,'eligible_frames':len(eligible),
            'local_indices':selected,'frame_indices':[int(base.hf_dataset[i]['frame_index']) for i in selected],'local_episode_start':int(local_ep[0]),'global_indices':global_indices,
            'config':config.name,'code_directory':str(a.source),'dataset_directory':str(a.datasets),
            'selection':'uniform unique training episode with >=100 eligible frames; then100 uniform frames without replacement',
            'video_source':'native LeRobotDataset.__getitem__, past-only timestamps; no standalone video decoder'}
    (a.output/'chosen-episode.json').write_text(json.dumps(chosen,indent=2))
    records=[];transform=compose(config.data.transform_pipeline.training_input_transforms())
    tokenizer=config.data.transform_pipeline._model_input_transforms()[-2].tokenizer
    cached={}
    def images(raw,local_index):
        """Save actual decoded camera arrays once; do not fabricate or resize a mosaic."""
        if local_index in cached:return cached[local_index]
        paths={};folder=a.output/'frames'/str(local_index);folder.mkdir(parents=True,exist_ok=True)
        for camera in cameras:
            image=Image.fromarray(image_uint8(raw['observation.images.'+camera]));image.thumbnail((640,640))
            p=folder/(camera+'.jpg');image.save(p,quality=88);paths[camera]=str(p.relative_to(a.output))
        cached[local_index]=paths;return paths
    class NativeAuditSubset:
        """Observe real raw samples before passing the SAME sample through the native transforms."""
        def __len__(self):return len(global_indices)
        def __getitem__(self,i):
            raw=native[global_indices[i]];local=selected[i]
            assert int(raw['episode_index'])==episode and int(raw['dataset_index'])==dataset_index
            assert int(raw['frame_index'])==int(base.hf_dataset[local]['frame_index'])
            now=float(raw['timestamp']);current=images(raw,local)
            record={'number':i+1,'global_index':global_indices[i],'local_index':local,'frame_index':int(raw['frame_index']),
                    'episode_index':episode,'time_s':now,'task':raw['task'],'current_images':current,
                    'annotation':dict(zip(('bottle_description','memory_annotation','current_task'),parse_annotation(raw['subtask']) or ('','',''))),
                    'teacher_state':json.loads(encode_memory(native.timelines[(dataset_index,episode)].at(now))),
                    'actual_training_memory':json.loads(raw['long_memory']),
                    'before_augmentation':json.loads(raw['long_memory_before_augmentation']),
                    'before_dropout':json.loads(raw['long_memory_before_dropout']),
                    'augmentation':raw['memory_augmentation']}
            before=transform({**raw,'long_memory':raw['long_memory_before_augmentation']})
            output=transform(dict(raw))
            assert record['before_augmentation']['current_task']==record['actual_training_memory']['current_task']==record['annotation']['current_task']
            assert record['before_augmentation']['bottle_description']==record['actual_training_memory']['bottle_description']
            record['model_prompt_before_augmentation']=tokenizer._tokenizer.decode(before['tokenized_prompt'][before['tokenized_prompt_mask']].tolist())
            tokens=output['tokenized_prompt'];mask=output['tokenized_prompt_mask']
            record['model_prompt']=tokenizer._tokenizer.decode(tokens[mask].tolist());record['model_token_count']=int(mask.sum())
            assert all(x not in record['model_prompt'] for x in ('Current time:', 'incomplete or stale', '\"events\"', '\"intent\"', '\"observation\"'))
            assert all(x in record['model_prompt'] for x in ('Bottle description:', 'Memory:', 'Current task:'))
            window=[]
            for j in video_indices(timestamps.tolist(),now,seconds=3,hz=2):
                li=int(local_ep[j]);history=base[li] if li!=local else None
                if history is not None:
                    assert int(history['episode_index'])==episode and float(history['timestamp'])<=now+1e-6
                window.append({'time_s':float(timestamps[j]),'images':images(history,li) if history is not None else current})
            record['video_frames']=window;records.append(record)
            print('CAPTURE',i+1,'frame',record['frame_index'],'tokens',record['model_token_count'],flush=True)
            return {"before":before,"after":output}
    loader=TorchDataLoader(NativeAuditSubset(),local_batch_size=1,shuffle=False,num_workers=0,num_batches=100,seed=seed)
    for batch in loader:
        assert batch['before']['tokenized_prompt'].shape==(1,1024)
        assert batch['after']['tokenized_prompt'].shape==(1,1024)
    assert len(records)==100 and len({r['frame_index'] for r in records})==100
    (a.output/'samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    (a.output/'summary.json').write_text(json.dumps({'chosen':chosen,'samples':len(records),'native_loader':'openpi.data.dataloaders.TorchDataLoader',
      'same_episode_verified':True,'past_only_history_verified':True,'phase_counts':dict(collections.Counter(r['annotation']['current_task'] for r in records)),
      'source_hashes':{str(p.relative_to(a.source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.source/'src/openpi/data/dataloaders.py',a.source/'src/openpi/data/long_memory.py',a.source/'src/openpi/training/config.py']}},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--datasets',type=pathlib.Path,required=True)
    p.add_argument('--chosen',type=pathlib.Path);p.add_argument('--config',required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--seed',type=int);main(p.parse_args())
