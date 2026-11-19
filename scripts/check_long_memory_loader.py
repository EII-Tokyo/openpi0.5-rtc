"""Exercise real LeRobot video/actions, memory joins and the native tokenizer on CPU."""
import json,pathlib,random,time
import numpy as np
import torch
from openpi.training.config import get_config
from openpi.data.datasets import create_torch_dataset,transform_dataset
from openpi.data.transforms import compose

NAME='eii_twist_cam3_fullft_25repo_h100x2_224_longmem3s_drop_bs16ga16_5k_20261119'

def main():
    random.seed(42);np.random.seed(42)
    c=get_config(NAME);d=create_torch_dataset(c.data,c.model.action_horizon)
    transform=compose(c.data.transform_pipeline.training_input_transforms())
    tokenizer=c.data.transform_pipeline._model_input_transforms()[-2].tokenizer
    report={'config':NAME,'weighted_entries':len(c.data.repo_ids),'dataset_frames':len(d),'eligible_training_frames':len(d.train_indices),'samples':[]}
    print('DATASET',len(d),len(d.train_indices),flush=True)
    seen=set();offset=0
    for base in d._dataset._dataset._datasets:
        if base.repo_id not in seen:
            seen.add(base.repo_id)
            valid=d.train_indices[(d.train_indices>=offset)&(d.train_indices<offset+len(base))]
            if not len(valid):raise ValueError(f'No train samples for weighted source {base.repo_id}')
            for idx in [valid[0],valid[len(valid)//2],valid[-1]]:
                raw=d[int(idx)];memory=json.loads(raw['long_memory']);now=float(raw['timestamp'])
                assert all(e['time_s']<=now for e in memory['events'])
                transformed=transform(raw)
                tokens=transformed['tokenized_prompt'];mask=transformed['tokenized_prompt_mask']
                decoded=tokenizer._tokenizer.decode(tokens[mask].tolist())
                assert 'Long memory' in decoded and len(tokens)==1024
                assert 'subtask' not in transformed and 'long_memory' not in transformed
                report['samples'].append({'repo_id':base.repo_id,'global_index':int(idx),'time_s':now,'memory':memory,'prompt':decoded,'tokens':int(mask.sum()),'action_shape':list(transformed['actions'].shape)})
            print('PASS',base.repo_id,flush=True)
        offset+=len(base)
    loader=torch.utils.data.DataLoader(transform_dataset(d,c.data),batch_size=2,num_workers=2)
    batch=next(iter(loader));assert batch['tokenized_prompt'].shape==(2,1024)
    report['worker_batch_shape']=list(batch['tokenized_prompt'].shape)
    report['max_observed_prompt_tokens']=max(r['tokens'] for r in report['samples'])
    p=pathlib.Path('../artifacts/loader-check.json');p.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print('DONE',len(report['samples']),report['max_observed_prompt_tokens'],flush=True)

if __name__=='__main__':main()
