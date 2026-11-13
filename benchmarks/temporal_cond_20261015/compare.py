import json, pathlib, re
import numpy as np
p=pathlib.Path('/out')
def read_array(name):
    z=np.load(p/name)
    if z.dtype.kind=='V':
        z=(z.view(np.uint16).astype(np.uint32)<<16).view(np.float32)
    return z
for r in [224,448]:
    x,y=[read_array(f'{m}_{r}_4.npy') for m in ['gate','cond']]
    print('OUTPUT',r,dict(max_abs=float(np.max(np.abs(x-y))),rmse=float(np.sqrt(np.mean((x-y)**2))),relative_l2=float(np.linalg.norm(x-y)/np.linalg.norm(x)),equal_fraction=float(np.mean(x==y))))
if (p/'cond_224_grad.npy').exists():
    x,y=[read_array(f'{m}_224_grad.npy') for m in ['gate','cond']]
    print('GRAD_SAMPLES',len(x),dict(max_abs=float(np.max(np.abs(x-y))),relative_l2=float(np.linalg.norm(x-y)/np.linalg.norm(x)),equal_fraction=float(np.mean(x==y))))
for name in ['gate_224','cond_224','gate_448','cond_448','gate_backward','cond_backward']:
    lines=(p/(name+'.log')).read_text().splitlines()
    results=[json.loads(z[7:]) for z in lines if z.startswith('RESULT ')]
    if not results:print('MISSING_RESULT',name);continue
    r=results[-1]
    mem=r['compiled_memory']
    tmp=int(re.search(r'(?<!host_)temp_size_in_bytes=(\d+)',mem).group(1))
    print('SUMMARY',name,'median_ms',round(r['median_ms'],3),'temp_MiB',round(tmp/2**20,2),'peak_live_MiB',round(r['device_memory']['peak_bytes_in_use']/2**20,2),'gpu_mib',r['gpu_mib'])
for name in ['gate_224','cond_224']:
    hlo=(p/(name+'.hlo')).read_text()
    print('HLO',name,'conditional_count',len(re.findall(r'\) conditional\(',hlo)),'scan_27', '"n":"27"' in hlo)
