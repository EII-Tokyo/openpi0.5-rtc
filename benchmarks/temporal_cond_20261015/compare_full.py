import pathlib,json,numpy as np
p=pathlib.Path('/out')
def compare(mode):
    a=p/'gate_full_grads';b=p/(mode+'_full_grads')
    ma=json.loads((a/'manifest.json').read_text());mb=json.loads((b/'manifest.json').read_text())
    assert ma==mb, 'Parameter paths or shapes differ'
    total=equal=0;error=reference=0.;mx=0.;leaves=[]
    for info in ma:
        x=np.load(a/info['file'],mmap_mode='r').ravel();y=np.load(b/info['file'],mmap_mode='r').ravel()
        e=r=0.;m=0.;eq=0
        for i in range(0,x.size,1048576):
            u=np.asarray(x[i:i+1048576],dtype=np.float64);v=np.asarray(y[i:i+1048576],dtype=np.float64)
            d=u-v;e+=float(d@d);r+=float(u@u);m=max(m,float(np.max(np.abs(d))));eq+=int(np.count_nonzero(u==v))
        total+=x.size;equal+=eq;error+=e;reference+=r;mx=max(mx,m)
        leaves.append(dict(path=info['path'],max_abs=m,relative_l2=(e/r)**.5 if r else 0,equal=eq,size=x.size))
    f=np.load(p/'gate_forward.npy');g=np.load(p/(mode+'_forward.npy'))
    result=dict(mode=mode,parameters=total,max_abs=mx,relative_l2=(error/reference)**.5,equal_fraction=equal/total,forward_max_abs=float(np.max(np.abs(f-g))),forward_relative_l2=float(np.linalg.norm(f-g)/np.linalg.norm(f)),worst_leaves=sorted(leaves,key=lambda z:z['relative_l2'],reverse=True)[:8])
    print(json.dumps(result),flush=True)
    (p/(mode+'_comparison.json')).write_text(json.dumps(result,indent=2))
for mode in ['cond','setup_gate']:
    if (p/(mode+'_full_grads/manifest.json')).exists():compare(mode)
