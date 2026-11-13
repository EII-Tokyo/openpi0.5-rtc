"""Restore the existing 22000-step ViT; no robot or policy server is started."""
import pathlib,types,time,json
import jax,jax.numpy as jnp,numpy as np
from openpi.models import siglip,model
old=types.ModuleType('old_siglip')
exec(compile(pathlib.Path('/out/siglip_before_nn_cond.py').read_text(),'old_siglip.py','exec'),old.__dict__)
root=pathlib.Path('/app/checkpoints/eii_with_rinse_cam4_fullft_from_april_original_38repo/6beba2e5-e00a-431f-b8a8-b95bf48ac168/22000')
params=model.restore_params(root/'params',dtype=jnp.bfloat16)['PaliGemma']['img']
variables={'params':params}
kwargs=dict(num_classes=2048,variant='So400m/14',pool_type='none',scan=True,dtype_mm='bfloat16')
models={'before':old.Module(**kwargs),'after':siglip.Module(**kwargs)}
for frames in [1,4]:
    shape=(1,224,224,3) if frames==1 else (1,frames,224,224,3)
    x=jax.random.normal(jax.random.key(8),shape,dtype=jnp.float32)
    outputs={};latency={}
    for name,vit in models.items():
        f=jax.jit(lambda v,x:vit.apply(v,x,train=False)[0])
        for _ in range(3):jax.block_until_ready(f(variables,x))
        samples=[]
        for _ in range(20):
            t=time.perf_counter();y=f(variables,x);jax.block_until_ready(y);samples.append((time.perf_counter()-t)*1000)
        outputs[name]=np.asarray(y,dtype=np.float32)
        latency[name]=float(np.median(samples))
    assert np.array_equal(outputs['before'],outputs['after']),float(np.max(np.abs(outputs['before']-outputs['after'])))
    print('CHECKPOINT PASS',json.dumps(dict(frames=frames,shape=list(outputs['after'].shape),median_ms=latency)),flush=True)
