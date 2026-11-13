"""Regression checks against the exact pre-change production implementation."""
import pathlib,types,json,sys
import jax,jax.numpy as jnp,numpy as np
from openpi.models import siglip as new
old=types.ModuleType('old_siglip')
exec(compile(pathlib.Path('/app/benchmarks/temporal_cond_20261015/siglip_before_nn_cond.py').read_text(),'old_siglip.py','exec'),old.__dict__)
def equal_tree(a,b):
    assert jax.tree.structure(a)==jax.tree.structure(b)
    assert all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)))
def check(scan,dtype,frames):
    args=dict(depth=8,mlp_dim=128,num_heads=4,scan=scan,dtype_mm=dtype)
    before=old.Encoder(**args);after=new.Encoder(**args)
    x=jax.random.normal(jax.random.key(1),(1,frames,8,32),dtype=jnp.bfloat16).astype(dtype)
    key=jax.random.key(2)
    v=jax.jit(before.init)(key,x,deterministic=True)
    fresh=jax.jit(after.init)(key,x,deterministic=True)
    equal_tree(v,fresh)
    f=lambda module:jax.jit(lambda p,x:module.apply(p,x,deterministic=True)[0])
    a=f(before)(v,x);b=f(after)(v,x)
    assert np.array_equal(np.asarray(a),np.asarray(b)),(scan,dtype,frames)
    grad=jax.jit(jax.grad(lambda p:jnp.mean(jnp.sin(after.apply(p,x,deterministic=True)[0].astype(jnp.float32)))))
    g=grad(v);jax.block_until_ready(g)
    assert all(np.all(np.isfinite(np.asarray(z))) for z in jax.tree.leaves(g))
    print('PASS encoder',scan,dtype,frames,flush=True)
if '--extras-only' not in sys.argv:
    for scan in [False,True]:
        for dtype in ['bfloat16','float32']:
            for frames in [1,4]:check(scan,dtype,frames)
# Nonzero dropout must work under lifted cond for both train and eval.
model=new.Encoder(depth=8,mlp_dim=128,num_heads=4,scan=True,dtype_mm='bfloat16',dropout=.1)
x=jax.random.normal(jax.random.key(9),(1,4,8,32),dtype=jnp.bfloat16)
v=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4)},x,deterministic=False)
run=jax.jit(lambda key:model.apply(v,x,deterministic=False,rngs={'dropout':key})[0])
a=run(jax.random.key(5));b=run(jax.random.key(6))
assert not np.array_equal(np.asarray(a),np.asarray(b))
jax.block_until_ready(jax.jit(lambda:model.apply(v,x,deterministic=True)[0])())
print('PASS dropout train/eval',flush=True)
# Exercise complete ViT initialization with image and video inputs.
vit=new._Module(width=32,depth=4,mlp_dim=128,num_heads=4,patch_size=(14,14),pool_type='none',scan=True,dtype_mm='bfloat16')
for shape in [(1,28,28,3),(1,4,28,28,3)]:
    image=jnp.ones(shape)
    v=jax.jit(vit.init)(jax.random.key(7),image)
    y=jax.jit(vit.apply)(v,image)[0];jax.block_until_ready(y)
    assert y.shape==(1,4,32)
    print('PASS ViT',shape,flush=True)
print('ALL PASSED',flush=True)
