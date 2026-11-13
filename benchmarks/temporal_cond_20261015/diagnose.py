import pathlib,types,json
import jax,jax.numpy as jnp,numpy as np
modules={}
for name in ['gate','cond']:
    mod=types.ModuleType(name)
    source_path='/app/src/openpi/models/siglip.py' if name=='gate' else '/app/benchmarks/temporal_cond_20261015/siglip_cond.py'
    exec(compile(pathlib.Path(source_path).read_text(),name+'.py','exec'),mod.__dict__)
    modules[name]=mod
def metrics(x,y):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    return dict(max_abs=float(np.max(np.abs(x-y))),relative_l2=float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-30)),equal=bool(np.array_equal(x,y)))
for dtype in ['bfloat16','float32']:
    jax.config.update('jax_default_matmul_precision','highest' if dtype=='float32' else 'default')
    x=jax.random.normal(jax.random.key(1),(1,4,8,32),dtype=jnp.bfloat16).astype(dtype)
    cot=jax.random.normal(jax.random.key(2),x.shape,dtype=jnp.float32)/x.size
    models={name:mod.Encoder1DBlock(mlp_dim=128,num_heads=4,dtype_mm=dtype) for name,mod in modules.items()}
    variables=models['gate'].init(jax.random.key(3),x[:,:1],deterministic=True)
    mask=jnp.tril(jnp.ones((4,4),dtype=bool))[None,None]
    for flag in [False,True]:
        outputs={};gradients={}
        for name,model in models.items():
            def f(params,value,pred):
                y=model.apply(params,value,deterministic=True,temporal_first=pred,temporal_attn_mask=mask)[0]
                return jnp.sum(y.astype(jnp.float32)*cot),y
            compiled=jax.jit(jax.value_and_grad(f,argnums=(0,1),has_aux=True)).lower(variables,x,jnp.asarray(flag)).compile()
            pathlib.Path('/out/'+name+'_'+dtype+'_'+str(flag)+'.hlo').write_text(compiled.as_text())
            (loss,y),g=compiled(variables,x,jnp.asarray(flag))
            outputs[name]=y;gradients[name]=g
        pg=[np.concatenate([np.asarray(z,dtype=np.float64).ravel() for z in jax.tree.leaves(gradients[n][0])]) for n in ['gate','cond']]
        print('BLOCK',json.dumps(dict(dtype=dtype,flag=flag,output=metrics(outputs['gate'],outputs['cond']),input_grad=metrics(gradients['gate'][1],gradients['cond'][1]),params_grad=metrics(*pg))),flush=True)
    # Minimal residual-only reproduction, with no Flax, attention, scan or remat.
    w=jax.random.normal(jax.random.key(4),(32,32),dtype=jnp.float32)
    def residual(mode,params,value,flag,downstream=False):
        def yes(v):
            t=v+jnp.tanh(v@params.astype(dtype))
            return v+(t-v)
        if mode=='gate':
            t=value+jnp.tanh(value@params.astype(dtype))
            y=value+flag.astype(dtype)*(t-value)
        else:y=jax.lax.cond(flag,yes,lambda v:v,value)
        if downstream:
            y=y+jnp.tanh(y@params.astype(dtype))
        return jnp.sum(y.astype(jnp.float32)*cot),y
    for flag in [False,True]:
        vals={}
        for mode in ['gate','cond']:
            vals[mode]=jax.jit(jax.value_and_grad(lambda p,v,pred:residual(mode,p,v,pred),argnums=(0,1),has_aux=True))(w,x,jnp.asarray(flag))
        print('RESIDUAL',json.dumps(dict(dtype=dtype,flag=flag,output=metrics(vals['gate'][0][1],vals['cond'][0][1]),input_grad=metrics(vals['gate'][1][1],vals['cond'][1][1]),params_grad=metrics(vals['gate'][1][0],vals['cond'][1][0]))),flush=True)
    for flag in [False,True]:
        vals={}
        for mode in ['gate','cond']:
            vals[mode]=jax.jit(jax.value_and_grad(lambda p,v,pred:residual(mode,p,v,pred,True),argnums=(0,1),has_aux=True))(w,x,jnp.asarray(flag))
        print('TWO_RESIDUALS',json.dumps(dict(dtype=dtype,flag=flag,output=metrics(vals['gate'][0][1],vals['cond'][0][1]),input_grad=metrics(vals['gate'][1][1],vals['cond'][1][1]),params_grad=metrics(vals['gate'][1][0],vals['cond'][1][0]))),flush=True)
