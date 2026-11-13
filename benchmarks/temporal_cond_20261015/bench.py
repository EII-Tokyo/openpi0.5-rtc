import argparse, json, time, types, pathlib, subprocess
import jax
import jax.numpy as jnp
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['gate','cond','setup_gate'], required=True)
p.add_argument('--dtype', choices=['bfloat16','float32'], default='bfloat16')
p.add_argument('--full-grads', action='store_true')
p.add_argument('--resolution', type=int, default=224)
p.add_argument('--frames', type=int, default=4)
p.add_argument('--count', action='store_true')
p.add_argument('--backward', action='store_true')
a = p.parse_args()
source = pathlib.Path('/app/src/openpi/models/siglip.py').read_text()
start = source.index('                temporal_x = jnp.swapaxes(x, 1, 2)')
end = source.index('\n            else:', start)
original = source[start:end]
if a.mode in ('cond','setup_gate'):
    # Move the unchanged, explicitly named children to setup so lifted cond can
    # access them through its cloned module while preserving checkpoint paths.
    setup_start = source.index('        ln0 = nn.LayerNorm')
    setup_end = source.index('\n        def _apply_attention', setup_start)
    setup_body = source[setup_start:setup_end]
    setup_body = setup_body.replace('ln0 =', 'self.ln0 =').replace('attn =', 'self.attn =').replace('ln1 =', 'self.ln1 =').replace('mlp =', 'self.mlp =').replace('deterministic=deterministic,', 'deterministic=True,')
    source = source[:setup_start] + '        ln0, attn, ln1, mlp = self.ln0, self.attn, self.ln1, self.mlp\n' + source[setup_end:]
    class_start = source.index('class Encoder1DBlock')
    insertion = source.index('    @nn.compact', class_start)
    source = source[:insertion] + '    def setup(self):\n' + setup_body + '\n\n' + source[insertion:]
    start = source.index('                temporal_x = jnp.swapaxes(x, 1, 2)')
    end = source.index('\n            else:', start)
    replacement = '''                def yes(modules, value):
                    seq = jnp.swapaxes(value, 1, 2).reshape(batch_size * seq_len, time_size, width)
                    seq = sharding.activation_sharding_constraint(seq)
                    y = modules.ln0(seq)
                    y = modules.attn(y, y, mask=temporal_attn_mask)
                    y = sharding.activation_sharding_constraint(y)
                    y = y.reshape(batch_size, seq_len, time_size, width).swapaxes(1, 2)
                    # Preserve the baseline's bf16 residual subtract/add order.
                    temporal_value = value + y
                    return value + (temporal_value - value), y
                def no(modules, value):
                    return value, jnp.zeros_like(value)
                x, out["temporal_sa"] = nn.cond(temporal_first, yes, no, self, x)
'''.rstrip()
    if a.mode == 'cond':
        source = source[:start] + replacement + source[end:]
if a.count:
    # A device-side callback in the actual attention path; counting is excluded from timings.
    needle = '                temporal_x, temporal_y = _apply_attention(temporal_x, temporal_attn_mask)'
    if a.mode == 'gate':
        source = source.replace(needle, '                jax.debug.callback(lambda: calls.append(1))\n' + needle)
    else:
        needle = '                    y = modules.attn(y, y, mask=temporal_attn_mask)'
        source = source.replace(needle, '                    jax.debug.callback(lambda: calls.append(1))\n' + needle)
if not a.count:
    pathlib.Path('/out/siglip_'+a.mode+'.py').write_text(source)
mod = types.ModuleType('bench_siglip')
calls = []
mod.__dict__['calls'] = calls
exec(compile(source, 'bench_siglip.py', 'exec'), mod.__dict__)
jax.config.update('jax_default_matmul_precision', 'highest' if a.dtype == 'float32' else 'default')
model = mod.Encoder(depth=27, mlp_dim=4304, num_heads=16, scan=True, dtype_mm=a.dtype)
# Generate identical numeric input for both dtypes, then cast.
x = jax.random.normal(jax.random.key(10), (1,a.frames,(a.resolution//14)**2,1152), dtype=jnp.bfloat16).astype(a.dtype)
# Initialize with one frame to share existing spatial parameters across both cond branches.
params = jax.jit(model.init)(jax.random.key(42), x[:,:1], deterministic=True)
forward = lambda v,z:model.apply(v,z,deterministic=True)[0]
if a.backward:
    f = jax.jit(jax.value_and_grad(lambda v,z:jnp.mean(jnp.sin(forward(v,z).astype(jnp.float32)))))
else:
    f = jax.jit(forward)
t=time.perf_counter()
compiled=f.lower(params,x).compile()
compile_s=time.perf_counter()-t
for _ in range(3): jax.block_until_ready(compiled(params,x))
jax.effects_barrier()
calls.clear()
if a.count:
    jax.block_until_ready(compiled(params,x));jax.effects_barrier()
    print('COUNT',a.mode,len(calls),flush=True)
else:
    pathlib.Path('/out/'+a.mode+'_'+str(a.resolution)+('_backward' if a.backward else '')+'.hlo').write_text(compiled.as_text())
    samples=[]
    for _ in range(30):
        t=time.perf_counter();y=compiled(params,x);jax.block_until_ready(y);samples.append((time.perf_counter()-t)*1000)
        del y
    y=compiled(params,x);jax.block_until_ready(y)
    if not a.backward:
        np.save('/out/'+a.mode+'_'+str(a.resolution)+'_'+str(a.frames)+'.npy',np.asarray(y))
    else:
        probes = jax.jit(lambda g:jnp.concatenate([z.astype(jnp.float32).ravel()[::max(1,z.size//100)] for z in jax.tree.leaves(g)]))(y[1])
        np.save('/out/'+a.mode+'_'+str(a.resolution)+'_grad.npy',np.asarray(probes))
        print('LOSS',float(y[0]),flush=True)
        if a.full_grads:
            folder=pathlib.Path('/out')/(a.mode+'_full_grads')
            folder.mkdir(exist_ok=True)
            manifest=[]
            for i,(path,leaf) in enumerate(jax.tree_util.tree_flatten_with_path(y[1])[0]):
                filename=str(i)+'.npy'
                np.save(folder/filename,np.asarray(leaf,dtype=np.float32))
                manifest.append(dict(path=str(path),shape=list(leaf.shape),file=filename))
            (folder/'manifest.json').write_text(json.dumps(manifest))
            np.save('/out/'+a.mode+'_forward.npy',np.asarray(jax.block_until_ready(jax.jit(forward)(params,x)),dtype=np.float32))
    mem=compiled.memory_analysis()
    print('RESULT',json.dumps(dict(mode=a.mode,backward=a.backward,resolution=a.resolution,frames=a.frames,compile_s=compile_s,median_ms=float(np.median(samples)),mean_ms=float(np.mean(samples)),p95_ms=float(np.percentile(samples,95)),compiled_memory=str(mem),device_memory=jax.devices()[0].memory_stats(),gpu_mib=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())),flush=True)
