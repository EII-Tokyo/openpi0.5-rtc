"""Run on one GPU. Compare compiled memory, synchronized latency and gradients.

No optimizer/donation: isolates gradient accumulation rather than allocator reserve.
The retained variants intentionally expose all micro-gradients as outputs.
"""
import json
import time
import jax
import jax.numpy as jnp
import numpy as np

def loss(p, x):
    h = jnp.tanh(x @ p['w1'])
    return jnp.mean((h @ p['w2']) ** 2)

grad = jax.grad(loss)

def build(mode):
    def f(p, xs):
        n = xs.shape[0]
        zero = jax.tree.map(jnp.zeros_like, p)
        if mode == 'for_accumulate':
            acc = zero
            for i in range(n):
                g = grad(p, xs[i])
                acc = jax.tree.map(lambda a, b: a + b / n, acc, g)
            return acc
        if mode.startswith('scan'):
            def body(acc, x):
                g = grad(p, x)
                acc = jax.tree.map(jnp.add, acc, g)
                return acc, g if mode == 'scan_return_all' else None
            acc, gs = jax.lax.scan(body, zero, xs)
            avg = jax.tree.map(lambda a: a / n, acc)
            return (avg, gs) if mode == 'scan_return_all' else avg
        if mode.startswith('append'):
            gs = [grad(p, xs[i]) for i in range(n)]
            gs = jax.tree.map(lambda *a: jnp.stack(a), *gs)
        else:
            gs = jax.vmap(grad, in_axes=(None, 0))(p, xs)
        avg = jax.tree.map(lambda a: a.mean(0), gs)
        return (avg, gs) if mode.endswith('return_all') else avg
    return f

def main():
    rng = np.random.default_rng(7)
    d = 2048
    p = {k: jnp.asarray(rng.normal(0, .02, (d,d)).astype('float32')) for k in ('w1','w2')}
    print(json.dumps({'jax':jax.__version__, 'device':str(jax.devices()[0]), 'gradient_MiB':32}))
    for n in (1,4,8):
        xs = jnp.asarray(rng.normal(size=(n,64,d)).astype('float32'))
        ref = jax.jit(jax.grad(lambda p,x: loss(p,x.reshape(-1,d))))(p,xs)
        for mode in ('for_accumulate','scan_carry_only','scan_return_all','append_reduce','append_return_all','vmap_reduce','vmap_return_all'):
            start = time.perf_counter()
            compiled = jax.jit(build(mode)).lower(p,xs).compile()
            compile_s = time.perf_counter()-start
            mem = compiled.memory_analysis()
            out = jax.block_until_ready(compiled(p,xs))
            avg = out[0] if mode.endswith('return_all') else out
            errs = [float(jnp.max(jnp.abs(a-b))) for a,b in zip(jax.tree.leaves(avg),jax.tree.leaves(ref))]
            for a,b in zip(jax.tree.leaves(avg),jax.tree.leaves(ref)):
                np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=2e-3,atol=2e-6)
            times=[]
            for _ in range(5):
                start=time.perf_counter()
                out=jax.block_until_ready(compiled(p,xs))
                times.append(time.perf_counter()-start)
            fields={k:getattr(mem,k) for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
            print(json.dumps(dict(n=n,mode=mode,compile_s=compile_s,mean_ms=np.mean(times)*1000,max_abs_error=max(errs),**fields)),flush=True)
            del out,avg,compiled
    print('ALL_CORRECT',flush=True)

if __name__ == '__main__':
    main()
