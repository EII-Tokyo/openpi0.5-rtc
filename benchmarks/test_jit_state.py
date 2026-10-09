"""Compare discarded and propagated NNX state changes. CPU only; no weights."""
import os

os.environ['JAX_PLATFORMS'] = 'cpu'

import flax
from flax import nnx
import jax
import jax.numpy as jnp
from openpi.shared.nnx_utils import module_jit


class Counter(nnx.Module):
    def __init__(self):
        self.count = nnx.Variable(jnp.array(0, dtype=jnp.int32))

    def tick(self):
        self.count.value += 1
        return self.count.value


def main():
    print(f'JAX={jax.__version__}, Flax={flax.__version__}, devices={jax.devices()}')

    # Captures the initial state. Updated state is not returned or saved.
    frozen_model = Counter()
    frozen_tick = module_jit(frozen_model.tick)

    # Pass the model explicitly so NNX can synchronize its updated state.
    mutable_model = Counter()
    mutable_tick = nnx.jit(Counter.tick)

    print('\nwrapper          call  returned  external_count')
    frozen_results = []
    mutable_results = []
    for i in range(1, 4):
        result = int(frozen_tick())
        count = int(frozen_model.count.value)
        frozen_results.append((result, count))
        print(f'module_jit       {i:4d}  {result:8d}  {count:14d}')
    for i in range(1, 4):
        result = int(mutable_tick(mutable_model))
        count = int(mutable_model.count.value)
        mutable_results.append((result, count))
        print(f'nnx.jit explicit {i:4d}  {result:8d}  {count:14d}')

    assert frozen_results == [(1, 0), (1, 0), (1, 0)], frozen_results
    assert mutable_results == [(1, 1), (2, 2), (3, 3)], mutable_results
    print('\nPASS: module_jit discards state updates; explicit nnx.jit preserves them.')


if __name__ == '__main__':
    main()
