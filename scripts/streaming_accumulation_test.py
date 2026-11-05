"""CPU-only numerical checks. No dataset, model checkpoint or W&B access.

Run with JAX_PLATFORMS=cpu and a four-device CPU backend for sharding checks.
The scan reference is extracted unchanged from the pre-microbatch trainer.
See benchmarks/microbatch_jit/README.md for CPU and full-model GPU commands.
"""
import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from flax import nnx, struct
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.data import transforms
from openpi.shared import nnx_utils
import streaming_accumulation as streaming


@struct.dataclass
class State:
    step: object
    params: object
    model_def: object
    opt_state: object
    tx: object = struct.field(pytree_node=False)
    ema_decay: object = struct.field(pytree_node=False)
    ema_params: object = None


class TinyModel(nnx.Module):
    def __init__(self):
        self.kernel = nnx.Param(jnp.arange(16, dtype=jnp.float32).reshape(4, 4) / 20)
        self.bias = nnx.Param(jnp.zeros(4))
        self.frozen = nnx.Param(jnp.eye(4) / 10)

    def compute_loss(self, rng, observation, actions, *, train):
        assert train
        prediction = observation @ (self.kernel.value + self.frozen.value) + self.bias.value
        prediction += jax.random.normal(rng, prediction.shape) * 0.01
        return (prediction - actions) ** 2


def load_reference():
    path = Path(__file__).parent / "testdata" / "scan_train_step.py.txt"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "train_step")
    for child in ast.walk(node):
        if isinstance(child, ast.FunctionDef):
            child.decorator_list = []
            child.returns = None
        if isinstance(child, ast.arg):
            child.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    scope = dict(nnx=nnx, jax=jax, jnp=jnp, optax=optax, dataclasses=dataclasses,
                 nnx_utils=nnx_utils, _transforms=transforms)
    scope["at"] = SimpleNamespace(typecheck=lambda f: f)
    exec(compile(module, str(path), "exec"), scope)
    return scope["train_step"]


def fixture(count, ema):
    model = TinyModel()
    graph, params = nnx.split(model)
    train_filter = nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex("frozen")))
    tx = optax.adamw(0.001, weight_decay=0.01)
    state = State(jnp.array(7), params, graph, tx.init(params.filter(train_filter)),
                  tx, ema, None if ema is None else params)
    config = SimpleNamespace(
        gradient_accumulation_steps=count, batch_size=8, trainable_filter=train_filter,
        model=SimpleNamespace(image_resolution=(448, 448)),
    )
    batches = [(jax.random.normal(jax.random.key(100 + i), (8, 4)),
                jax.random.normal(jax.random.key(200 + i), (8, 4))) for i in range(count)]
    return config, state, batches


class StreamingTests(unittest.TestCase):
    def assert_tree_close(self, left, right):
        self.assertEqual(jax.tree.structure(left), jax.tree.structure(right))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=3e-5, atol=3e-6)

    def run_case(self, count, ema):
        cfg, initial, batches = fixture(count, ema)
        devices = np.array(jax.devices()).reshape(-1)
        mesh = jax.sharding.Mesh(devices, ("data",))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
        stacked_data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, "data"))
        def layout(x):
            return (data if getattr(x, "ndim", 0) >= 2 and x.shape[0] % len(devices) == 0
                    else replicated)
        state_layout = jax.tree.map(layout, initial)
        ref_state = jax.device_put(initial, state_layout)
        new_state = jax.tree.map(lambda x: jax.device_put(np.array(x, copy=True)), initial)
        new_state = jax.device_put(new_state, state_layout)
        micros = [jax.device_put(b, data) for b in batches]
        reference_batch = (jax.tree.map(lambda *x: jnp.stack(x), *micros)
                           if count > 1 else micros[0])
        reference_fn = load_reference()
        reference = jax.jit(
            lambda rng, state, batch: reference_fn(cfg, rng, state, batch),
            in_shardings=(replicated, state_layout, stacked_data if count > 1 else data),
            out_shardings=(state_layout, replicated),
        )
        engine = streaming.StreamingAccumulator(cfg, state_layout, data, replicated)
        def preprocess(rng, obs, **kwargs):
            return obs + jax.random.normal(rng, obs.shape) * 0.005
        # Trace the real production/reference and experimental kernels with the
        # same deterministic lightweight preprocessing; no full vision model.
        with mock.patch.object(transforms.AlohaTransformPipeline,
                               "preprocess_observation", side_effect=preprocess):
            for update in range(2):
                rng = jax.random.key(42)
                expected, expected_info = reference(rng, ref_state, reference_batch)
                events = []
                original_accumulate = engine.accumulate
                def record_accumulate(*args):
                    events.append("compute")
                    return original_accumulate(*args)
                engine.accumulate = record_accumulate
                remaining = iter(micros[1:])
                def get_next():
                    events.append("read")
                    return next(remaining), 0.25
                actual, info, timing = engine.run(rng, new_state, micros[0], get_next, 0.1)
                engine.accumulate = original_accumulate
                self.assert_tree_close(expected, actual)
                self.assert_tree_close(expected_info, info)
                self.assertEqual(int(actual.step), 8 + update)
                np.testing.assert_array_equal(actual.params["frozen"].value,
                                              initial.params["frozen"].value)
                self.assertEqual(events, [x for i in range(count)
                                          for x in ((["read"] if i else []) + ["compute"])])
                self.assertAlmostEqual(timing["data_wait_time"], 0.1 + (count - 1) * 0.25)
                self.assertGreater(timing["streaming_update_wall_time"], 0)
                ref_state, new_state = expected, actual

    def test_ga4_ema_and_sharding(self):
        self.run_case(4, 0.99)

    def test_ga2_without_ema(self):
        self.run_case(2, None)

    def test_ga1_rng_compatibility(self):
        self.run_case(1, 0.99)

    def test_reject_invalid_ga(self):
        cfg, _, _ = fixture(0, None)
        with self.assertRaises(ValueError):
            streaming.StreamingAccumulator(cfg, None, None, None)



    def test_no_update_when_iterator_exhausts_mid_group(self):
        # Test the orchestration contract without compiling a model: a partial
        # group must never call update (no partial optimizer/EMA/step commit).
        engine = object.__new__(streaming.StreamingAccumulator)
        engine.count = 4
        engine.initialize = lambda state: (0, 0)
        engine.accumulate = lambda rng, state, batch, i, loss, grads: (loss + 1, grads + 1)
        engine.update = mock.Mock(side_effect=AssertionError("must not update"))
        def exhausted():
            raise StopIteration
        with self.assertRaises(StopIteration):
            engine.run(None, object(), object(), exhausted)
        engine.update.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
