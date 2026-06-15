# Lightweight RLT Token Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a clearer Pi0RL boundary and a smaller low-dimensional RL Token path while preserving compatibility with existing 2048-dimensional RL Token checkpoints.

**Architecture:** Keep the existing `RLTokenAutoencoder` API but add optional projection bottlenecks: `h_vla[2048] -> token_hidden_dim -> z_dim -> token_hidden_dim -> h_hat[2048]`. Add an explicit `Pi0RLConfig`/`Pi0RL` model boundary for future RL-token checkpoints, and make policy inference prefer a single model call that returns both reference actions and `z_rl`.

**Tech Stack:** JAX, Flax NNX, OpenPI `Pi0`, pytest.

---

## File Structure

- Modify `src/openpi/models/rl_token.py`: add low-dimensional bottleneck config and implementation.
- Modify `src/openpi/models/rl_token_test.py`: test old compatible shape and new low-dimensional `z_rl`.
- Create `src/openpi/models/pi0_rl.py`: explicit `Pi0RL` subclass boundary.
- Create `src/openpi/models/pi0_rl_config.py`: explicit `Pi0RLConfig` that constructs `Pi0RL`.
- Modify `src/openpi/models/pi0.py`: add `sample_actions_with_rl_token()` and `guided_inference_with_rl_token()` convenience methods.
- Modify `src/openpi/policies/policy.py`: prefer `*_with_rl_token()` methods; keep existing prefix-hidden fallback.
- Modify `src/openpi/policies/policy_test.py`: cover direct `sample_actions_with_rl_token()` path.
- Modify `src/openpi/training/config.py`: add a new lightweight RL-token config for the current rinse checkpoint/data source.

---

### Task 1: Low-Dimensional RL Token Bottleneck

**Files:**
- Modify: `src/openpi/models/rl_token.py`
- Modify: `src/openpi/models/rl_token_test.py`

- [ ] **Step 1: Add failing test for low-dimensional token output**

Add this test to `src/openpi/models/rl_token_test.py`:

```python
def test_rl_token_low_dimensional_bottleneck_shapes():
    config = rl_token.RLTokenConfig(
        hidden_dim=8,
        token_hidden_dim=4,
        z_dim=3,
        encoder_layers=1,
        decoder_layers=1,
        num_heads=2,
        mlp_dim=8,
        max_prefix_len=5,
    )
    model = rl_token.RLTokenAutoencoder(config, rngs=nnx.Rngs(0))
    h_vla = jnp.ones((2, 4, 8), dtype=jnp.bfloat16)
    prefix_mask = jnp.array([[True, True, True, False], [True, True, False, False]])

    z_rl, h_hat = model(h_vla, prefix_mask)
    loss = model.compute_loss(h_vla, prefix_mask)

    assert z_rl.shape == (2, 3)
    assert h_hat.shape == h_vla.shape
    assert loss.shape == (2,)
    assert jnp.all(jnp.isfinite(loss))
```

- [ ] **Step 2: Verify the test fails**

Run:

```bash
uv run pytest src/openpi/models/rl_token_test.py::test_rl_token_low_dimensional_bottleneck_shapes -q
```

Expected: fails because `RLTokenConfig` does not accept `token_hidden_dim` or `z_dim`.

- [ ] **Step 3: Implement bottleneck fields and projections**

In `RLTokenConfig`, add:

```python
token_hidden_dim: int | None = None
z_dim: int | None = None
```

In `__post_init__`, default them to `hidden_dim` with `object.__setattr__`, validate divisibility by `num_heads`, and keep old defaults exactly equivalent.

Change `RLTokenBlock` to operate on `config.token_hidden_dim`, not `config.hidden_dim`.

In `RLTokenAutoencoder.__init__`, create:

```python
self.input_proj = nnx.Linear(config.hidden_dim, config.token_hidden_dim, rngs=rngs)
self.z_proj = nnx.Linear(config.token_hidden_dim, config.z_dim, rngs=rngs)
self.z_to_decoder = nnx.Linear(config.z_dim, config.token_hidden_dim, rngs=rngs)
self.output_proj = nnx.Linear(config.token_hidden_dim, config.hidden_dim, rngs=rngs)
```

Make position embeddings/query embeddings use `token_hidden_dim`.

In `encode()`, project `h_vla` to token hidden before transformer and return `z_proj(x[:, -1, :])`.

In `decode()`, condition with `z_to_decoder(z_rl)`, project teacher-forced history with `input_proj`, and return output projected to `hidden_dim`.

- [ ] **Step 4: Verify RL token tests pass**

Run:

```bash
uv run pytest src/openpi/models/rl_token_test.py -q
```

Expected: all RL token tests pass.

---

### Task 2: Explicit Pi0RL Boundary and Single-Call Token Sampling

**Files:**
- Create: `src/openpi/models/pi0_rl.py`
- Create: `src/openpi/models/pi0_rl_config.py`
- Modify: `src/openpi/models/pi0.py`
- Modify: `src/openpi/policies/policy.py`
- Modify: `src/openpi/policies/policy_test.py`

- [ ] **Step 1: Add failing policy test for direct `sample_actions_with_rl_token()`**

In `src/openpi/policies/policy_test.py`, extend `_FakeRlTokenModel` with:

```python
self.sample_actions_with_rl_token_calls = 0

def sample_actions_with_rl_token(self, rng, observation, **kwargs):
    self.sample_actions_with_rl_token_calls += 1
    actions = jnp.ones((1, 2, 3), dtype=jnp.float32)
    z_rl = jnp.full((1, 4), 2.0, dtype=jnp.float32)
    return actions, z_rl
```

Then add:

```python
def test_infer_prefers_direct_sample_actions_with_rl_token():
    model = _FakeRlTokenModel()
    policy = _policy.Policy.__new__(_policy.Policy)
    policy._model = model
    policy._input_transform = lambda x: x
    policy._output_transform = lambda x: x
    policy._sample_kwargs = {}
    policy._metadata = {}
    policy._is_pytorch_model = False
    policy._sample_actions = model.sample_actions
    policy._guided_inference = model.guided_inference
    policy._sample_actions_with_prefix_hidden = model.sample_actions_with_prefix_hidden
    policy._guided_inference_with_prefix_hidden = model.guided_inference_with_prefix_hidden
    policy._sample_actions_with_rl_token = model.sample_actions_with_rl_token
    policy._guided_inference_with_rl_token = None
    policy._rng = jax.random.key(0)

    result = policy.infer({
        "state": np.zeros((3,), dtype=np.float32),
        "image": {"cam": np.zeros((2, 2, 3), dtype=np.float32)},
        "image_mask": {"cam": np.array(True)},
    })

    assert model.sample_actions_with_rl_token_calls == 1
    assert model.sample_actions_with_prefix_hidden_calls == 0
    assert model.embed_prefix_hidden_calls == 0
    assert result["actions"].shape == (2, 3)
    assert np.all(result["z_rl"] == 2.0)
```

- [ ] **Step 2: Verify the test fails**

Run:

```bash
uv run pytest src/openpi/policies/policy_test.py::test_infer_prefers_direct_sample_actions_with_rl_token -q
```

Expected: fails because `Policy.infer()` does not look for direct `*_with_rl_token()` methods.

- [ ] **Step 3: Add direct token sampling methods to `Pi0`**

In `src/openpi/models/pi0.py`, add:

```python
def sample_actions_with_rl_token(...):
    actions, prefix_hidden = self.sample_actions(..., return_prefix_hidden=True)
    if self.rl_token_autoencoder is None:
        raise ValueError("sample_actions_with_rl_token requires rl_token_autoencoder")
    prefix_out, prefix_mask = prefix_hidden
    if observation.tokenized_prompt is not None:
        image_token_count = prefix_out.shape[1] - observation.tokenized_prompt.shape[1]
        prefix_out = prefix_out[:, :image_token_count]
        prefix_mask = prefix_mask[:, :image_token_count]
    z_rl = self.rl_token_autoencoder.encode(jax.lax.stop_gradient(prefix_out), prefix_mask)
    return actions, z_rl
```

Add the analogous `guided_inference_with_rl_token()` using `guided_inference(..., return_prefix_hidden=True)`.

- [ ] **Step 4: Add explicit Pi0RL class and config**

Create `src/openpi/models/pi0_rl.py`:

```python
from openpi.models.pi0 import Pi0


class Pi0RL(Pi0):
    """Pi0 variant that explicitly exposes RL Token sampling APIs."""
```

Create `src/openpi/models/pi0_rl_config.py`:

```python
import dataclasses
from typing import TYPE_CHECKING

from flax import nnx
from typing_extensions import override

from openpi.models import pi0_config
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.pi0_rl import Pi0RL


@dataclasses.dataclass(frozen=True)
class Pi0RLConfig(pi0_config.Pi0Config):
    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0RL":
        from openpi.models.pi0_rl import Pi0RL

        if self.rl_token is None:
            raise ValueError("Pi0RLConfig requires rl_token to be configured.")
        return Pi0RL(self, rngs=nnx.Rngs(rng))
```

- [ ] **Step 5: Make policy prefer direct token methods**

In `Policy.__init__`, add jitted attributes when methods exist:

```python
self._sample_actions_with_rl_token = (
    nnx_utils.module_jit(model.sample_actions_with_rl_token)
    if hasattr(model, "sample_actions_with_rl_token")
    else None
)
self._guided_inference_with_rl_token = (
    nnx_utils.module_jit(model.guided_inference_with_rl_token)
    if hasattr(model, "guided_inference_with_rl_token")
    else None
)
```

In `infer()`, if direct token methods exist, call them and use returned `(actions, z_rl)` without re-encoding prefix hidden. Keep the old prefix-hidden fallback for old models.

- [ ] **Step 6: Verify policy tests pass**

Run:

```bash
uv run pytest src/openpi/policies/policy_test.py -q
```

Expected: all policy tests pass.

---

### Task 3: Lightweight Rinse RL Token Training Config

**Files:**
- Modify: `src/openpi/training/config.py`

- [ ] **Step 1: Add a new config using `Pi0RLConfig`**

Add import:

```python
import openpi.models.pi0_rl_config as pi0_rl_config
```

Add a helper or config for:

```python
name="eii_rinse_11repo_cam4_fullft_rl_token_small"
model=pi0_rl_config.Pi0RLConfig(
    pi05=True,
    rl_token=_rl_token.RLTokenConfig(
        hidden_dim=2048,
        token_hidden_dim=768,
        z_dim=512,
        encoder_layers=2,
        decoder_layers=2,
        num_heads=8,
        mlp_dim=3072,
        max_prefix_len=1224,
        decoder_mode="teacher_forced",
    ),
    rl_token_only=True,
)
```

Use repo ids from `_EII_RINSE_11REPO_INSERT_X5_REPO_IDS`, assets `trossen`, and init checkpoint:

```python
"/workspace/openpi0.5-rtc/checkpoints/eii_rinse_11repo_cam4_fullft/rinse_11repo_insertx5_fullft_bs256_nw64_fsdp8_20260513/9000/params"
```

Use weight loader missing regex for `rl_token_autoencoder`.

- [ ] **Step 2: Verify config can be loaded**

Run:

```bash
uv run python - <<'PY'
from openpi.training import config
cfg = config.get_config("eii_rinse_11repo_cam4_fullft_rl_token_small")
print(type(cfg.model).__name__)
print(cfg.model.rl_token.z_dim, cfg.model.rl_token.token_hidden_dim)
PY
```

Expected:

```text
Pi0RLConfig
512 768
```

---

### Task 4: Verification

**Files:**
- No new files.

- [ ] **Step 1: Run focused model/policy tests**

Run:

```bash
uv run pytest src/openpi/models/rl_token_test.py src/openpi/policies/policy_test.py -q
```

Expected: all pass.

- [ ] **Step 2: Run RLT shape-sensitive tests**

Run:

```bash
uv run pytest src/openpi/models/rlt_test.py src/openpi/training/rlt_replay_store_test.py src/openpi/training/rlt_training_test.py packages/openpi-client/src/openpi_client/rlt_actor_runtime_test.py -q
```

Expected: all pass.

- [ ] **Step 3: Check changed files**

Run:

```bash
git diff --stat
git status --short
```

Expected: only planned files plus pre-existing user changes are modified.
