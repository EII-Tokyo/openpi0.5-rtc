import sys
import types

import importlib


def _import_serve_policy(monkeypatch):
    monkeypatch.setitem(sys.modules, "tyro", types.SimpleNamespace(cli=lambda _: None))

    openpi = types.ModuleType("openpi")
    policies = types.ModuleType("openpi.policies")
    aloha_policy = types.ModuleType("openpi.policies.aloha_policy")
    policy = types.ModuleType("openpi.policies.policy")
    policy_config = types.ModuleType("openpi.policies.policy_config")
    serving = types.ModuleType("openpi.serving")
    websocket_policy_server = types.ModuleType("openpi.serving.websocket_policy_server")
    training = types.ModuleType("openpi.training")
    config = types.ModuleType("openpi.training.config")
    policy.Policy = type("Policy", (), {})
    policy.PolicyRecorder = type("PolicyRecorder", (), {})

    monkeypatch.setitem(sys.modules, "openpi", openpi)
    monkeypatch.setitem(sys.modules, "openpi.policies", policies)
    monkeypatch.setitem(sys.modules, "openpi.policies.aloha_policy", aloha_policy)
    monkeypatch.setitem(sys.modules, "openpi.policies.policy", policy)
    monkeypatch.setitem(sys.modules, "openpi.policies.policy_config", policy_config)
    monkeypatch.setitem(sys.modules, "openpi.serving", serving)
    monkeypatch.setitem(sys.modules, "openpi.serving.websocket_policy_server", websocket_policy_server)
    monkeypatch.setitem(sys.modules, "openpi.training", training)
    monkeypatch.setitem(sys.modules, "openpi.training.config", config)
    sys.modules.pop("scripts.serve_policy", None)
    return importlib.import_module("scripts.serve_policy")


class _FakeJaxConfig:
    def __init__(self):
        self.updates = []

    def update(self, key, value):
        self.updates.append((key, value))


def test_configure_jax_persistent_cache_from_env(monkeypatch):
    serve_policy = _import_serve_policy(monkeypatch)
    fake_config = _FakeJaxConfig()
    monkeypatch.setitem(sys.modules, "jax", types.SimpleNamespace(config=fake_config))
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/app/.jax_cache")
    monkeypatch.setenv("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    monkeypatch.setenv("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
    monkeypatch.setenv("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", "xla_gpu_per_fusion_autotune_cache_dir")

    assert serve_policy.configure_jax_persistent_cache_from_env()

    assert fake_config.updates == [
        ("jax_compilation_cache_dir", "/app/.jax_cache"),
        ("jax_persistent_cache_min_compile_time_secs", 0.0),
        ("jax_persistent_cache_min_entry_size_bytes", -1),
        ("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"),
    ]


def test_configure_jax_persistent_cache_is_noop_without_cache_dir(monkeypatch):
    serve_policy = _import_serve_policy(monkeypatch)
    fake_config = _FakeJaxConfig()
    monkeypatch.setitem(sys.modules, "jax", types.SimpleNamespace(config=fake_config))
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)

    assert not serve_policy.configure_jax_persistent_cache_from_env()
    assert fake_config.updates == []
