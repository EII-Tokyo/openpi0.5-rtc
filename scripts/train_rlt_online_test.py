import dataclasses
import json
import os

import jax
import numpy as np
import pytest

from openpi.models import rlt
from openpi.training import rlt_replay_store
from openpi.training import rlt_training
from scripts import train_rlt
from scripts import train_rlt_offline
from scripts import train_rlt_online


class _FakeRedis:
    def __init__(self):
        self.messages = []
        self.values = {}

    def publish(self, channel, payload):
        self.messages.append((channel, payload))

    def set(self, key, payload):
        self.values[key] = payload


class _BrokenRedis:
    def publish(self, channel, payload):
        raise RuntimeError("redis down")


class _FakePubSub:
    def __init__(self, messages):
        self._messages = list(messages)

    def subscribe(self, channel):
        self.channel = channel

    def get_message(self, timeout=0.0):
        if not self._messages:
            return None
        return self._messages.pop(0)

    def close(self):
        pass


class _FakeRedisWithPubSub:
    def __init__(self, messages):
        self._pubsub = _FakePubSub(messages)

    def pubsub(self):
        return self._pubsub


def _stats(
    *,
    replay_size: int = 123,
    num_shards: int = 4,
    success_episodes: int = 3,
    failure_episodes: int = 2,
    bad_shards: int = 1,
):
    return rlt_replay_store.ReplayStats(
        replay_size=replay_size,
        num_shards=num_shards,
        success_episodes=success_episodes,
        failure_episodes=failure_episodes,
        bad_shards=bad_shards,
    )


def _shape(action_horizon=10):
    return rlt_replay_store.ReplayShape(
        z_dim=8,
        proprio_dim=4,
        action_horizon=action_horizon,
        action_dim=3,
    )


def test_trainer_defaults_expect_c10_replay_and_train_horizon():
    assert train_rlt.Args(replay_npz="/tmp/replay.npz").expected_replay_action_horizon == 10
    assert train_rlt.Args(replay_npz="/tmp/replay.npz").train_action_horizon == 10
    assert train_rlt_online.Args(replay_dir="/tmp/replay").expected_replay_action_horizon == 10
    assert train_rlt_online.Args(replay_dir="/tmp/replay").train_action_horizon == 10
    assert train_rlt_offline.Args(replay_dir="/tmp/replay").expected_replay_action_horizon == 10
    assert train_rlt_offline.Args(replay_dir="/tmp/replay").train_action_horizon == 10


def test_offline_trainer_builds_recursive_replay_store_with_train_horizon(tmp_path):
    args = train_rlt_offline.Args(
        replay_dir=tmp_path / "replay" / "rlt_key_regions",
        recursive_scan=True,
        train_action_horizon=10,
        max_replay_samples=123,
    )

    store = train_rlt_offline._build_replay_store(args)

    assert store.sample_shape is None
    assert store._recursive is True
    assert store._sample_action_horizon == 10
    assert store._max_replay_samples == 123


def test_offline_trainer_builds_manifest_replay_store(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text("")
    args = train_rlt_offline.Args(
        replay_dir=tmp_path / "replay" / "rlt_key_regions_clean",
        manifest_path=manifest_path,
        recursive_scan=True,
    )

    store = train_rlt_offline._build_replay_store(args)

    assert store._manifest_path == manifest_path


def _write_online_manifest(path, shard_paths):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for shard_path in shard_paths:
            file.write(json.dumps({"shard_path": str(shard_path)}, sort_keys=True) + "\n")


def test_online_trainer_builds_manifest_replay_store(tmp_path):
    manifest_path = tmp_path / "train_manifest.jsonl"
    manifest_path.write_text("")
    args = train_rlt_online.Args(
        replay_dir=tmp_path / "replay" / "rlt_key_regions_clean",
        manifest_path=manifest_path,
        recursive_scan=True,
    )

    store = train_rlt_online._build_replay_store(args)

    assert store._manifest_path == manifest_path
    assert store._recursive is True


def test_online_candidate_holdout_paths_use_holdout_manifest_not_training_store(tmp_path):
    train_shard = tmp_path / "train" / "key_region_train.npz"
    holdout_shard = tmp_path / "holdout" / "key_region_holdout.npz"
    train_shard.parent.mkdir()
    holdout_shard.parent.mkdir()
    train_shard.write_bytes(b"train")
    holdout_shard.write_bytes(b"holdout")
    train_manifest = tmp_path / "train_manifest.jsonl"
    holdout_manifest = tmp_path / "holdout_manifest.jsonl"
    _write_online_manifest(train_manifest, [train_shard])
    _write_online_manifest(holdout_manifest, [holdout_shard])
    args = train_rlt_online.Args(
        replay_dir=tmp_path,
        manifest_path=train_manifest,
        holdout_manifest_path=holdout_manifest,
    )
    store = train_rlt_online._build_replay_store(args)

    assert train_rlt_online._candidate_holdout_paths(args=args, store=store, round_index=7) == [
        holdout_shard.resolve()
    ]


def test_online_train_holdout_manifest_overlap_is_rejected(tmp_path):
    shared_shard = tmp_path / "shared" / "key_region_shared.npz"
    shared_shard.parent.mkdir()
    shared_shard.write_bytes(b"shared")
    train_manifest = tmp_path / "train_manifest.jsonl"
    holdout_manifest = tmp_path / "holdout_manifest.jsonl"
    _write_online_manifest(train_manifest, [shared_shard])
    _write_online_manifest(holdout_manifest, [shared_shard])
    args = train_rlt_online.Args(
        replay_dir=tmp_path,
        manifest_path=train_manifest,
        holdout_manifest_path=holdout_manifest,
    )

    with pytest.raises(ValueError, match="overlap"):
        train_rlt_online._validate_train_holdout_disjoint(args, train_paths=None)


def test_online_segment_db_train_holdout_overlap_is_rejected(tmp_path):
    shared_shard = tmp_path / "shared" / "key_region_shared.npz"
    shared_shard.parent.mkdir()
    shared_shard.write_bytes(b"shared")
    holdout_manifest = tmp_path / "holdout_manifest.jsonl"
    _write_online_manifest(holdout_manifest, [shared_shard])
    args = train_rlt_online.Args(
        replay_dir=tmp_path,
        holdout_manifest_path=holdout_manifest,
    )

    with pytest.raises(ValueError, match="overlap"):
        train_rlt_online._validate_train_holdout_disjoint(args, train_paths=[shared_shard.resolve()])


def test_online_safety_requires_explicit_holdout_manifest(tmp_path):
    args = train_rlt_online.Args(replay_dir=tmp_path, online_safety_enabled=True)

    with pytest.raises(ValueError, match="holdout_manifest_path"):
        train_rlt_online._validate_online_safety_inputs(args)


def test_offline_trainer_builds_config_with_manual_beta():
    shape = rlt_replay_store.ReplayShape(z_dim=8, proprio_dim=4, action_horizon=10, action_dim=14)
    args = train_rlt_offline.Args(
        replay_dir="/tmp/replay",
        actor_lr=1e-5,
        critic_lr=2e-4,
        beta=12.0,
        policy_delay=3,
        actor_publish_interval=250,
    )

    config = train_rlt_offline._build_training_config(args, shape)

    assert config.actor_lr == 1e-5
    assert config.critic_lr == 2e-4
    assert config.policy_delay == 3
    assert config.actor_publish_interval == 250
    assert config.model.beta == 12.0
    assert config.model.action_horizon == 10
    assert config.model.action_dim == 14


def test_offline_trainer_actor_gate_respects_training_stage():
    args = train_rlt_offline.Args(replay_dir="/tmp/replay", critic_burn_in_steps=1000)
    stats = _stats(replay_size=4096, num_shards=40, success_episodes=20, failure_episodes=20, bad_shards=0)

    assert not train_rlt_offline._actor_updates_allowed(
        dataclasses.replace(args, training_stage="critic_only"),
        stats=stats,
        step=5000,
        critic_gate_open=True,
    )
    assert train_rlt_offline._actor_updates_allowed(
        dataclasses.replace(args, training_stage="actor_only"),
        stats=stats,
        step=1,
        critic_gate_open=True,
    )
    assert not train_rlt_offline._actor_updates_allowed(
        dataclasses.replace(args, training_stage="critic_actor"),
        stats=stats,
        step=999,
        critic_gate_open=True,
    )
    assert train_rlt_offline._actor_updates_allowed(
        dataclasses.replace(args, training_stage="critic_actor"),
        stats=stats,
        step=1000,
        critic_gate_open=True,
    )
    assert not train_rlt_offline._actor_updates_allowed(
        dataclasses.replace(args, training_stage="critic_actor"),
        stats=stats,
        step=1000,
        critic_gate_open=False,
    )


def test_offline_trainer_actor_gate_uses_critic_holdout_threshold():
    args = train_rlt_offline.Args(
        replay_dir="/tmp/replay",
        training_stage="critic_actor",
        critic_auc_threshold=0.70,
        require_positive_q_gap=True,
    )

    assert not train_rlt_offline._critic_gate_allows_actor(args, None)
    assert not train_rlt_offline._critic_gate_allows_actor(args, {"auc": 0.69, "q_gap": 0.2})
    assert not train_rlt_offline._critic_gate_allows_actor(args, {"auc": 0.71, "q_gap": 0.0})
    assert train_rlt_offline._critic_gate_allows_actor(args, {"auc": 0.71, "q_gap": 0.2})


def test_offline_actor_only_uses_zero_critic_learning_rate():
    shape = rlt_replay_store.ReplayShape(z_dim=8, proprio_dim=4, action_horizon=10, action_dim=14)
    args = train_rlt_offline.Args(replay_dir="/tmp/replay", training_stage="actor_only", critic_lr=3e-4)

    config = train_rlt_offline._build_training_config(args, shape)

    assert config.critic_lr == 0.0


def test_build_metrics_payload_is_json_serializable():
    payload = train_rlt_online._build_metrics_payload(
        step=50,
        reduced={
            "critic_loss": np.float32(1.25),
            "critic_q1_loss": np.float32(0.75),
            "critic_q2_loss": np.float32(0.5),
            "actor_loss": np.float64(0.5),
            "actor_q_value": np.float32(1.75),
            "reference_q_value": np.float32(1.25),
            "q_advantage": np.float32(0.5),
            "actor_delta_norm": np.float32(0.025),
            "q1_mean": np.float32(1.1),
            "q2_mean": np.float32(0.9),
            "target_q_mean": np.float32(1.0),
            "actor_updated": np.float32(1.0),
            "publish_actor": np.float32(0.0),
            "beta": np.float32(10.0),
            "auto_beta_enabled": True,
            "auto_beta_target_delta_norm": np.float32(0.05),
            "auto_beta_delta_norm_ema": np.float32(0.025),
            "auto_beta_q_advantage_ema": np.float32(0.5),
            "auto_beta_critic_loss_ema": np.float32(1.25),
            "auto_beta_reason": "delta_below_target_q_positive",
            "steps_per_sec": np.float32(12.0),
        },
        stats=_stats(),
        replay_shape=_shape(action_horizon=10),
        train_shape=_shape(action_horizon=10),
        actor_enabled=True,
        trainer_enabled=True,
        trainer_running=True,
        critic_burn_in_steps=1000,
        target_sync_step=1000,
        latest_actor_path="/tmp/actor",
        latest_actor_step=50,
        wandb_url="https://wandb.example/run",
    )

    assert json.loads(json.dumps(payload)) == payload
    assert payload["type"] == "rlt_trainer_metrics"
    assert payload["trainer_step"] == 50
    assert payload["critic_burn_in_steps"] == 1000
    assert payload["target_sync_step"] == 1000
    assert payload["critic_loss"] == 1.25
    assert payload["critic_q1_loss"] == 0.75
    assert payload["critic_q2_loss"] == 0.5
    assert payload["actor_loss"] == 0.5
    assert payload["actor_q_value"] == 1.75
    assert payload["reference_q_value"] == 1.25
    assert payload["q_advantage"] == 0.5
    assert payload["actor_delta_norm"] == pytest.approx(0.025)
    assert payload["q1_mean"] == pytest.approx(1.1)
    assert payload["q2_mean"] == pytest.approx(0.9)
    assert payload["target_q_mean"] == 1.0
    assert payload["q_gap"] == pytest.approx(0.2)
    assert payload["actor_updated"] is True
    assert payload["publish_actor"] is False
    assert payload["beta"] == 10.0
    assert payload["auto_beta_enabled"] is True
    assert payload["auto_beta_target_delta_norm"] == pytest.approx(0.05)
    assert payload["auto_beta_delta_norm_ema"] == pytest.approx(0.025)
    assert payload["auto_beta_q_advantage_ema"] == 0.5
    assert payload["auto_beta_critic_loss_ema"] == 1.25
    assert payload["auto_beta_reason"] == "delta_below_target_q_positive"
    assert payload["replay_size"] == 123
    assert payload["actor_enabled"] is True
    assert payload["trainer_enabled"] is True
    assert payload["trainer_running"] is True
    assert payload["latest_actor_path"] == "/tmp/actor"
    assert payload["latest_actor_step"] == 50
    assert payload["wandb_url"] == "https://wandb.example/run"
    assert payload["replay_shards"] == 4
    assert payload["success_episodes"] == 3
    assert payload["failure_episodes"] == 2
    assert payload["bad_shards"] == 1
    assert payload["replay_action_horizon"] == 10
    assert payload["train_action_horizon"] == 10
    assert payload["steps_per_sec"] == 12.0


def test_reduce_numeric_infos_allows_manual_beta_none_metrics():
    reduced = train_rlt_online._reduce_numeric_infos(
        [
            {
                "beta": np.asarray(10.0),
                "critic_loss": np.asarray(0.2),
                "auto_beta_enabled": False,
                "auto_beta_delta_norm_ema": None,
                "auto_beta_reason": "manual_beta",
            },
            {
                "beta": np.asarray(10.0),
                "critic_loss": np.asarray(0.4),
                "auto_beta_enabled": False,
                "auto_beta_delta_norm_ema": None,
                "auto_beta_reason": "manual_beta",
            },
        ]
    )

    assert reduced["beta"] == 10.0
    assert reduced["critic_loss"] == pytest.approx(0.3)
    assert reduced["auto_beta_enabled"] == 0.0
    assert "auto_beta_delta_norm_ema" not in reduced
    assert "auto_beta_reason" not in reduced



def test_auto_beta_controller_reduces_beta_when_delta_low_and_advantage_positive():
    controller = train_rlt_online.AutoBetaController(
        beta=10.0,
        target_delta_norm=0.05,
        beta_min=1.0,
        beta_max=15.0,
        lr=0.1,
        ema_decay=0.0,
        q_margin=0.005,
        update_interval=1,
    )

    result = controller.update(
        step=10,
        metrics={"actor_delta_norm": 0.025, "q_advantage": 0.02, "critic_loss": 0.01},
    )

    assert result.beta < 10.0
    assert result.reason == "delta_below_target_q_positive"
    assert result.metrics["auto_beta_delta_norm_ema"] == pytest.approx(0.025)
    assert result.metrics["auto_beta_q_advantage_ema"] == pytest.approx(0.02)
    assert result.metrics["auto_beta_enabled"] is True


def test_auto_beta_controller_increases_beta_when_delta_high_or_advantage_weak():
    controller = train_rlt_online.AutoBetaController(
        beta=4.0,
        target_delta_norm=0.05,
        beta_min=1.0,
        beta_max=15.0,
        lr=0.1,
        ema_decay=0.0,
        q_margin=0.005,
        update_interval=1,
    )

    high_delta = controller.update(
        step=10,
        metrics={"actor_delta_norm": 0.10, "q_advantage": 0.02, "critic_loss": 0.01},
    )
    weak_advantage = controller.update(
        step=11,
        metrics={"actor_delta_norm": 0.04, "q_advantage": -0.01, "critic_loss": 0.01},
    )

    assert high_delta.beta > 4.0
    assert high_delta.reason == "delta_above_target"
    assert weak_advantage.beta > high_delta.beta
    assert weak_advantage.reason == "q_advantage_below_margin"


def test_auto_beta_controller_skips_until_update_interval():
    controller = train_rlt_online.AutoBetaController(
        beta=4.0,
        target_delta_norm=0.05,
        beta_min=1.0,
        beta_max=15.0,
        lr=0.1,
        ema_decay=0.0,
        q_margin=0.005,
        update_interval=10,
    )

    result = controller.update(
        step=9,
        metrics={"actor_delta_norm": 0.10, "q_advantage": 0.02, "critic_loss": 0.01},
    )

    assert result.beta == 4.0
    assert result.reason == "waiting_for_update_interval"


def test_online_round_controller_waits_for_new_shards_and_budgets_steps():
    controller = train_rlt_online.OnlineSafetyController(
        min_new_shards_per_round=10,
        min_new_success_per_round=0,
        min_new_failure_per_round=0,
        auto_train_critic=True,
        auto_train_actor=True,
        critic_updates_per_round=3,
        actor_updates_per_round=2,
    )

    assert controller.phase == "idle_wait_new_data"
    assert not controller.maybe_start_round(_stats(num_shards=9))
    assert controller.maybe_start_round(_stats(num_shards=10))
    assert controller.phase == "critic_candidate_training"
    assert [controller.step_allocation()["actor_enabled"] for _ in range(3)] == [False, False, False]
    assert controller.phase == "critic_eval"
    assert controller.accept_critic({"auc": 0.75, "q_gap": 0.2})
    assert controller.phase == "actor_candidate_training"
    assert [controller.step_allocation()["actor_enabled"] for _ in range(2)] == [True, True]
    assert controller.phase == "actor_eval"
    assert controller.accept_actor({"q_advantage": 0.05, "actor_delta_norm": 0.05})
    assert controller.phase == "idle_wait_new_data"
    assert controller.last_committed_shards == 10


def test_online_round_controller_defaults_to_manual_critic_start():
    controller = train_rlt_online.OnlineSafetyController(
        min_new_shards_per_round=10,
        min_new_success_per_round=0,
        min_new_failure_per_round=0,
    )

    assert not controller.maybe_start_round(_stats(num_shards=10))
    assert controller.phase == "idle_wait_new_data"
    assert controller.critic_steps_remaining == 0
    assert controller.last_rejection_reason == "critic_auto_train_disabled"


def test_online_round_controller_defaults_to_manual_actor_start_after_critic_accept():
    controller = train_rlt_online.OnlineSafetyController(
        min_new_shards_per_round=10,
        min_new_success_per_round=0,
        min_new_failure_per_round=0,
        auto_train_critic=True,
        auto_train_actor=False,
        critic_updates_per_round=1,
        actor_updates_per_round=2,
    )

    assert controller.maybe_start_round(_stats(num_shards=10))
    controller.step_allocation()

    assert controller.accept_critic({"auc": 0.75, "q_gap": 0.2})
    assert controller.phase == "idle_wait_new_data"
    assert controller.actor_steps_remaining == 0
    assert controller.last_committed_shards == 10
    assert controller.last_rejection_reason == "actor_auto_train_disabled"


def test_online_round_controller_rejects_unstable_critic_and_keeps_old_best():
    controller = train_rlt_online.OnlineSafetyController(
        min_new_shards_per_round=10,
        min_new_success_per_round=0,
        min_new_failure_per_round=0,
        auto_train_critic=True,
        auto_train_actor=True,
        critic_updates_per_round=1,
        actor_updates_per_round=1,
        critic_auc_min=0.70,
        critic_max_auc_drop=0.02,
    )
    assert controller.maybe_start_round(_stats(num_shards=10))
    controller.step_allocation()

    assert not controller.accept_critic({"auc": 0.69, "q_gap": 0.2})
    assert controller.phase == "idle_wait_new_data"
    assert controller.best_critic_auc is None
    assert controller.last_rejection_reason == "critic_auc_below_min"

    assert controller.maybe_start_round(_stats(num_shards=20))
    controller.step_allocation()
    assert controller.accept_critic({"auc": 0.80, "q_gap": 0.3})
    controller.phase = "idle_wait_new_data"
    assert controller.maybe_start_round(_stats(num_shards=30))
    controller.step_allocation()
    assert not controller.accept_critic({"auc": 0.75, "q_gap": 0.3})
    assert controller.best_critic_auc == 0.80
    assert controller.last_rejection_reason == "critic_auc_regressed"


def test_online_beta_schedule_starts_conservative_and_opens_after_actor_accept():
    controller = train_rlt_online.OnlineSafetyController(
        beta_initial=30.0,
        beta_min=5.0,
        beta_max=30.0,
        beta_decay_on_actor_accept=0.9,
        beta_increase_on_reject=1.25,
        target_delta_initial=0.04,
        target_delta_max=0.10,
        target_delta_increment=0.01,
    )

    assert controller.beta == 30.0
    assert controller.target_delta_norm == 0.04
    controller.on_actor_accepted()
    assert controller.beta == pytest.approx(27.0)
    assert controller.target_delta_norm == pytest.approx(0.05)
    controller.on_actor_rejected()
    assert controller.beta == pytest.approx(30.0)
    assert controller.target_delta_norm == pytest.approx(0.04)


def test_init_wandb_uses_process_api_key_without_leaking_config(monkeypatch, tmp_path):
    calls = []

    class _FakeWandb:
        @staticmethod
        def init(**kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(train_rlt_online, "wandb", _FakeWandb)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    args = train_rlt_online.Args(
        replay_dir=tmp_path / "replay",
        output_dir=tmp_path / "run",
        wandb_api_key="secret-test-key",
        wandb_project="project",
        wandb_run_name="run",
    )

    train_rlt_online._init_wandb(args, _FakeStoreForWandb())

    assert os.environ["WANDB_API_KEY"] == "secret-test-key"
    assert calls[0]["project"] == "project"
    assert calls[0]["name"] == "run"
    assert calls[0]["config"]["wandb_api_key"] == "<set>"


class _FakeStoreForWandb:
    @property
    def stats(self):
        return _stats()

def test_redis_metrics_publisher_publishes_json():
    fake = _FakeRedis()
    publisher = train_rlt_online.RedisMetricsPublisher(
        enabled=True,
        channel="aloha_rlt_state",
        redis_client=fake,
    )

    publisher.publish({"type": "rlt_trainer_metrics", "replay_size": 7})

    assert len(fake.messages) == 1
    channel, payload = fake.messages[0]
    assert channel == "aloha_rlt_state"
    assert isinstance(payload, str)
    decoded = json.loads(payload)
    assert decoded["type"] == "rlt_trainer_metrics"
    assert decoded["replay_size"] == 7
    assert json.loads(fake.values["aloha_rlt_state:latest"]) == decoded


def test_redis_metrics_publisher_disabled_does_not_publish():
    fake = _FakeRedis()
    publisher = train_rlt_online.RedisMetricsPublisher(
        enabled=False,
        channel="aloha_rlt_state",
        redis_client=fake,
    )

    publisher.publish({"type": "rlt_trainer_metrics", "replay_size": 7})

    assert fake.messages == []


def test_redis_metrics_publisher_failure_does_not_raise():
    publisher = train_rlt_online.RedisMetricsPublisher(
        enabled=True,
        channel="aloha_rlt_state",
        redis_client=_BrokenRedis(),
    )

    publisher.publish({"type": "rlt_trainer_metrics", "replay_size": 7})


def test_redis_control_subscriber_reads_trainer_enabled_and_beta_updates():
    fake = _FakeRedisWithPubSub(
        [
            {"type": "subscribe", "data": "1"},
            {"type": "message", "data": json.dumps({"type": "ignored", "trainer_enabled": True})},
            {
                "type": "message",
                "data": json.dumps({"type": "config_update", "trainer_enabled": True, "critic_burn_in_steps": 1000}),
            },
            {"type": "message", "data": json.dumps({"type": "config_update", "beta": 7.5, "trainer_enabled": False})},
        ]
    )
    subscriber = train_rlt_online.RedisControlSubscriber(
        enabled=True,
        channel="aloha_rlt_control",
        redis_client=fake,
    )

    update = subscriber.poll_update()

    assert update == {"trainer_enabled": False, "critic_burn_in_steps": 1000, "beta": 7.5}


def test_redis_control_subscriber_reads_auto_beta_updates():
    fake = _FakeRedisWithPubSub(
        [
            {
                "type": "message",
                "data": json.dumps(
                    {
                        "type": "config_update",
                        "auto_beta_enabled": True,
                        "auto_beta_target_delta_norm": 0.06,
                        "auto_beta_min": 1.0,
                        "auto_beta_max": 30.0,
                        "auto_beta_lr": 0.03,
                        "auto_beta_ema_decay": 0.8,
                        "auto_beta_update_interval": 100,
                        "auto_beta_q_margin": 0.01,
                    }
                ),
            }
        ]
    )
    subscriber = train_rlt_online.RedisControlSubscriber(
        enabled=True,
        channel="aloha_rlt_control",
        redis_client=fake,
    )

    update = subscriber.poll_update()

    assert update == {
        "auto_beta_enabled": True,
        "auto_beta_target_delta_norm": 0.06,
        "auto_beta_min": 1.0,
        "auto_beta_max": 30.0,
        "auto_beta_lr": 0.03,
        "auto_beta_ema_decay": 0.8,
        "auto_beta_update_interval": 100,
        "auto_beta_q_margin": 0.01,
    }


def test_auto_beta_controller_accepts_runtime_config_update():
    controller = train_rlt_online.AutoBetaController(
        beta=10.0,
        target_delta_norm=0.13,
        beta_min=1.0,
        beta_max=15.0,
        lr=0.03,
        ema_decay=0.8,
        q_margin=0.001,
        update_interval=100,
    )

    controller.update_config(
        target_delta_norm=0.06,
        beta_min=2.0,
        beta_max=30.0,
        lr=0.05,
        ema_decay=0.7,
        q_margin=0.01,
        update_interval=50,
    )

    assert controller.target_delta_norm == 0.06
    assert controller.beta_min == 2.0
    assert controller.beta_max == 30.0
    assert controller.lr == 0.05
    assert controller.ema_decay == 0.7
    assert controller.q_margin == 0.01
    assert controller.update_interval == 50
    assert controller.metrics()["auto_beta_target_delta_norm"] == 0.06


def test_save_actor_for_inference_writes_runtime_metadata(tmp_path):
    config = rlt_training.RLTTrainingConfig(
        model=rlt.RLTConfig(
            z_dim=8,
            proprio_dim=4,
            action_horizon=10,
            action_dim=3,
            hidden_dim=16,
            num_layers=2,
        )
    )
    state = rlt_training.init_train_state(config, jax.random.key(0))

    actor_dir = train_rlt_online._save_actor_for_inference(
        state,
        tmp_path,
        5,
        action_horizon=10,
        replay_shape=_shape(action_horizon=10),
        train_shape=_shape(action_horizon=10),
        replay_stats=_stats(),
    )

    actor_bytes = (actor_dir / "actor.msgpack").read_bytes()
    critic_bytes = (actor_dir / "critic.msgpack").read_bytes()
    metadata = json.loads((actor_dir / "metadata.json").read_text())

    assert actor_bytes
    assert critic_bytes
    assert json.loads(json.dumps(metadata)) == metadata
    assert (tmp_path / "inference_actor" / "LATEST").read_text() == str(actor_dir)
    assert metadata["type"] == "rlt_inference_actor"
    assert metadata["step"] == 5
    assert metadata["action_horizon"] == 10
    assert metadata["replay_shape"] == dataclasses.asdict(_shape(action_horizon=10))
    assert metadata["train_shape"] == dataclasses.asdict(_shape(action_horizon=10))
    assert metadata["replay_stats"] == dataclasses.asdict(_stats())
    assert metadata["rlt_config"] == dataclasses.asdict(config.model)
    assert metadata["rlt_config"]["action_horizon"] == 10
    assert len(metadata["actor_sha256"]) == 64
    assert metadata["actor_sha256"] == train_rlt_online._sha256_bytes(actor_bytes)
    assert metadata["critic_file"] == "critic.msgpack"
    assert len(metadata["critic_sha256"]) == 64
    assert metadata["critic_sha256"] == train_rlt_online._sha256_bytes(critic_bytes)


def test_runtime_beta_update_changes_train_step_beta():
    config = rlt_training.RLTTrainingConfig(
        model=rlt.RLTConfig(
            z_dim=8,
            proprio_dim=4,
            action_horizon=10,
            action_dim=3,
            hidden_dim=16,
            num_layers=2,
            beta=10.0,
        ),
        policy_delay=1,
    )
    state = rlt_training.init_train_state(config, jax.random.key(0))

    state = train_rlt_online._with_runtime_beta(state, 5.0)
    model = jax.tree_util.tree_leaves(rlt_training.actor_params_for_inference(state))
    assert model

    batch = rlt_training.make_replay_batch(
        z_rl=np.zeros((2, 8), dtype=np.float32),
        proprio=np.zeros((2, 4), dtype=np.float32),
        action=np.zeros((2, 10, 3), dtype=np.float32),
        reference_action=np.zeros((2, 10, 3), dtype=np.float32),
        reward_seq=np.zeros((2, 10), dtype=np.float32),
        next_z_rl=np.zeros((2, 8), dtype=np.float32),
        next_proprio=np.zeros((2, 4), dtype=np.float32),
        next_reference_action=np.zeros((2, 10, 3), dtype=np.float32),
        done=np.ones((2,), dtype=np.bool_),
    )
    _, info = rlt_training.train_step(state, batch, jax.random.key(1))

    assert float(jax.device_get(info["beta"])) == pytest.approx(5.0)


def test_load_inference_checkpoint_initializes_actor_and_critic(tmp_path):
    config = rlt_training.RLTTrainingConfig(
        model=rlt.RLTConfig(
            z_dim=8,
            proprio_dim=4,
            action_horizon=10,
            action_dim=3,
            hidden_dim=16,
            num_layers=2,
        )
    )
    source_state = rlt_training.init_train_state(config, jax.random.key(1))
    fresh_state = rlt_training.init_train_state(config, jax.random.key(2))
    actor_dir = train_rlt_online._save_actor_for_inference(
        source_state,
        tmp_path,
        6000,
        action_horizon=10,
        replay_shape=_shape(action_horizon=10),
        train_shape=_shape(action_horizon=10),
        replay_stats=_stats(),
    )

    loaded_state, metadata = train_rlt_online._load_inference_actor_checkpoint(fresh_state, actor_dir)

    source_model = rlt_training.nnx.merge(source_state.model_def, source_state.params)
    loaded_model = rlt_training.nnx.merge(loaded_state.model_def, loaded_state.params)
    assert metadata["step"] == 6000
    assert int(loaded_state.step) == 0
    for key, value in rlt_training.nnx.state(loaded_model.actor).flat_state().items():
        assert np.allclose(value.value, rlt_training.nnx.state(source_model.actor).flat_state()[key].value)
    for key, value in rlt_training.nnx.state(loaded_model.critic).flat_state().items():
        assert np.allclose(value.value, rlt_training.nnx.state(source_model.critic).flat_state()[key].value)
    for key, value in rlt_training.nnx.state(loaded_model.target_critic).flat_state().items():
        assert np.allclose(value.value, rlt_training.nnx.state(source_model.critic).flat_state()[key].value)


def test_online_controller_can_treat_existing_replay_as_bootstrap_baseline():
    controller = train_rlt_online.OnlineSafetyController(
        min_new_shards_per_round=10,
        min_new_success_per_round=0,
        min_new_failure_per_round=0,
        auto_train_critic=True,
    )

    controller.mark_bootstrap_committed(_stats(num_shards=117))

    assert controller.last_committed_shards == 117
    assert not controller.maybe_start_round(_stats(num_shards=126))
    assert controller.maybe_start_round(_stats(num_shards=127))


def test_online_controller_requires_new_success_and_failure_counts():
    controller = train_rlt_online.OnlineSafetyController(
        min_new_shards_per_round=10,
        min_new_success_per_round=5,
        min_new_failure_per_round=5,
        auto_train_critic=True,
    )
    controller.mark_bootstrap_committed(_stats(num_shards=100, success_episodes=20, failure_episodes=20))

    assert not controller.maybe_start_round(_stats(num_shards=110, success_episodes=24, failure_episodes=25))
    assert controller.last_rejection_reason == "waiting_for_new_success"
    assert not controller.maybe_start_round(_stats(num_shards=110, success_episodes=25, failure_episodes=24))
    assert controller.last_rejection_reason == "waiting_for_new_failure"
    assert controller.maybe_start_round(_stats(num_shards=110, success_episodes=25, failure_episodes=25))
    assert controller.phase == "critic_candidate_training"


def test_prepare_output_dir_preserves_existing_run_without_overwrite(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    keep = output_dir / "inference_actor" / "00000010" / "actor.msgpack"
    keep.parent.mkdir(parents=True)
    keep.write_bytes(b"actor")
    args = train_rlt_online.Args(replay_dir=tmp_path, output_dir=output_dir, overwrite=False)

    train_rlt_online._prepare_output_dir(args)

    assert keep.read_bytes() == b"actor"


def test_runtime_control_subscriber_reads_beta_update():
    class _FakePubSub:
        def __init__(self):
            self.messages = [
                {"type": "message", "data": json.dumps({"type": "config_update", "beta": 5.0})},
                {"type": "message", "data": json.dumps({"type": "config_update", "beta": -1.0})},
            ]
            self.subscribed = []

        def subscribe(self, channel):
            self.subscribed.append(channel)

        def get_message(self, timeout=0.0):
            return self.messages.pop(0) if self.messages else None

        def close(self):
            pass

    class _FakeControlRedis:
        def __init__(self):
            self.pubsub_obj = _FakePubSub()

        def pubsub(self):
            return self.pubsub_obj

    subscriber = train_rlt_online.RedisControlSubscriber(
        enabled=True,
        channel="aloha_rlt_control",
        redis_client=_FakeControlRedis(),
    )

    assert subscriber.poll_beta_update() == 5.0
    assert subscriber.poll_beta_update() is None

def test_actor_updates_respect_replay_shard_gate():
    class _FakeStore:
        @property
        def stats(self):
            return _stats()

    args = train_rlt_online.Args(
        replay_dir="/tmp/replay",
        min_replay_samples=100,
        min_replay_shards=5,
        min_success_episodes=1,
        min_failure_episodes=1,
        critic_burn_in_steps=0,
    )

    assert not train_rlt_online._actor_updates_enabled(args, _FakeStore(), step=1)

    args.min_replay_shards = 4
    assert train_rlt_online._actor_updates_enabled(args, _FakeStore(), step=1)

    args.actor_min_replay_shards = 5
    assert not train_rlt_online._actor_updates_enabled(args, _FakeStore(), step=1)


def test_actor_updates_default_after_critic_burn_in():
    class _FakeStore:
        @property
        def stats(self):
            return _stats(replay_size=4096, num_shards=40, success_episodes=20, failure_episodes=20)

    args = train_rlt_online.Args(
        replay_dir="/tmp/replay",
        min_replay_samples=2048,
        min_replay_shards=40,
        min_success_episodes=10,
        min_failure_episodes=10,
        actor_min_replay_samples=4096,
        actor_min_replay_shards=40,
        actor_min_success_episodes=20,
        actor_min_failure_episodes=20,
    )

    assert args.critic_burn_in_steps == 1000
    assert not train_rlt_online._actor_updates_enabled(args, _FakeStore(), step=999)
    assert train_rlt_online._actor_updates_enabled(args, _FakeStore(), step=1000)
