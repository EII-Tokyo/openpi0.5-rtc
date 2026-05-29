import dataclasses
import json

import jax
import numpy as np

from openpi.models import rlt
from openpi.training import rlt_replay_store
from openpi.training import rlt_training
from scripts import train_rlt_online


class _FakeRedis:
    def __init__(self):
        self.messages = []

    def publish(self, channel, payload):
        self.messages.append((channel, payload))


class _BrokenRedis:
    def publish(self, channel, payload):
        raise RuntimeError("redis down")


def _stats():
    return rlt_replay_store.ReplayStats(
        replay_size=123,
        num_shards=4,
        success_episodes=3,
        failure_episodes=2,
        bad_shards=1,
    )


def _shape(action_horizon=10):
    return rlt_replay_store.ReplayShape(
        z_dim=8,
        proprio_dim=4,
        action_horizon=action_horizon,
        action_dim=3,
    )


def test_build_metrics_payload_is_json_serializable():
    payload = train_rlt_online._build_metrics_payload(
        step=50,
        reduced={"critic_loss": np.float32(1.25), "actor_loss": np.float64(0.5), "steps_per_sec": np.float32(12.0)},
        stats=_stats(),
        replay_shape=_shape(action_horizon=50),
        train_shape=_shape(action_horizon=10),
        actor_enabled=True,
        latest_actor_path="/tmp/actor",
        latest_actor_step=50,
        wandb_url="https://wandb.example/run",
    )

    assert json.loads(json.dumps(payload)) == payload
    assert payload["type"] == "rlt_trainer_metrics"
    assert payload["trainer_step"] == 50
    assert payload["critic_loss"] == 1.25
    assert payload["actor_loss"] == 0.5
    assert payload["replay_size"] == 123
    assert payload["actor_enabled"] is True
    assert payload["latest_actor_path"] == "/tmp/actor"
    assert payload["latest_actor_step"] == 50
    assert payload["wandb_url"] == "https://wandb.example/run"
    assert payload["replay_shards"] == 4
    assert payload["success_episodes"] == 3
    assert payload["failure_episodes"] == 2
    assert payload["bad_shards"] == 1
    assert payload["replay_action_horizon"] == 50
    assert payload["train_action_horizon"] == 10
    assert payload["steps_per_sec"] == 12.0


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
        replay_shape=_shape(action_horizon=50),
        train_shape=_shape(action_horizon=10),
        replay_stats=_stats(),
    )

    actor_bytes = (actor_dir / "actor.msgpack").read_bytes()
    metadata = json.loads((actor_dir / "metadata.json").read_text())

    assert actor_bytes
    assert json.loads(json.dumps(metadata)) == metadata
    assert (tmp_path / "inference_actor" / "LATEST").read_text() == str(actor_dir)
    assert metadata["type"] == "rlt_inference_actor"
    assert metadata["step"] == 5
    assert metadata["action_horizon"] == 10
    assert metadata["replay_shape"] == dataclasses.asdict(_shape(action_horizon=50))
    assert metadata["train_shape"] == dataclasses.asdict(_shape(action_horizon=10))
    assert metadata["replay_stats"] == dataclasses.asdict(_stats())
    assert metadata["rlt_config"] == dataclasses.asdict(config.model)
    assert metadata["rlt_config"]["action_horizon"] == 10
    assert len(metadata["actor_sha256"]) == 64
    assert metadata["actor_sha256"] == train_rlt_online._sha256_bytes(actor_bytes)
