from __future__ import annotations

import dataclasses

from pi_value_function.training import train as train_module
from pi_value_function.training.train_config import TrainConfig


def test_train_dispatches_to_qwen_path(monkeypatch):
    config = TrainConfig.debug_config()
    config = dataclasses.replace(
        config,
        model_config=dataclasses.replace(
            config.model_config,
            backbone="qwen3vl",
            hf_model_id="Qwen/Qwen3-VL-2B-Instruct",
        ),
        logging=dataclasses.replace(config.logging, wandb_enabled=False),
    )

    called = {"qwen": False}

    def _fake_qwen_train(cfg):
        called["qwen"] = True
        assert cfg.model_config.backbone == "qwen3vl"

    monkeypatch.setattr(train_module, "train_qwen_torch", _fake_qwen_train)
    train_module.train(config)

    assert called["qwen"]
