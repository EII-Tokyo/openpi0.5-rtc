from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import torch

from pi_value_function.config import PiValueConfig
from pi_value_function.pi_value_qwen3vl_torch import PiValueQwen3VLTorch
from pi_value_function.training.train_config import TrainConfig
from pi_value_function.training.train_qwen_torch import _load_checkpoint
from pi_value_function.training.train_qwen_torch import _save_checkpoint


class _FakeProcessor:
    pass


class _FakeBackbone(torch.nn.Module):
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=True, return_dict=True, **kwargs):
        del attention_mask, output_hidden_states, return_dict, kwargs
        batch_size, seq_len = input_ids.shape
        hidden = torch.ones(batch_size, seq_len, self.config.hidden_size, dtype=torch.float32, device=input_ids.device)
        hidden = self.proj(hidden)
        return SimpleNamespace(hidden_states=(hidden,))


def _make_model() -> PiValueQwen3VLTorch:
    model_config = PiValueConfig(
        value_dims=201,
        value_min=-1.0,
        value_max=0.0,
        backbone="qwen3vl",
        hf_model_id="Qwen/Qwen3-VL-2B-Instruct",
        backbone_dtype="float32",
        value_head_layers=2,
    )
    return PiValueQwen3VLTorch(
        model_config,
        device=torch.device("cpu"),
        backbone=_FakeBackbone(),
        processor=_FakeProcessor(),
    )


def test_qwen_checkpoint_save_and_load_roundtrip(tmp_path):
    config = TrainConfig.debug_config()
    config = dataclasses.replace(
        config,
        model_config=dataclasses.replace(
            config.model_config,
            backbone="qwen3vl",
            hf_model_id="Qwen/Qwen3-VL-2B-Instruct",
            backbone_dtype="float32",
        ),
    )

    model = _make_model()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)

    checkpoint_root = tmp_path / "ckpts"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    _save_checkpoint(model, optimizer, 7, checkpoint_root, config)

    model2 = _make_model()
    optimizer2 = torch.optim.AdamW(model2.trainable_parameters(), lr=1e-3)

    restored_step = _load_checkpoint(
        model2,
        optimizer2,
        checkpoint_root / "7",
        torch.device("cpu"),
        load_optimizer=True,
    )

    assert restored_step == 7
