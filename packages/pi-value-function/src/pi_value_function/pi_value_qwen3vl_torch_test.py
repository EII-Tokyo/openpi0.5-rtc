from __future__ import annotations

from types import SimpleNamespace

import torch

from pi_value_function.config import PiValueConfig
from pi_value_function.pi_value_qwen3vl_torch import PiValueQwen3VLTorch


class _FakeProcessor:
    pass


class _FakeBackbone(torch.nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=True, return_dict=True, **kwargs):
        del attention_mask, output_hidden_states, return_dict, kwargs
        batch_size, seq_len = input_ids.shape
        hidden = torch.ones(batch_size, seq_len, self.config.hidden_size, dtype=torch.float32, device=input_ids.device)
        hidden = self.proj(hidden)
        return SimpleNamespace(hidden_states=(hidden,))


def test_qwen_value_model_shapes_and_freeze():
    config = PiValueConfig(
        value_dims=201,
        value_min=-1.0,
        value_max=0.0,
        backbone="qwen3vl",
        hf_model_id="Qwen/Qwen3-VL-2B-Instruct",
        backbone_dtype="float32",
        value_head_layers=2,
    )

    model = PiValueQwen3VLTorch(
        config,
        device=torch.device("cpu"),
        backbone=_FakeBackbone(),
        processor=_FakeProcessor(),
    )

    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.trainable_parameters())

    batch = {
        "input_ids": torch.ones((2, 8), dtype=torch.long),
        "attention_mask": torch.ones((2, 8), dtype=torch.long),
    }
    returns = torch.tensor([-1.0, 0.0], dtype=torch.float32)

    logits = model.forward(batch)
    assert logits.shape == (2, 201)

    losses = model.compute_loss(batch, returns)
    assert losses.shape == (2,)

    values = model.predict_value(batch)
    assert values.shape == (2,)
