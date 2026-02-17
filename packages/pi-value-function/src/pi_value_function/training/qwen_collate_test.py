from __future__ import annotations

import numpy as np
import torch

from pi_value_function.training.qwen_collate import QwenVLCollateFn


class _FakeProcessor:
    def __init__(self):
        self.last_images = None
        self.last_texts = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        del tokenize, add_generation_prompt
        return f"templated::{messages[0]['content'][-1]['text']}"

    def __call__(self, text, images, return_tensors="pt", padding=True, **kwargs):
        del return_tensors, padding, kwargs
        self.last_texts = text
        self.last_images = images
        batch = len(text)
        return {
            "input_ids": torch.ones((batch, 12), dtype=torch.long),
            "attention_mask": torch.ones((batch, 12), dtype=torch.long),
            "pixel_values": torch.zeros((batch, 3, 224, 224), dtype=torch.float32),
        }


def _sample_item(return_value: float) -> dict:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    return {
        "prompt": "Put the battery bank in the orange box",
        "returns": np.array(return_value, dtype=np.float32),
        "image": {
            "base_0_rgb": image,
            "left_wrist_0_rgb": image,
            "right_wrist_0_rgb": image,
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        },
        "state": np.zeros(8, dtype=np.float32),
    }


def test_qwen_collate_builds_multimodal_batch():
    collate = QwenVLCollateFn(hf_model_id="Qwen/Qwen3-VL-2B-Instruct", max_len=128)
    fake_processor = _FakeProcessor()
    collate._processor = fake_processor

    batch = collate([_sample_item(-1.0), _sample_item(0.0)])

    assert batch["input_ids"].shape == (2, 12)
    assert batch["attention_mask"].shape == (2, 12)
    assert batch["returns"].shape == (2,)
    assert fake_processor.last_images is not None
    assert all(len(per_sample_images) == 3 for per_sample_images in fake_processor.last_images)
