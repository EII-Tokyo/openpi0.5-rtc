"""Qwen3-VL collate and dataloader helpers for value-function training."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import torch

from pi_value_function.pi_value_qwen3vl_torch import build_qwen_messages
from pi_value_function.pi_value_qwen3vl_torch import to_pil_image
from pi_value_function.training.data_loader import ValueFunctionDataset
from pi_value_function.training.data_loader import _worker_init_fn


class QwenVLCollateFn:
    """Collate fn that prepares multimodal Qwen3-VL inputs from prompts + 3 camera views."""

    def __init__(self, hf_model_id: str, max_len: int | None = None):
        self.hf_model_id = hf_model_id
        self.max_len = max_len
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.hf_model_id, trust_remote_code=True)
        return self._processor

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prompts: list[str] = []
        returns: list[float] = []
        image_triplets = []

        for item in items:
            item.pop("is_success", None)

            prompt = item.pop("prompt")
            if not isinstance(prompt, str):
                prompt = prompt.item()
            prompts.append(prompt)

            returns.append(float(np.asarray(item["returns"]).item()))

            images = item["image"]
            image_triplets.append(
                [
                    to_pil_image(images["base_0_rgb"]),
                    to_pil_image(images["left_wrist_0_rgb"]),
                    to_pil_image(images["right_wrist_0_rgb"]),
                ]
            )

        processor = self._get_processor()
        texts = []
        for prompt, triplet in zip(prompts, image_triplets, strict=True):
            messages = build_qwen_messages(prompt, num_images=len(triplet))
            if hasattr(processor, "apply_chat_template"):
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            else:
                text = prompt
            texts.append(text)

        processor_kwargs: dict[str, Any] = {
            "text": texts,
            "images": image_triplets,
            "return_tensors": "pt",
            "padding": True,
        }
        if self.max_len is not None:
            processor_kwargs["truncation"] = True
            processor_kwargs["max_length"] = self.max_len

        encoded = processor(**processor_kwargs)
        batch = {k: v for k, v in encoded.items() if isinstance(v, torch.Tensor)}
        batch["returns"] = torch.tensor(returns, dtype=torch.float32)
        return batch


def create_qwen_value_dataloader(
    hf_model_id: str,
    *,
    success_repo_ids: list[str] | None = None,
    failure_repo_ids: list[str] | None = None,
    batch_size: int = 64,
    failure_cost_json: str | pathlib.Path | None = None,
    default_c_fail: float = 100.0,
    success_sampling_ratio: float = 0.5,
    num_workers: int = 4,
    seed: int = 42,
    split: str = "train",
    train_split: float = 0.9,
    split_seed: int = 42,
    target_task: str | None = None,
    treat_other_tasks_as_failure: bool = False,
    max_token_len: int = 256,
    rank: int = 0,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader that emits multimodal Qwen3-VL batches plus returns."""

    dataset = ValueFunctionDataset(
        success_repo_ids=success_repo_ids,
        failure_repo_ids=failure_repo_ids,
        failure_cost_json=failure_cost_json,
        default_c_fail=default_c_fail,
        success_sampling_ratio=success_sampling_ratio,
        # Dataset sampling is RNG-driven in __getitem__, so offset per-rank seeds in DDP.
        seed=seed + rank * 10_000,
        split=split,
        train_split=train_split,
        split_seed=split_seed,
        target_task=target_task,
        treat_other_tasks_as_failure=treat_other_tasks_as_failure,
    )

    sampler = torch.utils.data.RandomSampler(
        dataset,
        replacement=True,
        num_samples=int(1e9),
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=QwenVLCollateFn(hf_model_id=hf_model_id, max_len=max_token_len),
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init_fn,
        prefetch_factor=4 if num_workers > 0 else None,
        multiprocessing_context="forkserver" if num_workers > 0 else None,
    )
