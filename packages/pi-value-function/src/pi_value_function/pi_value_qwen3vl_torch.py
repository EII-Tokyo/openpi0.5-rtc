"""PyTorch Pi value model using a Qwen3-VL backbone."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from pi_value_function.backbone_type import BACKBONE_QWEN3VL


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported backbone dtype: {dtype_name}")
    return mapping[dtype_name]


def _load_auto_processor(hf_model_id: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(hf_model_id, trust_remote_code=True)


def _from_pretrained_with_dtype(auto_cls, hf_model_id: str, dtype: torch.dtype):
    """Load HF model preferring `dtype=` and falling back for older transformers."""

    try:
        return auto_cls.from_pretrained(
            hf_model_id,
            dtype=dtype,
            trust_remote_code=True,
        )
    except TypeError as exc:
        if "unexpected keyword argument 'dtype'" not in str(exc):
            raise
        return auto_cls.from_pretrained(
            hf_model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )


def _load_qwen_backbone(
    hf_model_id: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    """Load a multimodal causal model, preferring dedicated VL auto classes."""

    errors: list[str] = []

    try:
        from transformers import AutoModelForImageTextToText

        model = _from_pretrained_with_dtype(AutoModelForImageTextToText, hf_model_id, dtype)
        return model.to(device)
    except Exception as exc:  # pragma: no cover - exercised in real environment
        errors.append(f"AutoModelForImageTextToText: {exc!s}")

    try:
        from transformers import AutoModelForVision2Seq

        model = _from_pretrained_with_dtype(AutoModelForVision2Seq, hf_model_id, dtype)
        return model.to(device)
    except Exception as exc:  # pragma: no cover - exercised in real environment
        errors.append(f"AutoModelForVision2Seq: {exc!s}")

    try:
        from transformers import AutoModelForCausalLM

        model = _from_pretrained_with_dtype(AutoModelForCausalLM, hf_model_id, dtype)
        return model.to(device)
    except Exception as exc:  # pragma: no cover - exercised in real environment
        errors.append(f"AutoModelForCausalLM: {exc!s}")

    joined = "\n".join(errors)
    raise RuntimeError(f"Failed to load Qwen3-VL backbone '{hf_model_id}':\n{joined}")


def _infer_hidden_size(backbone: nn.Module) -> int:
    config = getattr(backbone, "config", None)
    if config is None:
        raise ValueError("Backbone has no config; cannot infer hidden size")

    if hasattr(config, "hidden_size"):
        return int(config.hidden_size)

    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, "hidden_size"):
        return int(text_config.hidden_size)

    if hasattr(config, "d_model"):
        return int(config.d_model)

    raise ValueError("Could not infer hidden size from backbone config")


def build_qwen_messages(prompt: str, num_images: int = 3) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "image"} for _ in range(num_images)]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def to_pil_image(image: np.ndarray | torch.Tensor) -> Image.Image:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape={image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        # Supports either [0, 1] or [-1, 1].
        if image.min() < 0.0:
            image = (image + 1.0) / 2.0
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    return Image.fromarray(image)


class PiValueQwen3VLTorch(nn.Module):
    """Distributional value model with frozen Qwen3-VL backbone and trainable value head."""

    def __init__(
        self,
        config,
        *,
        device: torch.device | None = None,
        backbone: nn.Module | None = None,
        processor: Any | None = None,
    ):
        super().__init__()

        self.backbone_name = BACKBONE_QWEN3VL
        self.value_dims = int(config.value_dims)
        self.value_min = float(config.value_min)
        self.value_max = float(config.value_max)
        self.value_head_layers = int(getattr(config, "value_head_layers", 2))
        self.hf_model_id = str(getattr(config, "hf_model_id", "Qwen/Qwen3-VL-2B-Instruct"))
        self.backbone_dtype_name = str(getattr(config, "backbone_dtype", "bfloat16"))
        self.backbone_dtype = resolve_torch_dtype(self.backbone_dtype_name)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.processor = processor if processor is not None else _load_auto_processor(self.hf_model_id)
        self.backbone = (
            backbone
            if backbone is not None
            else _load_qwen_backbone(
                self.hf_model_id,
                dtype=self.backbone_dtype,
                device=self.device,
            )
        )
        self.backbone.to(self.device)

        hidden_size = _infer_hidden_size(self.backbone)

        if self.value_head_layers == 1:
            self.value_proj = nn.Linear(hidden_size, self.value_dims)
            self.value_mlp = None
        else:
            self.value_proj = None
            self.value_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, self.value_dims),
            )

        self.register_buffer(
            "value_support",
            torch.linspace(self.value_min, self.value_max, self.value_dims, dtype=torch.float32),
            persistent=False,
        )

        self._freeze_backbone()
        self.to(self.device)

    def _freeze_backbone(self) -> None:
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def prepare_inputs(
        self,
        prompts: Sequence[str],
        image_triplets: Sequence[Sequence[np.ndarray | torch.Tensor | Image.Image]],
    ) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        pil_images: list[list[Image.Image]] = []

        for prompt, triplet in zip(prompts, image_triplets, strict=True):
            prompt = prompt if isinstance(prompt, str) else str(prompt)
            messages = build_qwen_messages(prompt=prompt, num_images=len(triplet))
            if hasattr(self.processor, "apply_chat_template"):
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            else:
                text = prompt

            texts.append(text)
            pil_images.append([
                img if isinstance(img, Image.Image) else to_pil_image(img)
                for img in triplet
            ])

        encoded = self.processor(
            text=texts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        )
        return {k: v for k, v in encoded.items() if isinstance(v, torch.Tensor)}

    def _extract_last_hidden(self, outputs: Any, attention_mask: torch.Tensor | None) -> torch.Tensor:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is not None and len(hidden_states) > 0:
            last_hidden = hidden_states[-1]
        else:
            last_hidden = getattr(outputs, "last_hidden_state", None)

        if last_hidden is None:
            raise RuntimeError("Backbone output does not expose hidden states")

        if attention_mask is None:
            last_token_idx = torch.full(
                (last_hidden.shape[0],),
                fill_value=last_hidden.shape[1] - 1,
                device=last_hidden.device,
                dtype=torch.long,
            )
        else:
            last_token_idx = attention_mask.long().sum(dim=1) - 1
            last_token_idx = torch.clamp(last_token_idx, min=0)

        batch_idx = torch.arange(last_hidden.shape[0], device=last_hidden.device)
        return last_hidden[batch_idx, last_token_idx]

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        model_inputs = {
            key: value.to(self.device)
            for key, value in batch.items()
            if key != "returns" and isinstance(value, torch.Tensor)
        }

        attention_mask = model_inputs.get("attention_mask")

        with torch.no_grad():
            outputs = self.backbone(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        eos_hidden = self._extract_last_hidden(outputs, attention_mask)
        head = self.value_mlp if self.value_mlp is not None else self.value_proj
        if head is None:
            raise RuntimeError("Value head is not initialized")
        return head(eos_hidden.float())

    def discretize_returns(self, returns: torch.Tensor) -> torch.Tensor:
        normalized = (returns - self.value_min) / (self.value_max - self.value_min)
        bins = normalized * (self.value_dims - 1)
        bins = torch.clamp(torch.round(bins), 0, self.value_dims - 1)
        return bins.long()

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        returns: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if returns is None:
            if "returns" not in batch:
                raise ValueError("Returns must be provided either as argument or in batch['returns']")
            returns = batch["returns"]

        returns = returns.to(self.device, dtype=torch.float32)
        logits = self.forward(batch)
        target_bins = self.discretize_returns(returns)
        return F.cross_entropy(logits, target_bins, reduction="none")

    @torch.no_grad()
    def predict_value(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.forward(batch)
        probs = torch.softmax(logits, dim=-1)
        support = self.value_support.to(logits.device)
        return torch.sum(probs * support, dim=-1)
