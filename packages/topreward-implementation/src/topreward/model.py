from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from typing import Sequence

import time

import torch
from PIL import Image
import transformers
import warnings
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from transformers import AutoTokenizer
from transformers.models.auto.configuration_auto import CONFIG_MAPPING


_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


class TOPRewardModel:
    """Wrapper for Qwen3-VL log-prob extraction of the "True" token."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        trust_remote_code: bool = True,
        true_token: str = "True",
        use_chat_template: bool = False,
        *,
        model: Any | None = None,
        processor: Any | None = None,
        tokenizer: Any | None = None,
    ):
        if use_chat_template:
            raise ValueError("TOPReward must not use chat templates (paper reports severe VOC degradation).")

        self._check_model_compatibility(model_name)

        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.video_fps: float | None = None
        self._token_id_cache: dict[str, int] = {}

        torch_dtype = _DTYPE_MAP.get(dtype.lower())
        if torch_dtype is None:
            supported = ", ".join(sorted(_DTYPE_MAP))
            raise ValueError(f"Unsupported dtype '{dtype}'. Supported values: {supported}")
        self.torch_dtype = torch_dtype

        self.processor = processor or AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.tokenizer = tokenizer or getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)

        self.model = model
        if self.model is None:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            common_kwargs = {"dtype": self.torch_dtype, "trust_remote_code": trust_remote_code}
            requested_attn = attn_implementation

            try:
                if getattr(config, "model_type", None) == "qwen3_vl":
                    from transformers import Qwen3VLForConditionalGeneration

                    self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model_name,
                        attn_implementation=requested_attn,
                        **common_kwargs,
                    )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        attn_implementation=requested_attn,
                        **common_kwargs,
                    )
            except ImportError as exc:
                if requested_attn == "flash_attention_2" and "flash_attn" in str(exc):
                    fallback_attn = "sdpa"
                    warnings.warn(
                        "flash_attention_2 requested but flash_attn is unavailable. "
                        "Falling back to sdpa attention for compatibility.",
                        stacklevel=2,
                    )
                    if getattr(config, "model_type", None) == "qwen3_vl":
                        from transformers import Qwen3VLForConditionalGeneration

                        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                            model_name,
                            attn_implementation=fallback_attn,
                            **common_kwargs,
                        )
                    else:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            attn_implementation=fallback_attn,
                            **common_kwargs,
                        )
                    self.attn_implementation = fallback_attn
                else:
                    raise
            self.model.to(self.device)

        self.model.eval()
        self.true_token_id = self._resolve_true_token_id(true_token)

    def _check_model_compatibility(self, model_name: str) -> None:
        lower = model_name.lower()
        if "qwen3-vl" not in lower:
            return

        if "qwen3_vl" in CONFIG_MAPPING:
            return

        current = transformers.__version__
        raise RuntimeError(
            "This environment has transformers=="
            f"{current}, which does not support model_type='qwen3_vl'. "
            "Use a newer transformers build (for example 4.57.1+). "
            "If you use uv --no-sync, run: "
            "`uv run --no-sync --with transformers==4.57.1 python packages/topreward/scripts/evaluate_offline.py ...`. "
            "To make this permanent, update the workspace transformers pin in pyproject.toml and re-sync."
        )

    def _resolve_device(self, device: str) -> str:
        lowered = device.lower()
        if not lowered.startswith("cuda"):
            return device

        if not torch.cuda.is_available():
            warnings.warn("CUDA was requested but is not available. Falling back to CPU.", stacklevel=2)
            return "cpu"

        try:
            arch_list = set(torch.cuda.get_arch_list())
            capability = torch.cuda.get_device_capability(0)
            sm = f"sm_{capability[0]}{capability[1]}"
            if arch_list and sm not in arch_list:
                warnings.warn(
                    f"CUDA device capability {sm} is unsupported by this PyTorch build. "
                    "Falling back to CPU. Install a PyTorch build that supports this GPU architecture to use CUDA.",
                    stacklevel=2,
                )
                return "cpu"
        except Exception:  # noqa: BLE001
            pass

        return device

    def _resolve_token_id(self, token: str) -> int:
        cached = self._token_id_cache.get(token)
        if cached is not None:
            return cached

        token_ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Token '{token}' is not a single token for {self.model_name}. Tokenized ids: {token_ids}. "
                "Use a single-token answer token or adapt score extraction to multi-token decoding."
            )

        decoded = self.tokenizer.decode([token_ids[0]])
        if token.strip() not in decoded:
            raise ValueError(
                f"Decoded token mismatch for '{token}'. token_id={token_ids[0]}, decoded={decoded!r}."
            )
        token_id = int(token_ids[0])
        self._token_id_cache[token] = token_id
        return token_id

    def _resolve_true_token_id(self, token: str) -> int:
        return self._resolve_token_id(token)

    def _move_to_device(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in batch.items():
            if hasattr(value, "to"):
                output[key] = value.to(self.device)
            else:
                output[key] = value
        return output

    def _prepare_inputs(self, video_frames: Sequence[Image.Image], prompt: str) -> dict[str, Any]:
        if not video_frames:
            raise ValueError("video_frames must not be empty")

        if "<|video|>" in prompt and "<|video_pad|>" not in prompt:
            prompt = prompt.replace("<|video|>", "<|vision_start|><|video_pad|><|vision_end|>")

        # Qwen3-VL video preprocessors may require at least 2 temporal frames.
        # For tiny prefixes (t=1), duplicate the last frame to satisfy that constraint.
        video_inputs = list(video_frames)
        if len(video_inputs) < 2:
            video_inputs = [*video_inputs, video_inputs[-1]]

        video_kwargs: dict[str, Any] = {}
        if self.video_fps is not None:
            video_kwargs["fps"] = float(self.video_fps)

        attempts: list[tuple[str, dict[str, Any]]] = [
            (
                "videos_flat",
                {
                    "text": prompt,
                    "videos": video_inputs,
                    "do_sample_frames": False,
                    **video_kwargs,
                    "return_tensors": "pt",
                },
            ),
            (
                "videos_batched",
                {
                    "text": [prompt],
                    "videos": [video_inputs],
                    "do_sample_frames": False,
                    **video_kwargs,
                    "return_tensors": "pt",
                    "padding": True,
                },
            ),
            (
                "images_flat",
                {
                    "text": prompt,
                    "images": list(video_frames),
                    "return_tensors": "pt",
                },
            ),
            (
                "images_batched",
                {
                    "text": [prompt],
                    "images": [list(video_frames)],
                    "return_tensors": "pt",
                    "padding": True,
                },
            ),
        ]

        errors: list[str] = []
        for name, kwargs in attempts:
            try:
                batch = self.processor(**kwargs)
                if isinstance(batch, Mapping):
                    if len(batch) > 0:
                        return self._move_to_device(batch)
                    errors.append(f"{name}: processor returned empty mapping")
                else:
                    keys = list(batch.keys()) if hasattr(batch, "keys") else []
                    if keys:
                        mapping_batch = {key: batch[key] for key in keys}
                        return self._move_to_device(mapping_batch)
                    errors.append(f"{name}: processor returned unsupported/empty batch type {type(batch).__name__}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {exc}")

        msg = " ; ".join(errors)
        raise RuntimeError(
            "Unable to build model inputs via processor. "
            "Check Qwen3-VL processor input format for videos/images. "
            f"Tried variants: {msg}"
        )

    @torch.inference_mode()
    def get_log_prob(self, video_frames: Sequence[Image.Image], prompt: str) -> float:
        """Return log P("True" | video_frames, prompt)."""
        t0 = time.perf_counter()
        model_inputs = self._prepare_inputs(video_frames, prompt)
        t1 = time.perf_counter()
        outputs = self.model(**model_inputs, use_cache=False)
        t2 = time.perf_counter()
        logits = outputs.logits[:, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        n_tokens = model_inputs.get("input_ids", model_inputs.get("attention_mask", [None])).shape[-1]
        print(f"[timing]   prepare={t1-t0:.2f}s  forward={t2-t1:.2f}s  seq_len={n_tokens}  n_frames={len(video_frames)}")
        return float(log_probs[0, self.true_token_id].item())

    @torch.inference_mode()
    def get_log_prob_for_token(self, video_frames: Sequence[Image.Image], prompt: str, token: str) -> float:
        """Return log P(token | video_frames, prompt) for a single-token answer token."""
        token_id = self._resolve_token_id(token)
        t0 = time.perf_counter()
        model_inputs = self._prepare_inputs(video_frames, prompt)
        t1 = time.perf_counter()
        outputs = self.model(**model_inputs, use_cache=False)
        t2 = time.perf_counter()
        logits = outputs.logits[:, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        n_tokens = model_inputs.get("input_ids", model_inputs.get("attention_mask", [None])).shape[-1]
        print(f"[timing]   prepare={t1-t0:.2f}s  forward={t2-t1:.2f}s  seq_len={n_tokens}  n_frames={len(video_frames)}")
        return float(log_probs[0, token_id].item())

    @torch.inference_mode()
    def get_log_probs_for_tokens(
        self,
        video_frames: Sequence[Image.Image],
        prompt: str,
        tokens: Sequence[str],
    ) -> dict[str, float]:
        """Return log-probs for multiple single-token answer tokens in one forward pass."""
        if not tokens:
            raise ValueError("tokens must not be empty")

        token_ids = [self._resolve_token_id(token) for token in tokens]
        t0 = time.perf_counter()
        model_inputs = self._prepare_inputs(video_frames, prompt)
        t1 = time.perf_counter()
        outputs = self.model(**model_inputs, use_cache=False)
        t2 = time.perf_counter()
        logits = outputs.logits[:, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        n_tokens = model_inputs.get("input_ids", model_inputs.get("attention_mask", [None])).shape[-1]
        print(f"[timing]   prepare={t1-t0:.2f}s  forward={t2-t1:.2f}s  seq_len={n_tokens}  n_frames={len(video_frames)}")
        return {token: float(log_probs[0, token_id].item()) for token, token_id in zip(tokens, token_ids)}

    def get_log_prob_batch(self, videos: Sequence[Sequence[Image.Image]], prompts: Sequence[str]) -> list[float]:
        """Compute log-probs for a batch of videos/prompts."""
        if len(videos) != len(prompts):
            raise ValueError(f"videos and prompts must have same length. Got {len(videos)} and {len(prompts)}")

        return [self.get_log_prob(video_frames, prompt) for video_frames, prompt in zip(videos, prompts)]
