"""Torch serving policy for PiValue with Qwen3-VL backbone."""

from __future__ import annotations

from typing import Dict
from typing import Optional

import numpy as np
import torch

from openpi_client import base_policy
from pi_value_function.pi_value_qwen3vl_torch import PiValueQwen3VLTorch
from pi_value_function.pi_value_qwen3vl_torch import to_pil_image


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    return bool(value)


class ValuePolicyTorch(base_policy.BasePolicy):
    def __init__(
        self,
        model: PiValueQwen3VLTorch,
        *,
        return_distribution: bool = False,
        device: torch.device | None = None,
    ):
        self._model = model
        self._model.eval()
        self._return_distribution_default = return_distribution
        self._device = device or model.device

    def _extract_image_triplet(self, obs: Dict) -> list:
        image_aliases = {
            "base_0_rgb": ["observation/exterior_image_1_left", "exterior_image_1_left", "left_image", "base_0_rgb"],
            "left_wrist_0_rgb": ["observation/wrist_image_left", "wrist_image_left", "wrist_image", "left_wrist_0_rgb"],
            "right_wrist_0_rgb": ["observation/exterior_image_2_left", "exterior_image_2_left", "right_image", "right_wrist_0_rgb"],
        }

        output = []
        for model_key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
            selected = None
            for key in image_aliases[model_key]:
                if key in obs:
                    selected = obs[key]
                    break

            if selected is None:
                selected = np.zeros((224, 224, 3), dtype=np.uint8)

            if isinstance(selected, np.ndarray) and selected.ndim == 4:
                selected = selected[0]
            output.append(to_pil_image(selected))

        return output

    def infer(
        self,
        obs: Dict,
        prev_action: Optional[Dict] = None,
        use_rtc: bool = False,
        return_distribution: Optional[bool] = None,
    ) -> Dict:
        del prev_action, use_rtc

        prompt = obs.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0]
        if not isinstance(prompt, str):
            prompt = str(prompt)

        obs_return_distribution = obs.get("return_distribution")
        if obs_return_distribution is None:
            obs_return_distribution = obs.get("observation/return_distribution")

        if return_distribution is None:
            if obs_return_distribution is None:
                return_distribution = self._return_distribution_default
            else:
                return_distribution = _coerce_bool(obs_return_distribution)

        image_triplet = self._extract_image_triplet(obs)
        batch = self._model.prepare_inputs([prompt], [image_triplet])
        batch = {k: v.to(self._device) for k, v in batch.items()}

        with torch.no_grad():
            if return_distribution:
                logits = self._model.forward(batch)
                probs = torch.softmax(logits, dim=-1)
                support = self._model.value_support.to(probs.device)
                value = torch.sum(probs * support, dim=-1)
                return {
                    "value": float(value[0].cpu()),
                    "distribution": probs[0].cpu().tolist(),
                }

            value = self._model.predict_value(batch)
            return {"value": float(value[0].cpu())}
