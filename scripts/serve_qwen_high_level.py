#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any
import urllib.error
import urllib.request

import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText
from transformers import AutoProcessor
from qwen_high_level_utils import (
    DEFAULT_CLASS_SPEC_PATH,
    describe_task_mode,
    infer_task_mode_from_prompt,
    load_class_map,
    parse_class_id,
    system_prompt,
)

from openpi.serving.websocket_policy_server import WebsocketPolicyServer

DEFAULT_CAMERA_ORDER = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")


def _to_pil(value: Any) -> Image.Image:
    arr = np.asarray(value)
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim != 3:
        raise ValueError(f"expected image with 3 dims, got shape={arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


class QwenHighLevelPolicy:
    def __init__(
        self,
        *,
        model_id: str,
        adapter: str | None,
        device: str,
        camera_order: tuple[str, ...],
        max_new_tokens: int,
        bottle_description: str,
        class_spec: str | None,
    ) -> None:
        self._model_id = model_id
        self._adapter = adapter
        self._device = device
        self._camera_order = camera_order
        self._max_new_tokens = max_new_tokens
        self._bottle_description = bottle_description
        self._class_map = load_class_map(class_spec)
        self._runtime_config_url = ""
        self._runtime_config_cache_ttl_s = 1.0
        self._last_runtime_config_fetch_s = 0.0
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=device,
            low_cpu_mem_usage=True,
        )
        if adapter:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, adapter)
        self._model.eval()

    def set_runtime_config_source(self, *, runtime_config_url: str, cache_ttl_s: float = 1.0) -> None:
        self._runtime_config_url = str(runtime_config_url or "").strip()
        self._runtime_config_cache_ttl_s = max(0.1, float(cache_ttl_s))

    def _refresh_class_map_from_runtime_config(self) -> None:
        if not self._runtime_config_url:
            return
        now = time.monotonic()
        if now - self._last_runtime_config_fetch_s < self._runtime_config_cache_ttl_s:
            return
        self._last_runtime_config_fetch_s = now
        try:
            with urllib.request.urlopen(self._runtime_config_url, timeout=2.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            logging.warning("Could not fetch qwen runtime config from %s; using cached class map.", self._runtime_config_url)
            return
        except Exception:
            logging.exception("Failed to fetch qwen runtime config from %s", self._runtime_config_url)
            return

        raw_pairs = payload.get("state_subtask_pairs")
        if not isinstance(raw_pairs, list) or not raw_pairs:
            return
        class_map: dict[int, tuple[str, str]] = {}
        for idx, item in enumerate(raw_pairs):
            bottle_state = ""
            subtask = ""
            if isinstance(item, dict):
                bottle_state = str(item.get("bottle_state") or "").strip()
                subtask = str(item.get("subtask") or "").strip()
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                bottle_state = str(item[0]).strip()
                subtask = str(item[1]).strip()
            if bottle_state and subtask:
                class_map[idx] = (bottle_state, subtask)
        if class_map:
            self._class_map = class_map

    def reset(self) -> None:
        pass

    def infer_subtask(
        self,
        obs: dict[str, Any],
        *,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        self._refresh_class_map_from_runtime_config()
        images_dict = obs.get("images")
        if not isinstance(images_dict, dict):
            raise ValueError("Qwen high-level policy requires obs['images'] dict.")
        selected: list[tuple[str, Image.Image]] = []
        for name in self._camera_order:
            if name in images_dict:
                selected.append((name, _to_pil(images_dict[name])))
        if not selected:
            raise ValueError(f"No images available for camera_order={self._camera_order}")

        task_mode = str(obs.get("task_mode") or "").strip().lower() or infer_task_mode_from_prompt(str(obs.get("prompt") or ""))
        camera_names = ", ".join(name for name, _ in selected)
        sys_prompt = system_prompt(self._class_map, task_mode)
        user_prompt = (
            f"Use all {len(selected)} images in this order: {camera_names}. "
            f"Known task mode: {describe_task_mode(task_mode)}. Classify the current scene."
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for _, image in selected]
                + [{"type": "text", "text": user_prompt}],
            },
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image for _, image in selected], return_tensors="pt")
        inputs = {key: value.to(self._device) if hasattr(value, "to") else value for key, value in inputs.items()}

        started_at = time.perf_counter()
        generate_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens or self._max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature
        with torch.inference_mode():
            output_ids = self._model.generate(**generate_kwargs)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        new_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        raw_text = self._processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
        class_id = parse_class_id(raw_text, self._class_map)
        if class_id is None or class_id not in self._class_map:
            raise RuntimeError(f"Qwen output did not contain a valid class id: {raw_text!r}")

        bottle_state, subtask = self._class_map[class_id]
        payload = {
            "bottle_description": self._bottle_description or None,
            "bottle_state": bottle_state,
            "subtask": subtask,
        }
        return {
            "subtask_text": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            "server_timing": {
                "provider": "qwen",
                "model": self._model_id,
                "adapter": self._adapter,
                "device": self._device,
                "num_images": len(selected),
                "selected_cameras": [name for name, _ in selected],
                "raw_text": raw_text,
                "class_id": class_id,
                "task_mode": task_mode,
                "infer_ms": round(elapsed_ms, 1),
            },
            "trace_payload": {
                "provider": "qwen",
                "model": self._model_id,
                "adapter": self._adapter,
                "selected_cameras": [name for name, _ in selected],
                "task_mode": task_mode,
                "system_text": sys_prompt,
                "user_prompt": user_prompt,
                "raw_text": raw_text,
                "class_id": class_id,
                "payload": payload,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument(
        "--adapter",
        default="output/qwen35-2b-twist-lora/v3-20260421-091009/checkpoint-797",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera-order", nargs="+", default=list(DEFAULT_CAMERA_ORDER))
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--bottle-description", default="")
    parser.add_argument("--class-spec", default=str(DEFAULT_CLASS_SPEC_PATH))
    parser.add_argument("--runtime-config-url", default="http://127.0.0.1:8011/api/runtime/config")
    parser.add_argument("--runtime-config-cache-ttl-s", type=float, default=1.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    policy = QwenHighLevelPolicy(
        model_id=args.model_id,
        adapter=args.adapter,
        device=args.device,
        camera_order=tuple(args.camera_order),
        max_new_tokens=args.max_new_tokens,
        bottle_description=args.bottle_description,
        class_spec=args.class_spec,
    )
    policy.set_runtime_config_source(
        runtime_config_url=args.runtime_config_url,
        cache_ttl_s=args.runtime_config_cache_ttl_s,
    )
    metadata = {
        "provider": "qwen",
        "model": args.model_id,
        "adapter": args.adapter,
        "camera_order": list(args.camera_order),
        "bottle_description": args.bottle_description,
        "class_spec": args.class_spec,
        "runtime_config_url": args.runtime_config_url,
    }
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
