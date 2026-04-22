#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText
from transformers import AutoProcessor

from openpi.serving.websocket_policy_server import WebsocketPolicyServer


CLASS_MAP: dict[int, tuple[str, str]] = {
    0: ("Bottle on table, opening faces left", "Rotate so opening faces right"),
    1: ("Bottle on table, opening faces right", "Pick up with left hand"),
    2: ("Bottle in left hand and capped", "Unscrew cap"),
    3: ("Bottle in left hand, cap removed, and cap in right hand", "Bottle to left trash bin, cap to right trash bin"),
    4: ("Bottle in left hand, cap removed, and cap not in right hand", "Bottle to left trash bin"),
    5: ("Bottle in left hand and upside down", "Bottle to left trash bin"),
    6: ("Bottle stuck in left hand", "Use right hand to remove and place into left trash bin"),
    7: ("Cap on table", "Pick up cap and place into right trash bin"),
    8: ("No bottle on table", "Return to initial pose"),
}

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


def _system_prompt() -> str:
    lines = [
        "Classify the robot scene into exactly one of 9 classes.",
        "Output exactly one character: 0 or 1 or 2 or 3 or 4 or 5 or 6 or 7 or 8.",
        "Do not output words.",
        "Do not output punctuation.",
        "Do not explain your answer.",
        "Classes:",
    ]
    for cid, (state, subtask) in CLASS_MAP.items():
        lines.append(f"{cid}: state={state}; action={subtask}")
    return "\n".join(lines)


def _parse_class_id(raw_text: str) -> int | None:
    for ch in str(raw_text):
        if ch in "012345678":
            return int(ch)
    return None


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
    ) -> None:
        self._model_id = model_id
        self._adapter = adapter
        self._device = device
        self._camera_order = camera_order
        self._max_new_tokens = max_new_tokens
        self._bottle_description = bottle_description
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
        self._system_prompt = _system_prompt()

    def reset(self) -> None:
        pass

    def infer_subtask(
        self,
        obs: dict[str, Any],
        *,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        images_dict = obs.get("images")
        if not isinstance(images_dict, dict):
            raise ValueError("Qwen high-level policy requires obs['images'] dict.")
        selected: list[tuple[str, Image.Image]] = []
        for name in self._camera_order:
            if name in images_dict:
                selected.append((name, _to_pil(images_dict[name])))
        if not selected:
            raise ValueError(f"No images available for camera_order={self._camera_order}")

        camera_names = ", ".join(name for name, _ in selected)
        user_prompt = f"Use all {len(selected)} images in this order: {camera_names}. Classify the current scene."
        messages = [
            {"role": "system", "content": self._system_prompt},
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
        class_id = _parse_class_id(raw_text)
        if class_id is None or class_id not in CLASS_MAP:
            raise RuntimeError(f"Qwen output did not contain a class id 0-8: {raw_text!r}")

        bottle_state, subtask = CLASS_MAP[class_id]
        payload = {
            "bottle_description": self._bottle_description,
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
                "infer_ms": round(elapsed_ms, 1),
            },
            "trace_payload": {
                "provider": "qwen",
                "model": self._model_id,
                "adapter": self._adapter,
                "selected_cameras": [name for name, _ in selected],
                "system_text": self._system_prompt,
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
    parser.add_argument("--bottle-description", default="clear bottle with white label")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    policy = QwenHighLevelPolicy(
        model_id=args.model_id,
        adapter=args.adapter,
        device=args.device,
        camera_order=tuple(args.camera_order),
        max_new_tokens=args.max_new_tokens,
        bottle_description=args.bottle_description,
    )
    metadata = {
        "provider": "qwen",
        "model": args.model_id,
        "adapter": args.adapter,
        "camera_order": list(args.camera_order),
        "bottle_description": args.bottle_description,
    }
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
