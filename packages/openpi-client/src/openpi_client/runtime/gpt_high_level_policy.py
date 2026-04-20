import base64
import io
import json
import logging
import os
import time
import uuid
import urllib.error
import urllib.request
from typing import Any

import numpy as np
from PIL import Image

from openpi_client import websocket_client_policy as _websocket_client_policy

CAMERA_ORDER = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def _to_hwc_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim != 3:
        raise ValueError(f"expected image with 3 dims, got shape {arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def _jpeg_data_url(image: Any, *, quality: int = 90) -> str:
    arr = _to_hwc_uint8(image)
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_camera_subset(images: dict[str, Any], image_mode: str) -> list[tuple[str, Any]]:
    if image_mode == "high_only":
        keys = ("cam_high",)
    else:
        keys = CAMERA_ORDER
    out: list[tuple[str, Any]] = []
    for key in keys:
        if key in images:
            out.append((key, images[key]))
    return out


def _json_dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _extract_responses_output_text(response_body: dict[str, Any]) -> str:
    output_text = response_body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = response_body.get("output")
    if not isinstance(output, list):
        raise RuntimeError("Responses API payload does not contain output text.")

    text_chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "output_text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    text_chunks.append(text.strip())
    if text_chunks:
        return "\n".join(text_chunks)
    raise RuntimeError("Responses API output did not include any output_text blocks.")


def _extract_chat_output_text(response_body: dict[str, Any]) -> str:
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Chat Completions payload does not contain choices.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Chat Completions payload does not contain a message.")
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        text_chunks: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                text_chunks.append(text.strip())
        if text_chunks:
            return "\n".join(text_chunks)
    raise RuntimeError("Chat Completions payload did not include message content.")


def _default_api_key(api_base: str, api_key: str | None) -> str:
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()
    if "api.openai.com" in api_base:
        return os.getenv("OPENAI_API_KEY", "")
    return os.getenv("OPENAI_API_KEY", "EMPTY")


def _default_api_mode(api_base: str, api_mode: str | None) -> str:
    if api_mode in {"responses", "chat_completions"}:
        return api_mode
    if "api.openai.com" in api_base:
        return "responses"
    return "chat_completions"


class GptHighLevelPolicy:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-5.4",
        image_mode: str = "all_cameras",
        api_base: str | None = None,
        api_mode: str | None = None,
    ) -> None:
        self._api_base = (api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com")).rstrip("/")
        self._api_key = _default_api_key(self._api_base, api_key)
        self._api_mode = _default_api_mode(self._api_base, api_mode or os.getenv("OPENAI_API_MODE"))
        self._model = model
        self._image_mode = image_mode
        self._state_subtask_pairs: list[tuple[str, str]] = []
        self._start_subtasks: tuple[str, ...] = ()

    def set_config(
        self,
        *,
        model: str | None = None,
        image_mode: str | None = None,
        api_base: str | None = None,
        api_mode: str | None = None,
    ) -> None:
        if isinstance(model, str) and model.strip():
            self._model = model.strip()
        if image_mode in {"high_only", "all_cameras"}:
            self._image_mode = image_mode
        if isinstance(api_base, str) and api_base.strip():
            self._api_base = api_base.strip().rstrip("/")
            self._api_key = _default_api_key(self._api_base, self._api_key)
        if api_mode in {"responses", "chat_completions"}:
            self._api_mode = api_mode

    def set_low_level_schema(self, *, state_subtask_pairs: list[tuple[str, str]], start_subtasks: tuple[str, ...]) -> None:
        self._state_subtask_pairs = list(state_subtask_pairs)
        self._start_subtasks = tuple(start_subtasks)

    def reset(self) -> None:
        pass

    def infer_subtask(self, obs: dict[str, Any]) -> dict[str, Any]:
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured for GPT high-level policy.")
        images = obs.get("images")
        if not isinstance(images, dict) or not images:
            raise ValueError("GPT high-level policy requires obs['images'].")

        selected_images = _extract_camera_subset(images, self._image_mode)
        if not selected_images:
            raise ValueError(f"No images available for image_mode={self._image_mode}")

        allowed_states = sorted({bottle_state for bottle_state, _ in self._state_subtask_pairs})
        allowed_subtasks = sorted({subtask for _, subtask in self._state_subtask_pairs})
        runtime_context = obs.get("runtime_context")
        prompt = str(obs.get("prompt") or "").strip()
        payload_text = {
            "task": prompt,
            "current_bottle": (
                runtime_context.get("locked_bottle_description")
                if isinstance(runtime_context, dict)
                else None
            ),
            "previous_result": (
                runtime_context.get("previous_structured_result")
                if isinstance(runtime_context, dict)
                else None
            ),
            "allowed_pairs": [
                {"state": bottle_state, "subtask": subtask}
                for bottle_state, subtask in self._state_subtask_pairs
            ],
        }

        system_text = f"""# ALOHA Bottle High-Level Classifier

You are the high-level classifier for an ALOHA robot bottle task.

Return strict JSON only with exactly these keys:
- `bottle_description`
- `bottle_state`
- `subtask`

Allowed `(bottle_state, subtask)` pairs:
{json.dumps(payload_text["allowed_pairs"], ensure_ascii=False, indent=2)}

## 1. Primary Rule: Always Reason About `current_bottle` First

If `current_bottle` is not null, you must treat that bottle as the active bottle currently being processed.

- Do not switch to a different bottle while `current_bottle` is still visible on the table or still being held by the robot.
- Do not generate a new `bottle_description` while the current bottle is still being processed.
- Use `previous_result` only as supporting context. The current images are the source of truth.

## 2. If `current_bottle` Is Still On The Table

If the bottle described by `current_bottle` is still on the work table, decide only between:

- `Bottle on table, opening faces left` -> `Rotate so opening faces right`
- `Bottle on table, opening faces right` -> `Pick up with left hand`

Rules:

- Use `cam_high` as the main view to judge whether the opening of `current_bottle` faces left or right.
- The left/right judgment must be about the exact same bottle described by `current_bottle`.
- If the opening faces left, you must choose `Rotate so opening faces right`.
- Only if the opening faces right may you choose `Pick up with left hand`.

## 3. If `current_bottle` Is In The Left Hand

If the robot is holding any bottle in the left hand, you do not need to use `current_bottle` for this step.
At this stage, simply decide:

- Is the bottle upside down?
- Does the bottle still have a cap?

Rules:

- Use `cam_high` together with `cam_right_wrist` when available to judge upside-down versus upright and capped versus uncapped.
- Orientation has priority over cap detection: first decide whether the left-hand bottle is upside down, then decide cap state only if it is upright.
- If the left-hand bottle is upside down, you must produce:
  `Bottle in left hand and upside down` -> `Bottle to left trash bin`
- Never output `Unscrew cap` for an upside-down bottle, even if a cap is visible.
- Only an upright capped bottle may produce:
  `Bottle in left hand and capped` -> `Unscrew cap`
- If the left-hand bottle no longer has a cap, you must not output `Unscrew cap`.
- If the cap is removed, choose the correct cap-removed disposal state depending on whether the cap is in the right hand.
- Do not confuse an upside-down bottle in the left hand with `Bottle stuck in left hand`.
- `Bottle stuck in left hand` is only valid when the left gripper has already moved forward to the front trash-bin area, is at or next to the left trash bin, and the robot has just attempted to discard the bottle but the bottle visibly remains stuck in the left gripper.
- If the left gripper is not at the front trash-bin area, never output `Bottle stuck in left hand`; classify the held bottle by orientation and cap state instead.

## 4. Only After The Current Bottle Is Finished May A New Bottle Start

Only when `current_bottle` has finished processing may you introduce a new bottle.

Rules:

- Ignore bottles or caps that are already inside the three front trash bins.
- Only table bottles and robot-held bottles matter.
- Treat the current bottle as finished only when the robot has actually released it or its cap into a trash bin.
- Use the latest images to judge whether the gripper is positioned above a trash bin and whether the gripper has opened to release the object.
- If the gripper is above a bin, the gripper is open, and the bottle or cap has already dropped away, then the current bottle handling is finished and a new bottle may start.
- If release has not visibly happened yet, do not start a new bottle.
- If the current bottle has been finished and a different bottle should start next, then generate a new short `bottle_description` for that next bottle.
- If no new bottle is starting yet, set `bottle_description` to `null`.
- Never describe one bottle and classify a different bottle.
"""

        user_content: list[dict[str, Any]] = [{"type": "input_text", "text": _json_dumps(payload_text)}]
        for camera_name, image in selected_images:
            user_content.append({"type": "input_text", "text": f"camera={camera_name}"})
            user_content.append({"type": "input_image", "image_url": _jpeg_data_url(image)})

        output_schema = {
            "type": "object",
            "properties": {
                "bottle_description": {
                    "anyOf": [
                        {"type": "string", "minLength": 1},
                        {"type": "null"},
                    ]
                },
                "bottle_state": {
                    "type": "string",
                    "enum": allowed_states,
                },
                "subtask": {
                    "type": "string",
                    "enum": allowed_subtasks,
                },
            },
            "required": ["bottle_description", "bottle_state", "subtask"],
            "additionalProperties": False,
        }

        request_payload: dict[str, Any]
        endpoint_path: str
        if self._api_mode == "chat_completions":
            chat_user_content: list[dict[str, Any]] = []
            for item in user_content:
                if item["type"] == "input_text":
                    chat_user_content.append({"type": "text", "text": item["text"]})
                else:
                    chat_user_content.append({"type": "image_url", "image_url": {"url": item["image_url"]}})
            request_payload = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": chat_user_content},
                ],
                "temperature": 0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "aloha_high_level_subtask",
                        "strict": True,
                        "schema": output_schema,
                    },
                },
            }
            endpoint_path = "/v1/chat/completions"
        else:
            request_payload = {
                "model": self._model,
                "input": [
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_text}],
                    },
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0,
                "store": True,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "aloha_high_level_subtask",
                        "strict": True,
                        "schema": output_schema,
                    }
                },
            }
            endpoint_path = "/v1/responses"

        body = json.dumps(request_payload).encode("utf-8")
        client_request_id = f"aloha-high-level-{uuid.uuid4()}"
        headers = {
            "Content-Type": "application/json",
            "X-Client-Request-Id": client_request_id,
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(
            f"{self._api_base}{endpoint_path}",
            data=body,
            method="POST",
            headers=headers,
        )
        started_at = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                response_body = response.read()
                request_id = response.headers.get("x-request-id")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"High-level API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"High-level API request failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0

        parsed = json.loads(response_body.decode("utf-8"))
        if self._api_mode == "chat_completions":
            message_content = _extract_chat_output_text(parsed)
        else:
            message_content = _extract_responses_output_text(parsed)
        selected_image_payloads = {
            camera_name: _jpeg_data_url(image)
            for camera_name, image in selected_images
        }
        response_id = parsed.get("id")

        logging.info(
            "High-level inference ok api_mode=%s model=%s image_mode=%s cameras=%s elapsed_ms=%.1f request_id=%s client_request_id=%s",
            self._api_mode,
            self._model,
            self._image_mode,
            [name for name, _ in selected_images],
            elapsed_ms,
            request_id,
            client_request_id,
        )
        return {
            "subtask_text": message_content,
            "server_timing": {
                "provider": "gpt",
                "endpoint": endpoint_path.removeprefix("/v1/"),
                "model": self._model,
                "image_mode": self._image_mode,
                "api_base": self._api_base,
                "api_mode": self._api_mode,
                "num_images": len(selected_images),
                "elapsed_ms": round(elapsed_ms, 1),
                "response_id": response_id,
                "request_id": request_id,
                "client_request_id": client_request_id,
            },
            "trace_payload": {
                "provider": "gpt",
                "endpoint": endpoint_path.removeprefix("/v1/"),
                "model": self._model,
                "image_mode": self._image_mode,
                "api_base": self._api_base,
                "api_mode": self._api_mode,
                "selected_cameras": [name for name, _ in selected_images],
                "payload_text": payload_text,
                "system_text": system_text,
                "images": selected_image_payloads,
                "response_id": response_id,
                "request_id": request_id,
                "client_request_id": client_request_id,
                "response_body": parsed,
            },
        }


class RoutedHighLevelPolicy:
    def __init__(
        self,
        *,
        service_host: str,
        service_port: int,
        source: str = "gpt",
        gpt_model: str = "gpt-5.4",
        gpt_image_mode: str = "all_cameras",
    ) -> None:
        self._service_host = service_host
        self._service_port = service_port
        self._source = source if source in {"gpt", "service"} else "gpt"
        self._service_policy = None
        self._gpt_policy = GptHighLevelPolicy(model=gpt_model, image_mode=gpt_image_mode)
        self._state_subtask_pairs: list[tuple[str, str]] = []
        self._start_subtasks: tuple[str, ...] = ()

    def set_config(
        self,
        *,
        source: str | None = None,
        gpt_model: str | None = None,
        gpt_image_mode: str | None = None,
    ) -> None:
        if source in {"gpt", "service"}:
            self._source = source
        self._gpt_policy.set_config(model=gpt_model, image_mode=gpt_image_mode)

    def set_low_level_schema(self, *, state_subtask_pairs: list[tuple[str, str]], start_subtasks: tuple[str, ...]) -> None:
        self._state_subtask_pairs = list(state_subtask_pairs)
        self._start_subtasks = tuple(start_subtasks)
        self._gpt_policy.set_low_level_schema(
            state_subtask_pairs=self._state_subtask_pairs,
            start_subtasks=self._start_subtasks,
        )

    def reset(self) -> None:
        if self._service_policy is not None:
            self._service_policy.reset()
        self._gpt_policy.reset()

    def get_server_metadata(self) -> dict[str, Any]:
        return {
            "provider": self._source,
            "service_host": self._service_host,
            "service_port": self._service_port,
        }

    def _get_service_policy(self):
        if self._service_policy is None:
            self._service_policy = _websocket_client_policy.WebsocketClientPolicy(
                host=self._service_host,
                port=self._service_port,
                wait_for_server=False,
            )
        return self._service_policy

    def infer_subtask(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self._source == "service":
            try:
                result = self._get_service_policy().infer_subtask(obs)
            except Exception as exc:
                logging.warning(
                    "High-level service unavailable at ws://%s:%s, fallback to GPT: %s",
                    self._service_host,
                    self._service_port,
                    exc,
                )
                self._service_policy = None
                return self._gpt_policy.infer_subtask(obs)
            timing = result.get("server_timing", {})
            if isinstance(timing, dict):
                timing = {**timing, "provider": "service"}
            else:
                timing = {"provider": "service"}
            result["server_timing"] = timing
            return result
        return self._gpt_policy.infer_subtask(obs)
