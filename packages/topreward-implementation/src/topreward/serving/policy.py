from __future__ import annotations

from typing import Any
from typing import Optional

import numpy as np
from openpi_client import base_policy
from PIL import Image

from topreward.model import TOPRewardModel
from topreward.serving.real_time import RealTimeConfig
from topreward.serving.real_time import RealTimeScorer


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


def _extract_nested(container: dict[str, Any], key: str) -> Any:
    aliases = [key, key.replace("/", "."), key.replace(".", "/")]
    for alias in aliases:
        if alias in container:
            return container[alias]

    for sep in ("/", "."):
        parts = key.split(sep)
        cursor: Any = container
        found = True
        for part in parts:
            if isinstance(cursor, dict) and part in cursor:
                cursor = cursor[part]
            else:
                found = False
                break
        if found:
            return cursor

    return None


def _looks_like_image(value: Any) -> bool:
    if isinstance(value, Image.Image):
        return True

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()

    try:
        array = np.asarray(value)
    except Exception:  # noqa: BLE001
        return False

    return array.ndim in (2, 3)


class TOPRewardPolicy(base_policy.BasePolicy):
    """Websocket server policy that returns TOPReward progress scores."""

    def __init__(
        self,
        model: TOPRewardModel,
        instruction: str = "",
        camera_key: str = "observation/exterior_image_1_left",
        fallback_camera_keys: list[str] | None = None,
        config: RealTimeConfig | None = None,
        reset_on_prompt_change: bool = True,
    ):
        self._default_instruction = instruction.strip()
        self._camera_key = camera_key
        self._fallback_camera_keys = fallback_camera_keys or [
            "observation/exterior_image_1_left",
            "observation/wrist_image_left",
            "observation/exterior_image_2_left",
            "observation.images.top",
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ]
        self._reset_on_prompt_change = reset_on_prompt_change

        self._scorer = RealTimeScorer(model, self._default_instruction, config or RealTimeConfig())
        self._last_instruction = self._default_instruction
        self._last_progress = 0.0
        self._last_raw_score = 0.0

    def _extract_instruction(self, obs: dict[str, Any]) -> str:
        for key in (
            "prompt",
            "instruction",
            "language_instruction",
            "task_description",
            "observation/prompt",
            "observation/instruction",
        ):
            value = _extract_nested(obs, key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _resolve_frame(self, obs: dict[str, Any]) -> tuple[Any | None, str | None]:
        keys = [self._camera_key, *self._fallback_camera_keys]
        seen: set[str] = set()
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            value = _extract_nested(obs, key)
            if value is not None:
                return value, key

        for key, value in obs.items():
            if not isinstance(key, str):
                continue
            lowered = key.lower()
            if any(token in lowered for token in ("image", "rgb", "camera", "cam_")) and _looks_like_image(value):
                return value, key

        observation = obs.get("observation")
        if isinstance(observation, dict):
            images = observation.get("images")
            if isinstance(images, dict):
                for key, value in images.items():
                    if _looks_like_image(value):
                        return value, f"observation/images/{key}"

        return None, None

    def _wants_reset(self, obs: dict[str, Any]) -> bool:
        for key in (
            "reset",
            "observation/reset",
            "episode_start",
            "observation/episode_start",
            "new_episode",
            "observation/new_episode",
        ):
            value = _extract_nested(obs, key)
            if value is not None and _coerce_bool(value):
                return True
        return False

    def _response(
        self,
        *,
        status: str,
        camera_key: str | None,
        frame_idx: int | None = None,
        model_frames: int | None = None,
        progress: float | None = None,
        raw_score: float | None = None,
    ) -> dict[str, Any]:
        current_progress = float(self._last_progress if progress is None else progress)
        current_raw = float(self._last_raw_score if raw_score is None else raw_score)
        return {
            "value": current_progress,
            "progress": current_progress,
            "raw_score": current_raw,
            "status": status,
            "instruction": self._scorer.instruction,
            "camera_key": camera_key,
            "frame_idx": frame_idx,
            "model_frames": model_frames,
        }

    def infer(
        self,
        obs: dict[str, Any],
        prev_action: Optional[dict[str, Any]] = None,
        use_rtc: bool = False,
    ) -> dict[str, Any]:
        del prev_action, use_rtc

        if not isinstance(obs, dict):
            return self._response(status="invalid_obs", camera_key=None)

        instruction = self._extract_instruction(obs)
        if not instruction:
            instruction = self._default_instruction

        should_reset = self._wants_reset(obs)
        if instruction and self._reset_on_prompt_change and instruction != self._last_instruction:
            should_reset = True

        if should_reset:
            self._scorer.reset(new_instruction=instruction)
        elif instruction and instruction != self._scorer.instruction:
            self._scorer.instruction = instruction

        self._last_instruction = instruction

        frame_payload, used_camera_key = self._resolve_frame(obs)
        if frame_payload is None:
            return self._response(
                status="missing_frame",
                camera_key=None,
                frame_idx=len(self._scorer.raw_frame_buffer),
                model_frames=len(self._scorer.model_frame_buffer),
            )

        values = frame_payload if isinstance(frame_payload, list) else [frame_payload]
        result: dict[str, Any] | None = None
        for frame in values:
            if not _looks_like_image(frame):
                continue
            result = self._scorer.on_frame(frame)

        if result is None:
            return self._response(
                status="buffering",
                camera_key=used_camera_key,
                frame_idx=len(self._scorer.raw_frame_buffer),
                model_frames=len(self._scorer.model_frame_buffer),
            )

        self._last_progress = float(result["progress"])
        self._last_raw_score = float(result["raw_score"])

        return self._response(
            status="scored",
            camera_key=used_camera_key,
            frame_idx=int(result["frame_idx"]),
            model_frames=int(result["model_frames"]),
            progress=self._last_progress,
            raw_score=self._last_raw_score,
        )
