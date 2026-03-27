from __future__ import annotations

import asyncio
import base64
import io
import json
from dataclasses import dataclass
from typing import Any

from PIL import Image
import websockets

from topreward.serving.real_time import RealTimeScorer


@dataclass
class WebSocketBridgeConfig:
    uri: str
    frame_field: str = "frame"
    outgoing_type: str = "progress_score"


class ProgressWebSocketBridge:
    """Bridge incoming websocket frames to TOPReward real-time scores."""

    def __init__(self, scorer: RealTimeScorer, config: WebSocketBridgeConfig):
        self.scorer = scorer
        self.config = config

    def _decode_frame_payload(self, payload: Any) -> Image.Image | None:
        if isinstance(payload, bytes):
            return Image.open(io.BytesIO(payload)).convert("RGB")

        if not isinstance(payload, str):
            return None

        try:
            message = json.loads(payload)
        except json.JSONDecodeError:
            return None

        frame_b64 = message.get(self.config.frame_field)
        if not isinstance(frame_b64, str):
            return None

        data = base64.b64decode(frame_b64)
        return Image.open(io.BytesIO(data)).convert("RGB")

    async def run_forever(self) -> None:
        while True:
            try:
                async with websockets.connect(self.config.uri, max_size=None) as ws:
                    async for message in ws:
                        frame = self._decode_frame_payload(message)
                        if frame is None:
                            continue

                        result = self.scorer.on_frame(frame)
                        if not result:
                            continue

                        await ws.send(
                            json.dumps(
                                {
                                    "type": self.config.outgoing_type,
                                    "progress": result["progress"],
                                    "raw_score": result["raw_score"],
                                    "frame_idx": result["frame_idx"],
                                    "model_frames": result["model_frames"],
                                }
                            )
                        )
            except (OSError, websockets.WebSocketException):
                await asyncio.sleep(1.0)
