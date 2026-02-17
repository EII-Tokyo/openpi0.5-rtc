"""Backbone identifiers for Pi value models."""

from __future__ import annotations

from typing import Final
from typing import Literal


BACKBONE_SIGLIP_GEMMA3: Final[str] = "siglip_gemma3"
BACKBONE_QWEN3VL: Final[str] = "qwen3vl"

BackboneType = Literal["siglip_gemma3", "qwen3vl"]


def is_qwen_backbone(backbone: str) -> bool:
    return backbone == BACKBONE_QWEN3VL
