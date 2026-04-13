from __future__ import annotations

from typing import Any
import logging

from pymongo import MongoClient

from .config import settings
from .low_level_subtask_defaults import DEFAULT_STATE_SUBTASK_PAIRS, DEFAULT_SUBTASK_CATALOG
from .schemas import (
    RuntimeConfigPayload,
    StateSubtaskPairPayload,
    SubtaskCatalogEntryPayload,
)


def _clamp_camera_refresh_ms(raw: Any) -> int:
    try:
        return max(100, int(raw))
    except (TypeError, ValueError):
        return 100


class RuntimeConfigStore:
    def __init__(self) -> None:
        self._client = MongoClient(
            settings.mongo_uri,
            serverSelectionTimeoutMS=1000,
            connectTimeoutMS=1000,
            socketTimeoutMS=1000,
        )
        self._collection = self._client[settings.mongo_db][settings.mongo_runtime_config_collection]

    def load(self) -> RuntimeConfigPayload:
        try:
            doc = self._collection.find_one({"_id": "default"}) or {}
            ann = doc.get("announcement_language", "zh")
            announcement_language = "ja" if ann == "ja" else "zh"
            ui = doc.get("ui_language", "en")
            ui_language = ui if ui in ("en", "ja", "zh") else "en"
            raw_cat = doc.get("subtask_catalog")
            if isinstance(raw_cat, list) and len(raw_cat) > 0:
                subtask_catalog = [SubtaskCatalogEntryPayload.model_validate(x) for x in raw_cat]
            else:
                subtask_catalog = [SubtaskCatalogEntryPayload.model_validate(x) for x in DEFAULT_SUBTASK_CATALOG]

            raw_pairs = doc.get("state_subtask_pairs")
            state_subtask_pairs: list[StateSubtaskPairPayload] = []
            if isinstance(raw_pairs, list) and len(raw_pairs) > 0:
                for p in raw_pairs:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        state_subtask_pairs.append(
                            StateSubtaskPairPayload(bottle_state=str(p[0]).strip(), subtask=str(p[1]).strip())
                        )
                    elif isinstance(p, dict):
                        state_subtask_pairs.append(StateSubtaskPairPayload.model_validate(p))
            if not state_subtask_pairs:
                state_subtask_pairs = [
                    StateSubtaskPairPayload(bottle_state=a, subtask=b) for a, b in DEFAULT_STATE_SUBTASK_PAIRS
                ]

            return RuntimeConfigPayload(
                dataset_dir=str(doc.get("dataset_dir") or ""),
                manual_dataset_dir=str(doc.get("manual_dataset_dir") or ""),
                include_bottle_description=bool(doc.get("include_bottle_description", True)),
                lock_bottle_description=bool(doc.get("lock_bottle_description", True)),
                include_bottle_position=bool(doc.get("include_bottle_position", False)),
                include_bottle_state=bool(doc.get("include_bottle_state", True)),
                include_subtask=bool(doc.get("include_subtask", True)),
                forced_low_level_subtask=(str(doc["forced_low_level_subtask"]).strip() if doc.get("forced_low_level_subtask") else None),
                video_memory_num_frames=4 if doc.get("video_memory_num_frames") == 4 else 1,
                high_level_source="service" if doc.get("high_level_source") == "service" else "gpt",
                gpt_model=str(doc.get("gpt_model") or "gpt-5.4"),
                gpt_image_mode="high_only" if doc.get("gpt_image_mode") == "high_only" else "all_cameras",
                announcement_language=announcement_language,
                api_base=str(doc.get("api_base") or ""),
                ws_base=str(doc.get("ws_base") or ""),
                camera_refresh_ms=_clamp_camera_refresh_ms(doc.get("camera_refresh_ms", 100)),
                ui_language=ui_language,
                subtask_catalog=subtask_catalog,
                state_subtask_pairs=state_subtask_pairs,
            )
        except Exception:
            logging.exception("failed to load runtime config from mongo")
            return RuntimeConfigPayload()

    def save(self, payload: RuntimeConfigPayload) -> RuntimeConfigPayload:
        try:
            doc: dict[str, Any] = payload.model_dump(mode="json")
            doc["_id"] = "default"
            self._collection.replace_one({"_id": "default"}, doc, upsert=True)
        except Exception:
            logging.exception("failed to save runtime config to mongo")
        return payload
