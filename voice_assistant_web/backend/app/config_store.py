from __future__ import annotations

from dataclasses import asdict
from typing import Any
import logging

from pymongo import MongoClient

from .config import settings
from .schemas import RuntimeConfigPayload


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
            return RuntimeConfigPayload(
                dataset_dir=str(doc.get("dataset_dir") or ""),
                manual_dataset_dir=str(doc.get("manual_dataset_dir") or ""),
                include_bottle_description=bool(doc.get("include_bottle_description", True)),
                include_bottle_position=bool(doc.get("include_bottle_position", False)),
                include_bottle_state=bool(doc.get("include_bottle_state", True)),
                include_subtask=bool(doc.get("include_subtask", True)),
                forced_low_level_subtask=(str(doc["forced_low_level_subtask"]).strip() if doc.get("forced_low_level_subtask") else None),
                video_memory_num_frames=4 if doc.get("video_memory_num_frames") == 4 else 1,
            )
        except Exception:
            logging.exception("failed to load runtime config from mongo")
            return RuntimeConfigPayload()

    def save(self, payload: RuntimeConfigPayload) -> RuntimeConfigPayload:
        try:
            doc: dict[str, Any] = asdict(payload)
            doc["_id"] = "default"
            self._collection.replace_one({"_id": "default"}, doc, upsert=True)
        except Exception:
            logging.exception("failed to save runtime config to mongo")
        return payload
