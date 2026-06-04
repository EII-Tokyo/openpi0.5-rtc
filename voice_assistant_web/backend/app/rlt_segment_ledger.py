
from __future__ import annotations

from pathlib import Path
import sqlite3
import time
from typing import Any


class RLTSegmentLedger:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._memory_conn = sqlite3.connect(":memory:", timeout=5.0) if self._db_path == ":memory:" else None
        self._init_db()

    def _connect(self):
        conn = self._memory_conn or sqlite3.connect(self._db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS segments (
                    key_region_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    reward INTEGER,
                    shard_path TEXT,
                    num_replay_transitions INTEGER NOT NULL DEFAULT 0,
                    invalid_reason TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS segment_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_region_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    detail TEXT,
                    created_at REAL NOT NULL
                )
                """
            )

    def record_started(self, key_region_id: str, *, phase: str) -> None:
        self._upsert(key_region_id, status="started", phase=phase, event="started")

    def record_ended(self, key_region_id: str, *, phase: str) -> None:
        self._upsert(key_region_id, status="ended", phase=phase, event="ended")

    def record_accepted(self, key_region_id: str, *, reward: int, phase: str) -> None:
        self._upsert(key_region_id, status="accepted", phase=phase, reward=reward, event="accepted")

    def record_discarded(self, key_region_id: str, *, phase: str, reason: str) -> None:
        self._upsert(key_region_id, status="discarded", phase=phase, invalid_reason=reason, event="discarded")

    def record_committed(
        self,
        key_region_id: str,
        *,
        reward: int,
        phase: str,
        shard_path: str | None,
        num_replay_transitions: int,
    ) -> None:
        existing = self.get_segment(key_region_id)
        if existing and existing["status"] in {"committed", "voided"}:
            return
        self._upsert(
            key_region_id,
            status="committed",
            phase=phase,
            reward=reward,
            shard_path=shard_path,
            num_replay_transitions=num_replay_transitions,
            event="committed",
        )

    def record_cropped(
        self,
        key_region_id: str,
        *,
        reward: int,
        phase: str,
        shard_path: str,
        num_replay_transitions: int,
        reason: str,
    ) -> None:
        self._upsert(
            key_region_id,
            status="committed",
            phase=phase,
            reward=reward,
            shard_path=shard_path,
            num_replay_transitions=num_replay_transitions,
            invalid_reason=reason,
            event="cropped",
            force_transitions=True,
        )

    def record_rejected(self, key_region_id: str, *, phase: str, reason: str) -> None:
        existing = self.get_segment(key_region_id)
        if existing and existing["status"] in {"committed", "voided"}:
            return
        self._upsert(key_region_id, status="rejected", phase=phase, invalid_reason=reason, event="rejected")

    def void_segment(self, key_region_id: str, *, reason: str) -> None:
        existing = self.get_segment(key_region_id)
        phase = existing["phase"] if existing else "warmup"
        reward = existing.get("reward") if existing else None
        shard_path = existing.get("shard_path") if existing else None
        num_replay_transitions = existing.get("num_replay_transitions") if existing else 0
        self._upsert(
            key_region_id,
            status="voided",
            phase=phase,
            reward=reward,
            shard_path=shard_path,
            num_replay_transitions=num_replay_transitions,
            invalid_reason=reason,
            event="voided",
        )

    def get_segment(self, key_region_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM segments WHERE key_region_id = ?", (key_region_id,)).fetchone()
        return None if row is None else dict(row)

    def list_segments(self, *, limit: int = 500) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM segments
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def void_segments(self, key_region_ids: list[str], *, reason: str) -> list[str]:
        changed = []
        for key_region_id in key_region_ids:
            existing = self.get_segment(key_region_id) if key_region_id else None
            if not existing or existing.get("status") != "committed":
                continue
            self.void_segment(key_region_id, reason=reason)
            changed.append(key_region_id)
        return changed

    def restore_segments(self, key_region_ids: list[str], *, reason: str) -> list[str]:
        changed = []
        for key_region_id in key_region_ids:
            existing = self.get_segment(key_region_id)
            if not existing or existing.get("status") != "voided" or not existing.get("shard_path"):
                continue
            self._upsert(
                key_region_id,
                status="committed",
                phase=str(existing.get("phase") or "warmup"),
                reward=existing.get("reward"),
                shard_path=existing.get("shard_path"),
                num_replay_transitions=int(existing.get("num_replay_transitions") or 0),
                invalid_reason=reason,
                event="restored",
            )
            changed.append(key_region_id)
        return changed

    def delete_segments(self, key_region_ids: list[str]) -> list[str]:
        changed = []
        with self._connect() as conn:
            for key_region_id in key_region_ids:
                if not key_region_id:
                    continue
                row = conn.execute("SELECT key_region_id FROM segments WHERE key_region_id = ?", (key_region_id,)).fetchone()
                if row is None:
                    continue
                conn.execute("DELETE FROM segment_events WHERE key_region_id = ?", (key_region_id,))
                conn.execute("DELETE FROM segments WHERE key_region_id = ?", (key_region_id,))
                changed.append(key_region_id)
        return changed

    def stats(self) -> dict[str, int]:
        stats = {
            "warmup_count": 0,
            "warmup_success": 0,
            "warmup_failure": 0,
            "warmup_invalid": 0,
            "auto_rollout_count": 0,
            "auto_rollout_success": 0,
            "auto_rollout_failure": 0,
            "auto_rollout_invalid": 0,
        }
        with self._connect() as conn:
            rows = conn.execute("SELECT status, phase, reward FROM segments").fetchall()
        for row in rows:
            phase = "warmup" if row["phase"] == "warmup" else "auto_rollout"
            if row["status"] == "committed":
                stats[f"{phase}_count"] += 1
                if int(row["reward"] or 0) == 1:
                    stats[f"{phase}_success"] += 1
                else:
                    stats[f"{phase}_failure"] += 1
            elif row["status"] in {"discarded", "rejected", "voided"}:
                stats[f"{phase}_invalid"] += 1
        return stats

    def _upsert(self, key_region_id: str, *, status: str, phase: str, event: str, **values) -> None:
        now = time.time()
        existing = self.get_segment(key_region_id)
        reward = values.get("reward")
        shard_path = values.get("shard_path")
        transitions = int(values.get("num_replay_transitions") or 0)
        invalid_reason = values.get("invalid_reason")
        force_transitions = bool(values.get("force_transitions"))
        with self._connect() as conn:
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO segments (
                        key_region_id, status, phase, reward, shard_path, num_replay_transitions,
                        invalid_reason, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (key_region_id, status, phase, reward, shard_path, transitions, invalid_reason, now, now),
                )
            else:
                conn.execute(
                    """
                    UPDATE segments
                    SET status=?, phase=?, reward=COALESCE(?, reward), shard_path=COALESCE(?, shard_path),
                        num_replay_transitions=CASE WHEN ? OR ? > 0 THEN ? ELSE num_replay_transitions END,
                        invalid_reason=COALESCE(?, invalid_reason), updated_at=?
                    WHERE key_region_id=?
                    """,
                    (
                        status,
                        phase,
                        reward,
                        shard_path,
                        force_transitions,
                        transitions,
                        transitions,
                        invalid_reason,
                        now,
                        key_region_id,
                    ),
                )
            conn.execute(
                "INSERT INTO segment_events (key_region_id, event, detail, created_at) VALUES (?, ?, ?, ?)",
                (key_region_id, event, invalid_reason, now),
            )
