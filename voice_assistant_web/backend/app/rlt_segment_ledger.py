
from __future__ import annotations

from pathlib import Path
import sqlite3
import time
from typing import Any

JITTER_PENALTIES: dict[str, float] = {
    "smooth": 0.0,
    "mild_jitter": 0.3,
    "severe_jitter": 1.0,
}

ACTOR_TRAIN_MODES = {"auto", "exclude", "low_weight", "normal", "strong"}
QUALITY_JITTER_LAMBDA = 0.25


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
                    quality_score INTEGER,
                    quality_task REAL,
                    jitter_level TEXT,
                    jitter_penalty REAL,
                    quality_final REAL,
                    actor_train_mode TEXT NOT NULL DEFAULT 'auto',
                    quality_source TEXT NOT NULL DEFAULT 'legacy',
                    quality_version INTEGER NOT NULL DEFAULT 1,
                    quality_updated_at REAL,
                    quality_notes TEXT,
                    shard_path TEXT,
                    num_replay_transitions INTEGER NOT NULL DEFAULT 0,
                    invalid_reason TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._ensure_quality_columns(conn)
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

    def _ensure_quality_columns(self, conn) -> None:
        rows = conn.execute("PRAGMA table_info(segments)").fetchall()
        columns = {row[1] for row in rows}
        for name, definition in (
            ("quality_score", "INTEGER"),
            ("quality_task", "REAL"),
            ("jitter_level", "TEXT"),
            ("jitter_penalty", "REAL"),
            ("quality_final", "REAL"),
            ("actor_train_mode", "TEXT NOT NULL DEFAULT 'auto'"),
            ("quality_source", "TEXT NOT NULL DEFAULT 'legacy'"),
            ("quality_version", "INTEGER NOT NULL DEFAULT 1"),
            ("quality_updated_at", "REAL"),
            ("quality_notes", "TEXT"),
        ):
            if name not in columns:
                conn.execute(f"ALTER TABLE segments ADD COLUMN {name} {definition}")

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
        if existing and existing["status"] in {"committed", "voided", "deleted"}:
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

    def record_rescored(
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
            event="rescored",
            force_transitions=True,
        )

    def record_quality_review(
        self,
        key_region_id: str,
        *,
        quality_score: int,
        jitter_level: str,
        actor_train_mode: str,
        quality_final: float | None = None,
        source: str = "ui",
        notes: str | None = None,
    ) -> None:
        if quality_score < 0 or quality_score > 4:
            raise ValueError("quality_score must be between 0 and 4")
        if jitter_level not in JITTER_PENALTIES:
            raise ValueError(f"unsupported jitter_level={jitter_level!r}")
        if actor_train_mode not in ACTOR_TRAIN_MODES:
            raise ValueError(f"unsupported actor_train_mode={actor_train_mode!r}")
        quality_task = float(quality_score) / 4.0
        jitter_penalty = JITTER_PENALTIES[jitter_level]
        if quality_final is None:
            quality_final = max(0.0, min(1.0, quality_task - QUALITY_JITTER_LAMBDA * jitter_penalty))
        else:
            quality_final = max(0.0, min(1.0, float(quality_final)))
        now = time.time()
        existing = self.get_segment(key_region_id)
        if existing is None:
            raise ValueError(f"Unknown key_region_id: {key_region_id}")
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE segments
                SET quality_score=?, quality_task=?, jitter_level=?, jitter_penalty=?, quality_final=?,
                    actor_train_mode=?, quality_source='human', quality_version=1, quality_updated_at=?,
                    quality_notes=?, updated_at=?
                WHERE key_region_id=?
                """,
                (
                    quality_score,
                    quality_task,
                    jitter_level,
                    jitter_penalty,
                    quality_final,
                    actor_train_mode,
                    now,
                    notes,
                    now,
                    key_region_id,
                ),
            )
            conn.execute(
                "INSERT INTO segment_events (key_region_id, event, detail, created_at) VALUES (?, ?, ?, ?)",
                (key_region_id, "quality_reviewed", f"{source}:{quality_score}:{jitter_level}:{actor_train_mode}", now),
            )

    def record_rejected(self, key_region_id: str, *, phase: str, reason: str) -> None:
        existing = self.get_segment(key_region_id)
        if existing and existing["status"] in {"committed", "voided", "deleted"}:
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
        for key_region_id in key_region_ids:
            if not key_region_id:
                continue
            existing = self.get_segment(key_region_id)
            if existing and existing.get("status") == "deleted":
                continue
            self._upsert(
                key_region_id,
                status="deleted",
                phase=str((existing or {}).get("phase") or "warmup"),
                reward=None if existing is None else existing.get("reward"),
                shard_path=None if existing is None else existing.get("shard_path"),
                num_replay_transitions=0 if existing is None else int(existing.get("num_replay_transitions") or 0),
                invalid_reason="operator_delete",
                event="deleted",
            )
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
