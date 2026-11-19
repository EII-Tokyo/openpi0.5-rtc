"""Causal, episode-local memory shared by annotation training and online serving.

Annotation events are supervision, never an online sensor. Online inference must
receive memory from a high-level model; it cannot access demonstration labels.
"""

from __future__ import annotations

import bisect
import dataclasses
import hashlib
import json
import math
import random


@dataclasses.dataclass(frozen=True)
class LongMemoryConfig:
    update_seconds: float = 3.0
    video_seconds: float = 3.0
    video_hz: float = 2.0
    max_entries: int = 6
    empty_probability: float = 0.2
    partial_probability: float = 0.3
    stale_probability: float = 0.2
    max_stale_seconds: float = 9.0
    reset_on_home: bool = True
    split_seed: int = 20261119

    def __post_init__(self):
        if min(self.update_seconds, self.video_seconds, self.video_hz) <= 0 or self.max_entries < 1:
            raise ValueError("Invalid memory/video settings")
        if (not 0 <= self.empty_probability <= 1 or not 0 <= self.partial_probability <= 1
                or self.empty_probability + self.partial_probability > 1
                or not 0 <= self.stale_probability <= 1 or self.max_stale_seconds < 0):
            raise ValueError("Invalid memory corruption settings")


@dataclasses.dataclass(frozen=True)
class MemoryEntry:
    time_s: float
    bottle: str
    observation: str
    intent: str

    def __post_init__(self):
        if not math.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError("Memory timestamps must be finite and episode-relative")
        if not all(isinstance(v, str) for v in (self.bottle, self.observation, self.intent)):
            raise ValueError("Memory fields must be strings")


def parse_annotation(raw) -> tuple[str, str, str] | None:
    """Keep actual labeled descriptions/states; never infer completion from a future label."""
    obj = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(obj, dict):
        raise ValueError("Expected structured bottle annotation")
    description = obj.get("bottle_description")
    if isinstance(description, list):
        description = next((x.strip() for x in description if isinstance(x, str) and x.strip()), "")
    if description is None:
        description = ""
    observation, intent = obj.get("bottle_state") or "", obj.get("subtask") or ""
    if not all(isinstance(x, str) for x in (description, observation, intent)):
        raise ValueError("Malformed annotation fields")
    if not observation.strip() and not intent.strip():
        return None  # An explicitly empty label supplies no new evidence.
    return description.strip(), observation.strip() or "Unknown (not annotated)", intent.strip() or "Unknown (not annotated)"


def episode_split(repo_id: str, episode_index: int, seed: int = 20261119) -> str:
    """All weighted copies of an episode share one split, in both hierarchy levels."""
    value = int.from_bytes(hashlib.sha256(f"{seed}/{repo_id}/{episode_index}".encode()).digest()[:8], "big") / 2**64
    return "test" if value < .1 else "validation" if value < .2 else "train"


def is_return_home(intent: str) -> bool:
    """Recognize annotated returning intent, never equate its start with completion."""
    intent = intent.lower().replace("_", " ")
    return "return" in intent and ("home" in intent or "initial" in intent)


class MemoryTimeline:
    def __init__(self, timestamps, annotations, eligible=None, *, reset_on_home=False):
        if len(timestamps) != len(annotations) or not len(timestamps):
            raise ValueError("Memory timeline must contain one aligned episode")
        self.events = []
        self.cycle_starts = []
        self.terminal_home_complete_at = None
        previous = None
        previous_was_home = False
        last_time = -1.0
        for i, (timestamp, annotation) in enumerate(zip(timestamps, annotations, strict=True)):
            timestamp = float(timestamp)
            if not math.isfinite(timestamp) or timestamp < last_time or timestamp < 0:
                raise ValueError("Episode timestamps must be finite, nonnegative and sorted")
            last_time = timestamp
            if eligible is not None and not bool(eligible[i]):
                continue
            fields = parse_annotation(annotation)
            if fields is None:
                continue
            returning_home = is_return_home(fields[2])
            if reset_on_home and previous_was_home and not returning_home:
                # Only a newly observed next phase establishes this within-EP boundary.
                self.cycle_starts.append(timestamp)
                previous = None
            previous_was_home = returning_home
            if fields != previous:
                self.events.append(MemoryEntry(timestamp, *fields))
                previous = fields
        self.times = [entry.time_s for entry in self.events]
        if reset_on_home and previous_was_home:
            # EP end is an explicit trajectory boundary, not the beginning of return-home intent.
            self.terminal_home_complete_at = last_time

    def at(self, time_s: float, max_entries: int = 6) -> tuple[MemoryEntry, ...]:
        if self.terminal_home_complete_at is not None and time_s >= self.terminal_home_complete_at:
            return ()
        end = bisect.bisect_right(self.times, time_s)
        boundary = bisect.bisect_right(self.cycle_starts, time_s)
        start = bisect.bisect_left(self.times, self.cycle_starts[boundary-1]) if boundary else 0
        return tuple(self.events[max(start, end - max_entries):end])


def drop_memory(entries, rng: random.Random, *, empty_probability=0.2, partial_probability=0.3):
    """20% empty, 30% arbitrary proper subset, 50% full for >=2 entries.

    With one event, a nonempty proper subset is impossible: partial draws keep
    that event. Empty draws remain exactly 20% for nonempty input histories.
    """
    entries = tuple(entries)
    draw = rng.random()
    if not entries or draw < empty_probability:
        return ()
    if draw >= empty_probability + partial_probability or len(entries) < 2:
        return entries
    count = rng.randint(1, len(entries) - 1)
    return tuple(entries[i] for i in sorted(rng.sample(range(len(entries)), count)))


def condition_at(timeline, time_s, settings, *, training: bool, rng=None):
    """Simulate periodic updates, missing events and delayed high-level decisions."""
    update_time = math.floor(time_s / settings.update_seconds) * settings.update_seconds
    if training:
        if rng is None:
            raise ValueError("Training memory corruption requires an explicit RNG")
        # Include clean, empty and arbitrarily stale states, not only transition frames.
        if rng.random() < settings.stale_probability:
            update_time -= rng.uniform(0, settings.max_stale_seconds)
    entries = timeline.at(update_time, settings.max_entries)
    return drop_memory(entries, rng, empty_probability=settings.empty_probability, partial_probability=settings.partial_probability) if training else entries


def encode_memory(entries) -> str:
    return json.dumps({"events": [{**dataclasses.asdict(x), "time_s": math.floor(x.time_s * 1000) / 1000} for x in entries]}, ensure_ascii=False, separators=(",", ":"))


def decode_memory(text: str, *, now_s: float, max_entries=6):
    obj = json.loads(text)
    if not isinstance(obj, dict) or set(obj) != {"events"} or not isinstance(obj["events"], list):
        raise ValueError("Expected an events-only memory object")
    if len(obj["events"]) > max_entries:
        raise ValueError("Memory exceeds configured event budget")
    entries = tuple(MemoryEntry(**x) for x in obj["events"])
    if any(x.time_s > now_s + 1e-6 for x in entries):
        raise ValueError("Memory cannot contain future observations")
    if any(b.time_s < a.time_s for a, b in zip(entries, entries[1:])):
        raise ValueError("Memory events must be chronologically ordered")
    return entries


def memory_prompt(task: str, entries) -> str:
    return f"{task}\nLong memory (may be incomplete or stale; use current visual evidence): {encode_memory(entries)}"


def video_indices(timestamps, now_s, *, seconds=3.0, hz=2.0):
    """Causal nearest-previous frames from one episode; never select a future frame."""
    if seconds <= 0 or hz <= 0 or not timestamps:
        raise ValueError("Invalid video window")
    if any(b < a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Unsorted video timestamps")
    end = bisect.bisect_right(timestamps, now_s)
    if end == 0:
        raise ValueError("No frame at or before the query")
    start = max(float(timestamps[0]), now_s - seconds)
    targets = [start + i / hz for i in range(int((now_s - start) * hz) + 1)]
    if not targets or targets[-1] < now_s - 1e-6:
        targets.append(now_s)
    return tuple(dict.fromkeys(max(0, bisect.bisect_right(timestamps, t, 0, end) - 1) for t in targets))
