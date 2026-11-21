"""Build the rinse prompt from the same task catalog used by low-level conditioning."""
from pathlib import Path

from openpi.shared.completed_step_memory import SUBTASKS_BY_INDEX
from openpi.shared.high_level_index_protocol import INDEX_MEMORY_SCHEMA


def rinse_prompt(language: str = "en") -> str:
    """Include the canonical task indices and the exact completion record for each task."""
    templates = {"en": "sol_rinse_prompt_en.txt", "zh": "sol_rinse_prompt_zh.txt"}
    instructions = Path(__file__).with_name(templates[language]).read_text(encoding="utf-8")
    rows = ["Main cycle" if language == "en" else "主流程"]
    for index, task in SUBTASKS_BY_INDEX.items():
        if index == 8:
            rows.extend(("", "Special cases" if language == "en" else "特殊情况"))
        rows.append(f"{index}. {task['current_task']}")
        if task["memory_en"]:
            rows.append(f"   Memory: {task['memory_en']}")
    return instructions + "\n" + "\n".join(rows) + "\n"


def rinse_schema() -> dict:
    """Use the existing index-based output shape with the rinse branch indices."""
    return INDEX_MEMORY_SCHEMA


class InsertionDwell:
    """Accumulate continuous full-depth evidence for one bottle using capture timestamps."""

    def __init__(self, threshold_seconds: float = 1.5, max_gap_seconds: float = 1.25):
        if not 1 <= threshold_seconds <= 2 or not 0 < max_gap_seconds <= 2:
            raise ValueError("Invalid rinse dwell timing")
        self.threshold = threshold_seconds
        self.max_gap = max_gap_seconds
        self.reset()

    def reset(self) -> None:
        """Clear state when the active bottle changes or a rollout is reset."""
        self.started = None
        self.last_capture = None
        self.duration = 0.0
        self.completed = False

    def observe(self, captured_at: float, fully_inserted: bool | None) -> None:
        """Use fresh observations only. Unknown depth breaks an unfinished hold."""
        from math import isfinite

        if not isfinite(captured_at):
            raise ValueError("Capture timestamp must be finite")
        if self.last_capture is not None and captured_at <= self.last_capture:
            return
        gap = self.last_capture is not None and captured_at - self.last_capture > self.max_gap
        self.last_capture = captured_at
        if self.completed:
            return
        if fully_inserted is not True:
            self.started = None
            self.duration = 0.0
            return
        if self.started is None or gap:
            self.started = captured_at
        self.duration = captured_at - self.started
        self.completed = self.duration >= self.threshold

    def prompt_context(self) -> dict:
        """Provide measured evidence and the configured threshold without replaying images."""
        return {
            "verified_full_insertion_seconds": self.duration,
            "rinse_dwell_threshold_seconds": self.threshold,
        }
