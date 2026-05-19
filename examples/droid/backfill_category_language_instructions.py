#!/usr/bin/env python3
"""Backfill category prompts into language_instruction_2 for DROID LeRobot datasets.

The generated prompt teaches a two-category abstraction:

- Category 1: battery bank, phone, vape pen
- Category two: all other moved objects

Example:
    "Put the battery bank in the orange box and the phone in the blue box"
    -> "Category 1 (battery bank, phone, vape pen) in orange and blue boxes."

Usage:
    uv run python -m examples.droid.backfill_category_language_instructions --dry-run
    uv run python -m examples.droid.backfill_category_language_instructions --no-push
    uv run python -m examples.droid.backfill_category_language_instructions --repos michios/droid_xxjd_canonical
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re

from huggingface_hub import HfApi
import pandas as pd

log = logging.getLogger(__name__)

try:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
except ImportError:
    HF_LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"


DEFAULT_REPOS = [
    "michios/droid_xxjd_canonical",
    "michios/droid_xxjd_2_canonical",
    "michios/droid_xxjd_3_canonical",
    "michios/droid_xxjd_4_canonical",
    "michios/droid_xxjd_5_canonical",
    "michios/droid_xxjd_6_2",
    "michios/droid_xxjd_7_2",
    "michios/droid_xxjd_8_2_canonical",
    "michios/droid_xxjd_20260202",
    "michios/droid_xxjd_20260421",
    "michios/droid_xxjd_20260423",
    "michios/droid_xxjd_20260511_20260512",
]

CATEGORY_1_LABEL = "Category 1 (battery bank, phone, vape pen)"
CATEGORY_2_LABEL = "everything else"
CATEGORY_1_ITEMS = {
    "battery bank",
    "battery banks",
    "phone",
    "phones",
    "vape pen",
    "vape pens",
}

ASSIGNMENT_RE = re.compile(
    r"(?:^|\band\s+)(?:put\s+)?(?:the\s+)?(?P<item>.+?)\s+in\s+(?:the\s+)?(?P<container>.+?)(?=\.|$|\s+and\s+(?:put\s+)?(?:the\s+)?)",
    re.IGNORECASE,
)


def _clean_text(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if hasattr(value, "item"):
        value = value.item()
    return str(value).strip()


def _normalize_item(item: str) -> str:
    item = item.strip().lower()
    item = re.sub(r"^(?:a|an|the)\s+", "", item)
    return re.sub(r"\s+", " ", item)


def _normalize_container(container: str) -> str:
    container = container.strip().lower()
    container = re.sub(r"^(?:a|an|the)\s+", "", container)
    container = container.rstrip(" .")
    return re.sub(r"\s+", " ", container)


def _container_phrase(containers: list[str], *, definite: bool) -> str:
    unique: list[str] = []
    for container in containers:
        if container not in unique:
            unique.append(container)

    if not unique:
        return ""

    if len(unique) == 1:
        phrase = unique[0]
        return f"the {phrase}" if definite else phrase

    box_colors: list[str] = []
    all_boxes = True
    for container in unique:
        if container.endswith(" box"):
            box_colors.append(container[: -len(" box")])
        elif container.endswith(" boxes"):
            box_colors.append(container[: -len(" boxes")])
        else:
            all_boxes = False
            break

    if all_boxes:
        joined = ", ".join(box_colors[:-1]) + f" and {box_colors[-1]}" if len(box_colors) > 2 else " and ".join(box_colors)
        phrase = f"{joined} boxes"
    else:
        phrase = ", ".join(unique[:-1]) + f" and {unique[-1]}" if len(unique) > 2 else " and ".join(unique)

    return f"the {phrase}" if definite else phrase


def build_category_instruction(instruction: object) -> str:
    """Build a category-level prompt from one object-placement instruction."""
    text = _clean_text(instruction)
    if not text:
        return ""

    category_1_containers: list[str] = []
    category_2_containers: list[str] = []

    for match in ASSIGNMENT_RE.finditer(text):
        item = _normalize_item(match.group("item"))
        container = _normalize_container(match.group("container"))
        if not item or not container:
            continue
        if item in CATEGORY_1_ITEMS:
            category_1_containers.append(container)
        else:
            category_2_containers.append(container)

    if not category_1_containers and not category_2_containers:
        return ""

    if category_1_containers and not category_2_containers:
        return f"{CATEGORY_1_LABEL} in {_container_phrase(category_1_containers, definite=False)}."

    if category_1_containers and category_2_containers:
        return (
            f"{CATEGORY_1_LABEL} in {_container_phrase(category_1_containers, definite=True)} "
            f"and {CATEGORY_2_LABEL} in {_container_phrase(category_2_containers, definite=True)}."
        )

    return f"{CATEGORY_2_LABEL.capitalize()} in {_container_phrase(category_2_containers, definite=True)}."


def _language_source_column(df: pd.DataFrame) -> str:
    for column in ("language_instruction", "task"):
        if column in df.columns:
            return column
    raise ValueError("No language_instruction or task column found")


def _update_info_features(root: Path, *, dry_run: bool) -> None:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        log.warning("  meta/info.json not found; skipping feature metadata update")
        return

    info = json.loads(info_path.read_text())
    features = info.setdefault("features", {})
    if "language_instruction_2" in features:
        return

    features["language_instruction_2"] = {"dtype": "string", "shape": [1], "names": None}
    log.info("  [meta] adding language_instruction_2 feature")
    if not dry_run:
        info_path.write_text(json.dumps(info, indent=2) + "\n")


def update_data_parquets(root: Path, *, dry_run: bool) -> tuple[int, int, int]:
    data_paths = sorted((root / "data").glob("**/*.parquet"))
    if not data_paths:
        log.error("  No data parquet files found under %s -- skipping", root / "data")
        return 0, 0, 0

    total_rows = 0
    updated_rows = 0
    empty_rows = 0

    for path in data_paths:
        df = pd.read_parquet(path)
        source_column = _language_source_column(df)
        prompts = df[source_column].map(build_category_instruction)
        changed = "language_instruction_2" not in df.columns or not df["language_instruction_2"].equals(prompts)

        total_rows += len(df)
        updated_rows += int((prompts != "").sum())
        empty_rows += int((prompts == "").sum())

        if changed:
            log.info("  [data] %s: writing %d prompts (%d empty)", path.relative_to(root), (prompts != "").sum(), (prompts == "").sum())
            if not dry_run:
                df["language_instruction_2"] = prompts
                df.to_parquet(path, index=False)

    return total_rows, updated_rows, empty_rows


def push_to_hub(root: Path, repo_id: str, *, dry_run: bool) -> None:
    log.info("  [hub] pushing -> %s", repo_id)
    if not dry_run:
        HfApi().upload_folder(
            folder_path=str(root),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="add category language instructions",
        )


def process_repo(repo_id: str, *, lerobot_root: Path, push: bool, dry_run: bool) -> None:
    root = lerobot_root / repo_id
    log.info("")
    log.info("=== %s ===", repo_id)
    log.info("    root: %s", root)
    if not root.exists():
        log.error("  Dataset not found locally at %s -- skipping", root)
        return

    total_rows, updated_rows, empty_rows = update_data_parquets(root, dry_run=dry_run)
    if total_rows == 0:
        return

    _update_info_features(root, dry_run=dry_run)
    log.info("  [summary] rows=%d category_prompts=%d empty=%d", total_rows, updated_rows, empty_rows)

    if push:
        push_to_hub(root, repo_id, dry_run=dry_run)
    else:
        log.info("  [hub] skipping push (--no-push)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repos", nargs="+", default=DEFAULT_REPOS, metavar="REPO_ID")
    parser.add_argument("--lerobot-root", type=Path, default=Path(HF_LEROBOT_HOME))
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s: %(message)s")

    for repo_id in args.repos:
        process_repo(repo_id, lerobot_root=args.lerobot_root, push=not args.no_push, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
