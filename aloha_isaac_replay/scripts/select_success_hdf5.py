from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from aloha_isaac_replay.data.hdf5_reader import to_jsonable
from aloha_isaac_replay.data.select_episodes import select_success_episodes


def _write_markdown(path: Path, selected: list[object], rejected_count: int, root: str) -> None:
    lines = [
        "# Selected Successful Raw ALOHA HDF5 Episodes",
        "",
        f"- Search root: `{root}`",
        f"- Selected success episodes: {len(selected)}",
        f"- Rejected or non-success inspected episodes before limit: {rejected_count}",
        "- Selection rule: complete raw HDF5 episode with root attr `reward > 0.5`.",
        "",
        "| # | reward | frames | fps | phase | action semantics | confidence | path |",
        "|---:|---:|---:|---:|---|---|---:|---|",
    ]
    for idx, candidate in enumerate(selected, start=1):
        row = dataclasses.asdict(candidate)
        lines.append(
            "| {idx} | {reward} | {frames} | {fps} | {phase} | {semantics} | {confidence:.3f} | `{path}` |".format(
                idx=idx,
                reward=row["reward"],
                frames=row["episode_length"],
                fps=row["fps"],
                phase=row["phase"] or "",
                semantics=row["action_semantics"],
                confidence=row["action_semantics_confidence"],
                path=row["path"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Select successful complete raw ALOHA HDF5 episodes.")
    parser.add_argument("--root", required=True, help="Bounded dataset root to search")
    parser.add_argument("--limit", type=int, default=10, help="Number of success episodes to select")
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--markdown", required=True, help="Markdown report output path")
    args = parser.parse_args()

    selected, rejected = select_success_episodes(args.root, limit=args.limit)
    payload = {
        "search_root": args.root,
        "selection_rule": "complete raw HDF5 with reward > 0.5",
        "requested_limit": args.limit,
        "selected_count": len(selected),
        "selected": [to_jsonable(candidate) for candidate in selected],
        "rejected_or_non_success_before_limit_count": len(rejected),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _write_markdown(Path(args.markdown), selected, len(rejected), args.root)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if len(selected) == args.limit else 2


if __name__ == "__main__":
    raise SystemExit(main())

