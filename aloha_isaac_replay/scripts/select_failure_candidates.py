from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from aloha_isaac_replay.data.hdf5_reader import to_jsonable
from aloha_isaac_replay.data.select_failures import select_failure_candidates


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Selected Failure Candidates",
        "",
        f"- Search root: `{payload['search_root']}`",
        "- Labels are candidates only; no reward evaluator is implemented in this round.",
        "",
    ]
    for bucket, items in payload["buckets"].items():
        lines.extend(
            [
                f"## {bucket}",
                "",
                "| # | reward | frames | fps | phase | path |",
                "|---:|---:|---:|---:|---|---|",
            ]
        )
        for idx, item in enumerate(items, start=1):
            lines.append(
                f"| {idx} | {item.get('reward')} | {item.get('episode_length')} | {item.get('fps')} | {item.get('phase') or ''} | `{item.get('path')}` |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Select unconfirmed failure candidates from raw ALOHA HDF5 episodes.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--limit-per-bucket", type=int, default=5)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    buckets = select_failure_candidates(args.root, limit_per_bucket=args.limit_per_bucket)
    payload = {
        "search_root": args.root,
        "buckets": {
            name: [to_jsonable(dataclasses.asdict(candidate)) for candidate in candidates]
            for name, candidates in buckets.items()
        },
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    _write_markdown(Path(args.output_md), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

