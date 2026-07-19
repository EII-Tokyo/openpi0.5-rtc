from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from aloha_isaac_replay.data.grasp_candidate_scan import inspect_grasp_candidate
from aloha_isaac_replay.data.grasp_candidate_scan import to_jsonable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/hdf5_grasp_candidate_scan_20260719"
DEFAULT_ROOTS = [
    Path("/home/eii/project/high_level/video/main_s01_L_pick_bottle_capped"),
    REPO_ROOT / "local_rlt_data/raw_from_103/rollouts/key_regions",
]


def _iter_hdf5_files(roots: list[Path], *, max_files: int | None) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix in {".hdf5", ".h5"}:
            files.append(root)
        else:
            files.extend(sorted(root.rglob("*.hdf5")))
            files.extend(sorted(root.rglob("*.h5")))
        if max_files is not None and len(files) >= max_files:
            return files[:max_files]
    return files


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "rank",
        "score",
        "likely_full_pickup",
        "source_hint",
        "episode_length",
        "duration_s",
        "left_gripper_start_median",
        "left_gripper_min",
        "left_gripper_end_median",
        "close_frame",
        "post_close_frames",
        "gripper_close_delta",
        "left_arm_total_motion",
        "left_arm_pre_close_motion",
        "left_arm_post_close_motion",
        "reasons",
        "path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            out = {name: row.get(name) for name in fieldnames}
            out["rank"] = idx
            out["reasons"] = ";".join(row.get("reasons") or [])
            writer.writerow(out)


def _write_md(path: Path, rows: list[dict[str, object]], scanned: int) -> None:
    lines = [
        "# HDF5 Grasp Candidate Scan",
        "",
        f"- scanned files: `{scanned}`",
        f"- candidate rows: `{len(rows)}`",
        "",
        "This scan is a signal-level filter only. It does not prove visual bottle pose or Isaac contact.",
        "A strong active tabletop pickup candidate should start with the left gripper open, contain a clear close event, retain enough frames after close, and move the left arm before and after closing.",
        "",
        "| rank | score | source | len | duration | close frame | open start | min grip | post-close frames | reasons | path |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for idx, row in enumerate(rows[:30], start=1):
        duration = row.get("duration_s")
        duration_text = "" if duration is None else f"{float(duration):.2f}"
        reasons = "; ".join(row.get("reasons") or [])
        lines.append(
            "| "
            f"{idx} | "
            f"{float(row.get('score') or 0.0):.2f} | "
            f"`{row.get('source_hint')}` | "
            f"{row.get('episode_length')} | "
            f"{duration_text} | "
            f"{row.get('close_frame')} | "
            f"{float(row.get('left_gripper_start_median') or 0.0):.3f} | "
            f"{float(row.get('left_gripper_min') or 0.0):.3f} | "
            f"{row.get('post_close_frames')} | "
            f"{reasons} | "
            f"`{row.get('path')}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Scan raw HDF5 episodes for active left-gripper bottle-pickup candidates. "
            "The score is only a prefilter; visual and Isaac contact gates must still validate the selected episode."
        )
    )
    parser.add_argument("--root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    roots = args.root or list(DEFAULT_ROOTS)
    files = _iter_hdf5_files(roots, max_files=args.max_files)
    rows = []
    for path in files:
        try:
            rows.append(to_jsonable(inspect_grasp_candidate(path)))
        except Exception as exc:
            rows.append(
                {
                    "path": str(path),
                    "source_hint": "scan_error",
                    "score": 0.0,
                    "likely_full_pickup": False,
                    "reasons": [f"scan_error:{type(exc).__name__}:{exc}"],
                }
            )
    rows.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
    top_rows = rows[: args.top_k]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "grasp_candidate_scan.json").write_text(
        json.dumps({"roots": [str(p) for p in roots], "scanned": len(files), "rows": rows}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "grasp_candidate_scan_top.csv", top_rows)
    _write_md(args.output_dir / "grasp_candidate_scan.md", top_rows, len(files))
    print(json.dumps({"status": "PASS", "scanned": len(files), "top_csv": str(args.output_dir / "grasp_candidate_scan_top.csv")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
