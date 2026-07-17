from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from aloha_isaac_replay.adapters.standard_aloha import STANDARD_ALOHA_14D_NAMES
from aloha_isaac_replay.scripts.check_trossen_scaffold_fk_against_aloha1 import _rel


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HDF5_ROOT = REPO_ROOT / "local_rlt_data/raw_from_103/rollouts/key_regions"
DEFAULT_PHASE5_JSON = (
    REPO_ROOT / "reports/aloha1_isaac_adaptation/phase5_one_joint_static_validation_20260717/one_joint_static_validation.json"
)
DEFAULT_PHASE9_JSON = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase9_fk_mapping_holdout_20260717/fk_mapping_holdout.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase10_mapping_full_dataset_limits_20260718"


def _discover_hdf5(root: Path, limit: int | None) -> list[Path]:
    paths = sorted(root.rglob("episode.hdf5"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No episode.hdf5 files under {root}")
    return paths


def _load_phase5_limits(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["canonical_name"]: row for row in payload["adapter_rows"]}


def _load_combo(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["best"]["combo"]


def _iter_qpos(paths: list[Path]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    parts = []
    episodes = []
    for path in paths:
        with h5py.File(path, "r") as h5:
            if "observations/qpos" not in h5:
                episodes.append({"path": _rel(path), "status": "SKIP_NO_QPOS"})
                continue
            qpos = np.asarray(h5["observations/qpos"][:], dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] < 14:
            episodes.append({"path": _rel(path), "status": "SKIP_BAD_QPOS_SHAPE", "shape": list(qpos.shape)})
            continue
        parts.append(qpos[:, :14])
        episodes.append({"path": _rel(path), "status": "OK", "frames": int(qpos.shape[0])})
    if not parts:
        raise ValueError("No valid qpos arrays found")
    return np.concatenate(parts, axis=0), episodes


def _joint_stats(qpos: np.ndarray, canonical: str, combo: dict[str, Any], limit_row: dict[str, Any]) -> dict[str, Any]:
    source_idx = STANDARD_ALOHA_14D_NAMES.index(canonical)
    lower, upper = [float(x) for x in limit_row["trossen_runtime_limit"]]
    values = float(combo["sign"]) * qpos[:, source_idx] + float(combo["offset"])
    inside = (values >= lower) & (values <= upper)
    lower_margin = values - lower
    upper_margin = upper - values
    min_margin = np.minimum(lower_margin, upper_margin)
    return {
        "canonical_name": canonical,
        "trossen_dof": limit_row["trossen_candidate_dof"],
        "sign": int(combo["sign"]),
        "offset": float(combo["offset"]),
        "limit_lower": lower,
        "limit_upper": upper,
        "value_min": float(np.min(values)),
        "value_max": float(np.max(values)),
        "inside_fraction": float(np.mean(inside)),
        "outside_count": int(values.size - np.count_nonzero(inside)),
        "min_limit_margin": float(np.min(min_margin)),
        "p01_limit_margin": float(np.quantile(min_margin, 0.01)),
        "p50_limit_margin": float(np.quantile(min_margin, 0.50)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "canonical_name",
        "trossen_dof",
        "sign",
        "offset",
        "limit_lower",
        "limit_upper",
        "value_min",
        "value_max",
        "inside_fraction",
        "outside_count",
        "min_limit_margin",
        "p01_limit_margin",
        "p50_limit_margin",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 10 - Full-Dataset Trossen Mapping Limit Check - 2026-07-18",
        "",
        "## Scope",
        "",
        "This is a pure offline qpos limit check for the Phase 9 left-arm mapping candidate.",
        "",
        "It does not start Isaac Sim and does not touch the real robot.",
        "",
        "## Dataset",
        "",
        f"- HDF5 root: `{payload['inputs']['hdf5_root']}`",
        f"- valid episodes: `{payload['summary']['valid_episode_count']}`",
        f"- frames: `{payload['summary']['frame_count']}`",
        "",
        "## Gates",
        "",
    ]
    for key, value in payload["gates"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Joint Limit Table", ""])
    lines.append("| joint | sign | offset | mapped range | Trossen limit | inside fraction | outside | min margin |")
    lines.append("|---|---:|---:|---|---|---:|---:|---:|")
    for row in payload["joint_stats"]:
        lines.append(
            "| "
            f"`{row['canonical_name']}` | "
            f"{row['sign']} | "
            f"{row['offset']:.6f} | "
            f"[{row['value_min']:.6f}, {row['value_max']:.6f}] | "
            f"[{row['limit_lower']:.6f}, {row['limit_upper']:.6f}] | "
            f"{row['inside_fraction']:.6f} | "
            f"{row['outside_count']} | "
            f"{row['min_limit_margin']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A mapping candidate cannot be used for controller work if it pushes normal recorded ALOHA1 qpos outside the Trossen runtime limits.",
            "",
            "Passing this check still does not validate orientation, gripper mechanics, controller stability, or contact dynamics.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Phase 9 left-arm mapping candidate against Trossen limits on all local HDF5 qpos.")
    parser.add_argument("--hdf5-root", type=Path, default=DEFAULT_HDF5_ROOT)
    parser.add_argument("--phase5-json", type=Path, default=DEFAULT_PHASE5_JSON)
    parser.add_argument("--phase9-json", type=Path, default=DEFAULT_PHASE9_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode-limit", type=int, default=None)
    args = parser.parse_args()

    paths = _discover_hdf5(args.hdf5_root, args.episode_limit)
    qpos, episode_rows = _iter_qpos(paths)
    phase5_rows = _load_phase5_limits(args.phase5_json)
    combo = _load_combo(args.phase9_json)
    joint_stats = [_joint_stats(qpos, canonical, combo[canonical], phase5_rows[canonical]) for canonical in combo]

    all_inside = all(row["outside_count"] == 0 for row in joint_stats)
    min_margin = min(row["min_limit_margin"] for row in joint_stats)
    payload = {
        "inputs": {
            "hdf5_root": _rel(args.hdf5_root),
            "phase5_json": _rel(args.phase5_json),
            "phase9_json": _rel(args.phase9_json),
        },
        "summary": {
            "episode_count_total": len(paths),
            "valid_episode_count": sum(1 for row in episode_rows if row["status"] == "OK"),
            "frame_count": int(qpos.shape[0]),
            "min_limit_margin": min_margin,
        },
        "episode_rows": episode_rows,
        "joint_stats": joint_stats,
        "gates": {
            "real_robot_touched": "PASS_FALSE",
            "isaac_runtime_started": "PASS_FALSE",
            "qpos_loaded": "PASS",
            "all_mapped_values_inside_trossen_limits": "PASS" if all_inside else "FAIL",
            "controller": "BLOCKED_NOT_ATTEMPTED",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "mapping_full_dataset_limits.json"
    md_path = args.output_dir / "mapping_full_dataset_limits.md"
    csv_path = args.output_dir / "mapping_full_dataset_limits.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    _write_csv(csv_path, joint_stats)
    print(
        json.dumps(
            {
                "json": _rel(json_path),
                "markdown": _rel(md_path),
                "csv": _rel(csv_path),
                "summary": payload["summary"],
                "gates": payload["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if all_inside else 2


if __name__ == "__main__":
    raise SystemExit(main())
