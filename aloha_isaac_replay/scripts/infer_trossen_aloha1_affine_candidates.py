from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from aloha_isaac_replay.adapters.standard_aloha import STANDARD_ALOHA_14D_NAMES
from aloha_isaac_replay.scripts.validate_trossen_backed_aloha1_one_joint_mapping import ARM_JOINTS
from aloha_isaac_replay.scripts.validate_trossen_backed_aloha1_one_joint_mapping import CANONICAL_TO_TROSSEN


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHASE5_JSON = (
    REPO_ROOT / "reports/aloha1_isaac_adaptation/phase5_one_joint_static_validation_20260717/one_joint_static_validation.json"
)
DEFAULT_HDF5_ROOT = REPO_ROOT / "local_rlt_data/raw_from_103/rollouts/key_regions"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/aloha1_isaac_adaptation/phase6_affine_candidate_inference_20260717"


ARM_CANONICAL_NAMES = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _discover_hdf5(root: Path, limit: int | None) -> list[Path]:
    paths = sorted(root.rglob("episode.hdf5"))
    if limit is not None:
        paths = paths[:limit]
    return paths


def _episode_id(path: Path) -> str:
    if path.parent.name.startswith("key_region_"):
        return path.parent.name
    return path.stem


def _load_qpos(paths: list[Path], max_frames_per_episode: int | None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    parts: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for path in paths:
        with h5py.File(path, "r") as h5:
            if "observations/qpos" not in h5:
                rows.append({"path": _rel(path), "status": "SKIP_NO_OBSERVATIONS_QPOS"})
                continue
            qpos = np.asarray(h5["observations/qpos"][:], dtype=np.float64)
            if qpos.ndim != 2 or qpos.shape[1] < 14:
                rows.append({"path": _rel(path), "status": "SKIP_BAD_QPOS_SHAPE", "shape": list(qpos.shape)})
                continue
            if max_frames_per_episode is not None and len(qpos) > max_frames_per_episode:
                qpos = qpos[:max_frames_per_episode]
            parts.append(qpos[:, :14])
            rows.append(
                {
                    "path": _rel(path),
                    "episode_id": _episode_id(path),
                    "frames_used": int(len(qpos)),
                    "phase": str(h5.attrs.get("phase", "")),
                    "reward": int(h5.attrs.get("reward", -1)),
                    "status": "OK",
                }
            )
    if not parts:
        raise ValueError(f"No valid observations/qpos found under {paths[:3]}")
    return np.concatenate(parts, axis=0), rows


def _load_phase5(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    rows = {row["canonical_name"]: row for row in payload["adapter_rows"]}
    if not all(name in rows for name in ARM_CANONICAL_NAMES):
        missing = [name for name in ARM_CANONICAL_NAMES if name not in rows]
        raise ValueError(f"Phase 5 JSON missing adapter rows: {missing}")
    return payload


def _candidate_fit(
    samples: np.ndarray,
    *,
    sign: float,
    ros_reference: float,
    isaac_reference: float,
    isaac_lower: float,
    isaac_upper: float,
) -> dict[str, Any]:
    offset = isaac_reference - sign * ros_reference
    mapped = sign * samples + offset
    inside = (mapped >= isaac_lower) & (mapped <= isaac_upper)
    lower_margin = mapped - isaac_lower
    upper_margin = isaac_upper - mapped
    min_margin = np.minimum(lower_margin, upper_margin)
    return {
        "sign": int(sign),
        "offset": float(offset),
        "isaac_reference": float(isaac_reference),
        "ros_reference": float(ros_reference),
        "inside_fraction": float(np.mean(inside)),
        "outside_count": int(np.size(inside) - np.count_nonzero(inside)),
        "mapped_min": float(np.min(mapped)),
        "mapped_max": float(np.max(mapped)),
        "min_limit_margin": float(np.min(min_margin)),
        "p01_limit_margin": float(np.quantile(min_margin, 0.01)),
        "p50_limit_margin": float(np.quantile(min_margin, 0.50)),
    }


def _infer_candidates(qpos: np.ndarray, phase5: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for name in ARM_CANONICAL_NAMES:
        dataset_index = STANDARD_ALOHA_14D_NAMES.index(name)
        side, joint = name.split("_", 1)
        joint_index = ARM_JOINTS.index(joint)
        row = phase5["adapter_rows"][[r["canonical_name"] for r in phase5["adapter_rows"]].index(name)]
        isaac_lower, isaac_upper = row["trossen_runtime_limit"]
        ros_sleep = phase5["real_aloha1_facts"]["puppet"]["arm"]["sleep"][joint_index]
        samples = qpos[:, dataset_index]
        candidates = [
            _candidate_fit(
                samples,
                sign=sign,
                ros_reference=ros_sleep,
                isaac_reference=0.0,
                isaac_lower=float(isaac_lower),
                isaac_upper=float(isaac_upper),
            )
            for sign in (1.0, -1.0)
        ]
        good = [candidate for candidate in candidates if candidate["inside_fraction"] >= 1.0]
        if len(good) == 1:
            status = "PASS_LIMIT_FIT_UNIQUE_CANDIDATE"
            selected = good[0]
        elif len(good) > 1:
            status = "AMBIGUOUS_BOTH_SIGNS_FIT_LIMITS"
            selected = max(good, key=lambda c: c["min_limit_margin"])
        else:
            status = "FAIL_NO_SIGN_FITS_LIMITS"
            selected = max(candidates, key=lambda c: (c["inside_fraction"], c["min_limit_margin"]))
        result.append(
            {
                "canonical_name": name,
                "dataset_index": dataset_index,
                "trossen_dof": CANONICAL_TO_TROSSEN[name],
                "trossen_runtime_index": row["trossen_runtime_index"],
                "ros_reference_pose": "puppet_sleep",
                "isaac_reference_pose": "trossen_scaffold_zero",
                "ros_sample_min": float(np.min(samples)),
                "ros_sample_max": float(np.max(samples)),
                "ros_sample_mean": float(np.mean(samples)),
                "ros_sample_std": float(np.std(samples)),
                "candidate_plus": candidates[0],
                "candidate_minus": candidates[1],
                "selected_sign": selected["sign"],
                "selected_offset": selected["offset"],
                "selected_inside_fraction": selected["inside_fraction"],
                "selected_min_limit_margin": selected["min_limit_margin"],
                "status": status,
                "confidence": "LOW_LIMIT_FIT_ONLY",
                "blocked_reason": "Limit-fit cannot prove positive direction, zero pose geometry, or FK equivalence.",
            }
        )
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "canonical_name",
        "dataset_index",
        "trossen_dof",
        "trossen_runtime_index",
        "selected_sign",
        "selected_offset",
        "selected_inside_fraction",
        "selected_min_limit_margin",
        "ros_sample_min",
        "ros_sample_max",
        "ros_sample_std",
        "status",
        "confidence",
        "blocked_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# ALOHA1 to Trossen Affine Candidate Inference - 2026-07-17",
        "",
        "## Scope",
        "",
        "This is an offline limit-fit analysis. It does not start Isaac Sim and does not touch the real robot.",
        "",
        "Candidate form:",
        "",
        "```text",
        "q_isaac = sign * q_aloha + offset",
        "sign in {+1, -1}",
        "```",
        "",
        "The reference assumption is deliberately explicit:",
        "",
        "```text",
        "ALOHA1 puppet sleep pose maps to the Trossen scaffold zero pose",
        "```",
        "",
        "This assumption is useful for generating candidates, but it is not a proof of physical correctness.",
        "",
        "## Dataset",
        "",
        f"- HDF5 root: `{payload['inputs']['hdf5_root']}`",
        f"- valid episodes: `{payload['summary']['valid_episode_count']}`",
        f"- frames used: `{payload['summary']['frame_count']}`",
        f"- Phase 5 JSON: `{payload['inputs']['phase5_json']}`",
        "",
        "## Gates",
        "",
    ]
    for key, value in payload["gates"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Candidate Table",
            "",
            "| joint | selected sign | selected offset | inside fraction | min margin | status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in payload["candidates"]:
        lines.append(
            "| "
            f"`{row['canonical_name']}` | "
            f"{row['selected_sign']} | "
            f"{row['selected_offset']:.6f} | "
            f"{row['selected_inside_fraction']:.4f} | "
            f"{row['selected_min_limit_margin']:.6f} | "
            f"`{row['status']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A `PASS_LIMIT_FIT_UNIQUE_CANDIDATE` row only means one sign keeps all sampled HDF5 qpos values inside the Trossen runtime joint limit under the stated reference-pose assumption.",
            "",
            "It does **not** prove positive direction, zero offset geometry, end-effector FK, or real command safety.",
            "",
            "Rows marked `AMBIGUOUS_BOTH_SIGNS_FIT_LIMITS` are especially important: limits alone cannot determine the sign for those joints.",
            "",
            "## Next Gate",
            "",
            "The next gate must add geometric evidence: either a trusted ALOHA1 FK chain, matched reference poses with end-effector positions, or a separately reviewed real one-joint positive-direction test plan.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer conservative ALOHA1-to-Trossen sign/offset candidates from HDF5 qpos.")
    parser.add_argument("--hdf5-root", type=Path, default=DEFAULT_HDF5_ROOT)
    parser.add_argument("--phase5-json", type=Path, default=DEFAULT_PHASE5_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    args = parser.parse_args()

    paths = _discover_hdf5(args.hdf5_root, args.episode_limit)
    if not paths:
        raise ValueError(f"No episode.hdf5 files found under {args.hdf5_root}")
    qpos, episode_rows = _load_qpos(paths, args.max_frames_per_episode)
    phase5 = _load_phase5(args.phase5_json)
    candidates = _infer_candidates(qpos, phase5)
    good_count = sum(row["status"] == "PASS_LIMIT_FIT_UNIQUE_CANDIDATE" for row in candidates)
    ambiguous_count = sum(row["status"] == "AMBIGUOUS_BOTH_SIGNS_FIT_LIMITS" for row in candidates)
    fail_count = sum(row["status"] == "FAIL_NO_SIGN_FITS_LIMITS" for row in candidates)
    payload = {
        "inputs": {
            "hdf5_root": _rel(args.hdf5_root),
            "phase5_json": _rel(args.phase5_json),
            "episode_limit": args.episode_limit,
            "max_frames_per_episode": args.max_frames_per_episode,
        },
        "summary": {
            "valid_episode_count": sum(row["status"] == "OK" for row in episode_rows),
            "frame_count": int(qpos.shape[0]),
            "good_count": good_count,
            "ambiguous_count": ambiguous_count,
            "fail_count": fail_count,
        },
        "gates": {
            "real_robot_touched": "PASS_FALSE",
            "isaac_runtime_started": "PASS_FALSE",
            "hdf5_qpos_loaded": "PASS",
            "limit_fit_candidates_generated": "PASS",
            "mapping_candidates_complete": (
                "PASS_ALL_JOINTS_UNIQUE_LIMIT_FIT"
                if good_count == len(ARM_CANONICAL_NAMES) and ambiguous_count == 0 and fail_count == 0
                else f"BLOCKED_{fail_count}_FAIL_{ambiguous_count}_AMBIGUOUS"
            ),
            "sign": "BLOCKED_LIMIT_FIT_IS_NOT_POSITIVE_DIRECTION_EVIDENCE",
            "offset": "BLOCKED_REFERENCE_ASSUMPTION_NOT_GEOMETRIC_PROOF",
            "fk": "BLOCKED_REQUIRES_TRUSTED_FK_OR_REFERENCE_POSES",
        },
        "candidates": candidates,
        "episodes": episode_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "affine_candidates.json"
    csv_path = args.output_dir / "affine_candidates.csv"
    md_path = args.output_dir / "affine_candidates.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(csv_path, candidates)
    _write_markdown(md_path, payload)
    print(
        json.dumps(
            {
                "json": _rel(json_path),
                "csv": _rel(csv_path),
                "markdown": _rel(md_path),
                "summary": payload["summary"],
                "gates": payload["gates"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
