from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPLAY_KEYS = (
    "z_rl",
    "proprio",
    "action",
    "reference_action",
    "reward_seq",
    "next_z_rl",
    "next_proprio",
    "next_reference_action",
    "done",
)


@dataclasses.dataclass(frozen=True)
class AugmentArgs:
    manifest_path: Path
    output_root: Path
    output_manifest_path: Path
    dense_start_progress: float = 0.5
    dense_min_reward: float = 0.2
    create_hard_negatives: bool = True
    hard_negative_ratio: float = 1.0
    seed: int = 0
    overwrite: bool = False


@dataclasses.dataclass(frozen=True)
class AugmentSummary:
    copied_shards: int
    negative_shards: int
    skipped: dict[str, int]
    output_manifest_path: Path


def augment_no_actor_q_replay(args: AugmentArgs) -> AugmentSummary:
    if not 0.0 <= args.dense_start_progress <= 1.0:
        raise ValueError("dense_start_progress must be in [0, 1]")
    if not 0.0 <= args.dense_min_reward <= 1.0:
        raise ValueError("dense_min_reward must be in [0, 1]")
    if args.hard_negative_ratio < 0.0:
        raise ValueError("hard_negative_ratio must be non-negative")

    rows = _read_manifest(args.manifest_path)
    args.output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.output_manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_manifest_path} exists. Pass --overwrite to replace it.")

    source_rows: list[tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]] = []
    skipped: dict[str, int] = {}
    for row in rows:
        path = Path(row["shard_path"])
        try:
            arrays, manifest = _load_shard(path)
        except Exception:
            skipped["unreadable_source"] = skipped.get("unreadable_source", 0) + 1
            continue
        source_rows.append((row, arrays, manifest))

    rng = np.random.default_rng(args.seed)
    success_sources = [
        (row, arrays, manifest)
        for row, arrays, manifest in source_rows
        if _episode_reward(arrays, manifest) > 0.0
    ]

    output_rows: list[dict[str, Any]] = []
    copied = 0
    negatives = 0
    for index, (row, arrays, manifest) in enumerate(source_rows):
        reward = _episode_reward(arrays, manifest)
        copied_arrays = {key: np.asarray(value).copy() for key, value in arrays.items()}
        copied_manifest = dict(manifest)
        copied_manifest.update(
            {
                "reward": int(reward > 0.0),
                "reward_placement": "terminal_last_train_step",
                "augmentation": {
                    "dense_reward": bool(reward > 0.0),
                    "hard_negative": False,
                    "reward_mode": "dense_progress_terminal" if reward > 0.0 else "zero",
                    "source_manifest_path": str(args.manifest_path),
                },
            }
        )
        if reward > 0.0:
            copied_arrays["reward_seq"] = _dense_reward_seq(
                copied_arrays["reward_seq"].shape,
                start_progress=args.dense_start_progress,
                min_reward=args.dense_min_reward,
            )
        else:
            copied_arrays["reward_seq"] = np.zeros_like(copied_arrays["reward_seq"], dtype=np.float32)
        copied_arrays["done"] = _terminal_done(copied_arrays["done"].shape[0])
        copied_path = _output_path(args.output_root, Path(row["shard_path"]), suffix="dense")
        copied_manifest["shard_path"] = str(copied_path)
        copied_manifest["replay_array_shapes"] = _array_shapes(copied_arrays)
        _write_shard(copied_path, copied_arrays, copied_manifest, overwrite=args.overwrite)
        output_rows.append({**row, **copied_manifest, "shard_path": str(copied_path)})
        copied += 1

        if not args.create_hard_negatives or reward <= 0.0:
            continue
        if rng.random() > min(args.hard_negative_ratio, 1.0):
            continue
        donor = _choose_donor(success_sources, exclude_path=Path(row["shard_path"]), rng=rng)
        if donor is None:
            skipped["no_negative_donor"] = skipped.get("no_negative_donor", 0) + 1
            continue
        _donor_row, donor_arrays, _donor_manifest = donor
        negative_arrays = {key: np.asarray(value).copy() for key, value in arrays.items()}
        negative_arrays["action"] = _mismatched_action(
            negative_arrays["action"],
            np.asarray(donor_arrays["action"], dtype=np.float32),
        )
        negative_arrays["reference_action"] = np.asarray(arrays["reference_action"], dtype=np.float32).copy()
        negative_arrays["reward_seq"] = np.zeros_like(negative_arrays["reward_seq"], dtype=np.float32)
        negative_arrays["done"] = _terminal_done(negative_arrays["done"].shape[0])
        negative_manifest = dict(manifest)
        negative_manifest.update(
            {
                "reward": 0,
                "label": "hard_negative_action_mismatch",
                "source_type": "hard_negative",
                "phase": "human_expert_hard_negative",
                "batch": "hard_negative",
                "actor_enabled": False,
                "rlt_actor_applied_ratio": 0.0,
                "intervention_scale": 0.0,
                "reward_placement": "terminal_last_train_step",
                "reference_action_source": "source_human_action_no_actor",
                "action_source": "mismatched_human_action_from_other_success_crop",
                "augmentation": {
                    "dense_reward": False,
                    "hard_negative": True,
                    "reward_mode": "hard_negative_zero",
                    "source_manifest_path": str(args.manifest_path),
                    "source_shard_path": str(Path(row["shard_path"]).resolve()),
                    "donor_shard_path": str(Path(donor[0]["shard_path"]).resolve()),
                },
            }
        )
        delta = np.abs(negative_arrays["action"] - negative_arrays["reference_action"])
        negative_manifest["action_reference_delta"] = {
            "all_max_abs": float(np.max(delta)) if delta.size else 0.0,
            "all_p95_abs": float(np.percentile(delta, 95)) if delta.size else 0.0,
        }
        negative_path = _output_path(args.output_root, Path(row["shard_path"]), suffix=f"hard_negative_{index:06d}")
        negative_manifest["shard_path"] = str(negative_path)
        negative_manifest["replay_array_shapes"] = _array_shapes(negative_arrays)
        _write_shard(negative_path, negative_arrays, negative_manifest, overwrite=args.overwrite)
        output_rows.append({**row, **negative_manifest, "shard_path": str(negative_path)})
        negatives += 1

    with args.output_manifest_path.open("w", encoding="utf-8") as file:
        for row in output_rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return AugmentSummary(copied, negatives, dict(sorted(skipped.items())), args.output_manifest_path)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_shard(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in REPLAY_KEYS}
        raw = data["manifest"]
        value = raw.item() if raw.shape == () else raw.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return arrays, json.loads(str(value))


def _episode_reward(arrays: dict[str, np.ndarray], manifest: dict[str, Any]) -> float:
    if "reward" in manifest:
        return float(manifest["reward"])
    return float(np.max(np.asarray(arrays["reward_seq"], dtype=np.float32)))


def _dense_reward_seq(shape: tuple[int, ...], *, start_progress: float, min_reward: float) -> np.ndarray:
    rows, horizon = int(shape[0]), int(shape[1])
    reward_seq = np.zeros((rows, horizon), dtype=np.float32)
    if rows <= 0:
        return reward_seq
    start_row = min(max(int(np.ceil(start_progress * max(rows - 1, 1))), 0), rows - 1)
    denom = max(rows - 1 - start_row, 1)
    for row in range(rows):
        progress = 1.0 if rows == 1 else row / (rows - 1)
        if progress < start_progress:
            continue
        alpha = (row - start_row) / denom
        value = min_reward + (1.0 - min_reward) * alpha
        reward_seq[row, horizon - 1] = float(np.clip(value, 0.0, 1.0))
    reward_seq[-1, horizon - 1] = 1.0
    return reward_seq


def _terminal_done(rows: int) -> np.ndarray:
    done = np.zeros((rows,), dtype=np.bool_)
    if rows > 0:
        done[-1] = True
    return done


def _choose_donor(
    sources: list[tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]],
    *,
    exclude_path: Path,
    rng: np.random.Generator,
):
    candidates = [item for item in sources if Path(item[0]["shard_path"]).resolve() != exclude_path.resolve()]
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))]


def _mismatched_action(target: np.ndarray, donor: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32)
    donor = np.asarray(donor, dtype=np.float32)
    if donor.shape == target.shape:
        return donor.copy()
    rows, horizon, dim = target.shape
    out = np.empty_like(target, dtype=np.float32)
    for row in range(rows):
        donor_row = min(row, donor.shape[0] - 1)
        out[row] = donor[donor_row, :horizon, :dim]
    return out


def _array_shapes(arrays: dict[str, np.ndarray]) -> dict[str, list[int]]:
    return {key: list(np.asarray(value).shape) for key, value in arrays.items()}


def _output_path(output_root: Path, source_path: Path, *, suffix: str) -> Path:
    digest = hashlib.sha1(f"{source_path.resolve()}:{suffix}".encode("utf-8")).hexdigest()[:12]
    dataset = source_path.parent.name if source_path.parent.name else "shards"
    return (output_root / dataset / f"{source_path.stem}.{suffix}.{digest}.npz").resolve()


def _write_shard(path: Path, arrays: dict[str, np.ndarray], manifest: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {key: np.asarray(value) for key, value in arrays.items()}
    payload["manifest"] = np.asarray(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    with tmp_path.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    tmp_path.replace(path)


def _parse_args() -> AugmentArgs:
    parser = argparse.ArgumentParser(description="Augment no-actor Q replay with dense rewards and hard negatives.")
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest-path", type=Path, required=True)
    parser.add_argument("--dense-start-progress", type=float, default=0.5)
    parser.add_argument("--dense-min-reward", type=float, default=0.2)
    parser.add_argument("--no-hard-negatives", action="store_true")
    parser.add_argument("--hard-negative-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    ns = parser.parse_args()
    return AugmentArgs(
        manifest_path=ns.manifest_path,
        output_root=ns.output_root,
        output_manifest_path=ns.output_manifest_path,
        dense_start_progress=ns.dense_start_progress,
        dense_min_reward=ns.dense_min_reward,
        create_hard_negatives=not ns.no_hard_negatives,
        hard_negative_ratio=ns.hard_negative_ratio,
        seed=ns.seed,
        overwrite=ns.overwrite,
    )


def main() -> None:
    summary = augment_no_actor_q_replay(_parse_args())
    print(json.dumps(dataclasses.asdict(summary), ensure_ascii=False, default=str, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
