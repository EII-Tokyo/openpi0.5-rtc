from __future__ import annotations

import argparse
import dataclasses
import errno
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ALLOWED_KINDS = {"rlt_raw", "rlt_clean", "expert", "bootstrap"}
ALLOWED_SPLITS = {"unsplit", "train", "holdout"}
EXPECTED_Z_DIM = 2048


@dataclasses.dataclass(frozen=True)
class SourceSpec:
    kind: str
    split: str
    machine: str
    batch: str
    root: Path


@dataclasses.dataclass(frozen=True)
class BuildArgs:
    canonical_root: Path
    manifest_root: Path
    sources: list[SourceSpec]
    overwrite: bool = False
    copy_mode: str = "hardlink"


def parse_source_spec(value: str) -> SourceSpec:
    parts = value.split("|", 4)
    if len(parts) != 5:
        raise ValueError("source spec must be kind|split|machine|batch|root")
    kind, split, machine, batch, root = parts
    if kind not in ALLOWED_KINDS:
        raise ValueError(f"unsupported source kind {kind!r}; expected one of {sorted(ALLOWED_KINDS)}")
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"unsupported source split {split!r}; expected one of {sorted(ALLOWED_SPLITS)}")
    if not machine:
        raise ValueError("source machine must not be empty")
    if not batch:
        raise ValueError("source batch must not be empty")
    return SourceSpec(kind=kind, split=split, machine=machine, batch=batch, root=Path(root).expanduser())


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_manifest(npz: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "manifest" not in npz.files:
        return {}
    raw = npz["manifest"]
    if raw.shape == ():
        value = raw.item()
    else:
        value = raw.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    if isinstance(value, dict):
        return value
    raise ValueError("manifest is not a JSON object")


def _terminal_reward(npz: np.lib.npyio.NpzFile, manifest: dict[str, Any]) -> float | None:
    for key in ("reward", "rewards"):
        if key in npz.files:
            arr = np.asarray(npz[key])
            if arr.size:
                return float(arr.reshape(-1)[-1])
    for key in ("reward", "terminal_reward", "success"):
        if key in manifest:
            try:
                return float(manifest[key])
            except (TypeError, ValueError):
                return None
    return None


def _validate_shard(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with np.load(path, allow_pickle=False) as npz:
            if "z_rl" not in npz.files or "next_z_rl" not in npz.files:
                return None, "missing_z_rl"
            z_rl = np.asarray(npz["z_rl"])
            next_z_rl = np.asarray(npz["next_z_rl"])
            if z_rl.ndim < 2 or next_z_rl.ndim < 2:
                return None, "invalid_z_shape"
            if z_rl.shape[-1] != EXPECTED_Z_DIM or next_z_rl.shape[-1] != EXPECTED_Z_DIM:
                return None, "invalid_z_dim"
            if not np.isfinite(z_rl).all() or not np.isfinite(next_z_rl).all():
                return None, "non_finite_z_rl"
            manifest = _load_manifest(npz)
            z_rl_source = manifest.get("z_rl_source")
            replay_state_grain = manifest.get("replay_state_grain")
            if z_rl_source == "rl_token_reencoded_aligned_to_proprio_segments":
                return None, "fixed_segments_not_paper_subsampled"
            if z_rl_source == "rl_token_reencoded" and replay_state_grain != "paper_subsampled_anchor":
                return None, "missing_paper_subsampled_anchor_grain"
            key_region_id = str(
                manifest.get("key_region_id")
                or manifest.get("segment_id")
                or manifest.get("id")
                or path.stem.replace("key_region_", "")
            )
            return {
                "key_region_id": key_region_id,
                "rows": int(z_rl.shape[0]),
                "z_dim": int(z_rl.shape[-1]),
                "reward": _terminal_reward(npz, manifest),
                "rl_token_config": (
                    manifest.get("rl_token_config")
                    or manifest.get("rl_token_config_name")
                    or manifest.get("rlt_rl_token_config")
                ),
                "z_rl_source": z_rl_source,
                "replay_state_grain": replay_state_grain,
                "rl_token_checkpoint_path": manifest.get("rl_token_checkpoint_path")
                or manifest.get("rlt_rl_token_checkpoint_path"),
                "manifest": manifest,
            }, None
    except Exception:
        return None, "unreadable_npz"


def _discover_npz(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.npz") if path.is_file())


def _destination_path(canonical_root: Path, source: SourceSpec, shard: Path) -> Path:
    split_dir = source.split if source.split != "unsplit" else "all"
    return canonical_root / source.kind / source.batch / split_dir / "shards" / shard.name


def _link_or_copy(src: Path, dst: Path, *, overwrite: bool, copy_mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if overwrite:
            dst.unlink()
        else:
            return "exists"
    if copy_mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    if copy_mode != "hardlink":
        raise ValueError("copy_mode must be 'hardlink' or 'copy'")
    try:
        dst.hardlink_to(src)
        return "hardlink"
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES}:
            raise
        shutil.copy2(src, dst)
        return "copy"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")


def build_canonical_dataset(args: BuildArgs) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    copy_counts: Counter[str] = Counter()
    seen_keys: Counter[str] = Counter()

    for source in args.sources:
        for shard in _discover_npz(source.root):
            metadata, reason = _validate_shard(shard)
            if metadata is None:
                skipped[reason or "invalid"] += 1
                continue
            dst = _destination_path(args.canonical_root, source, shard)
            copy_result = _link_or_copy(shard, dst, overwrite=args.overwrite, copy_mode=args.copy_mode)
            copy_counts[copy_result] += 1
            key_region_id = metadata["key_region_id"]
            seen_keys[key_region_id] += 1
            row = {
                "key_region_id": key_region_id,
                "canonical_path": str(dst),
                "source_path": str(shard),
                "kind": source.kind,
                "split": source.split,
                "machine": source.machine,
                "batch": source.batch,
                "reward": metadata["reward"],
                "rows": metadata["rows"],
                "z_dim": metadata["z_dim"],
                "z_rl_source": metadata["z_rl_source"],
                "replay_state_grain": metadata["replay_state_grain"],
                "rl_token_config": metadata["rl_token_config"],
                "rl_token_checkpoint_path": metadata["rl_token_checkpoint_path"],
            }
            all_rows.append(row)

    all_rows.sort(key=lambda row: (row["kind"], row["batch"], row["split"], row["key_region_id"], row["canonical_path"]))
    train_rows = [row for row in all_rows if row["split"] == "train"]
    holdout_rows = [row for row in all_rows if row["split"] == "holdout"]

    _write_jsonl(args.manifest_root / "canonical_2048_all.jsonl", all_rows)
    _write_jsonl(args.manifest_root / "canonical_2048_train.jsonl", train_rows)
    _write_jsonl(args.manifest_root / "canonical_2048_holdout.jsonl", holdout_rows)

    by_kind = Counter(row["kind"] for row in all_rows)
    by_split = Counter(row["split"] for row in all_rows)
    by_reward = Counter("unknown" if row["reward"] is None else str(int(row["reward"])) for row in all_rows)
    duplicates = {key: count for key, count in sorted(seen_keys.items()) if count > 1}
    train_ids = {row["key_region_id"] for row in train_rows}
    holdout_ids = {row["key_region_id"] for row in holdout_rows}
    summary: dict[str, Any] = {
        "canonical_root": str(args.canonical_root),
        "manifest_root": str(args.manifest_root),
        "total_rows": len(all_rows),
        "total_transitions": int(sum(row["rows"] for row in all_rows)),
        "by_kind": dict(sorted(by_kind.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_reward": dict(sorted(by_reward.items())),
        "skipped": dict(sorted(skipped.items())),
        "copy_counts": dict(sorted(copy_counts.items())),
        "duplicate_key_region_ids": duplicates,
        "train_holdout_overlap_key_region_ids": sorted(train_ids & holdout_ids),
        "sources": [dataclasses.asdict(source) for source in args.sources],
    }
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    (args.manifest_root / "inventory.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args() -> BuildArgs:
    parser = argparse.ArgumentParser(
        description="Build a canonical lower+right 4-layer z_rl=2048 RLT replay directory and manifests."
    )
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True, help="kind|split|machine|batch|root")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-mode", choices=["hardlink", "copy"], default="hardlink")
    ns = parser.parse_args()
    return BuildArgs(
        canonical_root=ns.canonical_root,
        manifest_root=ns.manifest_root,
        sources=[parse_source_spec(value) for value in ns.source],
        overwrite=ns.overwrite,
        copy_mode=ns.copy_mode,
    )


def main() -> None:
    summary = build_canonical_dataset(_parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
