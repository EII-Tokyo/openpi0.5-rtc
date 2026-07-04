from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

from scripts.reencode_clean_no_actor_z_rl import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    ReencodeSummary,
    RLTokenPolicyEncoder,
    find_rollout_dir,
    load_manifest_from_npz,
    rewrite_shard_z_rl,
    _shard_rows,
)


@dataclasses.dataclass(frozen=True)
class ManifestReencodeArgs:
    manifest_path: Path
    output_root: Path
    rollout_roots: tuple[Path, ...]
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    config_name: str = DEFAULT_CONFIG
    prompt: str = "Twist off the bottle cap."
    convert_bgr_to_rgb: bool = False
    require_camera: tuple[str, ...] = ("cam_low", "cam_right_wrist")
    input_rewrite_from: str | None = None
    input_rewrite_to: str | None = None
    limit: int | None = None
    execute: bool = False
    overwrite: bool = False


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path} contains a non-object JSONL row")
                rows.append(payload)
    return rows


def _resolve_shard_path(row: dict[str, Any], args: ManifestReencodeArgs) -> Path:
    raw = str(row.get("local_shard_path") or row.get("shard_path") or "")
    if not raw:
        raise ValueError("manifest row is missing shard_path")
    if args.input_rewrite_from is not None and args.input_rewrite_to is not None and raw.startswith(args.input_rewrite_from):
        raw = args.input_rewrite_to + raw[len(args.input_rewrite_from) :]
    return Path(raw)


def _find_rollout(roots: tuple[Path, ...], manifest: dict[str, Any]) -> Path:
    errors: list[str] = []
    for root in roots:
        try:
            return find_rollout_dir(root, manifest)
        except Exception as exc:
            errors.append(f"{root}: {exc}")
    key_region_id = manifest.get("key_region_id")
    raise FileNotFoundError(f"cannot find rollout for key_region_id={key_region_id}: {'; '.join(errors)}")


def reencode_manifest(args: ManifestReencodeArgs) -> ReencodeSummary:
    rows = _read_jsonl(args.manifest_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not args.execute:
        return ReencodeSummary(planned=len(rows), converted=0, skipped={}, output_root=args.output_root)

    encoder = RLTokenPolicyEncoder(
        config_name=args.config_name,
        checkpoint_path=args.checkpoint_path,
        prompt=args.prompt,
        convert_bgr_to_rgb=args.convert_bgr_to_rgb,
        require_camera=args.require_camera,
    )
    converted = 0
    skipped: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        try:
            shard_path = _resolve_shard_path(row, args)
            if not shard_path.exists():
                raise FileNotFoundError(shard_path)
            manifest = load_manifest_from_npz(shard_path)
            rollout_dir = _find_rollout(args.rollout_roots, manifest)
            output_path = args.output_root / shard_path.name
            if output_path.exists() and not args.overwrite:
                skipped["output_exists"] = skipped.get("output_exists", 0) + 1
                continue
            current_z, current_proprio, next_z, next_proprio = encoder.encode_shard(
                rollout_dir,
                manifest,
                clean_rows=_shard_rows(shard_path),
            )
            rewrite_shard_z_rl(
                shard_path,
                output_path,
                z_rl=current_z,
                next_z_rl=next_z,
                proprio=current_proprio,
                next_proprio=next_proprio,
                checkpoint_path=args.checkpoint_path,
                config_name=args.config_name,
            )
            converted += 1
            logging.info("converted %s/%s shard=%s rollout=%s", index, len(rows), shard_path, rollout_dir)
        except Exception as exc:  # pragma: no cover - CLI diagnostics.
            key = type(exc).__name__
            skipped[key] = skipped.get(key, 0) + 1
            logging.exception("failed to reencode manifest row %s/%s: %s", index, len(rows), exc)
    return ReencodeSummary(planned=len(rows), converted=converted, skipped=dict(sorted(skipped.items())), output_root=args.output_root)


def _parse_args() -> ManifestReencodeArgs:
    parser = argparse.ArgumentParser(description="Strictly re-encode RLT replay shards listed in a frozen manifest.")
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--prompt", default="Twist off the bottle cap.")
    parser.add_argument("--convert-bgr-to-rgb", action="store_true")
    parser.add_argument("--require-camera", action="append", default=["cam_low", "cam_right_wrist"])
    parser.add_argument("--input-rewrite-from")
    parser.add_argument("--input-rewrite-to")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    ns = parser.parse_args()
    return ManifestReencodeArgs(
        manifest_path=ns.manifest_path,
        output_root=ns.output_root,
        rollout_roots=tuple(ns.rollout_root),
        checkpoint_path=ns.checkpoint_path,
        config_name=ns.config_name,
        prompt=ns.prompt,
        convert_bgr_to_rgb=ns.convert_bgr_to_rgb,
        require_camera=tuple(ns.require_camera),
        input_rewrite_from=ns.input_rewrite_from,
        input_rewrite_to=ns.input_rewrite_to,
        limit=ns.limit,
        execute=ns.execute,
        overwrite=ns.overwrite,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    summary = reencode_manifest(args)
    logging.info(
        "manifest reencode summary planned=%s converted=%s skipped=%s output_root=%s execute=%s",
        summary.planned,
        summary.converted,
        summary.skipped,
        summary.output_root,
        args.execute,
    )


if __name__ == "__main__":
    main()
