from __future__ import annotations

import argparse
import json
import pathlib
import shutil
from typing import Any

from voice_assistant_web.backend.app.rlt_key_region_crop import crop_key_region_files
from voice_assistant_web.backend.app.rlt_key_region_crop import rescore_key_region_files


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_edits(path: pathlib.Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    edits: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            key_region_id = str(payload.get("key_region_id") or "")
            if not key_region_id:
                raise ValueError(f"{path}:{line_number} missing key_region_id")
            edits[key_region_id] = payload
    return edits


def _relative_rollout_dir(raw_root: pathlib.Path, rollout_dir: pathlib.Path) -> pathlib.Path:
    try:
        return rollout_dir.relative_to(raw_root / "rollouts" / "key_regions")
    except ValueError:
        return rollout_dir.name


def _resolve_raw_shard(raw_root: pathlib.Path, rollout_dir: pathlib.Path, manifest: dict[str, Any]) -> pathlib.Path:
    shard_path = manifest.get("shard_path")
    if shard_path:
        path = pathlib.Path(str(shard_path))
        if path.exists():
            return path
        if path.is_absolute():
            task = str(manifest.get("task") or "")
            key_region_id = str(manifest.get("key_region_id") or "")
            date = rollout_dir.parent.parent.name
            candidate = raw_root / "replay" / "rlt_key_regions" / task / date / "shards" / f"key_region_{key_region_id}.npz"
            if candidate.exists():
                return candidate
        candidate = raw_root / path
        if candidate.exists():
            return candidate
    task = str(manifest.get("task") or "")
    key_region_id = str(manifest.get("key_region_id") or "")
    date = rollout_dir.parent.parent.name
    candidate = raw_root / "replay" / "rlt_key_regions" / task / date / "shards" / f"key_region_{key_region_id}.npz"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cannot resolve replay shard for key_region_id={key_region_id}")


def _write_manifest_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_offline_dataset(
    *,
    raw_root: pathlib.Path,
    output_root: pathlib.Path,
    edits_path: pathlib.Path | None = None,
    copy_videos: bool = False,
) -> dict[str, int]:
    edits = _load_edits(edits_path)
    raw_rollouts = raw_root / "rollouts" / "key_regions"
    rows_by_manifest: dict[pathlib.Path, list[dict[str, Any]]] = {}
    written = 0
    skipped = 0

    for manifest_path in sorted(raw_rollouts.glob("*/*/*/key_region_*/manifest.json")):
        raw_rollout_dir = manifest_path.parent
        manifest = _load_json(manifest_path)
        key_region_id = str(manifest.get("key_region_id") or "")
        edit = edits.get(key_region_id, {})
        if edit.get("voided") is True or manifest.get("voided") is True or manifest.get("train_eligible") is False:
            skipped += 1
            continue

        rel_rollout = _relative_rollout_dir(raw_root, raw_rollout_dir)
        output_rollout_dir = output_root / "rollouts" / "key_regions" / rel_rollout
        output_rollout_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, output_rollout_dir / "manifest.json")
        if copy_videos:
            for video_path in raw_rollout_dir.glob("*.mp4"):
                shutil.copy2(video_path, output_rollout_dir / video_path.name)

        raw_shard = _resolve_raw_shard(raw_root, raw_rollout_dir, manifest)
        task = str(manifest.get("task") or rel_rollout.parts[0])
        date = rel_rollout.parts[1]
        output_shard = output_root / "replay" / "rlt_key_regions" / task / date / "shards" / raw_shard.name

        if "start_sec" in edit or "end_sec" in edit:
            start_sec = float(edit.get("start_sec", 0.0))
            end_sec = float(edit.get("end_sec", manifest.get("duration_seconds") or 0.0))
            result = crop_key_region_files(output_rollout_dir, raw_shard, output_shard, start_sec=start_sec, end_sec=end_sec)
        else:
            output_shard.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_shard, output_shard)
            result = _load_json(output_rollout_dir / "manifest.json")
            result["shard_path"] = str(output_shard)

        if "reward" in edit:
            result = rescore_key_region_files(output_rollout_dir, output_shard, reward=int(edit["reward"]))
        else:
            current_manifest = _load_json(output_rollout_dir / "manifest.json")
            current_manifest["shard_path"] = str(output_shard)
            (output_rollout_dir / "manifest.json").write_text(
                json.dumps(current_manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            result = current_manifest if result is None else result

        manifest_jsonl = output_shard.parent.parent / "manifest.jsonl"
        rows_by_manifest.setdefault(manifest_jsonl, []).append({**result, "shard_path": str(output_shard)})
        written += 1

    for path, rows in rows_by_manifest.items():
        _write_manifest_jsonl(path, rows)

    return {"written": written, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare locally edited RLT replay shards for offline actor/critic training.")
    parser.add_argument("--raw-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--edits-path", type=pathlib.Path)
    parser.add_argument("--copy-videos", action="store_true")
    args = parser.parse_args()
    summary = prepare_offline_dataset(
        raw_root=args.raw_root,
        output_root=args.output_root,
        edits_path=args.edits_path,
        copy_videos=args.copy_videos,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
