from __future__ import annotations

import dataclasses
import json
import pathlib
from collections import Counter
from typing import Any

import tyro


@dataclasses.dataclass
class Args:
    source_manifest: pathlib.Path
    output_dir: pathlib.Path
    expected_count: int | None = None
    output_name: str = "no_actor_clean_bootstrap"
    remote_shard_root: str | None = None
    remote_manifest_name: str | None = None


@dataclasses.dataclass(frozen=True)
class BootstrapResult:
    manifest_path: pathlib.Path
    summary_path: pathlib.Path
    summary: dict[str, Any]
    skipped_by_reason: dict[str, int]
    remote_manifest_path: pathlib.Path | None = None


def prepare_bootstrap(args: Args) -> BootstrapResult:
    rows = _read_jsonl(args.source_manifest)
    output_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    seen_keys: set[str] = set()

    for row in rows:
        if not _is_no_actor_row(row):
            skipped["not_no_actor"] += 1
            continue
        shard_path = pathlib.Path(str(row.get("shard_path") or "")).expanduser()
        if not shard_path.exists():
            skipped["missing_shard"] += 1
            continue
        dedup_key = _dedup_key(row, shard_path)
        if dedup_key in seen_keys:
            if str(row.get("key_region_id") or ""):
                skipped["duplicate_key_region_id"] += 1
            else:
                skipped["duplicate_shard_path"] += 1
            continue
        seen_keys.add(dedup_key)
        output_rows.append(_bootstrap_row(row, shard_path))

    if args.expected_count is not None and len(output_rows) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} bootstrap shards, got {len(output_rows)}")

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{args.output_name}.jsonl"
    summary_path = output_dir / f"{args.output_name}.summary.json"
    remote_manifest_path = None
    _write_jsonl(manifest_path, output_rows)
    if args.remote_shard_root:
        remote_manifest_name = args.remote_manifest_name or f"{args.output_name}.remote.jsonl"
        remote_manifest_path = output_dir / remote_manifest_name
        _write_jsonl(remote_manifest_path, _remote_rows(output_rows, args.remote_shard_root))
    summary = _summarize(output_rows, skipped)
    summary.update(
        {
            "source_manifest": str(args.source_manifest.expanduser().resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "summary_path": str(summary_path.resolve()),
        }
    )
    if remote_manifest_path is not None:
        summary["remote_manifest_path"] = str(remote_manifest_path.resolve())
        summary["remote_shard_root"] = args.remote_shard_root
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return BootstrapResult(
        manifest_path=manifest_path,
        summary_path=summary_path,
        summary=summary,
        skipped_by_reason=dict(sorted(skipped.items())),
        remote_manifest_path=remote_manifest_path,
    )


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.expanduser().open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _is_no_actor_row(row: dict[str, Any]) -> bool:
    selection = row.get("selection")
    if isinstance(selection, dict) and selection.get("selected_for_rtc_only_no_actor") is False:
        return False
    inferred = row.get("inferred_behavior")
    if isinstance(inferred, dict) and inferred.get("no_actor_from_action_reference_equality") is False:
        return False
    runtime = row.get("runtime_config_record")
    if isinstance(runtime, dict):
        if runtime.get("actor_checkpoint_path"):
            return False
        applied_ratio = runtime.get("rlt_actor_applied_ratio")
        if applied_ratio is not None and float(applied_ratio) > 0.0:
            return False
    if row.get("actor_enabled") is True:
        return False
    return True


def _dedup_key(row: dict[str, Any], shard_path: pathlib.Path) -> str:
    key_region_id = str(row.get("key_region_id") or "")
    if key_region_id:
        return f"key_region_id:{key_region_id}"
    source_shard_path = str(row.get("source_shard_path") or "")
    if source_shard_path:
        return f"source_shard_path:{pathlib.Path(source_shard_path).expanduser().resolve()}"
    return f"shard_path:{shard_path.resolve()}"


def _bootstrap_row(row: dict[str, Any], shard_path: pathlib.Path) -> dict[str, Any]:
    result = {
        "bootstrap_source": "no_actor_clean",
        "key_region_id": row.get("key_region_id"),
        "batch": row.get("batch") or "unknown",
        "phase": row.get("phase"),
        "reward": row.get("reward"),
        "shard_path": str(shard_path.resolve()),
        "source_shard_path": row.get("source_shard_path") or str(shard_path.resolve()),
        "num_replay_transitions": int(row.get("num_replay_transitions") or 0),
        "success_episodes": int(row.get("success_episodes") or 0),
        "failure_episodes": int(row.get("failure_episodes") or 0),
        "selection_reason": _selection_reason(row),
    }
    return {key: value for key, value in result.items() if value is not None}


def _selection_reason(row: dict[str, Any]) -> str:
    selection = row.get("selection")
    if isinstance(selection, dict) and selection.get("selection_reason"):
        return str(selection["selection_reason"])
    inferred = row.get("inferred_behavior")
    if isinstance(inferred, dict) and inferred.get("inference_method"):
        return str(inferred["inference_method"])
    return "source_manifest_no_actor"


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _remote_rows(rows: list[dict[str, Any]], remote_shard_root: str) -> list[dict[str, Any]]:
    filenames: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    root = pathlib.PurePosixPath(remote_shard_root)
    for index, row in enumerate(rows):
        local_path = pathlib.Path(str(row["shard_path"]))
        filename = local_path.name
        filenames[filename] += 1
        if filenames[filename] > 1:
            filename = f"{index:05d}_{filename}"
        remote_row = dict(row)
        remote_row["local_shard_path"] = str(local_path)
        remote_row["shard_path"] = str(root / filename)
        output.append(remote_row)
    return output


def _summarize(rows: list[dict[str, Any]], skipped: Counter[str]) -> dict[str, Any]:
    by_batch: dict[str, Counter[str]] = {}
    total = Counter()
    for row in rows:
        batch = str(row.get("batch") or "unknown")
        bucket = by_batch.setdefault(batch, Counter())
        _accumulate(bucket, row)
        _accumulate(total, row)
    return {
        "num_shards": int(total["num_shards"]),
        "num_transitions": int(total["num_transitions"]),
        "success_episodes": int(total["success_episodes"]),
        "failure_episodes": int(total["failure_episodes"]),
        "by_batch": {batch: dict(sorted(counter.items())) for batch, counter in sorted(by_batch.items())},
        "skipped_by_reason": dict(sorted(skipped.items())),
    }


def _accumulate(counter: Counter[str], row: dict[str, Any]) -> None:
    counter["num_shards"] += 1
    counter["num_transitions"] += int(row.get("num_replay_transitions") or 0)
    counter["success_episodes"] += int(row.get("success_episodes") or 0)
    counter["failure_episodes"] += int(row.get("failure_episodes") or 0)


def main(args: Args) -> None:
    result = prepare_bootstrap(args)
    print(json.dumps({"manifest_path": str(result.manifest_path), "summary": result.summary}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
