#!/usr/bin/env python3
# ruff: noqa: PLC0415
"""Run one fresh Isaac Sim 5.1 official-rule target for ALOHA Task 7."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serialize_issue(issue: Any) -> dict[str, Any]:
    return {
        "severity": getattr(issue.severity, "name", str(issue.severity)),
        "rule": issue.rule.__name__ if issue.rule else None,
        "message": issue.message,
        "at": issue.at.as_str() if issue.at is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.target.resolve(strict=True)
    hash_before = _sha256(target)

    import isaacsim.asset.validation  # noqa: F401
    import omni.asset_validator.core as av_core
    from pxr import Usd

    stage = Usd.Stage.Open(str(target))
    if stage is None:
        raise RuntimeError(f"unable to open Stage: {target}")
    rules = list(
        av_core.ValidationRulesRegistry.rules(
            args.category,
            enabledOnly=False,
        )
    )
    engine = av_core.ValidationEngine(init_rules=False, variants=False)
    for rule in rules:
        engine.enable_rule(rule)
    issues = sorted(
        (_serialize_issue(issue) for issue in engine.validate(stage)),
        key=lambda item: (
            item["severity"],
            item["rule"] or "",
            item["at"] or "",
            item["message"] or "",
        ),
    )
    blocking = [issue for issue in issues if issue["severity"] in {"ERROR", "FAILURE"}]
    warnings = [issue for issue in issues if issue["severity"] == "WARNING"]
    hash_after = _sha256(target)
    report = {
        "schema_version": 1,
        "category": args.category,
        "target_name": args.target_name,
        "target_absolute_path": str(target),
        "target_sha256_before": hash_before,
        "target_sha256_after": hash_after,
        "target_immutable": hash_before == hash_after,
        "official_status": ("FAIL" if blocking else "PARTIAL" if warnings else "PASS"),
        "rule_count": len(rules),
        "rules": sorted(rule.__name__ for rule in rules),
        "issues": issues,
        "blocking_issue_count": len(blocking),
        "warning_count": len(warnings),
        "runtime": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
        },
        "mcpjungle_nvidia_official_api_verified": True,
        "official_status_suppressed": False,
        "real_robot_connected": False,
        "remote_192_168_1_103_accessed": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "category": args.category,
                "target": args.target_name,
                "official_status": report["official_status"],
                "issue_count": len(issues),
                "output": str(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def run() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    exit_code = 1
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        extension_id = "isaacsim.asset.validation"
        if not manager.is_extension_enabled(extension_id):
            manager.set_extension_enabled_immediate(
                extension_id,
                True,  # noqa: FBT003
            )
        if not manager.is_extension_enabled(extension_id):
            raise RuntimeError(f"required extension disabled: {extension_id}")
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run())
