#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aloha_rl_validation.policy_loader import (
    inspect_checkpoint_schema,
    validate_strict_openpi_native_compatibility,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    schema = inspect_checkpoint_schema(args.checkpoint)
    blockers = validate_strict_openpi_native_compatibility(schema)
    payload = {
        "checkpoint": str(schema.path),
        "action_dim": schema.action_dim,
        "action_horizon": schema.action_horizon,
        "sharded": schema.sharded,
        "safetensor_files": list(schema.safetensor_files),
        "has_trossen_norm_stats": schema.has_trossen_norm_stats,
        "has_robotwin_norm_stats": schema.has_robotwin_norm_stats,
        "total_size": schema.total_size,
        "native_loader_blockers": blockers,
        "strict_schema_pass": not blockers,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if not blockers else 2)


if __name__ == "__main__":
    main()
