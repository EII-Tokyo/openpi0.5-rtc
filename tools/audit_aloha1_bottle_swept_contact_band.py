#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.aloha1_mapping.bottle_swept_contact_band_certificate import build_certificate
from tools.aloha1_mapping.bottle_swept_contact_band_certificate import render_markdown

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "reports/aloha1_mapping/aloha1_bottle_swept_contact_band_collider_certificate.json"
DEFAULT_MARKDOWN = ROOT / "reports/aloha1_mapping/aloha1_bottle_swept_contact_band_collider_certificate.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    report = build_certificate(args.root)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "task_contact_band": report["task_contact_band"]["status"],
                "candidate_decision": report["candidate_decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
