from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

from openpi.training import rlt_trainable_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a frozen RLT trainable replay manifest from the segment DB.")
    parser.add_argument("--segment-db-path", required=True, type=pathlib.Path)
    parser.add_argument("--clean-root", required=True, type=pathlib.Path)
    parser.add_argument("--output-path", required=True, type=pathlib.Path)
    args = parser.parse_args()

    result = rlt_trainable_manifest.build_manifest_from_segment_db(
        args.segment_db_path,
        output_path=args.output_path,
        clean_root=args.clean_root,
    )
    print(
        json.dumps(
            {
                "output_path": str(result.output_path),
                "summary": dataclasses.asdict(result.summary),
                "skipped_by_reason": result.skipped_by_reason,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
