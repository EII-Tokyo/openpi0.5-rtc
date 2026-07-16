from __future__ import annotations

import argparse
import json
from pathlib import Path

from aloha_isaac_replay.data.hdf5_reader import inspect_episode
from aloha_isaac_replay.data.hdf5_reader import to_jsonable
from aloha_isaac_replay.validation.time_axis import validate_50hz_timestamps


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect one raw ALOHA HDF5 episode without loading images.")
    parser.add_argument("--episode", required=True, help="Path to episode.hdf5")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    inspection = inspect_episode(args.episode)
    payload = to_jsonable(inspection)
    payload["time_axis_50hz"] = validate_50hz_timestamps_from_summary_hint(inspection.path)

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
    print(text)
    return 0


def validate_50hz_timestamps_from_summary_hint(path: str) -> dict[str, float | bool | str]:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as h5:
        if "timestamps" not in h5:
            return {"valid": False, "reason_code": "missing_timestamps"}
        return validate_50hz_timestamps(np.asarray(h5["timestamps"]))


if __name__ == "__main__":
    raise SystemExit(main())

