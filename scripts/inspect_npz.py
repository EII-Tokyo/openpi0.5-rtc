from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from numpy.lib import format as npy_format


def _read_npy_header(npz: zipfile.ZipFile, member: str) -> tuple[tuple[int, ...], np.dtype]:
    with npz.open(member) as f:
        version = npy_format.read_magic(f)
        if version == (1, 0):
            shape, _, dtype = npy_format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, _, dtype = npy_format.read_array_header_2_0(f)
        else:
            # Newer numpy may create v3 headers for unicode field names. Fall back
            # to loading the array in that uncommon case.
            with npz.open(member) as fallback:
                array = npy_format.read_array(fallback, allow_pickle=False)
            return array.shape, array.dtype
    return shape, dtype


def _summarize_header(name: str, shape: tuple[int, ...], dtype: np.dtype) -> None:
    print(f"\n{name}")
    print(f"  shape: {shape}")
    print(f"  dtype: {dtype}")


def _summarize_value(name: str, value: np.ndarray, *, show_values: bool, stats: bool) -> None:
    if value.shape == ():
        item = value.item()
        print(f"  value: {item!r}")
        if name == "metadata_json":
            try:
                print("  parsed:")
                print(json.dumps(json.loads(str(item)), indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                pass
        return

    if value.size == 0:
        print("  empty")
        return

    if stats and (np.issubdtype(value.dtype, np.number) or value.dtype == np.bool_):
        print(f"  min: {np.min(value)}")
        print(f"  max: {np.max(value)}")
        if np.issubdtype(value.dtype, np.number):
            print(f"  mean: {float(np.mean(value)):.6g}")
            print(f"  std: {float(np.std(value)):.6g}")

    if show_values:
        print("  values:")
        print(value)
    else:
        flat = value.reshape(-1)
        print("  first_values:")
        print(flat[: min(10, flat.shape[0])])


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect keys and values in a .npz file.")
    parser.add_argument("npz_path", type=Path)
    parser.add_argument("--load", action="append", default=[], help="Load and print a specific key. Can be passed more than once.")
    parser.add_argument("--values", action="store_true", help="Print full values instead of only first 10 flattened values.")
    parser.add_argument("--stats", action="store_true", help="Compute min/max/mean/std. This can be slow for large image arrays.")
    args = parser.parse_args()

    with zipfile.ZipFile(args.npz_path) as npz:
        members = [name for name in npz.namelist() if name.endswith(".npy")]
        keys = [Path(name).stem for name in members]
        print(f"path: {args.npz_path}")
        print(f"keys ({len(keys)}):")
        for key in keys:
            print(f"  {key}")

        for member, key in zip(members, keys, strict=True):
            shape, dtype = _read_npy_header(npz, member)
            _summarize_header(key, shape, dtype)

    load_keys = set(args.load)
    if args.values or args.stats:
        load_keys.update(keys)

    if load_keys:
        with np.load(args.npz_path, allow_pickle=False) as data:
            for key in keys:
                if key not in load_keys:
                    continue
                print(f"\nloaded {key}")
                _summarize_value(key, data[key], show_values=args.values, stats=args.stats)


if __name__ == "__main__":
    main()
