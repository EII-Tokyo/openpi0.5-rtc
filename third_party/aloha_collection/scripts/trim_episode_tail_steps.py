#!/usr/bin/env python3
"""Trim the last N timesteps from every episode_*.hdf5 in a folder.

Default input:
  /home/eii/aloha-2.0/aloha_data/aloha_stationary/2026-04-21_direction

Default output:
  /home/eii/aloha-2.0/aloha_data/2026-04-21_direction_trim65

The original folder is not modified. Every dataset whose first dimension matches
/observations/qpos is trimmed by the same amount, so episode contents stay
time-aligned. Other datasets, groups, and attributes are copied unchanged.
"""

import argparse
from pathlib import Path

import h5py


DEFAULT_SOURCE = Path(
    "/home/eii/aloha-2.0/aloha_data/aloha_stationary/2026-04-21_direction"
)
DEFAULT_OUTPUT = Path("/home/eii/aloha-2.0/aloha_data/2026-04-21_direction_trim65")


def episode_sort_key(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 10**12


def copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def make_dataset_kwargs(src: h5py.Dataset, shape: tuple[int, ...]) -> dict:
    kwargs = {}
    if src.chunks is not None:
        kwargs["chunks"] = tuple(min(chunk, dim) if dim > 0 else chunk for chunk, dim in zip(src.chunks, shape))
    if src.compression is not None:
        kwargs["compression"] = src.compression
        kwargs["compression_opts"] = src.compression_opts
    if src.shuffle:
        kwargs["shuffle"] = True
    if src.fletcher32:
        kwargs["fletcher32"] = True
    if src.fillvalue is not None:
        kwargs["fillvalue"] = src.fillvalue
    return kwargs


def copy_dataset(
    src: h5py.Dataset,
    dst_group: h5py.Group,
    name: str,
    old_steps: int,
    new_steps: int,
    batch_size: int,
) -> None:
    trim_this = src.ndim >= 1 and src.shape[0] == old_steps
    dst_shape = (new_steps, *src.shape[1:]) if trim_this else src.shape
    dst = dst_group.create_dataset(
        name,
        shape=dst_shape,
        dtype=src.dtype,
        **make_dataset_kwargs(src, dst_shape),
    )
    copy_attrs(src, dst)

    if src.shape == ():
        dst[()] = src[()]
        return

    if src.ndim == 0:
        dst[...] = src[...]
        return

    rows = new_steps if trim_this else src.shape[0]
    for start in range(0, rows, batch_size):
        end = min(start + batch_size, rows)
        dst[start:end] = src[start:end]


def copy_group(
    src: h5py.Group,
    dst: h5py.Group,
    old_steps: int,
    new_steps: int,
    batch_size: int,
) -> None:
    copy_attrs(src, dst)
    for name, item in src.items():
        if isinstance(item, h5py.Group):
            child = dst.create_group(name)
            copy_group(item, child, old_steps, new_steps, batch_size)
        elif isinstance(item, h5py.Dataset):
            copy_dataset(item, dst, name, old_steps, new_steps, batch_size)
        else:
            raise TypeError(f"Unsupported HDF5 object: {item.name}")


def trim_episode(src_path: Path, dst_path: Path, trim_steps: int, batch_size: int) -> None:
    with h5py.File(src_path, "r") as src:
        if "/observations/qpos" not in src:
            raise KeyError(f"{src_path.name} missing /observations/qpos")

        old_steps = src["/observations/qpos"].shape[0]
        new_steps = old_steps - trim_steps
        if new_steps <= 0:
            raise ValueError(f"{src_path.name}: {old_steps} steps, cannot trim {trim_steps}")

        with h5py.File(dst_path, "w") as dst:
            copy_group(src, dst, old_steps, new_steps, batch_size)

    print(f"{src_path.name}: {old_steps} -> {new_steps}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trim tail timesteps from all episode_*.hdf5 files in a folder."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trim-steps", type=int, default=65)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.source.is_dir():
        raise FileNotFoundError(f"Source folder does not exist: {args.source}")
    if args.trim_steps < 0:
        raise ValueError("--trim-steps must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    episodes = sorted(args.source.glob("episode_*.hdf5"), key=episode_sort_key)
    if not episodes:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {args.source}")

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"source: {args.source}")
    print(f"output: {args.output}")
    print(f"episodes: {len(episodes)}")
    print(f"trim_steps: {args.trim_steps}")

    for src_path in episodes:
        dst_path = args.output / src_path.name
        if dst_path.exists() and not args.overwrite:
            print(f"skip existing: {dst_path.name}")
            continue
        trim_episode(src_path, dst_path, args.trim_steps, args.batch_size)

    print("done")


if __name__ == "__main__":
    main()
