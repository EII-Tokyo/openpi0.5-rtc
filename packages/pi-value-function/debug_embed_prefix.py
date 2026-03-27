"""Quick debugger for PiValue.embed_prefix without starting the server.

Usage:
  uv run python packages/pi-value-function/debug_embed_prefix.py --disable-jit
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


def _add_local_paths() -> None:
    """Allow running this script directly from repo root without install step."""
    this_file = pathlib.Path(__file__).resolve()
    pkg_src = this_file.parent / "src"
    repo_src = this_file.parents[2] / "src"
    for path in (pkg_src, repo_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_add_local_paths()

from pi_value_function.config import PiValueConfig  # noqa: E402


def _summarize_array(name: str, x) -> None:
    arr = np.asarray(jax.device_get(x))
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}")
    if arr.size == 0:
        return
    if arr.dtype == np.bool_:
        true_count = int(arr.sum())
        print(f"  bool counts: true={true_count}, false={arr.size - true_count}")
    else:
        print(
            f"  stats: min={arr.min():.6f}, max={arr.max():.6f}, "
            f"mean={arr.mean():.6f}, std={arr.std():.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug PiValue.embed_prefix quickly.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-jit", action="store_true", help="Set JAX_DISABLE_JIT for easier debugging.")
    parser.add_argument("--no-language", action="store_true", help="Remove tokenized prompt inputs.")
    parser.add_argument(
        "--disable-camera",
        action="append",
        default=[],
        choices=["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
        help="Mark a camera mask as False for all batch items. Can be used multiple times.",
    )
    parser.add_argument("--dump-npz", type=str, default=None, help="Optional output path for tokens/masks .npz.")
    args = parser.parse_args()

    if args.disable_jit:
        jax.config.update("jax_disable_jit", True)
        print("JAX JIT disabled for debugging.")

    rng = jax.random.PRNGKey(args.seed)
    config = PiValueConfig()

    t0 = time.time()
    model = config.create(rng)
    print(f"Model init done in {time.time() - t0:.2f}s")

    obs = config.fake_obs(batch_size=args.batch_size)

    if args.no_language:
        obs = dataclasses.replace(obs, tokenized_prompt=None, tokenized_prompt_mask=None)

    if args.disable_camera:
        image_masks = dict(obs.image_masks)
        for cam_name in args.disable_camera:
            image_masks[cam_name] = jnp.zeros((args.batch_size,), dtype=jnp.bool_)
        obs = dataclasses.replace(obs, image_masks=image_masks)

    print("\nObservation summary:")
    for key in sorted(obs.images.keys()):
        _summarize_array(f"  image[{key}]", obs.images[key])
        _summarize_array(f"  image_mask[{key}]", obs.image_masks[key])
    if obs.tokenized_prompt is None:
        print("  tokenized_prompt: None")
        print("  tokenized_prompt_mask: None")
    else:
        _summarize_array("  tokenized_prompt", obs.tokenized_prompt)
        _summarize_array("  tokenized_prompt_mask", obs.tokenized_prompt_mask)

    t1 = time.time()
    tokens, input_mask, ar_mask = model.embed_prefix(obs)
    print(f"\nembed_prefix done in {time.time() - t1:.2f}s")

    print("\nembed_prefix outputs:")
    _summarize_array("  tokens", tokens)
    _summarize_array("  input_mask", input_mask)
    _summarize_array("  ar_mask", ar_mask)
    print(f"  seq_len={tokens.shape[1]}, emb_dim={tokens.shape[2]}")

    if args.dump_npz:
        out_path = pathlib.Path(args.dump_npz).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            tokens=np.asarray(jax.device_get(tokens)),
            input_mask=np.asarray(jax.device_get(input_mask)),
            ar_mask=np.asarray(jax.device_get(ar_mask)),
        )
        print(f"\nSaved debug arrays to: {out_path}")


if __name__ == "__main__":
    main()
