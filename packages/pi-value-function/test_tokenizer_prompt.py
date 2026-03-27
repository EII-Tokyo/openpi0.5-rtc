"""Test Gemma3 tokenizer output for a user-provided prompt.

Examples:
  uv run python packages/pi-value-function/test_tokenizer_prompt.py \
    --prompt "Put the battery bank in the orange box"

  uv run python packages/pi-value-function/test_tokenizer_prompt.py \
    --prompt "pick up cup" --no-image-tag --no-bos --max-len 64
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np


def _add_local_paths() -> None:
    """Allow running this script directly from repo root."""
    this_file = pathlib.Path(__file__).resolve()
    pkg_src = this_file.parent / "src"
    repo_src = this_file.parents[2] / "src"
    for path in (pkg_src, repo_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_add_local_paths()

from openpi.models.tokenizer import Gemma3Tokenizer  # noqa: E402
from pi_value_function.training.checkpoint_downloader import download_gemma_from_kaggle  # noqa: E402


def _resolve_tokenizer_path(user_path: str | None) -> pathlib.Path:
    if user_path:
        path = pathlib.Path(user_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer path does not exist: {path}")
        return path

    # Fallback: use existing helper that finds cached tokenizer or downloads it.
    _, tokenizer_path = download_gemma_from_kaggle()
    return pathlib.Path(tokenizer_path).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Gemma3Tokenizer for a custom prompt.")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text to tokenize.")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer.model. If omitted, uses cached/downloaded Gemma tokenizer.",
    )
    parser.add_argument("--max-len", type=int, default=48, help="Tokenizer max output length.")
    parser.add_argument("--no-bos", action="store_true", help="Do not prepend BOS token.")
    parser.add_argument("--no-image-tag", action="store_true", help="Do not prepend <img> tag.")
    parser.add_argument("--show-padded", action="store_true", help="Print full padded token/mask arrays.")
    args = parser.parse_args()

    prompt = args.prompt
    if prompt is None:
        prompt = input("Enter prompt: ").strip()
    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    tokenizer_path = _resolve_tokenizer_path(args.tokenizer_path)
    tokenizer = Gemma3Tokenizer(path=tokenizer_path, max_len=args.max_len)

    add_bos = not args.no_bos
    include_image_tag = not args.no_image_tag
    tokens, mask = tokenizer.tokenize(prompt, add_bos=add_bos, include_image_tag=include_image_tag)

    valid_tokens = tokens[mask]
    valid_count = int(np.sum(mask))

    print("=" * 80)
    print("Gemma3 Tokenizer Test")
    print("=" * 80)
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Prompt: {prompt!r}")
    print(f"max_len={args.max_len}, add_bos={add_bos}, include_image_tag={include_image_tag}")
    print(f"Valid tokens: {valid_count}/{len(tokens)}")
    print("-" * 80)

    print("Valid token IDs:")
    print(valid_tokens.tolist())

    print("\nValid token pieces:")
    pieces = [tokenizer._tokenizer.id_to_piece(int(tok)) for tok in valid_tokens]
    print(pieces)

    if args.show_padded:
        print("\nFull padded token IDs:")
        print(tokens.tolist())
        print("\nFull mask:")
        print(mask.tolist())

    print("=" * 80)


if __name__ == "__main__":
    main()
