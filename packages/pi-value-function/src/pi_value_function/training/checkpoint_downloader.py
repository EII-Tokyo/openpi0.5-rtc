"""Checkpoint downloading utilities.

Handles automatic downloading of pretrained model checkpoints from remote URLs.
Uses ~/.cache/openpi as the default cache directory, following the openpi convention.
"""

import os
import pathlib
import subprocess
import tarfile
from typing import Tuple

from openpi.shared.download import maybe_download, get_cache_dir


# Checkpoint URLs
SIGLIP2_SO400M14_224_URL = "https://storage.googleapis.com/big_vision/siglip2/siglip2_so400m14_224.npz"
# Gemma checkpoint from Kaggle (requires KAGGLE_USERNAME and KAGGLE_KEY env vars)
GEMMA_3_KAGGLE_URL = "https://www.kaggle.com/api/v1/models/google/gemma-3/flax/gemma-3-270m/1/download"


def download_checkpoint(url: str, cache_dir: pathlib.Path | None = None) -> pathlib.Path:
    """Download checkpoint from URL if not already cached.

    Args:
        url: URL to download checkpoint from
        cache_dir: Optional directory to cache the checkpoint. If None, uses ~/.cache/openpi

    Returns:
        Path to the downloaded/cached checkpoint file
    """
    if cache_dir is not None:
        # Legacy behavior: download to specified directory
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = url.split("/")[-1]
        filepath = cache_dir / filename

        if not filepath.exists():
            import urllib.request
            print(f"Downloading {filename} from {url}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded to {filepath}")
        else:
            print(f"Using cached checkpoint: {filepath}")

        return filepath
    else:
        # Use openpi's shared download utility with ~/.cache/openpi
        return maybe_download(url)

def download_gemma_from_kaggle(extract_name: str = "gemma-3-270m") -> Tuple[pathlib.Path, pathlib.Path]:
    # 1. Use the correct base path and expand the ~
    base_cache = pathlib.Path("~/.cache/kagglehub/models/google/gemma-3/flax/gemma-3-270m").expanduser()
    
    # Use glob to find the version folder (e.g., '1') and the tokenizer inside it
    # This looks for any folder (*/) that contains a tokenizer.model
    existing_tokenizers = list(base_cache.glob("*/tokenizer.model"))

    if existing_tokenizers:
        # Get the directory where the tokenizer was found (this is the '1' folder)
        download_path = existing_tokenizers[0].parent
        model_path = download_path / extract_name
        
        if model_path.exists():
            print(f"Using cached Gemma checkpoint: {download_path}")
            return model_path, existing_tokenizers[0]

    # Fallback to kagglehub if not found
    import kagglehub
    kagglehub.login()
    
    # This returns the path to the version folder (e.g., .../gemma-3-270m/1)
    download_path = pathlib.Path(kagglehub.model_download("google/gemma-3/flax/gemma-3-270m"))
    
    model_path = download_path / extract_name
    tokenizer_path = download_path / "tokenizer.model"

    return model_path, tokenizer_path


def download_and_extract_checkpoint(url: str, extract_name: str | None = None) -> pathlib.Path:
    """Download checkpoint and extract if it's a tar.gz file.

    Args:
        url: URL to download checkpoint from (supports .tar.gz files)
        extract_name: Optional name for extracted directory. If None, uses filename without .tar.gz

    Returns:
        Path to the extracted checkpoint directory (or file if not tar.gz)
    """
    downloaded_path = maybe_download(url)

    # If it's a tar.gz, extract it
    if str(downloaded_path).endswith('.tar.gz') or str(downloaded_path).endswith('.tgz'):
        cache_dir = get_cache_dir()

        # Determine extraction directory name
        if extract_name is None:
            # Remove .tar.gz extension
            extract_name = downloaded_path.stem
            if extract_name.endswith('.tar'):
                extract_name = extract_name[:-4]

        extract_dir = cache_dir / extract_name

        if not extract_dir.exists():
            print(f"Extracting {downloaded_path.name} to {extract_dir}...")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(downloaded_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
            print(f"Extracted to {extract_dir}")
        else:
            print(f"Using cached extracted checkpoint: {extract_dir}")

        return extract_dir

    return downloaded_path
