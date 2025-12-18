"""Checkpoint downloading utilities.

Handles automatic downloading of pretrained model checkpoints from remote URLs.
Uses ~/.cache/openpi as the default cache directory, following the openpi convention.
"""

import os
import pathlib
import subprocess
import tarfile

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


def download_gemma_from_kaggle(extract_name: str = "gemma-3-270m") -> pathlib.Path:
    """Download Gemma checkpoint from Kaggle and extract it.

    Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables to be set.

    Args:
        extract_name: Name for the extracted checkpoint directory

    Returns:
        Path to the extracted checkpoint directory
    """
    cache_dir = get_cache_dir()
    extract_dir = cache_dir / extract_name

    if extract_dir.exists():
        print(f"Using cached Gemma checkpoint: {extract_dir}")
        return extract_dir

    # Check for Kaggle credentials
    kaggle_username = os.environ.get("KAGGLE_USERNAME")
    kaggle_key = os.environ.get("KAGGLE_KEY")

    if not kaggle_username or not kaggle_key:
        raise EnvironmentError(
            "Kaggle credentials not found. Please set KAGGLE_USERNAME and KAGGLE_KEY environment variables.\n"
            "You can get these from https://www.kaggle.com/settings -> API -> Create New Token"
        )

    # Download tar.gz to cache
    tar_path = cache_dir / "gemma-3-270m.tar.gz"

    if not tar_path.exists():
        print(f"Downloading Gemma checkpoint from Kaggle to {tar_path}...")
        result = subprocess.run(
            [
                "curl", "-L",
                "-u", f"{kaggle_username}:{kaggle_key}",
                "-o", str(tar_path),
                GEMMA_3_KAGGLE_URL
            ],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download Gemma checkpoint: {result.stderr}")
        print(f"Downloaded to {tar_path}")

    # Extract
    print(f"Extracting {tar_path.name} to {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_dir)
    print(f"Extracted to {extract_dir}")

    return extract_dir


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
