import dataclasses
from pathlib import Path
import numpy as np

import openpi.shared.array_typing as at
import openpi.shared.download as download
from openpi.training.weight_loaders import WeightLoader, _merge_params
import openpi.models.siglip as _siglip


@dataclasses.dataclass(frozen=True)
class SigLIP2WeightLoader(WeightLoader):
    """Loads SigLIP 2 weights from a checkpoint.

    Checkpoint format:
      - URL: https://storage.googleapis.com/big_vision/siglip2/siglip2_so400m14_224.npz
      - Keys: 'params/img/...' for image encoder weights

    Attributes:
        checkpoint_path: Path to the checkpoint file (local path or URL).
                        If None, uses default URL.
    """

    checkpoint_path: str | Path | None = None

    def map_siglip2_to_model_params(self, siglip2_weights: dict) -> dict:
        """Map SigLIP 2 checkpoint weights to model's expected parameter structure.

        SigLIP 2 checkpoint format: params/img/embedding/kernel
        Model expects: embedding/kernel

        We just strip the 'params/img/' prefix.
        """
        mapped_weights = {}

        for key, value in siglip2_weights.items():
            # Strip 'params/img/' prefix
            if key.startswith('params/img/'):
                new_key = key[11:]  # Remove 'params/img/'

                # Build nested dict structure
                parts = new_key.split('/')
                current = mapped_weights
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value

        return mapped_weights
    
    def load(self, params: at.Params) -> at.Params:
        # Use provided checkpoint path or default URL
        if self.checkpoint_path is None:
            checkpoint_url = "https://storage.googleapis.com/big_vision/siglip2/siglip2_so400m14_224.npz"
            checkpoint_path = download.maybe_download(checkpoint_url)
        else:
            # If it's a local path, use it directly; if it's a URL, download it
            checkpoint_path = Path(self.checkpoint_path)
            if not checkpoint_path.exists():
                # Try treating it as a URL
                checkpoint_path = download.maybe_download(str(self.checkpoint_path))

        # Use memory mapping for large checkpoints to reduce memory pressure
        siglip2_weights = np.load(checkpoint_path, mmap_mode='r')
        mapped_weights = self.map_siglip2_to_model_params(dict(siglip2_weights))
        # Merge loaded weights with existing params
        return _merge_params(mapped_weights, params, missing_regex=".*")