"""Value function data loader for training.

This module provides PyTorch datasets and data loaders for training the
value function model. It handles:
- Loading success/failure trajectories from LeRobot datasets
- Computing value targets using V(s_t) formula
- Balancing success/failure samples
- Failure cost lookup from JSON
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import einops
import jax
import numpy as np
import torch
import torch.utils.data
from lerobot.datasets import lerobot_dataset


def _parse_image(image: np.ndarray | Any) -> np.ndarray:
    """Parse image from LeRobot format to uint8 [H, W, C].

    LeRobot stores images as float32 [C, H, W] or uint8 [H, W, C].
    This function converts to uint8 [H, W, C] format.

    Args:
        image: Image array from LeRobot dataset

    Returns:
        Image as uint8 [H, W, C]
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def parse_observation(data: dict) -> dict:
    """Parse LeRobot observation to model format.

    Camera mapping (v3.0 DROID format to Pi0 expected format):
    - exterior_image_1_left → base_0_rgb
    - exterior_image_2_left → right_wrist_0_rgb (repurpose as right wrist)
    - wrist_image_left → left_wrist_0_rgb

    Args:
        data: Sample from LeRobot dataset

    Returns:
        Dictionary with standardized observation format
    """
    # State: concatenate joint + gripper
    # v3.0 format uses keys without "observation/" prefix
    gripper_pos = np.asarray(data["gripper_position"])
    if gripper_pos.ndim == 0:
        gripper_pos = gripper_pos[np.newaxis]
    state = np.concatenate([data["joint_position"], gripper_pos])

    # Images: parse to uint8 [H, W, C]
    # Map DROID cameras to Pi0 expected format
    base_0_image = _parse_image(data["exterior_image_1_left"])
    right_wrist_image = _parse_image(data["exterior_image_2_left"])  # Repurpose base_1 as right wrist
    left_wrist_image = _parse_image(data["wrist_image_left"])

    # Get task string (lerobot v3.0 adds this by looking up task_index in meta/tasks.parquet)
    task = data.get("task", "")

    return {
        "state": state,
        "image": {
            "base_0_rgb": base_0_image,
            "left_wrist_0_rgb": left_wrist_image,
            "right_wrist_0_rgb": right_wrist_image,
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        },
        "prompt": task,
    }


@dataclasses.dataclass
class EpisodeMetadata:
    """Metadata for an episode in the dataset."""
    episode_id: int
    start_idx: int      # Global dataset index where episode starts
    end_idx: int        # Global dataset index where episode ends (inclusive)
    length: int         # Episode length (T)
    success: bool       # Success/failure label
    prompt: str | None  # Task prompt for C_fail lookup


class FailureCostManager:
    """Manages failure costs per prompt from JSON configuration."""

    def __init__(self, json_path: str | pathlib.Path | None, default_c_fail: float = 100.0):
        """Initialize failure cost manager.

        Args:
            json_path: Path to JSON file with failure costs
            default_c_fail: Default failure cost if prompt not found
        """
        self._prompt_to_cost: dict[str, float] = {}
        self._default_c_fail = default_c_fail

        if json_path is not None:
            self._load_from_json(json_path)

    def _resolve_path(self, json_path: str | pathlib.Path) -> pathlib.Path:
        """Resolve JSON path, checking multiple locations.

        Tries in order:
        1. Path as-is (absolute or relative to cwd)
        2. Relative to package root (pi-value-function/)
        3. Relative to this script's directory
        """
        json_path = pathlib.Path(json_path)
        if json_path.exists():
            return json_path

        # Get package root (4 levels up from this file: training/ -> pi_value_function/ -> src/ -> pi-value-function/)
        script_dir = pathlib.Path(__file__).parent
        package_root = script_dir.parent.parent.parent

        # Try relative to package root
        package_relative = package_root / json_path
        if package_relative.exists():
            return package_relative

        # Try relative to script directory
        script_relative = script_dir / json_path
        if script_relative.exists():
            return script_relative

        raise FileNotFoundError(
            f"Failure cost JSON not found. Tried:\n"
            f"  - {json_path}\n"
            f"  - {package_relative}\n"
            f"  - {script_relative}"
        )

    def _load_from_json(self, json_path: str | pathlib.Path) -> None:
        """Load failure costs from JSON file.

        JSON format:
        [
            {"prompt": "pick up cup", "average_time": 45.0, "c_fail": 90.0},
            ...
        ]
        """
        json_path = self._resolve_path(json_path)

        with open(json_path) as f:
            data = json.load(f)

        for entry in data:
            prompt = entry["prompt"].lower().strip()
            c_fail = entry["c_fail"]
            self._prompt_to_cost[prompt] = c_fail

    def get_cost(self, prompt: str | None) -> float:
        """Get failure cost for a prompt.

        Args:
            prompt: Task prompt (case-insensitive)

        Returns:
            Failure cost for the prompt, or default if not found
        """
        if prompt is None:
            return self._default_c_fail

        # Normalize prompt (lowercase, strip)
        normalized = prompt.lower().strip()
        return self._prompt_to_cost.get(normalized, self._default_c_fail)


def compute_value_target(
    timestep: int,
    episode_length: int,
    is_success: bool,
    c_fail: float,
    raw_value_min: float,
    raw_value_max: float,
    value_min: float = -1.0,
    value_max: float = 0.0,
) -> float:
    """Compute value target and normalize to [value_min, value_max].

    Formula:
        V(s_t) = -(T - t)           for success
        V(s_t) = -(T - t) - C_fail  for failure

    Then normalize from [raw_value_min, raw_value_max] to [value_min, value_max].

    Args:
        timestep: Current timestep (t)
        episode_length: Episode length (T)
        is_success: Whether episode succeeded
        c_fail: Failure cost
        raw_value_min: Minimum raw value in dataset
        raw_value_max: Maximum raw value in dataset
        value_min: Target minimum value (default: -1.0)
        value_max: Target maximum value (default: 0.0)

    Returns:
        Normalized value target in [value_min, value_max]
    """
    # Compute raw value
    remaining_steps = episode_length - timestep
    if is_success:
        raw_value = -remaining_steps
    else:
        raw_value = -remaining_steps - c_fail

    # Normalize to [value_min, value_max]
    if raw_value_max - raw_value_min > 1e-6:
        normalized = (raw_value - raw_value_min) / (raw_value_max - raw_value_min)
        normalized = normalized * (value_max - value_min) + value_min
    else:
        # If no variation, return middle of range
        normalized = (value_min + value_max) / 2.0

    return np.clip(normalized, value_min, value_max)


class ValueFunctionDataset(torch.utils.data.Dataset):
    """PyTorch dataset for value function training.

    Samples random timesteps from episodes and computes value targets.
    Balances success/failure samples according to configured ratio.
    """

    def __init__(
        self,
        success_repo_ids: list[str] | None = None,
        failure_repo_ids: list[str] | None = None,
        failure_cost_json: str | pathlib.Path | None = None,
        default_c_fail: float = 100.0,
        success_sampling_ratio: float = 0.5,
        value_min: float = -1.0,
        value_max: float = 0.0,
        seed: int = 42,
    ):
        """Initialize value function dataset.

        Args:
            success_repo_ids: LeRobot repo IDs for success trajectories
            failure_repo_ids: LeRobot repo IDs for failure trajectories
            failure_cost_json: Path to JSON with failure costs per prompt
            default_c_fail: Default failure cost if prompt not in JSON
            success_sampling_ratio: Fraction of samples from success dataset
            value_min: Minimum value for normalization (default: -1.0)
            value_max: Maximum value for normalization (default: 0.0)
            seed: Random seed
        """
        # Validate inputs
        if not success_repo_ids and not failure_repo_ids:
            raise ValueError("At least one of success_repo_ids or failure_repo_ids must be provided")

        if not 0.0 <= success_sampling_ratio <= 1.0:
            raise ValueError(f"success_sampling_ratio must be in [0, 1], got {success_sampling_ratio}")

        # Load datasets
        self.success_dataset = self._load_multi_dataset(success_repo_ids) if success_repo_ids else None
        self.failure_dataset = self._load_multi_dataset(failure_repo_ids) if failure_repo_ids else None

        # Build episode indices
        self.success_episodes = self._build_episode_index(self.success_dataset, success=True) if self.success_dataset else []
        self.failure_episodes = self._build_episode_index(self.failure_dataset, success=False) if self.failure_dataset else []

        # Validate episode indices
        if not self.success_episodes and success_repo_ids:
            raise ValueError("Success dataset is empty or has no valid episodes")
        if not self.failure_episodes and failure_repo_ids:
            raise ValueError("Failure dataset is empty or has no valid episodes")

        # Failure cost manager
        self.cost_manager = FailureCostManager(failure_cost_json, default_c_fail)

        # Sampling
        self.success_ratio = success_sampling_ratio
        self.base_seed = seed  # Store base seed for worker initialization
        self.rng = np.random.RandomState(seed)

        # Value normalization params
        self.value_min = value_min
        self.value_max = value_max
        self._compute_normalization_stats()

    def _load_multi_dataset(self, repo_ids: list[str]) -> lerobot_dataset.MultiLeRobotDataset:
        """Load MultiLeRobotDataset from repo IDs."""
        if not repo_ids:
            raise ValueError("repo_ids cannot be empty")

        return lerobot_dataset.MultiLeRobotDataset(repo_ids)

    def _build_episode_index(
        self,
        dataset: lerobot_dataset.MultiLeRobotDataset,
        success: bool
    ) -> list[EpisodeMetadata]:
        """Build episode index from dataset metadata.

        Uses ds.meta.episodes for O(num_episodes) complexity instead of
        scanning all samples which would be O(num_samples).

        Args:
            dataset: LeRobot dataset
            success: Whether this is a success dataset

        Returns:
            List of episode metadata
        """
        episodes = []
        offset = 0  # Track global index offset for multi-dataset

        for ds in dataset._datasets:
            ep_meta = ds.meta.episodes
            for i in range(len(ep_meta)):
                ep = ep_meta[i]
                task = ep["tasks"][0] if ep["tasks"] else None
                episodes.append(EpisodeMetadata(
                    episode_id=ep["episode_index"],
                    start_idx=offset + ep["dataset_from_index"],
                    end_idx=offset + ep["dataset_to_index"] - 1,  # inclusive
                    length=ep["length"],
                    success=success,
                    prompt=task,
                ))
            offset += len(ds)

        return episodes

    def _compute_normalization_stats(self) -> None:
        """Compute normalization statistics from episode metadata."""
        # Find max episode length
        max_episode_length = max(
            max((ep.length for ep in self.success_episodes), default=0),
            max((ep.length for ep in self.failure_episodes), default=0),
        )

        # Find max C_fail
        max_c_fail = max(
            (self.cost_manager.get_cost(ep.prompt) for ep in self.failure_episodes),
            default=self.cost_manager._default_c_fail,
        )

        # Raw value range
        self.raw_value_max = 0.0  # End of successful episode
        self.raw_value_min = -(max_episode_length + max_c_fail)

    def __len__(self) -> int:
        """Return total number of timesteps across all episodes."""
        return sum(ep.length for ep in self.success_episodes + self.failure_episodes)

    def __getitem__(self, idx: int) -> dict:
        """Sample a random timestep and compute value target.

        Note: idx is ignored for random sampling. We sample:
        1. Success or failure dataset (based on success_ratio)
        2. Random episode from selected dataset
        3. Random timestep from selected episode

        Args:
            idx: Index (ignored, sampling is random)

        Returns:
            Dictionary with observation and value target
        """
        # Sample dataset
        is_success = self.rng.random() < self.success_ratio if self.success_episodes and self.failure_episodes else bool(self.success_episodes)

        if is_success:
            episodes = self.success_episodes
            dataset = self.success_dataset
        else:
            episodes = self.failure_episodes
            dataset = self.failure_dataset

        # Sample random episode
        episode_meta = episodes[self.rng.randint(len(episodes))]

        # Sample random timestep within episode
        timestep_offset = self.rng.randint(episode_meta.length)
        global_idx = episode_meta.start_idx + timestep_offset

        # Get observation from dataset
        sample = dataset[global_idx]

        # Parse observation to model format
        parsed = parse_observation(sample)

        # Compute value target
        c_fail = 0.0 if is_success else self.cost_manager.get_cost(episode_meta.prompt)
        value = compute_value_target(
            timestep=timestep_offset,
            episode_length=episode_meta.length,
            is_success=is_success,
            c_fail=c_fail,
            raw_value_min=self.raw_value_min,
            raw_value_max=self.raw_value_max,
            value_min=self.value_min,
            value_max=self.value_max,
        )

        # Add value target (named "returns" to match PiValue.compute_loss signature)
        parsed["returns"] = np.array(value, dtype=np.float32)

        return parsed


def _worker_init_fn(worker_id: int) -> None:
    """Initialize worker with unique random seed."""
    # Reseed RNG for this worker
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if isinstance(dataset, ValueFunctionDataset):
            # Generate worker-specific seed from base seed using hash
            # This is similar to key splitting but stays in pure Python
            worker_seed = hash((dataset.base_seed, worker_id)) % (2**32)
            dataset.rng = np.random.RandomState(worker_seed)


class CollateFnWithTokenizer:
    """Collate function that tokenizes prompts before batching.

    This follows the openpi approach of tokenizing prompts per-sample before
    batching, avoiding numpy string arrays that JAX can't handle.

    This is implemented as a class instead of a nested function to make it
    picklable for PyTorch's multiprocessing DataLoader.
    """

    def __init__(self, tokenizer):
        """Initialize collate function with tokenizer.

        Args:
            tokenizer: Tokenizer with a tokenize(prompt, state) method
        """
        self.tokenizer = tokenizer

    def __call__(self, items: list[dict]) -> dict:
        """Collate batch items, tokenizing prompts individually."""
        # Extract and tokenize prompts
        prompts = [item.pop("prompt") for item in items]
        tokenized_prompts = []
        tokenized_prompt_masks = []

        for prompt in prompts:
            # Handle numpy scalar strings (like openpi transforms.py:263)
            if not isinstance(prompt, str):
                prompt = prompt.item()

            # Tokenize (state=None for Pi0 format)
            tokens, mask = self.tokenizer.tokenize(prompt, state=None)
            tokenized_prompts.append(tokens)
            tokenized_prompt_masks.append(mask)

        # Stack tokenized prompts
        batch = {
            "tokenized_prompt": np.stack(tokenized_prompts, axis=0),
            "tokenized_prompt_mask": np.stack(tokenized_prompt_masks, axis=0),
        }

        # Stack all other fields (images, returns, etc.)
        for key in items[0].keys():
            if key == "state":
                continue  # Handle state separately for padding
            batch[key] = jax.tree.map(
                lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0),
                *[item[key] for item in items]
            )

        # Pad state from 8-dim (DROID) to 32-dim (expected by model)
        states = np.stack([item["state"] for item in items], axis=0)  # Shape: (batch, 8)
        batch_size = states.shape[0]
        padded_states = np.zeros((batch_size, 32), dtype=np.float32)
        padded_states[:, :states.shape[1]] = states
        batch["state"] = padded_states

        return batch


def create_value_dataloader(
    tokenizer,  # Tokenizer for prompts
    success_repo_ids: list[str] | None = None,
    failure_repo_ids: list[str] | None = None,
    batch_size: int = 64,
    failure_cost_json: str | pathlib.Path | None = None,
    default_c_fail: float = 100.0,
    success_sampling_ratio: float = 0.5,
    num_workers: int = 4,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """Create value function data loader.

    Returns DataLoader that:
    - Yields batches of (observation, returns) pairs
    - Iterates infinitely
    - Uses multiple workers for parallel loading

    Args:
        success_repo_ids: LeRobot repo IDs for success trajectories
        failure_repo_ids: LeRobot repo IDs for failure trajectories
        batch_size: Batch size
        failure_cost_json: Path to JSON with failure costs per prompt
        default_c_fail: Default failure cost if prompt not in JSON
        success_sampling_ratio: Fraction of samples from success dataset
        num_workers: Number of worker processes
        seed: Random seed

    Returns:
        DataLoader yielding batches of observations and value targets
    """
    import time
    start_time = time.time()
    print("Creating ValueFunctionDataset...")
    dataset = ValueFunctionDataset(
        success_repo_ids=success_repo_ids,
        failure_repo_ids=failure_repo_ids,
        failure_cost_json=failure_cost_json,
        default_c_fail=default_c_fail,
        success_sampling_ratio=success_sampling_ratio,
        seed=seed,
    )
    print(f"Dataset created in {time.time() - start_time:.2f} seconds.")
    start_time = time.time()
    print("Creating Sampler...")

    # Use RandomSampler for infinite iteration
    sampler = torch.utils.data.RandomSampler(
        dataset,
        replacement=True,  # Infinite sampling with replacement
        num_samples=int(1e9),  # Very large number
    )
    print(f"Sampler created in {time.time() - start_time:.2f} seconds.")

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=CollateFnWithTokenizer(tokenizer),
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init_fn,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
