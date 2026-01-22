"""Training configuration for Pi Value Function.

This module defines the training configuration for the Pi value function model,
following the same patterns as openpi.training.config.
"""

import dataclasses
import pathlib
from typing import Any, Literal

import flax.nnx as nnx

from pi_value_function.config import PiValueConfig
import openpi.training.optimizer as _optimizer

# Type alias for filter
Filter = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class ValueDataConfig:
    """Configuration for value function training data.

    This config defines the data source and preprocessing for value function training.
    The value function is trained to predict state values from observations.
    """

    # Path to the training data directory (RLDS, LeRobot, or custom format)
    data_path: str | None = None

    # LeRobot repo IDs for success trajectories (training)
    success_repo_ids: list[str] | None = None

    # LeRobot repo IDs for failure trajectories (training)
    failure_repo_ids: list[str] | None = None

    # Path to JSON file with failure costs per prompt
    failure_cost_json: str | None = None

    # Default failure cost if prompt not in JSON
    default_c_fail: float = 100.0

    # Fraction of samples from success dataset (0.0 to 1.0)
    success_sampling_ratio: float = 0.5

    # Train/validation split ratios
    train_split: float = 0.9
    val_split: float = 0.1

    # Random seed for deterministic episode shuffling in train/val split
    split_seed: int = 42

    # Task-specific training: only episodes with this task are treated as success
    target_task: str | None = None

    # If True with target_task, treat episodes with other tasks as failures
    treat_other_tasks_as_failure: bool = False

    # Dynamic task augmentation: randomly sample target task per sample (not per batch!)
    # This augments failures by treating successful episodes with different prompts as failures
    # Each sample independently picks a random target task, then checks if the episode matches
    use_dynamic_task_augmentation: bool = False

    # Minimum episodes per task for dynamic augmentation (filters out rare tasks)
    min_episodes_per_task: int = 10

    # Shuffle buffer size for data loading
    shuffle_buffer_size: int = 10_000

    # Whether to load prompts from the task field in the dataset
    prompt_from_task: bool = True

    # Data format type
    data_format: Literal["rlds", "lerobot", "custom"] = "lerobot"

    # For RLDS datasets
    rlds_data_dir: str | None = None

    # Horizon for computing value targets (e.g., n-step returns, TD targets)
    value_horizon: int = 10

    # Discount factor for computing returns
    gamma: float = 0.99

    # Whether to use reward normalization
    normalize_rewards: bool = True


@dataclasses.dataclass(frozen=True)
class CheckpointConfig:
    """Configuration for checkpointing during training."""

    # Base directory for checkpoints
    checkpoint_dir: str = "./checkpoints"

    # How often (in steps) to save checkpoints
    save_every_n_steps: int = 1_000

    # Maximum number of checkpoints to keep (oldest deleted first)
    # If None, keep all checkpoints
    keep_n_checkpoints: int | None = 5

    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted
    keep_period: int | None = 5_000

    # Path to checkpoint to resume from
    resume_checkpoint_path: str | None = None

    # If true, will overwrite the checkpoint directory if it already exists
    overwrite: bool = False

    # If true, will resume training from the last checkpoint in checkpoint_dir
    resume: bool = False


@dataclasses.dataclass(frozen=True)
class LoggingConfig:
    """Configuration for logging and experiment tracking."""

    # How often (in steps) to log training metrics
    log_every_n_steps: int = 100

    # Whether to enable wandb logging
    wandb_enabled: bool = True

    # Wandb project name
    wandb_project: str = "pi-value-function"

    # Wandb run name (if None, will auto-generate)
    wandb_run_name: str | None = None

    # Wandb entity (team/username)
    wandb_entity: str | None = None

    # Additional wandb config
    wandb_config: dict[str, Any] | None = None

    # How often (in steps) to log validation metrics
    val_every_n_steps: int = 500


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Main training configuration for Pi Value Function.

    This configuration defines all hyperparameters and settings needed for training
    the value function model. It follows the same patterns as openpi.training.config.TrainConfig.

    Example usage:
        ```python
        # Create config programmatically
        config = TrainConfig(
            model_config=PiValueConfig(value_dims=1),
            learning_rate=3e-4,
            batch_size=32,
            num_epochs=10,
            data=ValueDataConfig(data_path="/path/to/data"),
            checkpoint=CheckpointConfig(checkpoint_dir="./checkpoints"),
        )

        # Or use a debug config for quick experiments
        config = TrainConfig.debug_config()
        ```
    """

    # =========================================================================
    # Model configuration
    # =========================================================================

    # Model configuration (defines architecture, value dims, etc.)
    model_config: PiValueConfig = dataclasses.field(default_factory=PiValueConfig)

    # =========================================================================
    # Optimizer settings
    # =========================================================================

    # Learning rate schedule configuration
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(
        default_factory=lambda: _optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=3e-4,
            decay_steps=30_000,
            decay_lr=3e-5,
        )
    )

    # Optimizer configuration
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(
        default_factory=lambda: _optimizer.AdamW(
            weight_decay=0.01,
            clip_gradient_norm=1.0,
        )
    )

    # EMA decay rate (if None, EMA is disabled)
    ema_decay: float | None = 0.99

    # =========================================================================
    # Training loop settings
    # =========================================================================

    # Number of training epochs
    num_epochs: int = 10

    # Global batch size across all devices
    batch_size: int = 32

    # Number of steps per epoch (if None, determined by dataset size)
    steps_per_epoch: int | None = None

    # Total number of training steps (if provided, overrides num_epochs)
    num_train_steps: int | None = None

    # Number of workers for data loading
    num_workers: int = 4
    
    # Number of steps between validations (if 0, validation is disabled)
    num_steps_per_validation: int = 0

    # =========================================================================
    # Data settings
    # =========================================================================

    # Data configuration
    data: ValueDataConfig = dataclasses.field(default_factory=ValueDataConfig)

    # =========================================================================
    # Checkpoint settings
    # =========================================================================

    # Checkpoint configuration
    checkpoint: CheckpointConfig = dataclasses.field(default_factory=CheckpointConfig)

    # =========================================================================
    # Logging settings
    # =========================================================================

    # Logging configuration
    logging: LoggingConfig = dataclasses.field(default_factory=LoggingConfig)

    # =========================================================================
    # Device/hardware settings
    # =========================================================================

    # Number of devices to use (if None, uses all available)
    num_devices: int | None = None

    # If true, will enable FSDP sharding across devices
    use_sharding: bool = False

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices
    fsdp_devices: int = 1

    # Training dtype (must match model dtype)
    dtype: Literal["bfloat16", "float32", "float16"] = "bfloat16"

    # =========================================================================
    # Reproducibility
    # =========================================================================

    # Random seed for reproducibility
    seed: int = 42

    # =========================================================================
    # Value-specific settings
    # =========================================================================

    # Loss weight for value prediction (allows for multi-task learning)
    value_loss_weight: float = 1.0

    # Whether to freeze the entire backbone (SigLIP + Gemma)
    freeze_backbone: bool = False

    # Whether to freeze only the SigLIP vision encoder
    freeze_siglip: bool = False

    # Whether to freeze only the Gemma language model
    freeze_gemma: bool = False

    # Custom freeze filter (if provided, overrides freeze_backbone/siglip/gemma)
    freeze_filter: Filter = dataclasses.field(default_factory=nnx.Nothing)

    # =========================================================================
    # Experiment naming
    # =========================================================================

    # Experiment name (used for checkpoint directory)
    exp_name: str = "pi_value_experiment"

    # Project name
    project_name: str = "pi-value-function"

    # =========================================================================
    # Computed properties
    # =========================================================================

    @property
    def total_steps(self) -> int:
        """Compute total training steps.

        Returns:
            Total number of training steps.
        """
        if self.num_train_steps is not None:
            return self.num_train_steps

        if self.steps_per_epoch is None:
            raise ValueError(
                "Either num_train_steps or steps_per_epoch must be provided. "
                "Cannot compute total_steps without knowing dataset size."
            )

        return self.num_epochs * self.steps_per_epoch

    @property
    def checkpoint_dir_path(self) -> pathlib.Path:
        """Get the full checkpoint directory path.

        Returns:
            Path to checkpoint directory.
        """
        return (pathlib.Path(self.checkpoint.checkpoint_dir) / self.project_name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> Filter:
        """Get the filter for trainable parameters.

        This constructs a filter based on freeze settings, determining which
        parameters should be trained vs. frozen.

        Returns:
            Filter for trainable parameters.
        """
        # If custom freeze filter is provided, use it
        if self.freeze_filter is not nnx.Nothing():
            return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

        # Build freeze filter based on freeze settings
        freeze_filters = []

        if self.freeze_backbone:
            # Freeze everything except value head
            freeze_filters.append(nnx.PathContains("siglip"))
            freeze_filters.append(nnx.PathContains("gemma"))
        else:
            if self.freeze_siglip:
                freeze_filters.append(nnx.PathContains("siglip"))
            if self.freeze_gemma:
                freeze_filters.append(nnx.PathContains("gemma"))

        # Combine all freeze filters
        if freeze_filters:
            combined_freeze = nnx.Any(*freeze_filters)
            return nnx.All(nnx.Param, nnx.Not(combined_freeze))

        # If nothing is frozen, train all parameters
        return nnx.Param

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If configuration is invalid.
        """
        # Validate resume/overwrite
        if self.checkpoint.resume and self.checkpoint.overwrite:
            raise ValueError("Cannot set both resume=True and overwrite=True")

        # Validate splits
        if not 0 < self.data.train_split <= 1:
            raise ValueError(f"train_split must be in (0, 1], got {self.data.train_split}")

        if not 0 <= self.data.val_split < 1:
            raise ValueError(f"val_split must be in [0, 1), got {self.data.val_split}")

        if self.data.train_split + self.data.val_split > 1.0:
            raise ValueError(
                f"train_split ({self.data.train_split}) + val_split ({self.data.val_split}) "
                f"must be <= 1.0"
            )

        # Validate batch size
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        # Validate value loss weight
        if self.value_loss_weight <= 0:
            raise ValueError(f"value_loss_weight must be positive, got {self.value_loss_weight}")

        # Validate dtype matches model config
        if self.dtype != self.model_config.dtype:
            raise ValueError(
                f"Training dtype ({self.dtype}) must match model dtype ({self.model_config.dtype})"
            )

        # Validate discount factor
        if not 0 <= self.data.gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {self.data.gamma}")

    @classmethod
    def debug_config(cls) -> "TrainConfig":
        """Create a debug configuration for quick testing.

        This config uses tiny settings for fast iteration during development:
        - Small batch size
        - Few training steps
        - Frequent logging
        - Disabled wandb
        - Dummy model variants

        Returns:
            Debug training configuration.
        """
        return cls(
            exp_name="debug",
            model_config=PiValueConfig(
                value_dims=201,  # 201 bins for categorical distribution over [-1, 0]
                value_min=-1.0,
                value_max=0.0,
                gemma_variant="gemma-3-270m",  # Smallest variant
                siglip_variant="siglip2-so400m-patch16-384",
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=10,
                peak_lr=1e-3,
                decay_steps=100,
                decay_lr=1e-4,
            ),
            optimizer=_optimizer.AdamW(
                weight_decay=0.01,
                clip_gradient_norm=1.0,
            ),
            num_train_steps=100,
            batch_size=2,
            data=ValueDataConfig(
                data_path=None,  # Will use fake data
                shuffle_buffer_size=100,
            ),
            checkpoint=CheckpointConfig(
                checkpoint_dir="./debug_checkpoints",
                save_every_n_steps=50,
                keep_n_checkpoints=2,
                overwrite=True,
            ),
            logging=LoggingConfig(
                log_every_n_steps=10,
                wandb_enabled=False,
                val_every_n_steps=50,
            ),
            num_workers=0,
            seed=42,
        )

    @classmethod
    def quick_experiment_config(
        cls,
        data_path: str,
        exp_name: str = "quick_exp",
        learning_rate: float = 3e-4,
        batch_size: int = 16,
        num_steps: int = 1000,
    ) -> "TrainConfig":
        """Create a config for quick experiments.

        This is useful for testing on real data with minimal setup.

        Args:
            data_path: Path to training data.
            exp_name: Experiment name.
            learning_rate: Peak learning rate.
            batch_size: Batch size.
            num_steps: Number of training steps.

        Returns:
            Quick experiment configuration.
        """
        return cls(
            exp_name=exp_name,
            model_config=PiValueConfig(value_dims=1),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=min(100, num_steps // 10),
                peak_lr=learning_rate,
                decay_steps=num_steps,
                decay_lr=learning_rate / 10,
            ),
            num_train_steps=num_steps,
            batch_size=batch_size,
            data=ValueDataConfig(data_path=data_path),
            checkpoint=CheckpointConfig(
                checkpoint_dir="./quick_experiments",
                save_every_n_steps=num_steps // 5,
                keep_n_checkpoints=3,
            ),
            logging=LoggingConfig(
                log_every_n_steps=50,
                wandb_enabled=True,
                wandb_project="pi-value-quick-experiments",
            ),
        )
