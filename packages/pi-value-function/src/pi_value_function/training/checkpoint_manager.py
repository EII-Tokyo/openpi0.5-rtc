"""Checkpoint management for value function training.

Provides utilities for saving and restoring model/optimizer state during training.
Based on openpi.training.checkpoints but simplified for value function training.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import Any

from etils import epath
from flax import nnx
import jax
import orbax.checkpoint as ocp

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ValueTrainState:
    """Training state for value function.

    Simplified compared to openpi's TrainState - stores NNX model/optimizer state
    for checkpointing.
    """
    step: int
    model_state: nnx.State
    model_graphdef: nnx.GraphDef
    optimizer_state: nnx.State
    optimizer_graphdef: nnx.GraphDef


def initialize_checkpoint_manager(
    checkpoint_dir: epath.Path | str | pathlib.Path,
    *,
    max_to_keep: int | None = 5,
    keep_period: int | None = None,
    overwrite: bool = False,
    resume: bool = False,
) -> tuple[ocp.CheckpointManager, bool]:
    """Initialize checkpoint manager and directory.

    Args:
        checkpoint_dir: Directory to save checkpoints
        max_to_keep: Maximum number of checkpoints to keep (oldest deleted first).
                     If None, keep all checkpoints.
        keep_period: If set, checkpoints at steps divisible by this period are never deleted
        overwrite: If True, delete existing checkpoint directory
        resume: If True, resume from existing checkpoints

    Returns:
        Tuple of (checkpoint_manager, resuming)
        - checkpoint_manager: Orbax CheckpointManager instance
        - resuming: True if resuming from existing checkpoint

    Raises:
        FileExistsError: If checkpoint_dir exists and neither overwrite nor resume is True
    """
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False

    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
            logger.info(f"Resuming from checkpoint directory {checkpoint_dir}")
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. "
                "Set overwrite=True or resume=True in CheckpointConfig."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoint manager
    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "state": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=3600),
        ),
    )

    # Special case: directory exists but no valid checkpoints
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logger.info("Checkpoint directory exists but contains no valid checkpoints. Starting fresh.")
        resuming = False

    return mngr, resuming


def save_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    step: int,
) -> None:
    """Save model and optimizer state to checkpoint.

    Args:
        checkpoint_manager: Orbax CheckpointManager
        model: NNX model to save
        optimizer: NNX optimizer to save
        step: Training step number
    """
    # Extract state and graphdefs from NNX objects
    model_graphdef, model_state = nnx.split(model)
    optimizer_graphdef, optimizer_state = nnx.split(optimizer)

    # Create train state
    train_state = ValueTrainState(
        step=step,
        model_state=model_state,
        model_graphdef=model_graphdef,
        optimizer_state=optimizer_state,
        optimizer_graphdef=optimizer_graphdef,
    )

    # Save to checkpoint
    checkpoint_manager.save(step, {"state": train_state})
    logger.info(f"Saved checkpoint at step {step}")


def restore_checkpoint(
    checkpoint_manager: ocp.CheckpointManager,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    step: int | None = None,
) -> tuple[nnx.Module, nnx.Optimizer, int]:
    """Restore model and optimizer state from checkpoint.

    Args:
        checkpoint_manager: Orbax CheckpointManager
        model: NNX model (used as template for structure)
        optimizer: NNX optimizer (used as template for structure)
        step: Specific step to restore. If None, restores latest checkpoint.

    Returns:
        Tuple of (restored_model, restored_optimizer, step)

    Raises:
        ValueError: If no checkpoint exists at specified step
    """
    # Get step to restore
    if step is None:
        all_steps = tuple(checkpoint_manager.all_steps())
        if not all_steps:
            raise ValueError("No checkpoints found")
        step = max(all_steps)
        logger.info(f"Restoring latest checkpoint at step {step}")
    else:
        logger.info(f"Restoring checkpoint at step {step}")

    # Create template train state for restoration
    model_graphdef, model_state = nnx.split(model)
    optimizer_graphdef, optimizer_state = nnx.split(optimizer)

    template = ValueTrainState(
        step=0,
        model_state=model_state,
        model_graphdef=model_graphdef,
        optimizer_state=optimizer_state,
        optimizer_graphdef=optimizer_graphdef,
    )

    # Restore from checkpoint
    restored = checkpoint_manager.restore(step, {"state": template})
    train_state = restored["state"]

    # Reconstruct NNX objects from restored state
    restored_model = nnx.merge(train_state.model_graphdef, train_state.model_state)
    restored_optimizer = nnx.merge(train_state.optimizer_graphdef, train_state.optimizer_state)

    logger.info(f"✓ Restored checkpoint from step {train_state.step}")

    return restored_model, restored_optimizer, train_state.step


def get_latest_step(checkpoint_manager: ocp.CheckpointManager) -> int | None:
    """Get the latest checkpoint step.

    Args:
        checkpoint_manager: Orbax CheckpointManager

    Returns:
        Latest checkpoint step, or None if no checkpoints exist
    """
    all_steps = tuple(checkpoint_manager.all_steps())
    if not all_steps:
        return None
    return max(all_steps)
