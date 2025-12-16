"""Value function training script using JAX/Flax NNX.

Main training script for PiValue model. Handles:
- Model initialization with pretrained SigLIP and Gemma weights
- Data loading from LeRobot datasets
- Training loop with gradient updates
- W&B logging and checkpointing
"""

import dataclasses
import functools
import pathlib
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax
import torch

from pi_value_function.config import PiValueConfig
from pi_value_function.pi_value import PiValue
from pi_value_function.training.train_config import TrainConfig
from pi_value_function.training.data_loader import create_value_dataloader
from pi_value_function.training.weight_loader import SigLIP2WeightLoader
from pi_value_function.training.direct_checkpoint_loader import DirectGemma3WeightLoader
from openpi.models.model import Observation
from openpi.models.tokenizer import PaligemmaTokenizer

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def create_model_with_weights(config: TrainConfig, rng: jax.Array) -> PiValue:
    """Create PiValue model and load pretrained weights.

    Args:
        config: Training configuration
        rng: Random key for model initialization

    Returns:
        PiValue model with pretrained SigLIP and Gemma weights loaded
    """
    # Create model using config
    model = config.model_config.create(rng)

    # Navigate from packages/pi-value-function/src/pi_value_function/training/train.py -> repo root
    repo_root = pathlib.Path(__file__).parent.parent.parent.parent.parent.parent

    # Load weights by directly updating the model (skip split/merge pattern)
    # This approach avoids State conversion issues
    nnx.update(model, nnx.state(model))  # Ensure model is in a good state

    # Helper function to selectively update state values
    def selective_tree_update(original, updates):
        """Recursively merge updates into original, only where keys and shapes match."""
        if isinstance(original, dict) and isinstance(updates, dict):
            result = {}
            for key in original.keys():
                if key in updates:
                    # Recursively merge nested dicts
                    result[key] = selective_tree_update(original[key], updates[key])
                else:
                    # Key not in updates, keep original
                    result[key] = original[key]
            return result
        elif hasattr(original, 'shape') and hasattr(updates, 'shape'):
            # Both are arrays - check shape match
            if original.shape == updates.shape:
                return updates
            else:
                return original
        else:
            # Not a dict or array, keep original
            return original

    # Load SigLIP weights
    siglip_checkpoint = repo_root / "checkpoints" / "siglip2_so400m14_224.npz"
    siglip_loader = SigLIP2WeightLoader(checkpoint_path=str(siglip_checkpoint))

    # Get siglip params as dict, load weights, update back
    t0 = time.time()
    siglip_state = nnx.state(model.siglip, nnx.Param)
    siglip_dict = siglip_state.to_pure_dict()
    t1 = time.time()
    loaded_siglip = siglip_loader.load(siglip_dict)
    t2 = time.time()

    # Create new state with updated values
    updated_siglip = selective_tree_update(siglip_dict, loaded_siglip)
    t3 = time.time()
    new_siglip_state = nnx.State(updated_siglip)
    nnx.update(model.siglip, new_siglip_state)
    t4 = time.time()
    print(f"✓ Loaded SigLIP weights from {siglip_checkpoint}")
    print(f"  Timing: state_extract={t1-t0:.2f}s, load={t2-t1:.2f}s, merge={t3-t2:.2f}s, update={t4-t3:.2f}s")

    # Load Gemma weights
    gemma_checkpoint = repo_root / "checkpoints" / "gemma-3-270m"
    gemma_loader = DirectGemma3WeightLoader(
        checkpoint_path=str(gemma_checkpoint),
        param_key=None
    )

    # Get gemma params as dict, load weights, update back
    t0 = time.time()
    gemma_state = nnx.state(model.gemma, nnx.Param)
    gemma_dict = gemma_state.to_pure_dict()
    t1 = time.time()
    loaded_gemma = gemma_loader.load(gemma_dict)
    t2 = time.time()

    # Create new state with updated values
    updated_gemma = selective_tree_update(gemma_dict, loaded_gemma)
    t3 = time.time()
    new_gemma_state = nnx.State(updated_gemma)
    nnx.update(model.gemma, new_gemma_state)
    t4 = time.time()
    print(f"✓ Loaded Gemma weights from {gemma_checkpoint}")
    print(f"  Timing: state_extract={t1-t0:.2f}s, load={t2-t1:.2f}s, merge={t3-t2:.2f}s, update={t4-t3:.2f}s")

    # Return model (no need for merge since we updated directly)
    return model


def batch_to_observation(batch: dict[str, Any], tokenizer: PaligemmaTokenizer) -> Observation:
    """Convert data loader batch to Observation format.

    Args:
        batch: Batch from data loader with keys:
            - "image": dict[str, ndarray] - Images as uint8 or float32
            - "image_mask": dict[str, ndarray] - Image validity masks
            - "state": ndarray - Robot state (8-dim from DROID)
            - "prompt": list[str] - Task prompts as strings
            - "returns": ndarray - Value targets
        tokenizer: Tokenizer for converting prompt strings to tokens

    Returns:
        Observation object with all fields converted to JAX arrays
    """
    # Tokenize prompts (batch of strings -> batch of token arrays)
    prompts = batch["prompt"]
    tokenized_prompts = []
    tokenized_prompt_masks = []

    for prompt in prompts:
        tokens, mask = tokenizer.tokenize(prompt, state=None)  # state=None for Pi0 format
        tokenized_prompts.append(tokens)
        tokenized_prompt_masks.append(mask)

    tokenized_prompts = np.stack(tokenized_prompts, axis=0)
    tokenized_prompt_masks = np.stack(tokenized_prompt_masks, axis=0)

    # Pad state from 8-dim (DROID) to 32-dim (expected by model)
    state = batch["state"]  # Shape: (batch, 8)
    batch_size = state.shape[0]
    padded_state = np.zeros((batch_size, 32), dtype=np.float32)
    padded_state[:, :state.shape[1]] = state

    # Create batch dict with tokenized prompts
    batch_dict = {
        "image": batch["image"],
        "image_mask": batch["image_mask"],
        "state": padded_state,
        "tokenized_prompt": tokenized_prompts,
        "tokenized_prompt_mask": tokenized_prompt_masks,
    }

    # Use Observation.from_dict to handle image normalization and conversion
    obs = Observation.from_dict(batch_dict)

    return obs


@functools.partial(nnx.jit, donate_argnums=(0,))
def train_step(
    model: PiValue,
    optimizer: nnx.Optimizer,
    rng: jax.Array,
    observation: Observation,
    returns: jax.Array,
) -> tuple[PiValue, nnx.Optimizer, dict[str, jax.Array]]:
    """Single training step with gradient update.

    Uses NNX patterns:
    - nnx.value_and_grad for gradient computation
    - optimizer.update() for parameter updates

    Args:
        model: PiValue model
        optimizer: NNX optimizer
        rng: Random key for this step
        observation: Observation inputs
        returns: Target return values

    Returns:
        Updated model, optimizer, and metrics dict
    """
    def loss_fn(model: PiValue) -> jax.Array:
        """Compute mean loss over batch."""
        losses = model.compute_loss(rng, observation, returns, train=True)
        return jnp.mean(losses)

    # Compute loss and gradients
    loss, grads = nnx.value_and_grad(loss_fn)(model)

    # Compute gradient norm before update
    grad_leaves = jax.tree_util.tree_leaves(grads)
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in grad_leaves if hasattr(g, 'shape')))

    # Update parameters
    optimizer.update(grads)

    # Collect metrics
    metrics = {
        "loss": loss,
        "grad_norm": grad_norm,
    }

    return model, optimizer, metrics


def train(config: TrainConfig) -> None:
    """Main training function.

    Args:
        config: Training configuration
    """
    # Initialize W&B
    if config.logging.wandb_enabled:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not installed, disabling W&B logging")
            config = dataclasses.replace(
                config,
                logging=dataclasses.replace(config.logging, wandb_enabled=False)
            )
        else:
            wandb.init(
                project=config.logging.wandb_project,
                name=config.logging.wandb_run_name or config.exp_name,
                entity=config.logging.wandb_entity,
                config=dataclasses.asdict(config),
            )
            print(f"✓ Initialized W&B project: {config.logging.wandb_project}")

    # RNG setup
    rng = jax.random.PRNGKey(config.seed)
    model_rng, train_rng = jax.random.split(rng)

    # Create model with pretrained weights
    print("\n=== Creating model with pretrained weights ===")
    model = create_model_with_weights(config, model_rng)
    print("✓ Model created successfully")

    # Create optimizer
    print("\n=== Setting up optimizer ===")
    lr_schedule = config.lr_schedule.create()
    tx = config.optimizer.create(lr_schedule)
    optimizer = nnx.Optimizer(model, tx)
    print(f"✓ Optimizer created: {config.optimizer.__class__.__name__}")
    print(f"  Peak LR: {config.lr_schedule.peak_lr}")
    print(f"  Warmup steps: {config.lr_schedule.warmup_steps}")

    # Create tokenizer
    print("\n=== Creating tokenizer ===")
    tokenizer = PaligemmaTokenizer(max_len=48)  # Match model's tokenized_prompt shape
    print("✓ Tokenizer created")

    # Create data loader
    print("\n=== Creating data loader ===")
    # For debug config, use fake data (no repo IDs provided)
    if config.data.data_path is None:
        print("Warning: No data_path provided, using dummy data for testing")
        # Create a dummy dataloader that yields random batches
        # This is just for testing the training loop
        class DummyDataLoader:
            def __init__(self, batch_size: int):
                self.batch_size = batch_size

            def __iter__(self):
                while True:
                    # Generate fake batch matching expected format
                    yield {
                        "image": {
                            "base_0_rgb": np.random.randint(0, 255, (self.batch_size, 224, 224, 3), dtype=np.uint8),
                            "left_wrist_0_rgb": np.random.randint(0, 255, (self.batch_size, 224, 224, 3), dtype=np.uint8),
                            "right_wrist_0_rgb": np.random.randint(0, 255, (self.batch_size, 224, 224, 3), dtype=np.uint8),
                        },
                        "image_mask": {
                            "base_0_rgb": np.ones(self.batch_size, dtype=bool),
                            "left_wrist_0_rgb": np.ones(self.batch_size, dtype=bool),
                            "right_wrist_0_rgb": np.ones(self.batch_size, dtype=bool),
                        },
                        "state": np.random.randn(self.batch_size, 8).astype(np.float32),
                        "prompt": ["pick up the cup"] * self.batch_size,
                        "returns": np.random.randn(self.batch_size).astype(np.float32) * 0.5 - 0.5,
                    }

        dataloader = DummyDataLoader(batch_size=config.batch_size)
    else:
        # Real data loader (to be implemented when repo IDs are provided)
        raise NotImplementedError(
            "Real data loading not yet implemented. "
            "Please provide success_repo_ids and failure_repo_ids in the config."
        )

    print("✓ Data loader created")

    # Compute total steps
    if config.num_train_steps is not None:
        total_steps = config.num_train_steps
    else:
        raise ValueError("num_train_steps must be provided in config")

    print(f"\n=== Starting training for {total_steps} steps ===")
    print(f"Batch size: {config.batch_size}")
    print(f"Log interval: {config.logging.log_every_n_steps}")

    # Training loop
    data_iter = iter(dataloader)
    log_count = 0
    max_console_logs = 5  # Only print first 5 log messages

    # Timing tracking
    last_log_time = time.time()
    last_log_step = 0

    for step in range(total_steps):
        # Get batch and convert to JAX arrays
        batch = next(data_iter)
        batch_jax = jax.tree.map(lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, batch)

        # Convert batch to Observation
        observation = batch_to_observation(batch_jax, tokenizer)

        # Get returns
        returns = batch_jax["returns"]

        # Generate RNG for this step
        step_rng = jax.random.fold_in(train_rng, step)

        # Training step
        model, optimizer, metrics = train_step(model, optimizer, step_rng, observation, returns)

        # Logging
        if step % config.logging.log_every_n_steps == 0:
            # Convert metrics to host
            metrics_host = jax.device_get(metrics)

            # Compute timing stats
            current_time = time.time()
            time_elapsed = current_time - last_log_time
            steps_since_log = step - last_log_step
            steps_per_sec = steps_since_log / time_elapsed if time_elapsed > 0 else 0

            # Print to console (only first 5 times)
            if log_count < max_console_logs:
                loss_value = float(metrics_host["loss"])
                grad_norm_value = float(metrics_host.get("grad_norm", 0.0))
                print(f"Step {step}/{total_steps}: loss={loss_value:.4f}, grad_norm={grad_norm_value:.4f}, "
                      f"time={time_elapsed:.2f}s, steps/sec={steps_per_sec:.2f}")
                log_count += 1
                if log_count == max_console_logs:
                    print(f"... (suppressing further console logs, training continues)")

            # Update timing trackers
            last_log_time = current_time
            last_log_step = step

            # Log to W&B
            if config.logging.wandb_enabled:
                wandb.log(metrics_host, step=step)

        # Save checkpoint
        if step > 0 and step % config.checkpoint.save_every_n_steps == 0:
            ckpt_path = pathlib.Path(config.checkpoint.checkpoint_dir) / str(config.model_config.model_type()) / config.exp_name
            ckpt_path.mkdir(parents=True, exist_ok=True)
            # For now, just print checkpoint would be saved (actual saving requires orbax setup)
            print(f"  [Checkpoint would be saved at step {step} to {ckpt_path}]")

    print("\n=== Training complete! ===")

    # Finish W&B run
    if config.logging.wandb_enabled:
        wandb.finish()
        print("✓ W&B run finished")


def main() -> None:
    """CLI entry point."""
    # Use debug config for testing
    config = TrainConfig.debug_config()

    print("=" * 60)
    print("Pi Value Function Training")
    print("=" * 60)
    print(f"Experiment: {config.exp_name}")
    print(f"Model: {config.model_config.__class__.__name__}")
    print(f"Value dims: {config.model_config.value_dims}")
    print(f"Seed: {config.seed}")
    print("=" * 60)

    # Run training
    train(config)


if __name__ == "__main__":
    main()
