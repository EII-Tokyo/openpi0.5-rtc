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

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tqdm import tqdm
import wandb

from pi_value_function.config import PiValueConfig
from pi_value_function.pi_value import PiValue
from pi_value_function.training.train_config import TrainConfig, ValueDataConfig, CheckpointConfig, LoggingConfig
from pi_value_function.training.data_loader import create_value_dataloader
from pi_value_function.training.weight_loader import SigLIP2WeightLoader
from pi_value_function.training.direct_checkpoint_loader import DirectGemma3WeightLoader
from pi_value_function.training import checkpoint_manager as ckpt_manager
from pi_value_function.training.checkpoint_downloader import (
    download_checkpoint,
    download_gemma_from_kaggle,
    SIGLIP2_SO400M14_224_URL,
)
from openpi.models.model import Observation
from openpi.models.tokenizer import Gemma3Tokenizer
import openpi.training.optimizer as _optimizer


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

    # Load SigLIP weights (downloaded to ~/.cache/openpi)
    siglip_checkpoint = download_checkpoint(SIGLIP2_SO400M14_224_URL)
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

    # Load Gemma weights (downloaded from Kaggle to ~/.cache/openpi)
    gemma_model_path, gemma_tokenizer_path = download_gemma_from_kaggle()
    gemma_loader = DirectGemma3WeightLoader(
        checkpoint_path=str(gemma_model_path),
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
    print(f"✓ Loaded Gemma weights from {gemma_model_path}")
    print(f"  Timing: state_extract={t1-t0:.2f}s, load={t2-t1:.2f}s, merge={t3-t2:.2f}s, update={t4-t3:.2f}s")

    # Return model and tokenizer
    return model, gemma_tokenizer_path


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

def validate_step(
    model: PiValue,
    rng: jax.Array,
    observation: Observation,
    returns: jax.Array,
) -> dict[str, jax.Array]:  
    """Single validation step.

    Args:
        model: PiValue model
        rng: Random key for this step

    Returns:
        Validation metrics dict
    """
    def loss_fn(model: PiValue) -> jax.Array:
        """Compute mean loss over batch."""
        losses = model.compute_loss(rng, observation, returns, train=False)
        return jnp.mean(losses)

    # Compute validation loss
    loss = loss_fn(model)
    metrics = {
        "val_loss": loss,
    }
    return metrics


def train(config: TrainConfig) -> None:
    """Main training function.

    Args:
        config: Training configuration
    """
    # Initialize W&B
    if config.logging.wandb_enabled:
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
    model, tokenizer_path = create_model_with_weights(config, model_rng)
    print("✓ Model created successfully")

    # Create optimizer
    print("\n=== Setting up optimizer ===")
    lr_schedule = config.lr_schedule.create()
    tx = config.optimizer.create(lr_schedule)

    # Freeze SigLIP backbone and Gemma - only train projection heads
    # Get full model state for parameter counting
    full_state = nnx.state(model, nnx.Param)

    # Create a filter to select only trainable parameters:
    # 1. Value MLP head (model.value_mlp - Sequential with Linear + GELU + Linear)
    # 2. SigLIP projection head (the 'head' layer in model.siglip)
    def trainable_filter(path, value):
        """Filter to select only trainable parameters."""
        path_str = '/'.join(str(p) for p in path)
        # Train value_mlp (non-linear value head)
        if 'value_mlp' in path_str:
            return True
        # Train SigLIP's projection head (projects to Gemma dimension)
        # The head is typically named 'head' in the SigLIP module
        if 'siglip' in path_str and 'head' in path_str:
            return True
        return False

    # Split model into trainable and frozen parameters
    trainable_state = nnx.State({
        k: v for k, v in nnx.iter_graph(full_state)
        if trainable_filter(k, v)
    })

    # Create optimizer with only trainable parameters
    optimizer = nnx.Optimizer(trainable_state, tx)

    # Count parameters
    trainable_params = sum(x.size for x in jax.tree_util.tree_leaves(trainable_state) if hasattr(x, 'size'))
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(full_state) if hasattr(x, 'size'))
    frozen_params = total_params - trainable_params

    print(f"✓ Optimizer created: {config.optimizer.__class__.__name__}")
    print(f"  Peak LR: {config.lr_schedule.peak_lr}")
    print(f"  Warmup steps: {config.lr_schedule.warmup_steps}")
    print(f"  Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"  Frozen params: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    print(f"  ⚠️  Frozen: SigLIP backbone and Gemma")
    print(f"  ✓ Training: SigLIP projection head + Value projection head")

    # Initialize checkpoint manager
    print("\n=== Setting up checkpointing ===")
    ckpt_path = pathlib.Path(config.checkpoint.checkpoint_dir) / str(config.model_config.model_type()) / config.exp_name
    checkpoint_mgr, resuming = ckpt_manager.initialize_checkpoint_manager(
        ckpt_path,
        max_to_keep=config.checkpoint.keep_n_checkpoints,
        keep_period=config.checkpoint.keep_period,
        overwrite=config.checkpoint.overwrite,
        resume=config.checkpoint.resume,
    )
    print(f"✓ Checkpoint manager initialized: {ckpt_path}")

    # Restore from checkpoint if resuming
    start_step = 0
    if resuming:
        print("\n=== Restoring from checkpoint ===")
        model, optimizer, start_step = ckpt_manager.restore_checkpoint(
            checkpoint_mgr, model, optimizer
        )
        print(f"✓ Resumed from step {start_step}")

    # Create tokenizer
    print("\n=== Creating tokenizer ===")
    tokenizer = Gemma3Tokenizer(max_len=48, path=tokenizer_path)
    print("✓ Tokenizer created")

    # Create data loaders
    print("\n=== Creating data loaders ===")
    # Check if real LeRobot repo IDs are provided
    if config.data.success_repo_ids or config.data.failure_repo_ids:
        print("Using real LeRobot datasets:")
        if config.data.success_repo_ids:
            print(f"  Success repos: {config.data.success_repo_ids}")
        if config.data.failure_repo_ids:
            print(f"  Failure repos: {config.data.failure_repo_ids}")

        # Create training dataloader
        print("Creating training dataloader...")
        dataloader = create_value_dataloader(
            tokenizer=tokenizer,
            success_repo_ids=config.data.success_repo_ids,
            failure_repo_ids=config.data.failure_repo_ids,
            batch_size=config.batch_size,
            failure_cost_json=config.data.failure_cost_json,
            default_c_fail=config.data.default_c_fail,
            success_sampling_ratio=config.data.success_sampling_ratio,
            num_workers=config.num_workers,
            seed=config.seed,
            target_task=config.data.target_task,
            treat_other_tasks_as_failure=config.data.treat_other_tasks_as_failure,
        )
        print("✓ Training dataloader created")

        # Create validation dataloader if validation is enabled
        val_dataloader = None
        if config.num_steps_per_validation > 0:
            print("Creating validation dataloader...")
            print(f"  Using same repos as training with 'val' split")
            print(f"  Train/val ratio: {config.data.train_split:.0%} / {config.data.val_split:.0%}")

            val_dataloader = create_value_dataloader(
                tokenizer=tokenizer,
                success_repo_ids=config.data.success_repo_ids,  # Same as training
                failure_repo_ids=config.data.failure_repo_ids,  # Same as training
                batch_size=config.batch_size,
                failure_cost_json=config.data.failure_cost_json,
                default_c_fail=config.data.default_c_fail,
                success_sampling_ratio=config.data.success_sampling_ratio,
                num_workers=config.num_workers,
                seed=config.seed + 1,  # Different seed for validation
                split="val",  # Use validation split
                train_split=config.data.train_split,
                split_seed=config.data.split_seed,
                target_task=config.data.target_task,
                treat_other_tasks_as_failure=config.data.treat_other_tasks_as_failure,
            )
            print("✓ Validation dataloader created")
    else:
        print("Warning: No repo IDs provided, using dummy data for testing")
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
                        "returns": np.random.uniform(low=-1.0, high=0.0, size=self.batch_size).astype(np.float32)
                    }

        dataloader = DummyDataLoader(batch_size=config.batch_size)
        val_dataloader = None

    # Compute total steps
    if config.num_train_steps is not None:
        total_steps = config.num_train_steps
    else:
        raise ValueError("num_train_steps must be provided in config")

    print(f"\n=== Starting training for {total_steps} steps ===")
    print(f"Batch size: {config.batch_size}")
    print(f"Log interval: {config.logging.log_every_n_steps}")
    if resuming:
        print(f"Resuming from step: {start_step}")

    # Training loop
    data_iter = iter(dataloader)
    val_iter = iter(val_dataloader) if val_dataloader is not None else None

    # Create progress bar
    pbar = tqdm(range(start_step, total_steps), desc="Training", unit="step", initial=start_step, total=total_steps)

    for step in pbar:
        # Get batch (already has tokenized prompts from collate_fn)
        try:
            batch = next(data_iter)
        except IndexError as e:
            pbar.write(f"DataLoader IndexError at step {step}: {e}. Resetting data iterator and retrying...")
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Convert to JAX arrays
        batch_jax = jax.tree.map(lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, batch)

        # Convert batch to Observation (prompts already tokenized in collate_fn)
        observation = Observation.from_dict(batch_jax)

        # Get returns
        returns = batch_jax["returns"]

        # Generate RNG for this step
        step_rng = jax.random.fold_in(train_rng, step)

        # Training step
        model, optimizer, metrics = train_step(model, optimizer, step_rng, observation, returns)

        # Validation
        if val_iter is not None and config.num_steps_per_validation != 0 and step % config.num_steps_per_validation == 0 and step != 0:
            # Get validation batch
            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_dataloader) if val_dataloader is not None else None
                val_batch = next(val_iter)

            val_batch_jax = jax.tree.map(lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, val_batch)
            val_observation = Observation.from_dict(val_batch_jax)
            val_returns = val_batch_jax["returns"]

            # Run validation step
            val_metrics = validate_step(model, step_rng, val_observation, val_returns)

            # Convert to host and log
            val_metrics_host = jax.device_get(val_metrics)
            val_loss_value = float(val_metrics_host["val_loss"])
            pbar.write(f"  Validation at step {step}: val_loss={val_loss_value:.4f}")

            # Log to W&B
            if config.logging.wandb_enabled:
                wandb.log(val_metrics_host, step=step)

        # Logging
        if step % config.logging.log_every_n_steps == 0:
            # Convert metrics to host
            metrics_host = jax.device_get(metrics)

            # Update progress bar with metrics
            loss_value = float(metrics_host["loss"])
            grad_norm_value = float(metrics_host.get("grad_norm", 0.0))
            pbar.set_postfix({
                "loss": f"{loss_value:.4f}",
                "gn": f"{grad_norm_value:.2f}",
            })

            # Log to W&B
            if config.logging.wandb_enabled:
                wandb.log(metrics_host, step=step)

        # Save checkpoint
        if step > 0 and step % config.checkpoint.save_every_n_steps == 0:
            pbar.write(f"  Saving checkpoint at step {step}...")
            ckpt_manager.save_checkpoint(checkpoint_mgr, model, optimizer, step)
            pbar.write(f"  ✓ Checkpoint saved")

    pbar.close()

    # Save final checkpoint
    print(f"\nSaving final checkpoint at step {total_steps}...")
    ckpt_manager.save_checkpoint(checkpoint_mgr, model, optimizer, total_steps)
    print("✓ Final checkpoint saved")

    print("\n=== Training complete! ===")

    # Finish W&B run
    if config.logging.wandb_enabled:
        wandb.finish()
        print("✓ W&B run finished")


def main() -> None:
    """CLI entry point."""
    # Create config with real LeRobot data
    config = TrainConfig(
        exp_name="pi_value_droid_bs64_30k",
        model_config=PiValueConfig(
            value_dims=201,  # 201 bins for categorical distribution over [-1, 0]
            value_min=-1.0,
            value_max=0.0,
            gemma_variant="gemma-3-270m",
            siglip_variant="siglip2-so400m-patch16-384",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=3e-5,
            decay_steps=10_000,
            decay_lr=3e-6,
        ),
        optimizer=_optimizer.AdamW(
            weight_decay=0.01,
            clip_gradient_norm=1.0,
        ),
        num_train_steps=30_000,
        batch_size=64,
        data=ValueDataConfig(
            # Dataset repos (episodes will be split randomly into train/val)
            success_repo_ids=[
                "michios/droid_xxjd",
                "michios/droid_xxjd_2",
                "michios/droid_xxjd_3",
                "michios/droid_xxjd_4",
                "michios/droid_xxjd_5",
                "michios/droid_xxjd_6",
                "michios/droid_xxjd_7",
            ],
            failure_repo_ids=[
                "michios/droid_xxjd_fail_1"
            ],
            train_split=0.9,  # 90% train, 10% val
            split_seed=42,  # Random but deterministic episode shuffling
            failure_cost_json="configs/failure_costs.json",
            default_c_fail=100.0,
            success_sampling_ratio=0.5,
        ),
        checkpoint=CheckpointConfig(
            checkpoint_dir="./checkpoints",
            save_every_n_steps=2_500,
            keep_n_checkpoints=5,
            overwrite=True,
        ),
        logging=LoggingConfig(
            log_every_n_steps=50,
            wandb_enabled=True,
            wandb_project="pi-value-function",
            wandb_run_name="droid_value_training_v2",
        ),
        num_workers=4,
        num_steps_per_validation=500,  # Run validation every 500 steps
        seed=42,
    )
    
    # config = TrainConfig.debug_config()

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
