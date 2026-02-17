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

# Data parallelism sharding
BATCH_AXIS = "batch"


def _create_data_parallel_mesh() -> jax.sharding.Mesh:
    """Create a 1D mesh for data parallelism across all devices."""
    devices = jax.devices()
    return jax.sharding.Mesh(devices, (BATCH_AXIS,))


def _shard_batch(batch, mesh: jax.sharding.Mesh):
    """Shard a batch along the batch axis across devices."""
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(BATCH_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def _shard_leaf(x):
        if isinstance(x, (np.ndarray, jnp.ndarray)) and x.ndim > 0:
            return jax.device_put(x, data_sharding)
        return jax.device_put(x, replicated)

    return jax.tree.map(_shard_leaf, batch)



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


def _train_step_impl(
    model: PiValue,
    optimizer: nnx.Optimizer,
    rng: jax.Array,
    observation: Observation,
    returns: jax.Array,
) -> tuple[PiValue, nnx.Optimizer, dict[str, jax.Array]]:
    """Single training step with gradient update (pure implementation).

    Uses NNX patterns:
    - nnx.value_and_grad for gradient computation
    - optimizer.update() for parameter updates
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


def _validate_step_impl(
    model: PiValue,
    rng: jax.Array,
    observation: Observation,
    returns: jax.Array,
) -> dict[str, jax.Array]:
    """Single validation step (pure implementation)."""
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


def create_jitted_train_step(mesh: jax.sharding.Mesh | None, model, optimizer):
    """Create a JIT-compiled train step with proper sharding for multi-GPU.

    For single GPU: uses simple nnx.jit.
    For multi-GPU: uses jax.jit with explicit in/out shardings via the mesh.
    """
    if mesh is None or len(mesh.devices.flat) <= 1:
        # Single device - simple JIT
        return nnx.jit(_train_step_impl)

    # Multi-device: extract state, create shardings, JIT with explicit shardings
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(BATCH_AXIS))

    # Get the NNX state structure for sharding specs
    model_graphdef, model_state = nnx.split(model)
    opt_graphdef, opt_state = nnx.split(optimizer)

    # All model/optimizer state is replicated; batch data is sharded
    model_state_sharding = jax.tree.map(lambda _: replicated, model_state)
    opt_state_sharding = jax.tree.map(lambda _: replicated, opt_state)

    def _pure_train_step(model_state, opt_state, rng, observation, returns):
        """Pure function operating on state pytrees (not NNX objects)."""
        model = nnx.merge(model_graphdef, model_state)
        optimizer = nnx.merge(opt_graphdef, opt_state)

        model, optimizer, metrics = _train_step_impl(model, optimizer, rng, observation, returns)

        _, new_model_state = nnx.split(model)
        _, new_opt_state = nnx.split(optimizer)
        return new_model_state, new_opt_state, metrics

    # Build sharding specs for observation (batch-sharded for arrays, replicated for scalars)
    def _obs_sharding(x):
        if hasattr(x, 'ndim') and x.ndim > 0:
            return data_sharding
        return replicated

    @functools.wraps(_pure_train_step)
    def _jitted_wrapper(model_obj, optimizer_obj, rng, observation, returns):
        # Split NNX objects into state
        nonlocal model_graphdef, opt_graphdef
        model_graphdef, m_state = nnx.split(model_obj)
        opt_graphdef, o_state = nnx.split(optimizer_obj)

        # Build observation sharding from actual structure
        obs_sharding = jax.tree.map(_obs_sharding, observation)
        returns_sharding = data_sharding

        in_shardings = (model_state_sharding, opt_state_sharding, replicated, obs_sharding, returns_sharding)
        out_shardings = (model_state_sharding, opt_state_sharding, replicated)

        # JIT with explicit shardings (cached after first call)
        jitted_fn = jax.jit(_pure_train_step, in_shardings=in_shardings, out_shardings=out_shardings)
        new_m_state, new_o_state, metrics = jitted_fn(m_state, o_state, rng, observation, returns)

        # Merge state back into NNX objects
        nnx.update(model_obj, new_m_state)
        nnx.update(optimizer_obj, new_o_state)
        return model_obj, optimizer_obj, metrics

    return _jitted_wrapper


def create_jitted_validate_step(mesh: jax.sharding.Mesh | None, model):
    """Create a JIT-compiled validation step with proper sharding."""
    if mesh is None or len(mesh.devices.flat) <= 1:
        return nnx.jit(_validate_step_impl)

    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(BATCH_AXIS))

    model_graphdef, model_state = nnx.split(model)
    model_state_sharding = jax.tree.map(lambda _: replicated, model_state)

    def _obs_sharding(x):
        if hasattr(x, 'ndim') and x.ndim > 0:
            return data_sharding
        return replicated

    def _pure_validate_step(model_state, rng, observation, returns):
        model = nnx.merge(model_graphdef, model_state)
        return _validate_step_impl(model, rng, observation, returns)

    @functools.wraps(_pure_validate_step)
    def _jitted_wrapper(model_obj, rng, observation, returns):
        nonlocal model_graphdef
        model_graphdef, m_state = nnx.split(model_obj)

        obs_sharding = jax.tree.map(_obs_sharding, observation)
        in_shardings = (model_state_sharding, replicated, obs_sharding, data_sharding)
        out_shardings = replicated

        jitted_fn = jax.jit(_pure_validate_step, in_shardings=in_shardings, out_shardings=out_shardings)
        return jitted_fn(m_state, rng, observation, returns)

    return _jitted_wrapper


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
    # Define a Frozen variable type to mark parameters that shouldn't be trained
    class Frozen(nnx.Variable):
        """Parameters that are frozen and won't be optimized."""
        pass

    # Get all parameters and reclassify frozen ones
    # Use ... to capture all other variable types (RNG state, etc.)
    graphdef, params, other_vars = nnx.split(model, nnx.Param, ...)

    # Filter function to determine which parameters to train
    def should_freeze(path_tuple):
        """Return True if parameter should be frozen."""
        path_str = '/'.join(str(p.key) if hasattr(p, 'key') else str(p) for p in path_tuple)
        # Train value_mlp (non-linear value head)
        if 'value_mlp' in path_str:
            return False
        # Train SigLIP's projection head (projects to Gemma dimension)
        if 'siglip' in path_str and 'head' in path_str:
            return False
        # Freeze everything else (SigLIP backbone, Gemma)
        return True

    # Convert State to pure dict, reclassify, and convert back
    params_dict = params.to_pure_dict()

    # Debug: print first few paths to understand structure
    print("Debugging parameter paths (first 10):")
    for i, (path, _) in enumerate(jax.tree_util.tree_leaves_with_path(params_dict)):
        if i < 10:
            path_str = '/'.join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
            print(f"  Path {i}: {path_str}")
        else:
            break

    def reclassify_variable(path, value):
        """Convert arrays to Frozen or Param based on path."""
        path_str = '/'.join(str(p.key) if hasattr(p, 'key') else str(p) for p in path)
        freeze = should_freeze(path)
        # Debug: print decisions for value_mlp and siglip head params
        if 'value_mlp' in path_str or ('siglip' in path_str and 'head' in path_str):
            print(f"  {'FREEZE' if freeze else 'TRAIN'}: {path_str}")
        if freeze:
            return Frozen(value)
        else:
            return nnx.Param(value)

    # Apply reclassification to the pure dict
    print("Reclassifying parameters...")
    reclassified_dict = jax.tree_util.tree_map_with_path(reclassify_variable, params_dict)

    # Convert back to State
    params = nnx.State(reclassified_dict)

    # Merge back into model (include other_vars like RNG state)
    model = nnx.merge(graphdef, params, other_vars)

    # Create optimizer - it will only optimize nnx.Param (not Frozen)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Count parameters for logging
    all_params = nnx.state(model, nnx.Param, Frozen)
    trainable_params_state = nnx.state(model, nnx.Param)
    frozen_params_state = nnx.state(model, Frozen)

    trainable_count = sum(
        x.value.size if hasattr(x, 'value') and hasattr(x.value, 'size') else 0
        for x in jax.tree_util.tree_leaves(trainable_params_state)
    )
    total_count = sum(
        x.value.size if hasattr(x, 'value') and hasattr(x.value, 'size') else 0
        for x in jax.tree_util.tree_leaves(all_params)
    )
    frozen_count = total_count - trainable_count

    print(f"✓ Optimizer created: {config.optimizer.__class__.__name__}")
    print(f"  Peak LR: {config.lr_schedule.peak_lr}")
    print(f"  Warmup steps: {config.lr_schedule.warmup_steps}")
    print(f"  Trainable params: {trainable_count:,} ({trainable_count/total_count*100:.2f}%)")
    print(f"  Frozen params: {frozen_count:,} ({frozen_count/total_count*100:.2f}%)")
    print(f"  ⚠️  Frozen: SigLIP backbone and Gemma")
    print(f"  ✓ Training: SigLIP projection head + Value MLP head")

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

    # Set up data parallelism across all devices
    mesh = _create_data_parallel_mesh()
    num_devices = len(jax.devices())
    print(f"\n=== Data parallelism ===")
    print(f"  Devices: {num_devices} ({[str(d) for d in jax.devices()]})")
    print(f"  Mesh: {mesh.shape}")
    if num_devices > 1:
        # Replicate model/optimizer state across all devices
        print(f"  Replicating model across {num_devices} devices...")
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        graphdef, state = nnx.split(model)
        state = jax.device_put(state, replicated)
        model = nnx.merge(graphdef, state)
        graphdef, state = nnx.split(optimizer)
        state = jax.device_put(state, replicated)
        optimizer = nnx.merge(graphdef, state)
        print(f"  ✓ Model and optimizer replicated")

    # Create JIT-compiled train/validate steps with proper sharding
    train_step = create_jitted_train_step(mesh if num_devices > 1 else None, model, optimizer)
    validate_step = create_jitted_validate_step(mesh if num_devices > 1 else None, model)
    print(f"  ✓ JIT-compiled train/validate steps created ({'multi-device' if num_devices > 1 else 'single-device'})")

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

    # Validate batch size divisibility for data parallelism
    if num_devices > 1 and config.batch_size % num_devices != 0:
        raise ValueError(
            f"batch_size ({config.batch_size}) must be divisible by num_devices ({num_devices}) "
            f"for data parallelism. Use batch_size={config.batch_size // num_devices * num_devices} or {(config.batch_size // num_devices + 1) * num_devices}."
        )

    print(f"\n=== Starting training for {total_steps} steps ===")
    print(f"Batch size: {config.batch_size}" + (f" ({config.batch_size // num_devices} per device)" if num_devices > 1 else ""))
    print(f"Log interval: {config.logging.log_every_n_steps}")
    if resuming:
        print(f"Resuming from step: {start_step}")

    # Sanity check: log first 5 samples to W&B
    if config.logging.wandb_enabled:
        print("\n=== Sanity check: logging first 5 samples to W&B ===")
        n_sanity = 5
        sanity_dataset = dataloader.dataset
        camera_keys = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]

        # Build table
        table_columns = ["sample_idx", "prompt", "is_success", "return_target"]
        table = wandb.Table(columns=table_columns)

        # Collect per-camera images across samples for panel display
        camera_images = {cam: [] for cam in camera_keys}

        for i in range(n_sanity):
            sample = sanity_dataset[i]
            prompt = sample["prompt"]
            ret = float(sample["returns"])
            is_success = sample["is_success"]
            table.add_data(i, prompt, is_success, ret)

            for cam in camera_keys:
                img = sample["image"][cam]  # uint8 [H, W, C]
                camera_images[cam].append(
                    wandb.Image(img, caption=f"Sample {i}")
                )

        # Log table and camera views
        log_dict = {"sanity_check/samples": table}
        for cam in camera_keys:
            log_dict[f"sanity_check/camera_views/{cam}"] = camera_images[cam]

        wandb.log(log_dict, step=0)
        print("✓ Sanity check logged to W&B")

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

        # Convert to JAX arrays and shard across devices
        batch_jax = jax.tree.map(lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, batch)
        if num_devices > 1:
            batch_jax = _shard_batch(batch_jax, mesh)

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
            if num_devices > 1:
                val_batch_jax = _shard_batch(val_batch_jax, mesh)
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
    # Wait for async checkpoint writes to complete before exiting
    checkpoint_mgr.wait_until_finished()
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
