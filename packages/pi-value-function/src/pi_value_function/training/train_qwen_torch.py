"""PyTorch training loop for PiValue with Qwen3-VL backbone."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import shutil
import time

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import wandb

from pi_value_function.backbone_type import BACKBONE_QWEN3VL
from pi_value_function.pi_value_qwen3vl_torch import PiValueQwen3VLTorch
from pi_value_function.training.qwen_collate import create_qwen_value_dataloader
from pi_value_function.training.train_config import TrainConfig


def _checkpoint_dir(config: TrainConfig) -> pathlib.Path:
    return pathlib.Path(config.checkpoint.checkpoint_dir) / str(config.model_config.model_type()) / config.exp_name


def _available_steps(checkpoint_dir: pathlib.Path) -> list[int]:
    if not checkpoint_dir.exists():
        return []

    steps = []
    for child in checkpoint_dir.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        if (child / "model.safetensors").exists() and (child / "metadata.pt").exists():
            steps.append(int(child.name))

    return sorted(steps)


def _save_checkpoint(
    model: PiValueQwen3VLTorch | DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_dir: pathlib.Path,
    config: TrainConfig,
) -> None:
    tmp_dir = checkpoint_dir / f"tmp_{step}"
    final_dir = checkpoint_dir / str(step)

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if isinstance(model, DistributedDataParallel) else model
    safetensors.torch.save_model(model_to_save, tmp_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")

    metadata = {
        "step": step,
        "timestamp": time.time(),
        "backbone": BACKBONE_QWEN3VL,
        "hf_model_id": config.model_config.hf_model_id,
        "value_head_layers": config.model_config.value_head_layers,
        "value_dims": config.model_config.value_dims,
        "value_min": config.model_config.value_min,
        "value_max": config.model_config.value_max,
        "config": dataclasses.asdict(config),
    }
    torch.save(metadata, tmp_dir / "metadata.pt")

    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)


def _load_checkpoint(
    model: PiValueQwen3VLTorch | DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    step_dir: pathlib.Path,
    device: torch.device,
    *,
    load_optimizer: bool,
) -> int:
    model_to_load = model.module if isinstance(model, DistributedDataParallel) else model
    safetensors.torch.load_model(model_to_load, step_dir / "model.safetensors", device=str(device))
    if load_optimizer and (step_dir / "optimizer.pt").exists():
        try:
            optimizer_state = torch.load(step_dir / "optimizer.pt", map_location=device, weights_only=False)
        except TypeError:
            optimizer_state = torch.load(step_dir / "optimizer.pt", map_location=device)
        optimizer.load_state_dict(optimizer_state)

    try:
        metadata = torch.load(step_dir / "metadata.pt", map_location=device, weights_only=False)
    except TypeError:
        metadata = torch.load(step_dir / "metadata.pt", map_location=device)
    if isinstance(metadata, dict) and "step" in metadata:
        return int(metadata["step"])
    return int(step_dir.name)


def _lr_at_step(step: int, *, warmup_steps: int, peak_lr: float, decay_steps: int, decay_lr: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        init_lr = peak_lr / (warmup_steps + 1)
        return init_lr + (peak_lr - init_lr) * step / warmup_steps

    progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
    cosine = 0.5 * (1 + np.cos(np.pi * progress))
    return decay_lr + (peak_lr - decay_lr) * cosine


@dataclasses.dataclass(frozen=True)
class _DistributedContext:
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def _setup_distributed() -> _DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ctx = _DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size)

    if ctx.is_distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA GPUs")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    return ctx


def _cleanup_distributed(ctx: _DistributedContext) -> None:
    if ctx.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def _distributed_mean(value: torch.Tensor | float, *, device: torch.device, world_size: int) -> float:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=device, dtype=torch.float32)
    else:
        tensor = torch.tensor(float(value), device=device, dtype=torch.float32)

    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return float(tensor.item())


def _compute_loss_vector(
    model: PiValueQwen3VLTorch | DistributedDataParallel,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    base_model = model.module if isinstance(model, DistributedDataParallel) else model
    logits = model(batch)
    returns = batch["returns"].to(logits.device, dtype=torch.float32)
    target_bins = base_model.discretize_returns(returns)
    return F.cross_entropy(logits, target_bins, reduction="none")


def train_qwen_torch(config: TrainConfig) -> None:
    if config.model_config.backbone != BACKBONE_QWEN3VL:
        raise ValueError(
            f"train_qwen_torch only supports backbone='{BACKBONE_QWEN3VL}', got '{config.model_config.backbone}'"
        )
    dist_ctx = _setup_distributed()
    wandb_started = False

    try:
        if torch.cuda.is_available():
            if dist_ctx.is_distributed:
                torch.cuda.set_device(dist_ctx.local_rank)
                device = torch.device(f"cuda:{dist_ctx.local_rank}")
            else:
                device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        if dist_ctx.is_main_process:
            print(f"Using device: {device} (world_size={dist_ctx.world_size})")

        if config.logging.wandb_enabled and dist_ctx.is_main_process:
            wandb.init(
                project=config.logging.wandb_project,
                name=config.logging.wandb_run_name or config.exp_name,
                entity=config.logging.wandb_entity,
                config=dataclasses.asdict(config),
            )
            wandb_started = True

        torch.manual_seed(config.seed + dist_ctx.rank)
        np.random.seed(config.seed + dist_ctx.rank)

        model: PiValueQwen3VLTorch | DistributedDataParallel = PiValueQwen3VLTorch(config.model_config, device=device)
        trainable_params = model.trainable_parameters()
        if not trainable_params:
            raise ValueError("No trainable parameters found for Qwen value model")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)
        if dist_ctx.is_main_process:
            print(f"Total params: {total_params:,}")
            print(f"Trainable params: {trainable_count:,} ({100 * trainable_count / total_params:.2f}%)")

        warmup_steps = int(getattr(config.lr_schedule, "warmup_steps", 0))
        peak_lr = float(getattr(config.lr_schedule, "peak_lr", 3e-4))
        decay_steps = int(getattr(config.lr_schedule, "decay_steps", config.num_train_steps or 1))
        decay_lr = float(getattr(config.lr_schedule, "decay_lr", peak_lr / 10.0))

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=peak_lr,
            betas=(
                float(getattr(config.optimizer, "b1", 0.9)),
                float(getattr(config.optimizer, "b2", 0.999)),
            ),
            eps=float(getattr(config.optimizer, "eps", 1e-8)),
            weight_decay=float(getattr(config.optimizer, "weight_decay", 0.0)),
        )

        checkpoint_dir = _checkpoint_dir(config)
        if config.checkpoint.overwrite and checkpoint_dir.exists() and not config.checkpoint.resume and dist_ctx.is_main_process:
            shutil.rmtree(checkpoint_dir)
        if dist_ctx.is_distributed:
            dist.barrier()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        start_step = 0

        if config.checkpoint.resume_checkpoint_path:
            source_dir = pathlib.Path(config.checkpoint.resume_checkpoint_path).expanduser().resolve()
            source_steps = _available_steps(source_dir)
            if not source_steps:
                raise ValueError(f"No valid checkpoints found at resume_checkpoint_path={source_dir}")
            source_step_dir = source_dir / str(source_steps[-1])
            start_step = _load_checkpoint(model, optimizer, source_step_dir, device, load_optimizer=config.checkpoint.resume)
            if not config.checkpoint.resume:
                start_step = 0
            if dist_ctx.is_main_process:
                print(f"Loaded warm-start checkpoint from {source_step_dir}")
        elif config.checkpoint.resume:
            steps = _available_steps(checkpoint_dir)
            if steps:
                step_dir = checkpoint_dir / str(steps[-1])
                start_step = _load_checkpoint(model, optimizer, step_dir, device, load_optimizer=True)
                if dist_ctx.is_main_process:
                    print(f"Resumed from checkpoint: {step_dir}")

        if dist_ctx.is_distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[dist_ctx.local_rank],
                output_device=dist_ctx.local_rank,
                find_unused_parameters=False,
                broadcast_buffers=False,
            )

        max_token_len = int(getattr(config.model_config, "max_token_len", 256))

        train_loader = create_qwen_value_dataloader(
            hf_model_id=config.model_config.hf_model_id,
            success_repo_ids=config.data.success_repo_ids,
            failure_repo_ids=config.data.failure_repo_ids,
            batch_size=config.batch_size,
            failure_cost_json=config.data.failure_cost_json,
            default_c_fail=config.data.default_c_fail,
            success_sampling_ratio=config.data.success_sampling_ratio,
            num_workers=config.num_workers,
            seed=config.seed,
            split="train",
            train_split=config.data.train_split,
            split_seed=config.data.split_seed,
            target_task=config.data.target_task,
            treat_other_tasks_as_failure=config.data.treat_other_tasks_as_failure,
            max_token_len=max_token_len,
            rank=dist_ctx.rank,
        )

        val_loader = None
        if config.num_steps_per_validation > 0:
            val_loader = create_qwen_value_dataloader(
                hf_model_id=config.model_config.hf_model_id,
                success_repo_ids=config.data.success_repo_ids,
                failure_repo_ids=config.data.failure_repo_ids,
                batch_size=config.batch_size,
                failure_cost_json=config.data.failure_cost_json,
                default_c_fail=config.data.default_c_fail,
                success_sampling_ratio=config.data.success_sampling_ratio,
                num_workers=config.num_workers,
                seed=config.seed + 1,
                split="val",
                train_split=config.data.train_split,
                split_seed=config.data.split_seed,
                target_task=config.data.target_task,
                treat_other_tasks_as_failure=config.data.treat_other_tasks_as_failure,
                max_token_len=max_token_len,
                rank=dist_ctx.rank,
            )

        grad_clip_norm = float(getattr(config.optimizer, "clip_gradient_norm", 1.0))
        amp_enabled = device.type == "cuda" and config.model_config.backbone_dtype in {"bfloat16", "float16"}
        amp_dtype = torch.bfloat16 if config.model_config.backbone_dtype == "bfloat16" else torch.float16

        train_iter = iter(train_loader)
        val_iter = iter(val_loader) if val_loader is not None else None

        total_steps = int(config.num_train_steps) if config.num_train_steps is not None else 0
        if total_steps <= 0:
            raise ValueError("num_train_steps must be a positive integer")

        log_every = max(1, config.logging.log_every_n_steps)

        model.train()
        start_time = time.time()
        if dist_ctx.is_main_process:
            step_iterator = tqdm(
                range(start_step, total_steps),
                desc="Training",
                unit="step",
                initial=start_step,
                total=total_steps,
            )
        else:
            step_iterator = range(start_step, total_steps)

        for step in step_iterator:
            try:
                batch = next(train_iter)
            except (StopIteration, IndexError):
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }

            lr = _lr_at_step(
                step,
                warmup_steps=warmup_steps,
                peak_lr=peak_lr,
                decay_steps=decay_steps,
                decay_lr=decay_lr,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    losses = _compute_loss_vector(model, batch)
                    loss = losses.mean()
            else:
                losses = _compute_loss_vector(model, batch)
                loss = losses.mean()

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=grad_clip_norm)
            optimizer.step()

            loss_value = _distributed_mean(loss, device=device, world_size=dist_ctx.world_size)
            grad_norm_value = _distributed_mean(grad_norm, device=device, world_size=dist_ctx.world_size)

            if step % log_every == 0:
                elapsed = time.time() - start_time
                if dist_ctx.is_main_process:
                    step_iterator.set_postfix({"loss": f"{loss_value:.4f}", "gn": f"{grad_norm_value:.2f}", "lr": f"{lr:.2e}"})
                    if config.logging.wandb_enabled:
                        wandb.log(
                            {
                                "loss": loss_value,
                                "grad_norm": grad_norm_value,
                                "learning_rate": lr,
                                "time_per_step": elapsed / log_every,
                            },
                            step=step,
                        )
                start_time = time.time()

            if (
                val_iter is not None
                and config.num_steps_per_validation > 0
                and step > 0
                and step % config.num_steps_per_validation == 0
            ):
                model.eval()
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader) if val_loader is not None else None
                    val_batch = next(val_iter)

                val_batch = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in val_batch.items()
                }

                with torch.no_grad():
                    val_loss = _compute_loss_vector(model, val_batch).mean()
                val_loss_value = _distributed_mean(val_loss, device=device, world_size=dist_ctx.world_size)

                if dist_ctx.is_main_process:
                    step_iterator.write(f"  Validation at step {step}: val_loss={val_loss_value:.4f}")
                    if config.logging.wandb_enabled:
                        wandb.log({"val_loss": val_loss_value}, step=step)
                model.train()

            if step > 0 and step % config.checkpoint.save_every_n_steps == 0:
                if dist_ctx.is_main_process:
                    step_iterator.write(f"  Saving checkpoint at step {step}...")
                    _save_checkpoint(model, optimizer, step, checkpoint_dir, config)
                    step_iterator.write("  ✓ Checkpoint saved")
                if dist_ctx.is_distributed:
                    dist.barrier()

        if dist_ctx.is_main_process and isinstance(step_iterator, tqdm):
            step_iterator.close()

        if dist_ctx.is_main_process:
            print(f"Saving final checkpoint at step {total_steps}...")
            _save_checkpoint(model, optimizer, total_steps, checkpoint_dir, config)
            print("✓ Final checkpoint saved")
        if dist_ctx.is_distributed:
            dist.barrier()

        if config.logging.wandb_enabled and dist_ctx.is_main_process and wandb_started:
            wandb.finish()
    finally:
        _cleanup_distributed(dist_ctx)
