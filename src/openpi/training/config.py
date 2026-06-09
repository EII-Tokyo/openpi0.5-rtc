"""See _CONFIGS for the list of available configs."""

import dataclasses
import difflib
import pathlib
from typing import Any, Literal, TypeAlias

import flax.nnx as nnx
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
from openpi.data import transforms as _transforms

Filter: TypeAlias = nnx.filterlib.Filter

_TROSSEN_RESET_POSE = {"reset_pose": [0, -1.5, 1.5, 0, 0, 0]}
_PI05_BASE_PARAMS = "gs://openpi-assets/checkpoints/pi05_base/params"

_EII_RINSE_11REPO_REPO_IDS = [
    "lyl472324464/2026-05-01_turn_over-lerobot-with-rinse",
    "lyl472324464/2026-05-01_water1-lerobot-with-rinse",
    "lyl472324464/2026-05-03_turn_over-lerobot-with-rinse",
    "lyl472324464/2026-05-04_direction-twist-water-lerobot-with-rinse",
    "lyl472324464/2026-05-04_turn_over-lerobot-with-rinse",
    "lyl472324464/2026-05-04_direction-lerobot-with-rinse",
    "lyl472324464/2026-05-05_direction-water-lerobot-with-rinse",
    "lyl472324464/2026-05-05_water-lerobot-with-rinse",
    "lyl472324464/2026-05-07_water-lerobot-with-rinse",
    "lyl472324464/2026-05-12_insert-to-nozzle_realign-lerobot-with-rinse",
    "lyl472324464/2026-05-13-insert-to-nozzle-no-cap-with-rinse",
]

_EII_RINSE_INSERT_REALIGN_REPO_ID = "lyl472324464/2026-05-12_insert-to-nozzle_realign-lerobot-with-rinse"
_EII_RINSE_INSERT_NO_CAP_REPO_ID = "lyl472324464/2026-05-13-insert-to-nozzle-no-cap-with-rinse"

_EII_RINSE_11REPO_INSERT_X5_REPO_IDS = [
    *_EII_RINSE_11REPO_REPO_IDS,
    *([_EII_RINSE_INSERT_REALIGN_REPO_ID] * 4),
    *([_EII_RINSE_INSERT_NO_CAP_REPO_ID] * 4),
]

_EII_DATA_SYSTEM_WITHOUT_RINSE_41_REPO_IDS = [
    "lyl472324464/2025-09-15-twist-one-bottle-no-box-in-the-front-without-rinse",
    "lyl472324464/2025-11-06-twist-many-bottles-without-rinse",
    "lyl472324464/2025-11-14-twist-two-bottles",
    "lyl472324464/2025-11-18-twist-two-bottles",
    "lyl472324464/2025-11-26-twist-two-bottles",
    "lyl472324464/2025-12-10-twist-one-bottle",
    "lyl472324464/2025-12-23-twist-one-bottle",
    "lyl472324464/2026-01-20-twist-one-bottle",
    "lyl472324464/2026-01-28-twist-many-bottle",
    "lyl472324464/2026-02-03-no-cap-and-direction-without-rinse",
    "lyl472324464/2026-03-04-one-direction-lerobot-without-rinse",
    "lyl472324464/2026-03-05-two-direction-lerobot-without-rinse",
    "lyl472324464/2026-03-12-one-havent-cap",
    "lyl472324464/2026-03-12-one-havent-cap-direction",
    "lyl472324464/2026-04-21_direction-lerobot-without-rinse",
    "lyl472324464/2026-04-21_direction_2-lerobot-without-rinse",
    "lyl472324464/2026-04-21_direction_haven-t_cap-lerobot-without-rinse",
    "lyl472324464/2026-04-21_direction_havent_cap_water-lerobot-without-rinse",
    "lyl472324464/2026-04-23_direction_have_cap_water-lerobot-without-rinse",
    "lyl472324464/2026-04-23_direction_havent_cap_water-lerobot-without-rinse",
    "lyl472324464/2026-04-27_direction_have_cap_water2-lerobot-without-rinse",
    "lyl472324464/2026-04-27direction_have_cap_water-lerobot-without-rinse",
    "lyl472324464/2026-04-28_direction_have_cap_water-lerobot-without-rinse",
    "lyl472324464/2026-04-28_direction_have_cap_water2-lerobot-without-rinse",
    "lyl472324464/2026-05-01_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-03_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-04_direction-lerobot-without-rinse",
    "lyl472324464/2026-05-04_direction-twist-water-lerobot-without-rinse",
    "lyl472324464/2026-05-04_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-05_direction-water-lerobot-without-rinse",
    "lyl472324464/2026-05-11_cap-lerobot-without-rinse",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse",
    "lyl472324464/2026-05-11_twist-lerobot-without-rinse",
    "lyl472324464/2026-05-12_twist-lerobot-without-rinse",
    "lyl472324464/2026-05-12_twist2-lerobot-without-rinse",
    "lyl472324464/2026.03.12_one_have_cap-lerobot-without-rinse",
    "lyl472324464/2026.03.12_one_have_cap_direction-lerobot-without-rinse",
    "lyl472324464/2026.03.12_two_have_all_left-lerobot-without-rinse",
    "lyl472324464/2026.03.12_two_have_cap_all_right-lerobot-without-rinse",
    "lyl472324464/2026.03.12_two_have_cap_one_right-lerobot-without-rinse",
    "lyl472324464/2026.03.16_twist_many-lerobot-without-rinse",
]

_EII_DATA_SYSTEM_WITHOUT_RINSE_RETURN_HOME_29_REPO_IDS = [
    "lyl472324464/2025-09-15-twist-one-bottle-no-box-in-the-front-without-rinse-merged-adjust-pickup",
    "lyl472324464/2025-12-10-twist-one-bottle-merged-adjust-pickup",
    "lyl472324464/2025-12-23-twist-one-bottle-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-01-20-twist-one-bottle-merged-adjust-pickup",
    "lyl472324464/2026-02-03-no-cap-and-direction-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-03-04-one-direction-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026.03.12_one_have_cap-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026.03.12_one_have_cap_direction-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-03-12-one-havent-cap-merged-adjust-pickup",
    "lyl472324464/2026-03-12-one-havent-cap-direction-merged-adjust-pickup",
    "lyl472324464/2026-04-21_direction-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-21_direction_2-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-21_direction_haven-t_cap-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-21_direction_havent_cap_water-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-23_direction_havent_cap_water-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-28_direction_have_cap_water-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-28_direction_have_cap_water2-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_cap-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_twist-lerobot-truncated-return-home-exp-truncated-return-home-20260520-095140",
    "lyl472324464/2026-05-12_twist-lerobot-truncated-return-home-exp-truncated-return-home-20260520-095140",
    "lyl472324464/2026-05-12_twist2-lerobot-truncated-return-home-exp-truncated-return-home-20260520-095140",
    "lyl472324464/2026-05-04_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-03_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-01_turn_over-lerobot-without-rinse",
]

_EII_DATA_SYSTEM_WITHOUT_RINSE_HAVE_CAP_WATER_RETURN_HOME_49_REPO_IDS = [
    "lyl472324464/2025-09-15-twist-one-bottle-no-box-in-the-front-without-rinse-merged-adjust-pickup",
    "lyl472324464/2025-12-10-twist-one-bottle-merged-adjust-pickup",
    "lyl472324464/2025-12-23-twist-one-bottle-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-01-20-twist-one-bottle-merged-adjust-pickup",
    "lyl472324464/2026-02-03-no-cap-and-direction-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-03-04-one-direction-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026.03.12_one_have_cap-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026.03.12_one_have_cap_direction-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-03-12-one-havent-cap-merged-adjust-pickup",
    "lyl472324464/2026-03-12-one-havent-cap-direction-merged-adjust-pickup",
    "lyl472324464/2026-04-21_direction-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-21_direction_2-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-21_direction_haven-t_cap-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-21_direction_havent_cap_water-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-23_direction_have_cap_water-lerobot-without-rinse",
    "lyl472324464/2026-04-23_direction_havent_cap_water-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-27direction_have_cap_water-lerobot-without-rinse",
    "lyl472324464/2026-04-27_direction_have_cap_water2-lerobot-without-rinse",
    "lyl472324464/2026-04-28_direction_have_cap_water-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-04-28_direction_have_cap_water2-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-01_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-03_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-04_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-01_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-03_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-04_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-01_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-03_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-04_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-01_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-03_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-04_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-01_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-03_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-04_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_cap-lerobot-without-rinse-merged-adjust-pickup",
    "lyl472324464/2026-05-11_twist-lerobot-truncated-return-home-exp-truncated-return-home-20260520-095140",
    "lyl472324464/2026-05-12_twist-lerobot-truncated-return-home-exp-truncated-return-home-20260520-095140",
    "lyl472324464/2026-05-12_twist2-lerobot-truncated-return-home-exp-truncated-return-home-20260520-095140",
]

_EII_TURN_OVER_WITHOUT_RINSE_REPO_IDS = [
    "lyl472324464/2026-05-04_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-03_turn_over-lerobot-without-rinse",
    "lyl472324464/2026-05-01_turn_over-lerobot-without-rinse",
]

# Used by `eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo`:
# the base list above has 29 entries, then the three turn_over repos are repeated
# four extra times each, and free-spinning is repeated ten extra times.
_EII_DATA_SYSTEM_WITHOUT_RINSE_RETURN_HOME_TURN_OVER_X5_REPO_IDS = [
    *_EII_DATA_SYSTEM_WITHOUT_RINSE_RETURN_HOME_29_REPO_IDS,
    *[repo_id for repo_id in _EII_TURN_OVER_WITHOUT_RINSE_REPO_IDS for _ in range(4)],
]

_EII_FREE_SPINNING_MERGED_ADJUST_PICKUP_REPO_ID = (
    "lyl472324464/2026-05-11_free-spinning-lerobot-without-rinse-merged-adjust-pickup"
)

_EII_DATA_SYSTEM_WITHOUT_RINSE_RETURN_HOME_TURN_OVER_X5_FREE_SPIN_PLUS10_REPO_IDS = [
    *_EII_DATA_SYSTEM_WITHOUT_RINSE_RETURN_HOME_TURN_OVER_X5_REPO_IDS,
    *([_EII_FREE_SPINNING_MERGED_ADJUST_PICKUP_REPO_ID] * 10),
]

@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig:
    repo_ids: list[str]
    transform_pipeline: _transforms.AlohaTransformPipeline


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: tyro.conf.Suppress[str]
    project_name: str = "openpi"
    exp_name: str = tyro.MISSING
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)
    data: LeRobotAlohaDataConfig = tyro.MISSING
    checkpoint_base_dir: str = "./checkpoints"
    seed: int = 42
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    num_workers: int = 2
    num_train_steps: int = 30_000
    log_interval: int = 100
    save_interval: int = 1000
    keep_period: int | None = 5000
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True
    policy_metadata: dict[str, Any] | None = None
    fsdp_devices: int = 1

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

def _make_twist_train_config(
    name: str,
    *,
    repo_ids: list[str],
    lora: bool,
    batch_size: int,
    num_workers: int,
    fsdp_devices: int = 1,
    include_low: bool = True,
    include_subtask: bool = True,
    gradient_accumulation_steps: int = 1,
    image_resolution: tuple[int, int] = (224, 224),
    max_token_len: int | None = None,
    video_memory_num_frames: int = 1,
    video_memory_stride_seconds: float = 1.0,
    training_time_rtc: bool = False,
    rtc_max_delay: int = 10,
    assets: _transforms.AssetsConfig = tyro.MISSING,
    exp_name: str = tyro.MISSING,
    checkpoint_base_dir: str = "./checkpoints",
    wandb_enabled: bool = True,
    overwrite: bool = False,
    resume: bool = False,
    num_train_steps: int = 40_000,
) -> TrainConfig:
    if lora:
        model = pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            image_resolution=image_resolution,
            max_token_len=max_token_len,
            training_time_rtc=training_time_rtc,
            rtc_max_delay=rtc_max_delay,
        )
        freeze_filter = pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter()
        ema_decay = None
    else:
        model = pi0_config.Pi0Config(
            image_resolution=image_resolution,
            max_token_len=max_token_len,
            training_time_rtc=training_time_rtc,
            rtc_max_delay=rtc_max_delay,
        )
        freeze_filter = nnx.Nothing()
        ema_decay = 0.99

    return TrainConfig(
        name=name,
        exp_name=exp_name,
        model=model,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=2.5e-5,
            decay_steps=40_000,
            decay_lr=2.5e-6,
        ),
        log_interval=10,
        data=LeRobotAlohaDataConfig(
            repo_ids=repo_ids,
            transform_pipeline=_transforms.AlohaTransformPipeline(
                include_low=include_low,
                include_subtask=include_subtask,
                image_resolution=model.image_resolution,
                max_token_len=model.max_token_len,
                discrete_state_input=model.discrete_state_input,
                assets=assets,
                use_quantile_norm=True,
                video_memory_num_frames=video_memory_num_frames,
                video_memory_stride_seconds=video_memory_stride_seconds,
                adapt_to_pi=True,
                use_delta_joint_actions=True,
                action_dim=model.action_dim,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE_PARAMS),
        freeze_filter=freeze_filter,
        ema_decay=ema_decay,
        save_interval=1000,
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        num_workers=num_workers,
        fsdp_devices=fsdp_devices,
        gradient_accumulation_steps=gradient_accumulation_steps,
        checkpoint_base_dir=checkpoint_base_dir,
        wandb_enabled=wandb_enabled,
        overwrite=overwrite,
        resume=resume,
    )


_CONFIGS = [
    _make_twist_train_config(
        "eii_data_system_without_rinse_cam3_fullft_h200_41repo",
        repo_ids=_EII_DATA_SYSTEM_WITHOUT_RINSE_41_REPO_IDS,
        lora=False,
        batch_size=256,
        num_workers=128,
        fsdp_devices=8,
        include_low=False,
        include_subtask=False,
        gradient_accumulation_steps=1,
        exp_name="no_rinse_cam3_fullft_41repo_bs256_nw128_fsdp8_20260513",
        checkpoint_base_dir="/workspace/openpi0.5-rtc/checkpoints",
        wandb_enabled=True,
        overwrite=True,
        resume=False,
        # 2026-05-13 setup for the 41-repo production run. Matches the
        # 36-repo H200 full fine-tune shape: 3 cameras, no temporal memory,
        # full fine-tune, fsdp=8, bs=256, nw=128, 40k steps.
        assets=_transforms.AssetsConfig(
            assets_dir="assets/eii_data_system_without_rinse_cam3_fullft_h200_41repo",
            asset_id="trossen",
        ),
    ),
    _make_twist_train_config(
        "eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo",
        repo_ids=_EII_DATA_SYSTEM_WITHOUT_RINSE_RETURN_HOME_TURN_OVER_X5_FREE_SPIN_PLUS10_REPO_IDS,
        lora=False,
        batch_size=256,
        num_workers=64,
        fsdp_devices=4,
        include_low=False,
        include_subtask=False,
        gradient_accumulation_steps=1,
        exp_name="no_rinse_cam3_fullft_return_home_29repo_bs256_nw64_fsdp4_20260520",
        checkpoint_base_dir="/workspace/openpi0.5-rtc/checkpoints",
        wandb_enabled=True,
        overwrite=False,
        resume=True,
        # 2026-05-22 resume setup: keep the same checkpoint directory and
        # continue from the latest saved step. The three turn-over no-rinse
        # repos remain weighted to five total copies each, and free-spinning
        # merged-adjust-pickup has ten additional copies.
        num_train_steps=60_000,
        assets=_transforms.AssetsConfig(
            assets_dir="assets/eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo",
            asset_id="trossen",
        ),
    ),
    _make_twist_train_config(
        "eii_data_system_without_rinse_cam3_fullft_h200_have_cap_water_return_home_49repo",
        repo_ids=_EII_DATA_SYSTEM_WITHOUT_RINSE_HAVE_CAP_WATER_RETURN_HOME_49_REPO_IDS,
        lora=False,
        batch_size=256,
        num_workers=64,
        fsdp_devices=4,
        include_low=False,
        include_subtask=True,
        gradient_accumulation_steps=1,
        exp_name="no_rinse_cam3_fullft_have_cap_water_return_home_49repo_bs256_nw64_fsdp4_20260527",
        checkpoint_base_dir="/workspace/openpi0.5-rtc/checkpoints",
        wandb_enabled=True,
        overwrite=True,
        resume=False,
        num_train_steps=60_000,
        assets=_transforms.AssetsConfig(
            assets_dir="assets/eii_data_system_without_rinse_cam3_fullft_h200_have_cap_water_return_home_49repo",
            asset_id="aloha",
        ),
    ),
    _make_twist_train_config(
        "debug_training_time_rtc_lora",
        repo_ids=["lyl472324464/2026-04-23_direction_have_cap_water-lerobot-without-rinse"],
        lora=True,
        batch_size=1,
        num_workers=0,
        fsdp_devices=1,
        include_low=False,
        include_subtask=True,
        gradient_accumulation_steps=1,
        training_time_rtc=True,
        rtc_max_delay=10,
        video_memory_num_frames=5,
        video_memory_stride_seconds=1.0,
        exp_name="debug_training_time_rtc_lora_bs1",
        checkpoint_base_dir="/home/eii/openpi0.5-rtc/checkpoints",
        wandb_enabled=False,
        overwrite=True,
        resume=False,
        num_train_steps=10,
        assets=_transforms.AssetsConfig(
            assets_dir="assets/eii_data_system_without_rinse_cam3_fullft_h200_have_cap_water_return_home_49repo",
            asset_id="aloha",
        ),
    ),
    _make_twist_train_config(
        "debug_image_resolution_448_lora",
        repo_ids=["lyl472324464/2026-04-23_direction_have_cap_water-lerobot-without-rinse"],
        lora=True,
        batch_size=1,
        num_workers=0,
        fsdp_devices=1,
        include_low=False,
        include_subtask=True,
        gradient_accumulation_steps=1,
        image_resolution=(448, 448),
        exp_name="debug_image_resolution_448_lora_bs1",
        checkpoint_base_dir="/home/eii/openpi0.5-rtc/checkpoints",
        wandb_enabled=False,
        overwrite=True,
        resume=False,
        num_train_steps=10,
        assets=_transforms.AssetsConfig(
            assets_dir="assets/eii_data_system_without_rinse_cam3_fullft_h200_have_cap_water_return_home_49repo",
            asset_id="aloha",
        ),
    ),
    _make_twist_train_config(
        "eii_rinse_11repo_cam4_fullft",
        repo_ids=_EII_RINSE_11REPO_INSERT_X5_REPO_IDS,
        lora=False,
        batch_size=256,
        num_workers=128,
        include_low=True,
        include_subtask=True,
        gradient_accumulation_steps=1,
        assets=_transforms.AssetsConfig(
            assets_dir="assets/eii_rinse_11repo_cam4_fullft",
            asset_id="trossen",
        ),
    ),
    TrainConfig(
        name="debug",
        data=LeRobotAlohaDataConfig(
            repo_ids=["lyl472324464/2026-03-09-inference-with-and-without-cap"],
            transform_pipeline=_transforms.AlohaTransformPipeline(
                include_low=False,
                include_subtask=False,
                image_resolution=(224, 224),
                max_token_len=200,
                discrete_state_input=True,
                assets=_transforms.AssetsConfig(
                    assets_dir="assets/debug",
                    asset_id="trossen",
                ),
                use_quantile_norm=True,
                video_memory_num_frames=1,
                video_memory_stride_seconds=1.0,
                adapt_to_pi=True,
                use_delta_joint_actions=True,
                action_dim=32,
            ),
        ),
        batch_size=2,
        model=pi0_config.Pi0Config(
            paligemma_variant="dummy",
            action_expert_variant="dummy",
        ),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
]

_CONFIGS[0] = dataclasses.replace(_CONFIGS[0], policy_metadata=_TROSSEN_RESET_POSE)

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")

_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")
    return _CONFIGS_DICT[config_name]
