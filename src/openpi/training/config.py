"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter

_TROSSEN_RESET_POSE = {"reset_pose": [0, -1.5, 1.5, 0, 0, 0]}
_PI05_BASE_ASSETS_DIR = "gs://openpi-assets/checkpoints/pi05_base/assets"
_PI05_BASE_PARAMS = "gs://openpi-assets/checkpoints/pi05_base/params"

_TWIST_AND_STATIC_REPO_IDS = [
    "lyl472324464/2026-03-09-inference-with-and-without-cap",
    "lyl472324464/2026-03-09-no-cap-inference",
    "lyl472324464/2026-03-05-two-direction",
    "lyl472324464/2026-03-04-one-direction",
    "lyl472324464/2026-02-03-no-cap-and-direction",
    "lyl472324464/2026-01-28-twist-many-bottle",
    "lyl472324464/2026-01-20-twist-one-bottle",
    "lyl472324464/2025-12-23-twist-one-bottle",
    "lyl472324464/2025-12-10-twist-one-bottle",
    "lyl472324464/2025-11-26-twist-two-bottles",
    "lyl472324464/2025-11-18-twist-two-bottles",
    "lyl472324464/2025-11-14-twist-two-bottles",
    "lyl472324464/2025-11-06-twist-many-bottles",
    "lyl472324464/2025-09-15-twist-one-bottle-no-box-in-the-front",
    "lyl472324464/aloha_static_battery",
    "lyl472324464/aloha_static_candy",
    "lyl472324464/aloha_static_coffee",
    "lyl472324464/aloha_static_coffee_new",
    "lyl472324464/aloha_static_cups_open",
    "lyl472324464/aloha_static_fork_pick_up",
    "lyl472324464/aloha_static_pingpong_test",
    "lyl472324464/aloha_static_pro_pencil",
    "lyl472324464/aloha_static_screw_driver",
    "lyl472324464/aloha_static_tape",
    "lyl472324464/aloha_static_thread_velcro",
    "lyl472324464/aloha_static_towel",
    "lyl472324464/aloha_static_vinh_cup",
    "lyl472324464/aloha_static_vinh_cup_left",
    "lyl472324464/aloha_static_ziploc_slide",
]

_TWIST_ONLY_REPO_IDS = [
    "lyl472324464/2026-03-09-inference-with-and-without-cap",
    "lyl472324464/2026-03-09-no-cap-inference",
    "lyl472324464/2026-03-05-two-direction",
    "lyl472324464/2026-03-04-one-direction",
    "lyl472324464/2026-02-03-no-cap-and-direction",
    "lyl472324464/2026-01-28-twist-many-bottle",
    "lyl472324464/2026-01-20-twist-one-bottle",
    "lyl472324464/2025-12-23-twist-one-bottle",
    "lyl472324464/2025-12-10-twist-one-bottle",
    "lyl472324464/2025-11-26-twist-two-bottles",
    "lyl472324464/2025-11-18-twist-two-bottles",
    "lyl472324464/2025-11-14-twist-two-bottles",
    "lyl472324464/2025-11-06-twist-many-bottles",
    "lyl472324464/2025-09-15-twist-one-bottle-no-box-in-the-front",
]

_TWIST_WATER_TEAR_REPO_IDS = [
    "lyl472324464/2026.04.10_twist_tear_water-direction_tabacco-top",
    "lyl472324464/2026.04.10_twist-and-tear-and-water-direction-and-tabacco-top",
    "lyl472324464/2026.04.13_twist_tear_water-direction_tabacco-top-2",
    "lyl472324464/2026-04-14_tear_twist_water-direction_tabacco-mid_lower",
    "lyl472324464/2026-04-14_twist_tear_water-direction_tabacco-mid_lower",
    "lyl472324464/2026-04-14_twist_tear_water-direction_tabacco-all"
]

# Hugging Face Hub dataset roots under `eii-data-system-prod/data/huggingface/hub` on 192.168.1.40
# (datasets--org--name), excluding any repo_id whose name contains "tear" (case-insensitive).
_EII_DATA_SYSTEM_HUB_NO_TEAR_REPO_IDS = [
    "lyl472324464/2025-09-15-twist-one-bottle-no-box-in-the-front",
    "lyl472324464/2025-11-06-twist-many-bottles",
    "lyl472324464/2025-11-14-twist-two-bottles",
    "lyl472324464/2025-11-18-twist-two-bottles",
    "lyl472324464/2025-11-26-twist-two-bottles",
    "lyl472324464/2025-12-10-twist-one-bottle",
    "lyl472324464/2025-12-23-twist-one-bottle",
    "lyl472324464/2026-01-20-twist-one-bottle",
    "lyl472324464/2026-01-28-twist-many-bottle",
    "lyl472324464/2026-02-03-no-cap-and-direction",
    "lyl472324464/2026-03-04-one-direction",
    "lyl472324464/2026-03-05-two-direction",
    # 2026-03-09 two repos ~28.3k frames total on Hub; repeat 4x each (~113k weighted) to target ~100k.
    "lyl472324464/2026-03-09-inference-with-and-without-cap",
    "lyl472324464/2026-03-09-no-cap-inference",
    "lyl472324464/2026-03-09-inference-with-and-without-cap",
    "lyl472324464/2026-03-09-no-cap-inference",
    "lyl472324464/2026-03-09-inference-with-and-without-cap",
    "lyl472324464/2026-03-09-no-cap-inference",
    "lyl472324464/2026-03-09-inference-with-and-without-cap",
    "lyl472324464/2026-03-09-no-cap-inference",
    "lyl472324464/2026-03-12-one-have-cap",
    "lyl472324464/2026-03-12-one-have-cap-direction",
    "lyl472324464/2026-03-12-one-havent-cap",
    "lyl472324464/2026-03-12-one-havent-cap-direction",
    "lyl472324464/2026-03-12-two-have-all-left",
    "lyl472324464/2026-03-12-two-have-cap-all-right",
    "lyl472324464/2026-03-12-two-have-cap-one-right",
    "lyl472324464/2026.03.16_twist_many",
    # 2026-04-21 (HF Hub ids use `-lerobot` suffix) — rotate-heavy; listed twice to up-weight in training.
    "lyl472324464/2026-04-21_direction-lerobot",
    "lyl472324464/2026-04-21_direction_2-lerobot",
    "lyl472324464/2026-04-21_direction_havent_cap-lerobot",
    "lyl472324464/2026-04-21_direction_havent_cap_water-lerobot",
    "lyl472324464/2026-04-21_direction-lerobot",
    "lyl472324464/2026-04-21_direction_2-lerobot",
    "lyl472324464/2026-04-21_direction_havent_cap-lerobot",
    "lyl472324464/2026-04-21_direction_havent_cap_water-lerobot",
]


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    assets_dir: str | None = None
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    repo_id: str | None = None
    repo_ids: list[str] | None = None
    asset_id: str | None = None
    norm_stats: dict[str, _transforms.NormStats] | None = None
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    use_quantile_norm: bool = False
    action_sequence_keys: Sequence[str] = ("actions",)
    prompt_from_task: bool = False
    video_memory_num_frames: int = 1
    video_memory_stride_seconds: float = 1.0
    rlds_data_dir: str | None = None
    action_space: Any | None = None
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a transform group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    default_prompt: str | None = None
    image_size: tuple[int, int] = (224, 224)

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        if model_config.model_type != _model.ModelType.PI05:
            raise NotImplementedError(f"Unsupported model type: {model_config.model_type}")
        assert isinstance(model_config, pi0_config.Pi0Config)
        image_height, image_width = self.image_size
        return _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(self.default_prompt),
                _transforms.ResizeImages(image_height, image_width),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input,
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    repo_id: str | None = None
    repo_ids: list[str] = tyro.MISSING
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        repo_ids = self.repo_ids if self.repo_ids is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            repo_ids=repo_ids,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        data_assets_dir = str(assets_dir / asset_id)
        try:
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info("Loaded norm stats from %s", data_assets_dir)
            return norm_stats
        except FileNotFoundError:
            logging.info("Norm stats not found in %s, skipping.", data_assets_dir)
            return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


def _aloha_lerobot_repack_transforms() -> _transforms.Group:
    return _transforms.Group(
        inputs=[
            _transforms.RepackTransform(
                {
                    "images": {"cam_high": "observation.images.top"},
                    "state": "observation.state",
                    "actions": "action",
                }
            )
        ]
    )


def _aloha_real_repack_transforms(*, include_low: bool, include_prompt: bool, include_subtask: bool) -> _transforms.Group:
    image_mapping = {
        "cam_high": "observation.images.cam_high",
        "cam_left_wrist": "observation.images.cam_left_wrist",
        "cam_right_wrist": "observation.images.cam_right_wrist",
    }
    if include_low:
        image_mapping["cam_low"] = "observation.images.cam_low"

    mapping: dict[str, Any] = {
        "images": image_mapping,
        "state": "observation.state",
        "actions": "action",
    }
    if include_subtask:
        mapping["subtask"] = "subtask"
    if include_prompt:
        mapping["prompt"] = "prompt"
    return _transforms.Group(inputs=[_transforms.RepackTransform(mapping)])


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    use_delta_joint_actions: bool = True
    default_prompt: str | None = None
    image_size: tuple[int, int] = (224, 224)
    adapt_to_pi: bool = True
    video_memory_num_frames: int = 1
    video_memory_stride_seconds: float = 1.0
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default_factory=_aloha_lerobot_repack_transforms
    )
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory(
                default_prompt=self.default_prompt,
                image_size=self.image_size,
            )(model_config),
            action_sequence_keys=self.action_sequence_keys,
            video_memory_num_frames=self.video_memory_num_frames,
            video_memory_stride_seconds=self.video_memory_stride_seconds,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: tyro.conf.Suppress[str]
    project_name: str = "openpi"
    exp_name: str = tyro.MISSING
    model: _model.BaseModelConfig = dataclasses.field(
        default_factory=lambda: pi0_config.Pi0Config(pi05=True)
    )
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)
    pytorch_weight_path: str | None = None
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)
    assets_base_dir: str = "./assets"
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
    def assets_dirs(self) -> pathlib.Path:
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1.")
        image_size = getattr(self.data, "image_size", None)
        image_resolution = getattr(self.model, "image_resolution", None)
        if image_size is not None and image_resolution != image_size:
            object.__setattr__(self, "model", dataclasses.replace(self.model, image_resolution=image_size))


def _pi05_base_assets() -> AssetsConfig:
    return AssetsConfig(assets_dir=_PI05_BASE_ASSETS_DIR, asset_id="trossen")


def _make_twist_train_config(
    name: str,
    *,
    repo_ids: list[str],
    lora: bool,
    batch_size: int,
    num_workers: int,
    include_low: bool = True,
    include_subtask: bool = True,
    gradient_accumulation_steps: int = 1,
    assets: AssetsConfig | None = None,
) -> TrainConfig:
    if lora:
        model = pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )
        freeze_filter = pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter()
        ema_decay = None
    else:
        model = pi0_config.Pi0Config(pi05=True)
        freeze_filter = nnx.Nothing()
        ema_decay = 0.99

    return TrainConfig(
        name=name,
        model=model,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=2.5e-5,
            decay_steps=40_000,
            decay_lr=2.5e-6,
        ),
        log_interval=10,
        data=LeRobotAlohaDataConfig(
            adapt_to_pi=True,
            image_size=(224, 224),
            video_memory_num_frames=1,
            video_memory_stride_seconds=1.0,
            repo_ids=repo_ids,
            assets=assets if assets is not None else _pi05_base_assets(),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_aloha_real_repack_transforms(
                include_low=include_low,
                include_prompt=True,
                include_subtask=include_subtask,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE_PARAMS),
        freeze_filter=freeze_filter,
        ema_decay=ema_decay,
        save_interval=1000,
        num_train_steps=40_000,
        batch_size=batch_size,
        num_workers=num_workers,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )


_CONFIGS = [
    _make_twist_train_config(
        "twist_off_the_bottle_cap",
        repo_ids=_TWIST_AND_STATIC_REPO_IDS,
        lora=False,
        batch_size=256,
        num_workers=16,
    ),
    _make_twist_train_config(
        "twist_off_the_bottle_cap_lora",
        repo_ids=_TWIST_ONLY_REPO_IDS,
        lora=True,
        batch_size=32,
        num_workers=4,
    ),
    _make_twist_train_config(
        "twist_water_tear_promptfix_lora",
        repo_ids=_TWIST_WATER_TEAR_REPO_IDS,
        lora=True,
        batch_size=64,
        num_workers=4,
    ),
    _make_twist_train_config(
        "eii_data_system_no_tear_cam3_lora",
        repo_ids=_EII_DATA_SYSTEM_HUB_NO_TEAR_REPO_IDS,
        lora=True,
        # micro-batch 64 x 2 accum = 128 effective (scripts/train.py). H100 80GB + LoRA + 3x224 cams: 64 is used elsewhere (tear_lora); OOM则改 32x4。
        batch_size=64,
        num_workers=4,
        include_low=False,
        include_subtask=False,
        gradient_accumulation_steps=2,
        # Load norm stats from ./assets/<config.name>/trossen (compute_norm_stats), not gs:// pi05_base.
        assets=AssetsConfig(assets_dir=None, asset_id="trossen"),
    ),
    TrainConfig(
        name="pi05_aloha_sim",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_PI05_BASE_PARAMS),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(
            pi05=True,
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
