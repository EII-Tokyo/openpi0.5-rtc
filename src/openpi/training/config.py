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
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # LeRobot repo ids. If None, fake data will be created.
    repo_ids: list[str] | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False
    # Number of temporally spaced frames to stack for each camera observation.
    video_memory_num_frames: int = 1
    # Temporal stride between stacked frames, in seconds.
    video_memory_stride_seconds: float = 1.0

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset (RLDS path only; unused when no RLDS configs).
    action_space: Any | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None
    # If provided, always overwrite any existing prompt before tokenization.
    force_prompt: str | None = None
    image_size: tuple[int, int] = (224, 224)
    include_bottle_description: bool = True
    include_bottle_position: bool = True
    include_bottle_state: bool = True
    include_subtask: bool = True

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        image_height, image_width = self.image_size
        match model_config.model_type:
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                tokenizer = _tokenizer.PaligemmaTokenizer(
                    model_config.max_token_len,
                    subtask_max_len=model_config.subtask_max_token_len,
                    fast_tokenizer_path=model_config.fast_tokenizer_path,
                )
                discrete_state_input = model_config.discrete_state_input
                transforms = [
                    _transforms.ForcePrompt(self.force_prompt),
                    _transforms.InjectDefaultPrompt(self.default_prompt),
                    _transforms.ResizeImages(image_height, image_width),
                    _transforms.FilterSubtaskPayload(
                        include_bottle_description=self.include_bottle_description,
                        include_bottle_position=self.include_bottle_position,
                        include_bottle_state=self.include_bottle_state,
                        include_subtask=self.include_subtask,
                    ),
                ]
                transforms += [
                    _transforms.TokenizePrompt(
                        tokenizer,
                        discrete_state_input=discrete_state_input,
                    ),
                ]
                transforms += [_transforms.PadStatesAndActions(model_config.action_dim)]
                return _transforms.Group(inputs=transforms)
            case _:
                raise NotImplementedError(f"Unsupported model_type for training: {model_config.model_type}")


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str | None = None
    repo_ids: list[str] = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
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
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If provided, always overwrite prompt before tokenization.
    force_prompt: str | None = None
    # Training/inference image resize target (height, width).
    image_size: tuple[int, int] = (224, 224)
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True
    force_cache_sync: bool = False
    video_memory_num_frames: int = 1
    video_memory_stride_seconds: float = 1.0
    include_bottle_description: bool = True
    include_bottle_position: bool = True
    include_bottle_state: bool = True
    include_subtask: bool = True
    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default_factory=lambda: _transforms.Group(
            inputs=[
                _transforms.InjectDefaultField("train_action", True),
                _transforms.InjectDefaultField("cam_high_mask", 1),
                _transforms.InjectDefaultField("cam_low_mask", 1),
                _transforms.InjectDefaultField("cam_left_wrist_mask", 1),
                _transforms.InjectDefaultField("cam_right_wrist_mask", 1),
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "image_masks": {
                            "cam_high": "cam_high_mask",
                            "cam_low": "cam_low_mask",
                            "cam_left_wrist": "cam_left_wrist_mask",
                            "cam_right_wrist": "cam_right_wrist_mask",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "actions_mask": "train_action",
                        "subtask": "subtask",
                    }
                ),
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if self.repo_id == "fake":
            model_transforms = ModelTransformFactory(
                default_prompt=self.default_prompt,
                force_prompt=self.force_prompt,
                image_size=self.image_size,
                include_bottle_description=self.include_bottle_description,
                include_bottle_position=self.include_bottle_position,
                include_bottle_state=self.include_bottle_state,
                include_subtask=self.include_subtask,
            )(model_config)
            return DataConfig(
                repo_id="fake",
                model_transforms=model_transforms,
                use_quantile_norm=model_config.model_type != ModelType.PI0,
            )
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

        model_transforms = ModelTransformFactory(
            default_prompt=self.default_prompt,
            force_prompt=self.force_prompt,
            image_size=self.image_size,
            include_bottle_description=self.include_bottle_description,
            include_bottle_position=self.include_bottle_position,
            include_bottle_state=self.include_bottle_state,
            include_subtask=self.include_subtask,
        )(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            video_memory_num_frames=self.video_memory_num_frames,
            video_memory_stride_seconds=self.video_memory_stride_seconds,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(
        default_factory=lambda: LeRobotAlohaDataConfig(repo_id="fake", repo_ids=[])
    )

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    # Subtask (bottle_state, subtask) holdout + periodic val accuracy (JAX LeRobot training only).
    subtask_eval_enabled: bool = False
    subtask_eval_holdout_fraction: float = 0.1
    subtask_eval_interval_steps: int = 1000
    subtask_eval_batch_size: int = 8
    subtask_eval_max_samples_per_class: int | None = None
    """Cap val frames per class each eval pass; None = use full val split."""
    subtask_eval_canonical_pairs: tuple[tuple[str, str], ...] | None = None
    """If None, use `DEFAULT_STATE_SUBTASK_PAIRS` from openpi_client (nine classes)."""

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")
        if not (0.0 < self.subtask_eval_holdout_fraction < 1.0):
            raise ValueError("subtask_eval_holdout_fraction must be in (0, 1).")
        image_size = getattr(self.data, "image_size", None)
        image_resolution = getattr(self.model, "image_resolution", None)
        if image_size is not None and image_resolution != image_size:
            object.__setattr__(self, "model", dataclasses.replace(self.model, image_resolution=image_size))


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    TrainConfig(
        name="twist_and_static_mixture_full_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            max_token_len=96,
            subtask_loss_weight=0.1,
            subtask_max_token_len=64,
            fast_tokenizer_path="physical-intelligence/fast",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=2.5e-5,
            decay_steps=40_000,
            decay_lr=2.5e-6,
        ),
        log_interval=10,
        data=LeRobotAlohaDataConfig(
            image_size=(224, 224),
            include_bottle_description=True,
            include_bottle_position=False,
            include_bottle_state=True,
            include_subtask=True,
            video_memory_num_frames=1,
            video_memory_stride_seconds=1.0,
            repo_ids=[
                "lyl472324464/2025-11-18-twist-two-bottles",
                "lyl472324464/2025-11-26-twist-two-bottles",
                "lyl472324464/2025-12-10-twist-one-bottle",
                "lyl472324464/2025-12-23-twist-one-bottle",
                "lyl472324464/2026-01-20-twist-one-bottle",
                "lyl472324464/2026-01-28-twist-many-bottle",
                "lyl472324464/2026-02-03-no-cap-and-direction",
                "lyl472324464/2026-03-04-one-direction",
                "lyl472324464/2026-03-05-two-direction",
                "lyl472324464/2026-03-09-no-cap-inference",
                "lyl472324464/2026-03-09-inference-with-and-without-cap",
                "lyl472324464/2026-03-12-one-have-cap",
                "lyl472324464/2026-03-12-one-have-cap-direction",
                "lyl472324464/2026-03-12-one-havent-cap",
                "lyl472324464/2026-03-12-one-havent-cap-direction",
                "lyl472324464/2026-03-12-two-have-all-left",
                "lyl472324464/2026-03-12-two-have-cap-all-right",
                "lyl472324464/2026-03-12-two-have-cap-one-right",
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
            ],
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.InjectDefaultField("train_action", True),
                    _transforms.InjectDefaultField("cam_high_mask", 1),
                    _transforms.InjectDefaultField("cam_low_mask", 1),
                    _transforms.InjectDefaultField("cam_left_wrist_mask", 1),
                    _transforms.InjectDefaultField("cam_right_wrist_mask", 1),
                    _transforms.InjectDefaultField("subtask", None),
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_low": "observation.images.cam_low",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "image_masks": {
                                "cam_high": "cam_high_mask",
                                "cam_low": "cam_low_mask",
                                "cam_left_wrist": "cam_left_wrist_mask",
                                "cam_right_wrist": "cam_right_wrist_mask",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "actions_mask": "train_action",
                            "prompt": "prompt",
                            "subtask": "subtask",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        save_interval=1000,
        num_train_steps=40_000,
        batch_size=128,
        num_workers=4,
        subtask_eval_enabled=True,
        subtask_eval_interval_steps=1000,
        # Cap per-class val frames so periodic eval finishes in bounded time (full split is ~75k disk reads).
        subtask_eval_max_samples_per_class=128,
    ),
    TrainConfig(
        name="twist_and_static_mixture_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            max_token_len=96,
            subtask_loss_weight=0.1,
            subtask_max_token_len=64,
            fast_tokenizer_path="physical-intelligence/fast",
            image_resolution=(224, 224),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=2.5e-5,
            decay_steps=40_000,
            decay_lr=2.5e-6,
        ),
        log_interval=10,
        data=LeRobotAlohaDataConfig(
            image_size=(224, 224),
            include_bottle_description=True,
            include_bottle_position=False,
            include_bottle_state=True,
            include_subtask=True,
            video_memory_num_frames=1,
            video_memory_stride_seconds=1.0,
            repo_ids=[
                "lyl472324464/2025-11-18-twist-two-bottles",
                "lyl472324464/2025-11-26-twist-two-bottles",
                "lyl472324464/2025-12-10-twist-one-bottle",
                "lyl472324464/2025-12-23-twist-one-bottle",
                "lyl472324464/2026-01-20-twist-one-bottle",
                "lyl472324464/2026-01-28-twist-many-bottle",
                "lyl472324464/2026-02-03-no-cap-and-direction",
                "lyl472324464/2026-03-04-one-direction",
                "lyl472324464/2026-03-05-two-direction",
                "lyl472324464/2026-03-09-no-cap-inference",
                "lyl472324464/2026-03-09-inference-with-and-without-cap",
                "lyl472324464/2026-03-12-one-have-cap",
                "lyl472324464/2026-03-12-one-have-cap-direction",
                "lyl472324464/2026-03-12-one-havent-cap",
                "lyl472324464/2026-03-12-one-havent-cap-direction",
                "lyl472324464/2026-03-12-two-have-all-left",
                "lyl472324464/2026-03-12-two-have-cap-all-right",
                "lyl472324464/2026-03-12-two-have-cap-one-right",
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
            ],
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.InjectDefaultField("train_action", True),
                    _transforms.InjectDefaultField("cam_high_mask", 1),
                    _transforms.InjectDefaultField("cam_low_mask", 1),
                    _transforms.InjectDefaultField("cam_left_wrist_mask", 1),
                    _transforms.InjectDefaultField("cam_right_wrist_mask", 1),
                    _transforms.InjectDefaultField("subtask", None),
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_low": "observation.images.cam_low",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "image_masks": {
                                "cam_high": "cam_high_mask",
                                "cam_low": "cam_low_mask",
                                "cam_left_wrist": "cam_left_wrist_mask",
                                "cam_right_wrist": "cam_right_wrist_mask",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "actions_mask": "train_action",
                            "prompt": "prompt",
                            "subtask": "subtask",
                        }
                    )
                ]
            ),
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            subtask_loss_weight=1.0,
            subtask_max_token_len=64,
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        save_interval=1000,
        num_train_steps=40_000,
        batch_size=128,
        num_workers=4,
        subtask_eval_enabled=True,
        subtask_eval_interval_steps=1000,
        subtask_eval_max_samples_per_class=128,
    ),
    TrainConfig(
        name="twist_only_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            max_token_len=96,
            subtask_loss_weight=0.1,
            subtask_max_token_len=64,
            fast_tokenizer_path="physical-intelligence/fast",
            image_resolution=(224, 224),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=2.5e-5,
            decay_steps=40_000,
            decay_lr=2.5e-6,
        ),
        log_interval=10,
        data=LeRobotAlohaDataConfig(
            image_size=(224, 224),
            include_bottle_description=True,
            include_bottle_position=False,
            include_bottle_state=True,
            include_subtask=True,
            video_memory_num_frames=1,
            video_memory_stride_seconds=1.0,
            repo_ids=[
                "lyl472324464/2025-11-18-twist-two-bottles",
                "lyl472324464/2025-11-26-twist-two-bottles",
                "lyl472324464/2025-12-10-twist-one-bottle",
                "lyl472324464/2025-12-23-twist-one-bottle",
                "lyl472324464/2026-01-20-twist-one-bottle",
                "lyl472324464/2026-01-28-twist-many-bottle",
                "lyl472324464/2026-02-03-no-cap-and-direction",
                "lyl472324464/2026-03-04-one-direction",
                "lyl472324464/2026-03-05-two-direction",
                "lyl472324464/2026-03-09-no-cap-inference",
                "lyl472324464/2026-03-09-inference-with-and-without-cap",
                "lyl472324464/2026-03-12-one-have-cap",
                "lyl472324464/2026-03-12-one-have-cap-direction",
                "lyl472324464/2026-03-12-one-havent-cap",
                "lyl472324464/2026-03-12-one-havent-cap-direction",
                "lyl472324464/2026-03-12-two-have-all-left",
                "lyl472324464/2026-03-12-two-have-cap-all-right",
                "lyl472324464/2026-03-12-two-have-cap-one-right",
            ],
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.InjectDefaultField("train_action", True),
                    _transforms.InjectDefaultField("cam_high_mask", 1),
                    _transforms.InjectDefaultField("cam_low_mask", 1),
                    _transforms.InjectDefaultField("cam_left_wrist_mask", 1),
                    _transforms.InjectDefaultField("cam_right_wrist_mask", 1),
                    _transforms.InjectDefaultField("subtask", None),
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_low": "observation.images.cam_low",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "image_masks": {
                                "cam_high": "cam_high_mask",
                                "cam_low": "cam_low_mask",
                                "cam_left_wrist": "cam_left_wrist_mask",
                                "cam_right_wrist": "cam_right_wrist_mask",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "actions_mask": "train_action",
                            "prompt": "prompt",
                            "subtask": "subtask",
                        }
                    )
                ]
            ),
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            subtask_loss_weight=1.0,
            subtask_max_token_len=64,
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        save_interval=1000,
        num_train_steps=40_000,
        batch_size=8,
        num_workers=4,
        subtask_eval_enabled=True,
        subtask_eval_interval_steps=1000,
        subtask_eval_max_samples_per_class=128,
    ),
    TrainConfig(
        name="twist_only_lora_triplet_10k",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            max_token_len=96,
            subtask_loss_weight=0.1,
            subtask_max_token_len=512,
            fast_tokenizer_path="physical-intelligence/fast",
            image_resolution=(448, 448),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=2.5e-5,
            decay_steps=40_000,
            decay_lr=2.5e-6,
        ),
        log_interval=10,
        data=LeRobotAlohaDataConfig(
            image_size=(448, 448),
            include_bottle_description=True,
            include_bottle_position=False,
            include_bottle_state=True,
            include_subtask=True,
            video_memory_num_frames=1,
            video_memory_stride_seconds=1.0,
            repo_ids=[
                "lyl472324464/openai_logs_gpt54_process_all_bottles_1054_224",
                "lyl472324464/vqav2_masked_10k_224",
                "lyl472324464/twist_subset_10k_224_from_2025_12_10",
            ],
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.InjectDefaultField("train_action", True),
                    _transforms.InjectDefaultField("cam_high_mask", 1),
                    _transforms.InjectDefaultField("cam_low_mask", 1),
                    _transforms.InjectDefaultField("cam_left_wrist_mask", 1),
                    _transforms.InjectDefaultField("cam_right_wrist_mask", 1),
                    _transforms.InjectDefaultField("subtask", None),
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_low": "observation.images.cam_low",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "image_masks": {
                                "cam_high": "cam_high_mask",
                                "cam_low": "cam_low_mask",
                                "cam_left_wrist": "cam_left_wrist_mask",
                                "cam_right_wrist": "cam_right_wrist_mask",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "actions_mask": "train_action",
                            "prompt": "prompt",
                            "subtask": "subtask",
                        }
                    ),
                ]
            ),
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            subtask_loss_weight=1.0,
            subtask_max_token_len=64,
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        save_interval=1000,
        num_train_steps=40_000,
        batch_size=2,
        num_workers=4,
        subtask_eval_enabled=True,
        subtask_eval_interval_steps=1000,
        subtask_eval_max_samples_per_class=128,
    ),
    TrainConfig(
        name="twist_only_lora_triplet_100k",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            max_token_len=96,
            subtask_loss_weight=0.1,
            subtask_max_token_len=512,
            fast_tokenizer_path="physical-intelligence/fast",
            image_resolution=(448, 448),
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=2.5e-5,
            decay_steps=40_000,
            decay_lr=2.5e-6,
        ),
        log_interval=10,
        data=LeRobotAlohaDataConfig(
            image_size=(448, 448),
            include_bottle_description=True,
            include_bottle_position=False,
            include_bottle_state=True,
            include_subtask=True,
            video_memory_num_frames=1,
            video_memory_stride_seconds=1.0,
            repo_ids=[
                "lyl472324464/openai_logs_gpt54_process_all_bottles_1054_224",
                "lyl472324464/vqav2_100k_224",
                "lyl472324464/twist_subset_100k_224_from_2025_12_10",
            ],
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            base_config=DataConfig(prompt_from_task=True),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.InjectDefaultField("train_action", True),
                    _transforms.InjectDefaultField("cam_high_mask", 1),
                    _transforms.InjectDefaultField("cam_low_mask", 1),
                    _transforms.InjectDefaultField("cam_left_wrist_mask", 1),
                    _transforms.InjectDefaultField("cam_right_wrist_mask", 1),
                    _transforms.InjectDefaultField("subtask", None),
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_low": "observation.images.cam_low",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "image_masks": {
                                "cam_high": "cam_high_mask",
                                "cam_low": "cam_low_mask",
                                "cam_left_wrist": "cam_left_wrist_mask",
                                "cam_right_wrist": "cam_right_wrist_mask",
                            },
                            "state": "observation.state",
                            "actions": "action",
                            "actions_mask": "train_action",
                            "prompt": "prompt",
                            "subtask": "subtask",
                        }
                    ),
                ]
            ),
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            subtask_loss_weight=1.0,
            subtask_max_token_len=64,
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        save_interval=5000,
        keep_period=None,
        num_train_steps=40_000,
        batch_size=2,
        num_workers=4,
        subtask_eval_enabled=True,
        subtask_eval_interval_steps=1000,
        subtask_eval_max_samples_per_class=128,
    ),
]

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
