"""The new training recipe owns its objects instead of inheriting another recipe."""
import ast
from pathlib import Path

from openpi.training import config


def test_independent_pool256_recipe():
    name = 'eii_with_rinse_cam4_fullft_all_69repo_h100_448_mem5s_5hz_pool256'
    cfg = config.get_config(name)
    assert cfg.model.image_resolution == cfg.data.transform_pipeline.image_resolution == (448,448)
    assert cfg.model.image_tokens_per_image == 256
    assert cfg.data.transform_pipeline.video_memory_num_frames == 26
    assert cfg.data.transform_pipeline.video_memory_stride_seconds == 0.2
    assert cfg.batch_size * cfg.gradient_accumulation_steps == 256
    assert cfg.num_train_steps == 60000 and cfg.fsdp_devices == 4
    assert not cfg.resume and not cfg.overwrite
    assert cfg.weight_loader.params_path == 'gs://openpi-assets/checkpoints/pi05_base/params'
    assert len(cfg.data.repo_ids) == 83 and len(set(cfg.data.repo_ids)) == 69
    for other in config._CONFIGS:
        if other is cfg:
            continue
        assert other.model is not cfg.model
        assert other.data is not cfg.data
        assert other.data.repo_ids is not cfg.data.repo_ids
        assert other.data.transform_pipeline is not cfg.data.transform_pipeline
        assert other.lr_schedule is not cfg.lr_schedule
        assert other.optimizer is not cfg.optimizer
    tree = ast.parse(Path(config.__file__).read_text())
    recipes = [node for node in ast.walk(tree) if isinstance(node,ast.Call)
               and isinstance(node.func,ast.Name) and node.func.id == 'TrainConfig'
               and any(kw.arg=='name' and isinstance(kw.value,ast.Constant)
                       and kw.value.value==name for kw in node.keywords)]
    assert len(recipes) == 1
    assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                   and node.func.attr in {'replace','get_config'} for node in ast.walk(recipes[0]))
