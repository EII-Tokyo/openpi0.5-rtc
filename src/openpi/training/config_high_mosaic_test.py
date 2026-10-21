import ast
from pathlib import Path

from openpi.training import config

NAME = 'eii_with_rinse_cam4_fullft_all_69repo_h100_448_high_mem5s4hz_mosaic2x2_pool256'


def test_independent_new_mosaic_recipe():
    cfg = config.get_config(NAME)
    pipeline = cfg.data.transform_pipeline
    assert cfg.model.image_resolution == pipeline.image_resolution == (448,448)
    assert cfg.model.image_tokens_per_image == 256
    assert pipeline.video_memory_num_frames == 6 and pipeline.video_memory_stride_seconds == 1
    assert pipeline.high_camera_mosaic.history_seconds == 5
    assert pipeline.high_camera_mosaic.sample_hz == 4 and pipeline.high_camera_mosaic.tile_size == 224
    assert cfg.batch_size == 64 and cfg.gradient_accumulation_steps == 4
    assert cfg.fsdp_devices == 4 and cfg.num_train_steps == 60000 and cfg.ema_decay == .99
    assert not cfg.resume and not cfg.overwrite
    assert cfg.weight_loader.params_path == 'gs://openpi-assets/checkpoints/pi05_base/params'
    assert len(cfg.data.repo_ids) == 83 and len(set(cfg.data.repo_ids)) == 69
    assert pipeline.assets.assets_dir == 'assets/' + NAME
    for other in config._CONFIGS:
        if other is cfg:
            continue
        assert other.data.transform_pipeline.high_camera_mosaic is None
        for field in ('model','data','optimizer','lr_schedule'):
            assert getattr(other,field) is not getattr(cfg,field)
        assert other.data.repo_ids is not cfg.data.repo_ids
    tree = ast.parse(Path(config.__file__).read_text())
    recipes = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == 'TrainConfig' and any(k.arg=='name' and isinstance(k.value,ast.Constant)
               and k.value.value==NAME for k in n.keywords)]
    assert len(recipes)==1
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                   and n.func.attr in {'replace','get_config'} for n in ast.walk(recipes[0]))
