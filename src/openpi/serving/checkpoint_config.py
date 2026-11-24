"""Load and validate the input contract exported with inference weights."""
import dataclasses
import hashlib
import json
import logging
from pathlib import Path
import runpy

from openpi.data import transforms
from openpi.training import config as configs
from openpi.shared import download


def checkpoint_config(checkpoint_dir, config_name=None, *, verify=True):
    root = download.maybe_download(str(checkpoint_dir))
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    settings_path = root / 'export_metadata/training-settings.json'
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    names = [value for value in (manifest.get('config'), settings.get('config'), config_name) if value]
    if len(set(names)) > 1:
        raise ValueError(f'Conflicting checkpoint config names: {names}')
    name = names[0] if names else root.parent.parent.name
    for step in (manifest.get('step'), settings.get('checkpoint_step')):
        if step is not None and str(step) != root.name:
            raise ValueError(f'Checkpoint step {root.name} does not match metadata {step}')
    entries = manifest.get('files', [])
    if entries and not isinstance(entries, list):
        raise ValueError('manifest.files must contain per-file records')
    for entry in entries:
        file = root / entry['path']
        if not file.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'Manifest path escapes checkpoint: {file}')
        if not file.is_file() or file.stat().st_size != entry['size']:
            raise ValueError(f'Checkpoint file missing or wrong size: {file}')
        if verify and entry.get('sha256'):
            digest = hashlib.sha256()
            with file.open('rb') as stream:
                for block in iter(lambda: stream.read(16*1024*1024), b''):
                    digest.update(block)
            if digest.hexdigest() != entry['sha256']:
                raise ValueError(f'Checkpoint SHA256 mismatch: {file}')
    snapshot = root / 'export_metadata/config.py'
    if snapshot.exists():
        config = runpy.run_path(str(snapshot))['get_config'](name)
        source = str(snapshot)
    else:
        config = configs.get_config(name)
        source = 'registered training config (checkpoint has no exported config.py)'
    pipeline = config.data.transform_pipeline
    if pipeline is None:
        raise ValueError('Checkpoint needs an ALOHA transform pipeline')
    pipeline = dataclasses.replace(pipeline, assets=dataclasses.replace(pipeline.assets, assets_dir=str(root/'assets')))
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, transform_pipeline=pipeline))
    if tuple(config.model.image_keys) != tuple(pipeline.raw_image_keys):
        raise ValueError('Model cameras differ from training input pipeline')
    if tuple(config.model.image_resolution) != tuple(pipeline.image_resolution):
        raise ValueError('Model resolution differs from training input pipeline')
    if config.model.action_dim != pipeline.action_dim:
        raise ValueError('Model action dimension differs from training input pipeline')
    pipeline._require_norm_stats()
    for relative in ('params/_METADATA','params/manifest.ocdbt'):
        if not (root/relative).is_file():
            raise ValueError(f'Missing required checkpoint file: {relative}')
    logging.info('Checkpoint config=%s step=%s source=%s files=%d sha256=%s',name,root.name,source,len(entries),verify)
    return root, config


def runtime_metadata(config, root):
    pipeline = config.data.transform_pipeline
    return {
        'adapt_to_pi': bool(pipeline.adapt_to_pi),
        'long_memory': dataclasses.asdict(pipeline.long_memory) if getattr(pipeline, "long_memory", None) is not None else None,
        'task_catalog_version': (getattr(config, 'policy_metadata', None) or {}).get('conditioning', ''),
        'action_horizon': config.model.action_horizon,
        'action_dim': config.model.action_dim,
        'joint_position_history': dataclasses.asdict(pipeline.joint_position_history) if pipeline.joint_position_history is not None else None,
        'video_input': {
            'config_name': config.name, 'checkpoint_step': root.name,
            'camera_keys': list(pipeline.raw_image_keys),
            'image_resolution': list(pipeline.image_resolution),
            'num_frames': pipeline.video_memory_num_frames,
            'stride_seconds': pipeline.video_memory_stride_seconds,
            'mosaic': dataclasses.asdict(pipeline.high_camera_mosaic) if pipeline.high_camera_mosaic is not None else None,
        },
    }
