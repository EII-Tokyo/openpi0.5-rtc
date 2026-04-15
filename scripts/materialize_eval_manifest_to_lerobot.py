#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import time
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDataset as SrcLeRobotDataset


def build_features(image_size: int):
    image_feature = {
        "dtype": "image",
        "shape": (image_size, image_size, 3),
        "names": ["height", "width", "channel"],
    }
    return {
        "observation.images.cam_high": image_feature,
        "observation.images.cam_low": image_feature,
        "observation.images.cam_left_wrist": image_feature,
        "observation.images.cam_right_wrist": image_feature,
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": ["action"],
        },
        "subtask": {"dtype": "string", "shape": (1,), "names": None},
        "bottle_state": {"dtype": "string", "shape": (1,), "names": None},
        "class_id": {"dtype": "int64", "shape": (1,), "names": None},
        "source_repo_id": {"dtype": "string", "shape": (1,), "names": None},
        "source_episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "source_frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    }


def np_img(x):
    arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def resize_img(arr, size=(224,224)):
    return np.asarray(Image.fromarray(arr).resize(size, resample=Image.BICUBIC), dtype=np.uint8)


def _repo_cache_root(repo_id: str) -> Path:
    owner, name = repo_id.split('/', 1)
    return Path.home()/'.cache'/'huggingface'/'lerobot'/'hub'/f'datasets--{owner}--{name}'


def _open_source_dataset(repo_id: str) -> SrcLeRobotDataset:
    return SrcLeRobotDataset(repo_id, revision='main', force_cache_sync=False, download_videos=True, delta_timestamps=None)


def _refresh_source_dataset(repo_id: str) -> SrcLeRobotDataset:
    cache_root = _repo_cache_root(repo_id)
    shutil.rmtree(cache_root, ignore_errors=True)
    gc.collect()
    return _open_source_dataset(repo_id)


def _get_source_row(ds: SrcLeRobotDataset, repo_id: str, frame_index: int):
    tries = 0
    while True:
        try:
            return ds[int(frame_index)], ds
        except FileNotFoundError:
            if tries >= 2:
                raise
            print(f'[retry] missing source file for {repo_id}, refreshing cache', flush=True)
            ds = _refresh_source_dataset(repo_id)
            tries += 1
        except OSError as e:
            if getattr(e, 'errno', None) == 24:
                if tries >= 4:
                    raise
                print(f'[retry] too many open files for {repo_id}, reopening dataset', flush=True)
                del ds
                gc.collect()
                time.sleep(0.5)
                ds = _open_source_dataset(repo_id)
                tries += 1
                continue
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--repo-id', required=True)
    ap.add_argument('--root', type=Path, default=None)
    ap.add_argument('--fps', type=int, default=50)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--data-files-size-in-mb', type=int, default=300)
    ap.add_argument('--force-override', action='store_true')
    args = ap.parse_args()

    obj = json.loads(args.manifest.read_text())
    rows = obj['rows']
    rows.sort(key=lambda r: (r['repo_id'], int(r['episode_index']), int(r['frame_index'])))
    root = args.root or (Path.home()/'.cache'/'huggingface'/'lerobot'/args.repo_id)
    if root.exists() and args.force_override:
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)

    out = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=root,
        robot_type='aloha',
        features=build_features(args.image_size),
        use_videos=False,
        image_writer_threads=1,
        image_writer_processes=1,
    )
    if hasattr(out, 'meta') and hasattr(out.meta, 'update_chunk_settings'):
        out.meta.update_chunk_settings(data_files_size_in_mb=args.data_files_size_in_mb)

    active_repo_id = None
    active_ds = None
    current_episode = None
    frames_in_episode = 0
    for i, row in enumerate(rows):
        repo_id = row['repo_id']
        if repo_id != active_repo_id:
            if active_ds is not None:
                del active_ds
                gc.collect()
            active_ds = _open_source_dataset(repo_id)
            active_repo_id = repo_id
        ds = active_ds
        sample, active_ds = _get_source_row(ds, repo_id, int(row['frame_index']))
        episode_key = (repo_id, int(row['episode_index']))
        if current_episode is None:
            current_episode = episode_key
        elif episode_key != current_episode:
            out.save_episode()
            current_episode = episode_key
            frames_in_episode = 0
        frame = {
            'observation.images.cam_high': resize_img(np_img(sample['observation.images.cam_high']), size=(args.image_size, args.image_size)),
            'observation.images.cam_low': resize_img(np_img(sample['observation.images.cam_low']), size=(args.image_size, args.image_size)),
            'observation.images.cam_left_wrist': resize_img(np_img(sample['observation.images.cam_left_wrist']), size=(args.image_size, args.image_size)),
            'observation.images.cam_right_wrist': resize_img(np_img(sample['observation.images.cam_right_wrist']), size=(args.image_size, args.image_size)),
            'observation.state': np.asarray(sample['observation.state'], dtype=np.float32),
            'action': np.asarray(sample['action'], dtype=np.float32),
            'task': str(sample.get('task', 'Evaluate subtask')),
            'subtask': str(row['subtask']),
            'bottle_state': str(row['bottle_state']),
            'class_id': np.asarray([int(row['class_id'])], dtype=np.int64),
            'source_repo_id': str(repo_id),
            'source_episode_index': np.asarray([int(row['episode_index'])], dtype=np.int64),
            'source_frame_index': np.asarray([int(row['frame_index'])], dtype=np.int64),
        }
        out.add_frame(frame)
        frames_in_episode += 1
        if (i + 1) % 200 == 0:
            print(f'processed {i+1}/{len(rows)}', flush=True)
    if current_episode is not None and frames_in_episode:
        out.save_episode()
    out.finalize()
    shutil.rmtree(root / 'images', ignore_errors=True)
    print(f'done {args.repo_id} rows={len(rows)} root={root}', flush=True)

if __name__ == '__main__':
    main()
