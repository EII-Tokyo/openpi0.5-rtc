#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重构版 ALOHA 可视化与视频导出脚本（颜色行为与官方 visualize_episodes.py 保持一致）

- 支持：
  - 单个 episode 可视化
  - 批量导出所有 episode 的视频
  - tiled（横向拼接多相机）/ separate（每个相机一个视频）
  - compressed / 未压缩图片
- 颜色逻辑（关键）：
  - 不管图片来源 / 是否压缩，
    在写入视频前统一做：
        image = image[:, :, [2, 1, 0]]
    这与 ALOHA 官方 visualize_episodes.py 完全一致。
"""

import argparse
import os
import re
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np

from aloha.robot_utils import (
    JOINT_NAMES,
    load_yaml_file,
)

STATE_NAMES = JOINT_NAMES + ['gripper']
BASE_STATE_NAMES = ['linear_vel', 'angular_vel']

# 匹配 episode_*.hdf5 / mirror_episode_*.hdf5 / new_episode_*.hdf5
EP_RE = re.compile(r'episode_(\d+)\.hdf5$')


# =========================
# HDF5 读取（只读元数据，不读图片）
# =========================
def load_hdf5_metadata(dataset_dir: str, dataset_name: str, is_mobile: bool):
    """
    只读取关节/动作等数据，不读图片，避免一次性占用大量内存。
    """
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset does not exist:\n{dataset_path}")

    with h5py.File(dataset_path, 'r') as root:
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        effort = root['/observations/effort'][()] if 'effort' in root else None
        action = root['/action'][()]
        if is_mobile and '/base_action' in root:
            base_action = root['/base_action'][()]
        else:
            base_action = None
        compressed = bool(root.attrs.get('compress', False))

    return qpos, qvel, effort, action, base_action, dataset_path, compressed


# =========================
# 视频写入工具函数
# =========================
def _open_writer(video_path: str, fps: float, frame_size: tuple[int, int]):
    """
    创建 VideoWriter（视频写入器）
    优先尝试 H.264 -> XVID -> mp4v
    """
    codecs_to_try = [
        ('avc1', 'H.264'),
        ('XVID', 'XVID'),
        ('mp4v', 'MPEG-4'),
    ]
    for fourcc_str, _name in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(video_path, fourcc, max(1, int(fps)), frame_size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Failed to create VideoWriter for {video_path}")


def _add_segment_suffix(path: str, idx: int):
    stem, ext = os.path.splitext(path)
    return f"{stem}_part{idx:03d}{ext}"


# =========================
# 核心：流式导出单个 episode 视频
# =========================
def export_episode_streamed(
    dataset_path: str,
    video_path: str,
    mode: str = "tiled",          # "tiled" 或 "separate"
    fps: float = 30.0,
    step: int = 1,                 # 1 = 每帧导出
    segment_minutes: int | None = None,
):
    """
    从 HDF5 流式读取图片并写成视频，避免一次性加载所有帧进内存。

    关键颜色逻辑（与官方保持一致）：
        1）从 HDF5 读出图像：
              - 若 root.attrs['compress'] == True:
                    arr 为压缩字节，先用 cv2.imdecode(..., IMREAD_COLOR) 解成 BGR，再转成 RGB
              - 否则：
                    arr 已经是未压缩图像（通常为 RGB），直接使用
        2）在写入视频之前，统一执行：
              image = image[:, :, [2, 1, 0]]
           这一步无论原始是 RGB 还是 BGR，都与官方脚本一致。
    """
    with h5py.File(dataset_path, 'r') as f:
        if '/observations/images' not in f:
            print(f"[Skip] No images in {dataset_path}")
            return

        images_grp = f['/observations/images']
        cam_names = sorted(images_grp.keys())
        if len(cam_names) == 0:
            print(f"[Skip] No cameras in {dataset_path}")
            return

        compressed = bool(f.attrs.get('compress', False))
        n_frames = images_grp[cam_names[0]].shape[0]

        # 读取首帧，确定尺寸（此处不做通道变换，仅用于获取大小）
        first_imgs = []
        for cam in cam_names:
            arr0 = images_grp[cam][0]
            if compressed:
                img0_bgr = cv2.imdecode(arr0, cv2.IMREAD_COLOR)
                img0 = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB)
            else:
                img0 = arr0                               # 通常为 RGB
            first_imgs.append(img0)

        # 初始化 VideoWriter
        writers: dict[str, cv2.VideoWriter] = {}
        seg_idx = 0
        frames_per_segment = None
        if segment_minutes is not None:
            frames_per_segment = int(segment_minutes * 60 * fps)

        def close_writers():
            for w in writers.values():
                w.release()
            writers.clear()

        def open_writers_for_segment(seg_idx_local: int):
            close_writers()
            if mode == "tiled":
                h, w, _ = first_imgs[0].shape
                total_w = w * len(cam_names)
                out_path = (
                    _add_segment_suffix(video_path, seg_idx_local)
                    if frames_per_segment else video_path
                )
                writers['tiled'] = _open_writer(out_path, fps, (total_w, h))
            else:  # "separate"
                for cam, img0 in zip(cam_names, first_imgs):
                    h, w, _ = img0.shape
                    stem, _ = os.path.splitext(video_path)
                    base = f"{stem}_{cam}.mp4"
                    out_path = (
                        _add_segment_suffix(base, seg_idx_local)
                        if frames_per_segment else base
                    )
                    writers[cam] = _open_writer(out_path, fps, (w, h))

        open_writers_for_segment(seg_idx)
        written_in_this_segment = 0

        # 逐帧写入
        for t in range(0, n_frames, step):
            imgs = []
            for cam in cam_names:
                arr = images_grp[cam][t]
                if compressed:
                    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                else:
                    img = arr                                 # 通常为 RGB
                imgs.append(img)

            if mode == "tiled":
                frame = np.concatenate(imgs, axis=1)  # (H, W * N, 3)
                # 与官方脚本一致：写入前统一 swap 通道
                frame_out = frame[:, :, [2, 1, 0]]
                writers['tiled'].write(frame_out)
            else:
                for cam, img in zip(cam_names, imgs):
                    img_out = img[:, :, [2, 1, 0]]
                    writers[cam].write(img_out)

            written_in_this_segment += 1
            if frames_per_segment and written_in_this_segment >= frames_per_segment:
                seg_idx += 1
                open_writers_for_segment(seg_idx)
                written_in_this_segment = 0

        close_writers()
        print(f"[OK] Exported video(s) for {os.path.basename(dataset_path)}")


# =========================
# 批量导出：所有 episodes
# =========================
def export_all_episodes(
    dataset_dir: str,
    robot_base: str,
    mode: str = "tiled",
    step: int = 1,
    segment_minutes: int | None = None,
):
    """
    遍历 dataset_dir 中所有 episode_*.hdf5 / mirror_episode_*.hdf5，
    批量导出视频。
    """
    base_path = Path(__file__).resolve().parent.parent / "config"
    config = load_yaml_file('robot', robot_base, base_path).get('robot', {})
    fps = config.get('fps', 50)

    for name in sorted(os.listdir(dataset_dir)):
        if name.endswith('.hdf5') and EP_RE.search(name):
            ep_name = name[:-5]
            dataset_path = os.path.join(dataset_dir, name)
            video_path = os.path.join(dataset_dir, ep_name + '_video.mp4')
            try:
                export_episode_streamed(
                    dataset_path=dataset_path,
                    video_path=video_path,
                    mode=mode,
                    fps=fps,
                    step=step,
                    segment_minutes=segment_minutes,
                )
            except Exception as e:
                print(f"[ERR] {name}: {e}")


# =========================
# 可视化关节 / effort / base
# =========================
def visualize_joints(qpos_list,
                     command_list,
                     plot_path=None,
                     ylim=None,
                     label_overwrite=None,
                     config: dict = {},
                     ):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list)      # (T, D)
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape

    fig, axs = plt.subplots(num_dim, 1, figsize=(8, 2 * num_dim))

    leader_robots = {arm['name']: arm for arm in config.get('leader_arms', [])}
    follower_robots = {arm['name']: arm for arm in config.get('follower_arms', [])}

    valid_suffixes = []
    for leader_name in leader_robots.keys():
        suffix = leader_name.split('_', 1)[1]
        if f"follower_{suffix}" in follower_robots:
            valid_suffixes.append(suffix)

    all_names = [f"{name}_{suffix}" for suffix in valid_suffixes for name in STATE_NAMES]
    if len(all_names) < num_dim:
        all_names += [f"dim_{i}" for i in range(len(all_names), num_dim)]

    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.plot(command[:, dim_idx], label=label2)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        if ylim:
            ax.set_ylim(ylim)
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()


def visualize_single(efforts_list,
                     label,
                     plot_path=None,
                     ylim=None,
                     label_overwrite=None,
                     config: dict = {}):
    efforts = np.array(efforts_list)   # (T, D)
    num_ts, num_dim = efforts.shape
    fig, axs = plt.subplots(num_dim, 1, figsize=(8, 2 * num_dim))

    leader_robots = {arm['name']: arm for arm in config.get('leader_arms', [])}
    follower_robots = {arm['name']: arm for arm in config.get('follower_arms', [])}

    valid_suffixes = []
    for leader_name in leader_robots.keys():
        suffix = leader_name.split('_', 1)[1]
        if f"follower_{suffix}" in follower_robots:
            valid_suffixes.append(suffix)

    all_names = [f"{name}_{suffix}" for suffix in valid_suffixes for name in STATE_NAMES]
    if len(all_names) < num_dim:
        all_names += [f"dim_{i}" for i in range(len(all_names), num_dim)]

    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(efforts[:, dim_idx], label=label)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        if ylim:
            ax.set_ylim(ylim)
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved effort plot to: {plot_path}')
    plt.close()


def visualize_base(readings, plot_path=None):
    readings = np.array(readings)   # (T, D)
    num_ts, num_dim = readings.shape
    fig, axs = plt.subplots(num_dim, 1, figsize=(8, 2 * num_dim))

    all_names = BASE_STATE_NAMES
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(readings[:, dim_idx], label='raw')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(20)/20, mode='same'),
                label='smoothed_20')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(10)/10, mode='same'),
                label='smoothed_10')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(5)/5, mode='same'),
                label='smoothed_5')
        title = all_names[dim_idx] if dim_idx < len(all_names) else f"dim_{dim_idx}"
        ax.set_title(f'Joint {dim_idx}: {title}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved base plot to: {plot_path}')
    plt.close()


# =========================
# 主流程
# =========================
def main(args: dict):
    dataset_dir = args['dataset_dir']
    episode_idx = args.get('episode_idx')
    robot_base = args['robot']
    ismirror = args.get('ismirror', False)
    no_video = args.get('no_video', False)
    mode = args.get('mode', 'tiled')
    step = args.get('step', 1)
    segment_minutes = args.get('segment_minutes', None)
    export_all = args.get('all', False)

    base_path = Path(__file__).resolve().parent.parent / "config"
    config = load_yaml_file('robot', robot_base, base_path).get('robot', {})
    is_mobile = bool(config.get('base', False))
    fps = config.get('fps', 50)
    dt = 1.0 / fps

    # 批量模式
    if export_all:
        if not no_video:
            export_all_episodes(
                dataset_dir=dataset_dir,
                robot_base=robot_base,
                mode=mode,
                step=step,
                segment_minutes=segment_minutes,
            )
        print("[OK] All episodes processed.")
        return

    # 单 episode 模式
    if episode_idx is None:
        raise ValueError("For single episode mode, --episode_idx is required.")

    dataset_name = f'{"mirror_" if ismirror else ""}episode_{episode_idx}'
    qpos, _, _, action, base_action, dataset_path, compressed = load_hdf5_metadata(
        dataset_dir, dataset_name, is_mobile
    )
    print('hdf5 loaded!')

    # 视频（流式导出）
    if not no_video:
        export_episode_streamed(
            dataset_path=dataset_path,
            video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'),
            mode=mode,
            fps=fps,
            step=step,
            segment_minutes=segment_minutes,
        )

    # 关节 / action 曲线
    visualize_joints(
        qpos,
        action,
        plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'),
        config=config,
    )

    # 移动底盘曲线
    if is_mobile and base_action is not None:
        visualize_base(
            base_action,
            plot_path=os.path.join(dataset_dir, dataset_name + '_base_action.png'),
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='Dataset dir.')
    parser.add_argument('--episode_idx', type=int, required=False, help='Episode index.')
    parser.add_argument(
        '-r', '--robot',
        required=True,
        help='Robot config: aloha_solo, aloha_stationary, or aloha_mobile.'
    )
    parser.add_argument('--ismirror', action='store_true', help='Use mirror_episode_* name')

    parser.add_argument('--all', action='store_true', help='Export all episodes in dataset_dir')
    parser.add_argument('--no_video', action='store_true', help='Skip exporting video(s)')
    parser.add_argument(
        '--mode',
        choices=['tiled', 'separate'],
        default='tiled',
        help='Video export mode: tiled (concat horizontally) or separate per camera'
    )
    parser.add_argument('--step', type=int, default=1, help='Frame step (1 = export every frame)')
    parser.add_argument(
        '--segment-minutes',
        type=int,
        default=None,
        help='Split output every N minutes (None = no split)'
    )

    main(vars(parser.parse_args()))
