#!/usr/bin/env python3
"""极简 HDF5 处理：交换左右臂（前7个和后7个互换），转换摄像头名称"""

import h5py
import sys
import argparse
import glob
import shutil
from pathlib import Path

def process_file(fname, backup=True):
    fname = Path(fname).resolve()  # 转换为绝对路径
    if not fname.exists():
        print(f'✗ 文件不存在: {fname}')
        return

    # 自动备份
    if backup:
        backup_path = fname.parent / (fname.stem + '.hdf5.backup')
        if not backup_path.exists():
            print(f'[备份] {fname.name} -> {backup_path.name}')
            try:
                shutil.copy2(str(fname), str(backup_path))
            except PermissionError as e:
                print(f'⚠ 备份失败（权限不足，继续处理）: {e}')
            except Exception as e:
                print(f'⚠ 备份失败: {e}')
        else:
            print(f'[跳过] 备份已存在: {backup_path.name}')

    print(f'[处理] {fname.name}')
    try:
        with h5py.File(str(fname), 'r+') as f:
            # 交换左右臂：前7个(右)和后7个(左)互换 -> 前7个(左)和后7个(右)
            for key in ['/observations/qpos', '/observations/qvel', '/observations/effort', '/action']:
                if key in f:
                    data = f[key]
                    data[:, :7], data[:, 7:14] = data[:, 7:14].copy(), data[:, :7].copy()

            # 转换摄像头名称：camera_* -> cam_*
            if '/observations/images' in f:
                images = f['/observations/images']
                for old_name in list(images.keys()):
                    if old_name.startswith('camera_'):
                        new_name = 'cam_' + old_name[7:]
                        images.move(old_name, new_name)
        print(f'[完成] {fname.name}\n')
    except Exception as e:
        print(f'✗ 处理失败: {e}\n')
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDF5处理：交换左右臂，转换摄像头名称')
    parser.add_argument('files', nargs='+', help='HDF5文件路径（支持通配符）')
    parser.add_argument('--no-backup', action='store_true', help='不备份原文件')
    args = parser.parse_args()

    # 展开通配符
    file_list = []
    for pattern in args.files:
        matches = glob.glob(pattern)
        if not matches:
            print(f'⚠ 未找到匹配: {pattern}')
        else:
            file_list.extend(matches)

    if not file_list:
        print('✗ 没有找到文件')
        print(f'提示: 检查路径是否正确，当前工作目录: {Path.cwd()}')
        sys.exit(1)

    print(f'找到 {len(file_list)} 个文件\n')
    for fname in file_list:
        try:
            process_file(fname, backup=not args.no_backup)
        except Exception as e:
            print(f'✗ 处理失败 {fname}: {e}\n')