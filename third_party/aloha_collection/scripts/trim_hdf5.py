#!/usr/bin/env python3
"""
裁剪HDF5文件最后1秒的数据
Trim the last 1 second from HDF5 file
"""

import argparse
import os
import h5py
import numpy as np
from pathlib import Path


def trim_hdf5(input_path, output_path, seconds_to_trim=1, fps=50):
    """
    裁剪HDF5文件，去掉最后N秒的数据
    
    参数:
        input_path: 输入HDF5文件路径
        output_path: 输出HDF5文件路径
        seconds_to_trim: 要裁剪的秒数（默认1秒）
        fps: 帧率（默认50 FPS）
    """
    
    # 计算要去掉的帧数
    frames_to_remove = int(fps * seconds_to_trim)
    
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"帧率: {fps} FPS")
    print(f"裁剪时长: {seconds_to_trim} 秒")
    print(f"要去掉的帧数: {frames_to_remove}")
    
    # 读取原始文件
    with h5py.File(input_path, 'r') as src:
        # 获取原始帧数
        original_frames = src['/observations/qpos'].shape[0]
        new_frames = original_frames - frames_to_remove
        
        if new_frames <= 0:
            print(f"错误: 文件只有 {original_frames} 帧，无法裁剪 {frames_to_remove} 帧！")
            return False
        
        print(f"原始帧数: {original_frames}")
        print(f"裁剪后帧数: {new_frames}")
        
        # 创建新文件
        with h5py.File(output_path, 'w') as dst:
            # 复制属性
            for attr_name in src.attrs:
                dst.attrs[attr_name] = src.attrs[attr_name]
            
            # 复制并裁剪数据
            # 1. observations/qpos
            qpos = src['/observations/qpos'][:new_frames]
            obs = dst.create_group('observations')
            obs.create_dataset('qpos', data=qpos)
            
            # 2. observations/qvel
            if '/observations/qvel' in src:
                qvel = src['/observations/qvel'][:new_frames]
                obs.create_dataset('qvel', data=qvel)
            
            # 3. observations/effort
            if '/observations/effort' in src:
                effort = src['/observations/effort'][:new_frames]
                obs.create_dataset('effort', data=effort)
            
            # 4. action
            if '/action' in src:
                action = src['/action'][:new_frames]
                dst.create_dataset('action', data=action)
            
            # 5. base_action (如果有)
            if '/base_action' in src:
                base_action = src['/base_action'][:new_frames]
                dst.create_dataset('base_action', data=base_action)
            
            # 6. images (如果有)
            if '/observations/images' in src:
                image_group = obs.create_group('images')
                for cam_name in src['/observations/images'].keys():
                    print(f"  处理相机: {cam_name}")
                    images = src[f'/observations/images/{cam_name}'][:new_frames]
                    image_group.create_dataset(cam_name, data=images)
            
            print(f"\n✅ 成功！裁剪后的文件已保存到: {output_path}")
            print(f"   去掉了最后 {seconds_to_trim} 秒 ({frames_to_remove} 帧)")
            return True


def main():
    parser = argparse.ArgumentParser(
        description='裁剪HDF5文件最后N秒的数据'
    )
    parser.add_argument(
        '--input',
        '-i',
        type=str,
        required=True,
        help='输入HDF5文件路径（例如: episode_0.hdf5）'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='输出HDF5文件路径（默认: 在输入文件名后加_trimmed）'
    )
    parser.add_argument(
        '--seconds',
        '-s',
        type=float,
        default=1.0,
        help='要裁剪的秒数（默认: 1秒）'
    )
    parser.add_argument(
        '--fps',
        '-f',
        type=int,
        default=50,
        help='帧率（默认: 50 FPS）'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.isfile(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    # 确定输出文件路径
    if args.output:
        output_path = args.output
    else:
        # 默认在输入文件名后加 _trimmed
        input_path = Path(args.input)
        output_path = str(input_path.parent / f"{input_path.stem}_trimmed{input_path.suffix}")
    
    # 执行裁剪
    trim_hdf5(args.input, output_path, args.seconds, args.fps)


if __name__ == '__main__':
    main()

