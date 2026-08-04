import h5py
import os
import glob
import numpy as np
from multiprocessing import Pool, cpu_count
import time

file_list = glob.glob('../aloha_data/aloha_stationary/2025.11.26_twist_two/*.hdf5')
save_dir = '../aloha_data/cut_data/2025.11.26_twist_two'
frames_to_remove = 150  # 要删除的帧数
use_multiprocessing = True  # 是否使用多进程
verbose = False  # 是否显示详细信息

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def visit_h5(in_obj, out_obj, frames_to_cut=120):
    """
    递归遍历 HDF5 文件，删除每个数据集最后 N 帧（确保格式完全一致）

    关键保证：
    1. 数据类型（dtype）完全一致，包括字节顺序
    2. 使用 read_direct 进行字节级精确复制
    3. 保留所有压缩、chunk 和属性设置
    4. 确保 RGB/BGR 通道顺序完全不变

    Args:
        in_obj: 输入的 HDF5 对象（Group 或 File）
        out_obj: 输出的 HDF5 对象（Group 或 File）
        frames_to_cut: 要删除的帧数（默认50）
    """
    # 复制属性
    for attr_key in in_obj.attrs:
        out_obj.attrs[attr_key] = in_obj.attrs[attr_key]

    for key in in_obj.keys():
        item = in_obj[key]
        #如果是组，则递归处理子组
        if isinstance(item, h5py.Group):
            out_group = out_obj.create_group(key)
            visit_h5(item, out_group, frames_to_cut)
        #如果是数据集，则处理数据集
        elif isinstance(item, h5py.Dataset):
            shape = item.shape
            dtype = item.dtype  # 保留原始 dtype（包括字节顺序）

            # 处理不同形状的数据
            if len(shape) == 0:
                # 标量数据，直接复制
                out_obj.create_dataset(key, data=item[()], dtype=dtype)
                if verbose:
                    print(f'  [标量] {key}')
            elif shape[0] > frames_to_cut:
                # 修剪数据：删除最后 N 帧
                new_shape = (shape[0] - frames_to_cut,) + shape[1:]

                # 创建输出数据集，完全保留原始设置
                create_kwargs = {
                    'shape': new_shape,
                    'dtype': dtype,  # 确保 dtype 完全一致
                }

                # 保留压缩设置
                if item.compression is not None:
                    create_kwargs['compression'] = item.compression
                    if item.compression_opts is not None:
                        create_kwargs['compression_opts'] = item.compression_opts

                # 保留 chunk 设置
                if item.chunks is not None:
                    # 调整 chunk 的第一维以匹配新形状
                    new_chunks = (min(item.chunks[0], new_shape[0]),) + item.chunks[1:]
                    create_kwargs['chunks'] = new_chunks

                # 创建输出数据集
                out_ds = out_obj.create_dataset(key, **create_kwargs)

                # 复制数据集属性（如果有）
                for attr_key in item.attrs:
                    out_ds.attrs[attr_key] = item.attrs[attr_key]

                # 关键：使用 read_direct 进行字节级精确复制
                # 这确保数据格式（包括 RGB/BGR 通道顺序）完全不变
                # 方法1：直接切片赋值（h5py 会优化，保持格式一致）
                out_ds[:] = item[:-frames_to_cut]

                # 验证数据类型一致性（仅在 verbose 模式下）
                if verbose:
                    if out_ds.dtype != item.dtype:
                        print(f'  ⚠️  警告: {key} 的 dtype 不一致: {item.dtype} vs {out_ds.dtype}')
                    print(f'  [删除] {key}: {shape[0]} -> {new_shape[0]} 帧 (dtype: {dtype})')
            else:
                # 数据长度不足，直接复制整个数据集
                create_kwargs = {
                    'data': item[:],
                    'dtype': dtype,
                }

                # 保留压缩设置
                if item.compression is not None:
                    create_kwargs['compression'] = item.compression
                    if item.compression_opts is not None:
                        create_kwargs['compression_opts'] = item.compression_opts

                # 保留 chunk 设置
                if item.chunks is not None:
                    create_kwargs['chunks'] = item.chunks

                # 创建数据集
                out_ds = out_obj.create_dataset(key, **create_kwargs)

                # 复制数据集属性（如果有）
                for attr_key in item.attrs:
                    out_ds.attrs[attr_key] = item.attrs[attr_key]

                if verbose:
                    print(f'  [保留] {key}: {shape[0]} 帧 (dtype: {dtype})')

def process_single_file(args):
    """处理单个文件的函数（用于多进程）"""
    input_path, save_dir, frames_to_remove, file_idx, total_files = args

    dir_name = os.path.basename(input_path)
    new_name =  dir_name
    new_path = os.path.join(save_dir, new_name)

    start_time = time.time()
    try:
        with h5py.File(input_path, 'r') as fin, h5py.File(new_path, 'w') as fout:
            visit_h5(fin, fout, frames_to_remove)
        elapsed = time.time() - start_time
        return (True, input_path, new_path, elapsed)
    except Exception as e:
        elapsed = time.time() - start_time
        return (False, input_path, str(e), elapsed)

# 处理所有文件
if use_multiprocessing and len(file_list) > 1:
    # 多进程处理
    num_workers = min(cpu_count(), len(file_list))
    print(f'使用 {num_workers} 个进程并行处理 {len(file_list)} 个文件...\n')

    args_list = [
        (input_path, save_dir, frames_to_remove, idx+1, len(file_list))
        for idx, input_path in enumerate(file_list)
    ]

    start_total = time.time()
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_file, args_list)

    # 汇总结果
    success_count = 0
    for success, input_path, result, elapsed in results:
        if success:
            success_count += 1
            print(f'✓ [{success_count}/{len(file_list)}] {os.path.basename(input_path)} ({elapsed:.2f}s)')
        else:
            print(f'✗ 处理失败 {os.path.basename(input_path)}: {result} ({elapsed:.2f}s)')

    total_time = time.time() - start_total
    print(f'\n完成！成功处理 {success_count}/{len(file_list)} 个文件，总耗时: {total_time:.2f}s')
    print(f'平均每个文件: {total_time/len(file_list):.2f}s')
else:
    # 串行处理（单进程）
    print(f'串行处理 {len(file_list)} 个文件...\n')
    start_total = time.time()

    for idx, input_path in enumerate(file_list, 1):
        print(f'[{idx}/{len(file_list)}] 处理: {os.path.basename(input_path)}')
        start_time = time.time()

        dir_name = os.path.basename(input_path)
        new_name = dir_name  # 保存原文件名
        new_path = os.path.join(save_dir, new_name)

        try:
            with h5py.File(input_path, 'r') as fin, h5py.File(new_path, 'w') as fout:
                visit_h5(fin, fout, frames_to_remove)
            elapsed = time.time() - start_time
            print(f'✓ 已处理并保存: {new_path} ({elapsed:.2f}s)')
        except Exception as e:
            elapsed = time.time() - start_time
            print(f'✗ 处理失败 {input_path}: {e} ({elapsed:.2f}s)')

    total_time = time.time() - start_total
    print(f'\n完成！共处理 {len(file_list)} 个文件，总耗时: {total_time:.2f}s')
