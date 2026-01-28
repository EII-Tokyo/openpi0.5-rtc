#!/usr/bin/env python3
"""脚本用于查看和修改 lerobot 数据集中的 tasks.parquet 文件"""

import argparse
import pathlib
import sys

import pandas as pd
import lerobot.datasets.lerobot_dataset as lerobot_dataset


def read_tasks_parquet(file_path: str) -> pd.DataFrame:
    """读取 tasks.parquet 文件"""
    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        print(f"错误: 无法读取文件 {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


def display_tasks(df: pd.DataFrame):
    """显示所有任务"""
    print("=" * 80)
    print("当前任务列表")
    print("=" * 80)
    if len(df) == 0:
        print("没有任务")
        return
    
    for idx, (task_desc, row) in enumerate(df.iterrows(), 1):
        print(f"\n任务 #{idx}:")
        print(f"  描述: {task_desc}")
        print(f"  索引: {row}")
    print("=" * 80)


def modify_task(df: pd.DataFrame, old_desc: str, new_desc: str) -> pd.DataFrame:
    """修改任务描述"""
    # 先尝试精确匹配
    if old_desc in df.index:
        df_new = df.copy()
        df_new = df_new.rename(index={old_desc: new_desc})
        print(f"✓ 已修改任务描述:")
        print(f"  旧: {old_desc}")
        print(f"  新: {new_desc}")
        return df_new
    
    # 如果精确匹配失败，尝试去除首尾空格后匹配
    for idx in df.index:
        if idx.strip() == old_desc.strip():
            df_new = df.copy()
            df_new = df_new.rename(index={idx: new_desc})
            print(f"✓ 已修改任务描述（去除空格后匹配）:")
            print(f"  旧: {idx}")
            print(f"  新: {new_desc}")
            return df_new
    
    # 如果都找不到，打印调试信息
    print(f"\n错误: 找不到任务描述 '{old_desc}'", file=sys.stderr)
    print(f"当前索引列表:", file=sys.stderr)
    for idx in df.index:
        print(f"  - '{idx}' (长度: {len(idx)})", file=sys.stderr)
        print(f"    查找的字符串长度: {len(old_desc)}", file=sys.stderr)
    
    return df


def save_tasks_parquet(df: pd.DataFrame, file_path: str):
    """保存 tasks.parquet 文件"""
    try:
        # 确保目录存在
        pathlib.Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        # 保存文件，保持索引
        df.to_parquet(file_path, index=True, engine='pyarrow')
        print(f"\n✓ 已保存到: {file_path}")
    except Exception as e:
        print(f"错误: 无法保存文件 {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


def load_dataset(repo_id: str, force_download: bool = False):
    """加载数据集，如果需要则下载"""
    try:
        if force_download:
            print(f"正在从 Hugging Face Hub 下载数据集: {repo_id}")
            print("这可能需要一些时间，请耐心等待...")
        else:
            print(f"正在加载数据集: {repo_id}")
        
        dataset = lerobot_dataset.LeRobotDataset(repo_id)
        
        if force_download:
            print(f"\n✓ 数据集下载完成: {repo_id}")
        print(f"数据集路径: {dataset.root}")
        
        return dataset
    except Exception as e:
        print(f"错误: 无法加载数据集 {repo_id}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="查看和修改 lerobot 数据集中的 tasks.parquet 文件"
    )
    parser.add_argument(
        "repo_id",
        type=str,
        nargs="?",
        default="lyl472324464/2025.12.23_twist_one",
        help="数据集仓库 ID (格式: username/dataset_name)"
    )
    parser.add_argument(
        "--modify",
        action="store_true",
        help="修改任务描述"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="如果数据集不存在则从 Hugging Face Hub 下载"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="将修改后的数据集推送到 Hugging Face Hub"
    )
    
    args = parser.parse_args()
    
    # 加载数据集（如果需要下载会自动下载）
    dataset = load_dataset(args.repo_id, force_download=args.download)
    
    # 获取 tasks.parquet 文件路径
    file_path = str(pathlib.Path(dataset.root) / "meta" / "tasks.parquet")
    
    # 读取文件
    df = read_tasks_parquet(file_path)
    
    # 显示当前内容
    display_tasks(df)
    
    # 如果需要修改，则修改并保存
    modified = False
    if args.modify:
        # 注意：索引末尾可能有空格，所以需要精确匹配
        old_desc1 = 'Do the followings: 1. Pick up the bottle. 2. Twist off the bottle cap. '
        old_desc2 = '[bad action] Do the followings: 1. Pick up the bottle. 2. Twist off the bottle cap. '
        new_desc1 = 'Do the followings: 1. If the bottle cap is facing left, rotate the bottle 180 degrees. 2. Pick up the bottle. 3. Twist off the bottle cap.'
        new_desc2 = '[bad action] Do the followings: 1. If the bottle cap is facing left, rotate the bottle 180 degrees. 2. Pick up the bottle. 3. Twist off the bottle cap.'
        df_before = df.copy()
        df = modify_task(df, old_desc1, new_desc1)
        df = modify_task(df, old_desc2, new_desc2)
        save_tasks_parquet(df, file_path)
        modified = True
    
    # 如果需要推送到 Hub
    if args.push:
        if modified:
            print(f"\n正在将修改后的数据集推送到 Hugging Face Hub: {args.repo_id}")
        else:
            print(f"\n正在将数据集推送到 Hugging Face Hub: {args.repo_id}")
        try:
            # 重新加载数据集以确保使用最新的数据
            dataset = lerobot_dataset.LeRobotDataset(args.repo_id)
            dataset.push_to_hub()
            print(f"\n✓ 已成功推送到 Hub: {args.repo_id}")
        except Exception as e:
            print(f"错误: 无法推送到 Hub: {e}", file=sys.stderr)
            sys.exit(1)
        
    

if __name__ == "__main__":
    main()

