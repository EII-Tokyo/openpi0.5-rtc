#!/usr/bin/env python3
"""脚本用于查看和修改 lerobot 数据集中的 tasks.parquet 文件"""

import argparse
import pathlib
import sys

import pandas as pd


def read_tasks_parquet(file_path: str) -> pd.DataFrame:
    """读取 tasks.parquet 文件"""
    try:
        df = pd.read_parquet(file_path)
        print(df.index)
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
    if old_desc not in df.index:
        print(f"错误: 找不到任务描述 '{old_desc}'", file=sys.stderr)
        return df
    
    # 创建新的 DataFrame，使用新的索引
    df_new = df.copy()
    df_new.index = df_new.index.str.replace(old_desc, new_desc, regex=False)
    
    print(f"✓ 已修改任务描述:")
    print(f"  旧: {old_desc}")
    print(f"  新: {new_desc}")
    
    return df_new


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


def main():
    parser = argparse.ArgumentParser(
        description="查看和修改 lerobot 数据集中的 tasks.parquet 文件"
    )
    parser.add_argument(
        "--modify",
        action="store_true",
        help="修改任务描述"
    )
    
    args = parser.parse_args()
    file_path="/home/ubuntu/.cache/huggingface/lerobot/lyl472324464/2025.12.23_twist_one/meta/tasks.parquet"
    # 读取文件
    df = read_tasks_parquet(file_path)
    
    # 显示当前内容
    display_tasks(df)
    
    if args.modify is True:
        old_desc = 'Do the followings: 1. Twist off the bottle cap. 2. Throw the bottle to the box on the left. 3. Throw the cap to the box on the right. 4. Return to home position.'
        new_desc = 'Do the followings: 1. Twist off the bottle cap. 2. Put the bottle into the box on the left. 3. Put the cap into the box on the right. 4. Return to home position.'
        # 命令行模式
        df = modify_task(df, old_desc, new_desc)
        save_tasks_parquet(df, file_path)
        
    

if __name__ == "__main__":
    main()

