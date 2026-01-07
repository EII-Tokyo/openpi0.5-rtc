"""
清理 Hugging Face 缓存的工具脚本

用法:
    python clean_cache.py                    # 查看缓存大小
    python clean_cache.py --clean           # 清理所有缓存
    python clean_cache.py --keep <repo_id>   # 清理除指定数据集外的所有缓存
"""

import shutil
from pathlib import Path
import argparse


def get_dir_size(path: Path) -> int:
    """获取目录大小（字节）"""
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except (PermissionError, OSError) as e:
        print(f"警告: 无法访问某些文件: {e}")
    return total


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def scan_cache():
    """扫描并显示缓存信息"""
    cache_dir = Path.home() / ".cache" / "huggingface"
    
    print("=" * 60)
    print("Hugging Face 缓存扫描")
    print("=" * 60)
    
    # LeRobot 缓存
    lerobot_cache = cache_dir / "lerobot"
    if lerobot_cache.exists():
        print(f"\n📁 LeRobot 数据集缓存: {lerobot_cache}")
        total_size = 0
        datasets = []
        
        for item in lerobot_cache.iterdir():
            if item.is_dir():
                size = get_dir_size(item)
                total_size += size
                datasets.append((item.name, size))
                print(f"  - {item.name}: {format_size(size)}")
        
        print(f"\n  总计: {format_size(total_size)} ({len(datasets)} 个数据集)")
    else:
        print(f"\n📁 LeRobot 缓存目录不存在: {lerobot_cache}")
    
    # Hugging Face Hub 缓存
    hub_cache = cache_dir / "hub"
    if hub_cache.exists():
        hub_size = get_dir_size(hub_cache)
        print(f"\n📁 Hugging Face Hub 缓存: {hub_cache}")
        print(f"  大小: {format_size(hub_size)}")
    else:
        print(f"\n📁 Hub 缓存目录不存在: {hub_cache}")
    
    # 其他缓存
    other_caches = []
    for item in cache_dir.iterdir():
        if item.is_dir() and item.name not in ["lerobot", "hub"]:
            size = get_dir_size(item)
            if size > 0:
                other_caches.append((item.name, size))
    
    if other_caches:
        print(f"\n📁 其他缓存:")
        for name, size in other_caches:
            print(f"  - {name}: {format_size(size)}")
    
    # 总缓存大小
    total_cache_size = get_dir_size(cache_dir)
    print(f"\n{'=' * 60}")
    print(f"总缓存大小: {format_size(total_cache_size)}")
    print(f"{'=' * 60}")


def clean_cache(keep_repo_id: str = None):
    """清理缓存
    
    Args:
        keep_repo_id: 要保留的数据集 repo_id（格式: org/dataset_name）
    """
    cache_dir = Path.home() / ".cache" / "huggingface"
    lerobot_cache = cache_dir / "lerobot"
    
    if not lerobot_cache.exists():
        print("LeRobot 缓存目录不存在，无需清理")
        return
    
    print("\n开始清理缓存...")
    
    if keep_repo_id:
        # 只清理除了指定数据集外的其他缓存
        keep_path = lerobot_cache / keep_repo_id
        cleaned_size = 0
        cleaned_count = 0
        
        for item in lerobot_cache.iterdir():
            if item.is_dir() and item != keep_path:
                size = get_dir_size(item)
                try:
                    shutil.rmtree(item)
                    cleaned_size += size
                    cleaned_count += 1
                    print(f"  ✓ 已删除: {item.name} ({format_size(size)})")
                except Exception as e:
                    print(f"  ✗ 删除失败 {item.name}: {e}")
        
        print(f"\n清理完成:")
        print(f"  删除数据集数: {cleaned_count}")
        print(f"  释放空间: {format_size(cleaned_size)}")
        print(f"  保留数据集: {keep_repo_id}")
    else:
        # 清理所有 lerobot 缓存
        total_size = get_dir_size(lerobot_cache)
        try:
            shutil.rmtree(lerobot_cache)
            print(f"  ✓ 已清理所有 LeRobot 缓存")
            print(f"  释放空间: {format_size(total_size)}")
        except Exception as e:
            print(f"  ✗ 清理失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清理 Hugging Face 缓存")
    parser.add_argument("--clean", action="store_true",
                       help="清理所有 LeRobot 缓存")
    parser.add_argument("--keep", type=str,
                       help="保留指定的数据集（格式: org/dataset_name），清理其他缓存")
    
    args = parser.parse_args()
    
    # 先扫描缓存
    scan_cache()
    
    # 执行清理
    if args.clean:
        response = input("\n⚠️  确定要清理所有 LeRobot 缓存吗？(y/n): ")
        if response.lower() == 'y':
            clean_cache()
        else:
            print("已取消")
    elif args.keep:
        response = input(f"\n⚠️  确定要清理除 {args.keep} 外的所有缓存吗？(y/n): ")
        if response.lower() == 'y':
            clean_cache(keep_repo_id=args.keep)
        else:
            print("已取消")
    else:
        print("\n提示: 使用 --clean 清理所有缓存，或 --keep <repo_id> 保留指定数据集")

