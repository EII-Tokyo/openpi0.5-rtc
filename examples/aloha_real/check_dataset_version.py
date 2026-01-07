"""
查看 LeRobot 数据集的 codebase version 的几种方法
"""

import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

def check_local_version(dataset_path: str):
    """从本地缓存查看 codebase version"""
    info_json = Path(dataset_path) / "meta" / "info.json"
    if info_json.exists():
        with open(info_json, 'r') as f:
            info = json.load(f)
        return info.get("codebase_version", "Not found")
    return None

def check_hub_version(repo_id: str):
    """从 Hugging Face Hub 查看 codebase version"""
    try:
        # 方法1: 下载 info.json 文件
        info_path = hf_hub_download(
            repo_id=repo_id,
            filename="meta/info.json",
            repo_type="dataset"
        )
        with open(info_path, 'r') as f:
            info = json.load(f)
        return info.get("codebase_version", "Not found")
    except Exception as e:
        print(f"从 Hub 下载失败: {e}")
        return None

def check_hub_tags(repo_id: str):
    """查看 Hub 上的标签（可能包含版本标签）"""
    try:
        from huggingface_hub import list_repo_refs
        refs = list_repo_refs(repo_id=repo_id, repo_type="dataset")
        tags = [ref.name for ref in refs if ref.ref_type == "tag"]
        return tags
    except Exception as e:
        print(f"获取标签失败: {e}")
        return []

if __name__ == "__main__":
    repo_id = "lyl472324464/twist_two_202511"
    local_path = Path.home() / ".cache" / "huggingface" / "lerobot" / "lyl472324464" / "twist_two_202511"
    
    print("=" * 60)
    print("查看数据集 Codebase Version")
    print("=" * 60)
    print(f"数据集: {repo_id}\n")
    
    # 方法1: 从本地缓存查看
    print("1. 从本地缓存查看:")
    local_version = check_local_version(str(local_path))
    if local_version:
        print(f"   ✓ Codebase Version: {local_version}")
        print(f"   路径: {local_path / 'meta' / 'info.json'}")
    else:
        print("   ✗ 本地缓存中未找到数据集")
    print()
    
    # 方法2: 从 Hub 查看
    print("2. 从 Hugging Face Hub 查看:")
    hub_version = check_hub_version(repo_id)
    if hub_version:
        print(f"   ✓ Codebase Version: {hub_version}")
    else:
        print("   ✗ 无法从 Hub 获取")
    print()
    
    # 方法3: 查看 Hub 标签
    print("3. Hub 上的标签:")
    tags = check_hub_tags(repo_id)
    if tags:
        print(f"   标签: {', '.join(tags)}")
        version_tags = [t for t in tags if t.startswith('v')]
        if version_tags:
            print(f"   版本标签: {', '.join(version_tags)}")
    else:
        print("   无标签或无法获取")
    print()
    
    print("=" * 60)
    print("提示: 也可以直接查看文件:")
    print(f"  cat {local_path / 'meta' / 'info.json'} | grep codebase_version")
    print("=" * 60)

