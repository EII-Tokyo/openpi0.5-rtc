#!/usr/bin/env python3
"""比较两个模型的 embedding 参数，检查是否相同。"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util

from openpi.models.model import restore_params


def find_embedding_key(flat_params):
    """查找 embedding 参数的键。"""
    embedding_key = None
    for key in flat_params:
        key_str = "/".join(key)
        if "embedder" in key_str and "input_embedding" in key_str:
            embedding_key = key
            break
    
    if embedding_key is None:
        # 尝试其他可能的路径
        for key in flat_params:
            if "input_embedding" in "/".join(key):
                embedding_key = key
                break
    
    if embedding_key is None:
        raise ValueError("找不到 embedding 参数。可用的键: " + str(list(flat_params.keys())[:10]))
    
    return embedding_key


def compare_embeddings(params_path1: str, params_path2: str):
    """比较两个模型的 embedding 参数。"""
    print(f"加载模型 1: {params_path1}")
    params1 = restore_params(params_path1, restore_type=np.ndarray)
    flat_params1 = traverse_util.flatten_dict(params1)
    embedding_key1 = find_embedding_key(flat_params1)
    embedding1 = np.array(flat_params1[embedding_key1])
    
    print(f"加载模型 2: {params_path2}")
    params2 = restore_params(params_path2, restore_type=np.ndarray)
    flat_params2 = traverse_util.flatten_dict(params2)
    embedding_key2 = find_embedding_key(flat_params2)
    embedding2 = np.array(flat_params2[embedding_key2])
    
    print(f"\n模型 1 embedding 键: {'/'.join(embedding_key1)}")
    print(f"模型 2 embedding 键: {'/'.join(embedding_key2)}")
    print(f"\n模型 1 embedding 形状: {embedding1.shape}")
    print(f"模型 2 embedding 形状: {embedding2.shape}")
    
    if embedding1.shape != embedding2.shape:
        print("⚠️  警告: 两个模型的 embedding 形状不同！")
        return
    
    # 计算差异
    diff = embedding1 - embedding2
    max_diff = np.abs(diff).max()
    mean_diff = np.abs(diff).mean()
    num_different = np.sum(np.abs(diff) > 1e-6)
    total_params = embedding1.size
    
    print(f"\n{'='*80}")
    print(f"Embedding 参数比较结果:")
    print(f"{'='*80}")
    print(f"最大差异: {max_diff:.10f}")
    print(f"平均差异: {mean_diff:.10f}")
    print(f"不同参数数量: {num_different} / {total_params} ({100*num_different/total_params:.2f}%)")
    
    # 计算 L2 范数差异
    l2_norms1 = np.linalg.norm(embedding1, axis=1)
    l2_norms2 = np.linalg.norm(embedding2, axis=1)
    norm_diff = np.abs(l2_norms1 - l2_norms2)
    
    print(f"\nL2 范数比较:")
    print(f"  模型 1 - 最小值: {l2_norms1.min():.6f}, 最大值: {l2_norms1.max():.6f}, 平均值: {l2_norms1.mean():.6f}")
    print(f"  模型 2 - 最小值: {l2_norms2.min():.6f}, 最大值: {l2_norms2.max():.6f}, 平均值: {l2_norms2.mean():.6f}")
    print(f"  L2 范数最大差异: {norm_diff.max():.10f}")
    print(f"  L2 范数平均差异: {norm_diff.mean():.10f}")
    
    # 检查是否完全相同
    if max_diff < 1e-10:
        print(f"\n✓ 两个模型的 embedding 参数完全相同（差异 < 1e-10）")
        print(f"  这可能意味着 embedding 在训练时被冻结了！")
    elif max_diff < 1e-6:
        print(f"\n⚠️  两个模型的 embedding 参数非常相似（差异 < 1e-6）")
        print(f"  这可能意味着 embedding 几乎没有更新。")
    else:
        print(f"\n✓ 两个模型的 embedding 参数有显著差异")
        print(f"  embedding 在训练时应该被更新了。")
    
    # 找出差异最大的 token
    if max_diff > 1e-6:
        max_diff_per_token = np.abs(diff).max(axis=1)
        top_diff_indices = np.argsort(max_diff_per_token)[-10:][::-1]
        print(f"\n差异最大的 10 个 token:")
        for i, idx in enumerate(top_diff_indices, 1):
            print(f"  {i:2d}. Token ID: {idx:6d}, 最大差异: {max_diff_per_token[idx]:.10f}")
    
    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "num_different": num_different,
        "total_params": total_params,
        "norm_diff_max": norm_diff.max(),
        "norm_diff_mean": norm_diff.mean(),
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python compare_embeddings.py <model1_params_path> <model2_params_path>")
        print("\n示例:")
        print("  python compare_embeddings.py checkpoints/20260205/params/ /home/eii/.cache/openpi/openpi-assets/checkpoints/pi05_base/params")
        sys.exit(1)
    
    params_path1 = sys.argv[1]
    params_path2 = sys.argv[2]
    
    compare_embeddings(params_path1, params_path2)
