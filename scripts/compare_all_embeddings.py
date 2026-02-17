#!/usr/bin/env python3
"""比较三个模型的 embedding 参数：finetune 模型、pi05_base 和 pi0_base。"""

import sys
from compare_embeddings import compare_embeddings

def main():
    finetune_path = "checkpoints/20260205/params/"
    pi05_base_path = "/home/eii/.cache/openpi/openpi-assets/checkpoints/pi05_base/params"
    pi0_base_path = "/home/eii/.cache/openpi/openpi-assets/checkpoints/pi0_base/params"
    
    print("="*80)
    print("模型 Embedding 对比分析")
    print("="*80)
    
    print("\n【对比 1】Finetune 模型 vs pi05_base（预训练基础模型）")
    print("-"*80)
    result1 = compare_embeddings(finetune_path, pi05_base_path)
    
    print("\n\n【对比 2】Finetune 模型 vs pi0_base（另一个基础模型）")
    print("-"*80)
    result2 = compare_embeddings(finetune_path, pi0_base_path)
    
    print("\n\n【对比 3】pi05_base vs pi0_base（两个基础模型之间的差异）")
    print("-"*80)
    result3 = compare_embeddings(pi05_base_path, pi0_base_path)
    
    print("\n\n" + "="*80)
    print("总结")
    print("="*80)
    print(f"\n1. Finetune 模型 vs pi05_base:")
    print(f"   最大差异: {result1['max_diff']:.6f}")
    print(f"   平均差异: {result1['mean_diff']:.6f}")
    print(f"   不同参数: {result1['num_different']}/{result1['total_params']} ({100*result1['num_different']/result1['total_params']:.2f}%)")
    
    print(f"\n2. Finetune 模型 vs pi0_base:")
    print(f"   最大差异: {result2['max_diff']:.6f}")
    print(f"   平均差异: {result2['mean_diff']:.6f}")
    print(f"   不同参数: {result2['num_different']}/{result2['total_params']} ({100*result2['num_different']/result2['total_params']:.2f}%)")
    
    print(f"\n3. pi05_base vs pi0_base:")
    print(f"   最大差异: {result3['max_diff']:.6f}")
    print(f"   平均差异: {result3['mean_diff']:.6f}")
    print(f"   不同参数: {result3['num_different']}/{result3['total_params']} ({100*result3['num_different']/result3['total_params']:.2f}%)")
    
    print("\n" + "="*80)
    print("结论")
    print("="*80)
    
    if result1['max_diff'] < 0.1:
        print("⚠️  Finetune 模型和 pi05_base 的 embedding 几乎完全相同！")
        print("   这说明在 finetune 过程中，embedding 几乎没有更新。")
        print("   可能的原因：")
        print("   1. 学习率太小（当前 peak_lr=2.5e-5）")
        print("   2. Embedding 的梯度很小")
        print("   3. 训练步数不够多")
    else:
        print("✓  Finetune 模型和 pi05_base 的 embedding 有显著差异")
        print("   embedding 在训练时被更新了。")
    
    if abs(result1['max_diff'] - result3['max_diff']) < 0.01:
        print("\n⚠️  注意：Finetune 模型和 pi05_base 的差异，与 pi05_base 和 pi0_base 的差异几乎相同！")
        print("   这说明 finetune 模型可能没有从 pi05_base 进一步更新。")

if __name__ == "__main__":
    main()
