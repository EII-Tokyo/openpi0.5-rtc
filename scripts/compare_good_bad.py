#!/usr/bin/env python3
"""对比三个模型中 'good' 和 'bad' 的 embedding 内积。"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from scripts.check_embedding_norms import get_word_embeddings_and_dot_product

def main():
    models = {
        "pi0_base": "/home/eii/.cache/openpi/openpi-assets/checkpoints/pi0_base/params",
        "pi05_base": "/home/eii/.cache/openpi/openpi-assets/checkpoints/pi05_base/params",
        "finetune": "checkpoints/20260205/params/",
    }
    
    results = {}
    
    print("="*80)
    print("'good' 和 'bad' 的 Embedding 内积对比")
    print("="*80)
    
    for name, path in models.items():
        print(f"\n【{name}】")
        print("-"*80)
        result = get_word_embeddings_and_dot_product("good", "bad", path)
        results[name] = result
    
    print("\n\n" + "="*80)
    print("对比总结")
    print("="*80)
    
    print(f"\n{'模型':<15} {'内积 (点积)':<15} {'余弦相似度':<15} {'good L2范数':<15} {'bad L2范数':<15}")
    print("-"*80)
    
    for name in ["pi0_base", "pi05_base", "finetune"]:
        r = results[name]
        print(f"{name:<15} {r['dot_product']:<15.6f} {r['cosine_similarity']:<15.6f} {r['norm1']:<15.6f} {r['norm2']:<15.6f}")
    
    print("\n" + "="*80)
    print("分析")
    print("="*80)
    
    pi0_dot = results["pi0_base"]["dot_product"]
    pi05_dot = results["pi05_base"]["dot_product"]
    finetune_dot = results["finetune"]["dot_product"]
    
    print(f"\n1. pi0_base 内积: {pi0_dot:.6f}")
    print(f"2. pi05_base 内积: {pi05_dot:.6f}")
    print(f"3. finetune 内积: {finetune_dot:.6f}")
    
    print(f"\n差异分析:")
    print(f"  pi05_base vs pi0_base: {abs(pi05_dot - pi0_dot):.6f}")
    print(f"  finetune vs pi05_base: {abs(finetune_dot - pi05_dot):.6f}")
    print(f"  finetune vs pi0_base: {abs(finetune_dot - pi0_dot):.6f}")
    
    if abs(finetune_dot - pi05_dot) < 0.01:
        print(f"\n⚠️  finetune 模型和 pi05_base 的内积几乎完全相同！")
        print(f"   差异仅为: {abs(finetune_dot - pi05_dot):.6f}")
        print(f"   这进一步证实了 embedding 在 finetune 过程中几乎没有更新。")
    else:
        print(f"\n✓  finetune 模型和 pi05_base 的内积有差异")
        print(f"   说明 embedding 在 finetune 过程中有更新。")

if __name__ == "__main__":
    main()
