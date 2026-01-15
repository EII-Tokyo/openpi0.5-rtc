#!/usr/bin/env python3
"""计算 embedding 矩阵中每个向量的 L2 范数，并找出最小值和最大值。"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util

import openpi.models.gemma as gemma
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training import data_loader as _data_loader
from openpi.training import config as _config


def compute_embedding_norms(params_path: str | None = None):
    """计算 embedding 矩阵的 L2 范数统计信息。
    
    Args:
        params_path: 可选的检查点路径。如果为 None，则使用随机初始化的参数。
    """
    if params_path:
        # 如果提供了检查点路径，直接加载参数
        from openpi.models.model import restore_params
        loaded_params = restore_params(params_path, restore_type=np.ndarray)
        print(f"Loaded params: {loaded_params.keys()}")
        
        # 查找 embedding 参数
        flat_params = traverse_util.flatten_dict(loaded_params)
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
        
        embedding_table = np.array(flat_params[embedding_key])
    else:
        # 如果没有提供检查点，初始化模型
        config = gemma.get_config("gemma_300m")
        
        # 直接初始化 Embedder 模块
        rng = jax.random.PRNGKey(0)
        embedder = gemma.Embedder(
            vocab_size=gemma.PALIGEMMA_VOCAB_SIZE,
            embed_dim=config.width,
        )
        # 初始化 embedder - 调用 encode 来触发参数初始化
        # 在 Flax 中，参数在第一次调用时初始化
        # 在 Flax Linen 中获取参数的标准方式
        embedder_vars = embedder.init(rng, jnp.zeros((1, 1), dtype=jnp.int32), method=embedder.encode)
        embedding_table = np.array(embedder_vars["params"]["input_embedding"])
        
        print("注意: 使用的是随机初始化的参数。如果要使用训练好的模型，请提供检查点路径。")
    
    # 转换为 numpy 数组以便计算
    if isinstance(embedding_table, jax.Array):
        embedding_table = np.array(embedding_table)
    else:
        embedding_table = np.array(embedding_table)
    
    print(f"Embedding 矩阵形状: {embedding_table.shape}")
    print(f"词汇表大小: {embedding_table.shape[0]}")
    print(f"Embedding 维度: {embedding_table.shape[1]}")
    
    # 计算每个 embedding 向量的 L2 范数
    # L2 范数 = sqrt(sum(x^2))
    l2_norms = np.linalg.norm(embedding_table, axis=1)
    
    print(f"\nL2 范数统计:")
    print(f"  最小值: {l2_norms.min():.6f}")
    print(f"  最大值: {l2_norms.max():.6f}")
    print(f"  平均值: {l2_norms.mean():.6f}")
    print(f"  中位数: {np.median(l2_norms):.6f}")
    print(f"  标准差: {l2_norms.std():.6f}")
    
    # 找出最小的10个和最大的10个 L2 范数对应的索引
    top_k = 10
    min_indices = np.argsort(l2_norms)[:top_k]  # 最小的10个
    max_indices = np.argsort(l2_norms)[-top_k:][::-1]  # 最大的10个（从大到小排序）
    
    # 初始化 tokenizer 以将 token ID 转换为单词
    tokenizer = None
    try:
        tokenizer = PaligemmaTokenizer()
    except Exception as e:
        print(f"\n警告: 无法加载 tokenizer 来显示单词: {e}")
    
    def get_token_info(token_id, tokenizer_obj):
        """获取 token 的信息
        
        Args:
            token_id: Token 的 ID
            tokenizer_obj: SentencePiece tokenizer 对象
            
        Returns:
            tuple: (token_piece, decoded_text)
            - token_piece: tokenizer 内部的原始 piece（可能包含特殊标记，如 ▁ 表示词首空格）
            - decoded_text: 解码后的可读文本（去除特殊标记，处理空格等）
            
        区别：
        - id_to_piece: 返回原始 piece，例如 "▁hello"（带词首空格标记）
        - decode: 返回可读文本，例如 "hello"（去除标记）
        """
        if tokenizer_obj is None:
            return None, None
        try:
            # id_to_piece: 返回 tokenizer 内部的原始 piece（可能包含特殊标记）
            token_piece = tokenizer_obj._tokenizer.id_to_piece(int(token_id))
            try:
                # decode: 将 token ID 解码为可读文本（去除特殊标记，处理空格等）
                decoded = tokenizer_obj._tokenizer.decode([int(token_id)])
            except:
                decoded = token_piece
            return token_piece, decoded
        except:
            return None, None
    
    # 输出最小的10个
    print(f"\n最小的 {top_k} 个 L2 范数对应的 token:")
    print("-" * 80)
    min_results = []
    for i, idx in enumerate(min_indices, 1):
        token_piece, decoded = get_token_info(idx, tokenizer)
        norm_value = l2_norms[idx]
        print(f"{i:2d}. Token ID: {idx:6d}, L2 范数: {norm_value:.6f}", end="")
        if token_piece:
            print(f", Token piece: {token_piece:20s}, 解码: {repr(decoded)}")
        else:
            print()
        min_results.append({
            "idx": int(idx),
            "norm": float(norm_value),
            "token_piece": token_piece,
            "decoded": decoded
        })
    
    # 输出最大的10个
    print(f"\n最大的 {top_k} 个 L2 范数对应的 token:")
    print("-" * 80)
    max_results = []
    for i, idx in enumerate(max_indices, 1):
        token_piece, decoded = get_token_info(idx, tokenizer)
        norm_value = l2_norms[idx]
        print(f"{i:2d}. Token ID: {idx:6d}, L2 范数: {norm_value:.6f}", end="")
        if token_piece:
            print(f", Token piece: {token_piece:20s}, 解码: {repr(decoded)}")
        else:
            print()
        max_results.append({
            "idx": int(idx),
            "norm": float(norm_value),
            "token_piece": token_piece,
            "decoded": decoded
        })
    
    return {
        "min": l2_norms.min(),
        "max": l2_norms.max(),
        "mean": l2_norms.mean(),
        "median": np.median(l2_norms),
        "std": l2_norms.std(),
        "min_top10": min_results,
        "max_top10": max_results,
    }


def get_word_embeddings_and_dot_product(word1: str, word2: str, params_path: str | None = None):
    """获取两个单词的 embedding 并计算它们的内积。
    
    Args:
        word1: 第一个单词（例如 "good"）
        word2: 第二个单词（例如 "bad"）
        params_path: 可选的检查点路径。如果为 None，则使用随机初始化的参数。
    """
    # 加载 embedding 表
    if params_path:
        from openpi.models.model import restore_params
        loaded_params = restore_params(params_path, restore_type=np.ndarray)
        flat_params = traverse_util.flatten_dict(loaded_params)
        embedding_key = None
        for key in flat_params:
            key_str = "/".join(key)
            if "embedder" in key_str and "input_embedding" in key_str:
                embedding_key = key
                break
        
        if embedding_key is None:
            for key in flat_params:
                if "input_embedding" in "/".join(key):
                    embedding_key = key
                    break
        
        if embedding_key is None:
            raise ValueError("找不到 embedding 参数。")
        
        embedding_table = np.array(flat_params[embedding_key])
    else:
        # 使用随机初始化的参数
        # 在 Flax Linen 中，获取参数的标准方式：
        # 1. 调用 init() 初始化模块，返回包含所有变量的字典
        # 2. 结构: {"params": {...}, "batch_stats": {...}, ...}
        # 3. 参数在 ["params"] 键下
        # 4. 对于独立模块: vars["params"]["param_name"]
        # 5. 对于嵌套模块: vars["params"]["module_name"]["param_name"]
        config = gemma.get_config("gemma_300m")
        rng = jax.random.PRNGKey(0)
        embedder = gemma.Embedder(
            vocab_size=gemma.PALIGEMMA_VOCAB_SIZE,
            embed_dim=config.width,
        )
        # init() 返回的字典结构: {"params": {"input_embedding": Array(...)}, ...}
        embedder_vars = embedder.init(rng, jnp.zeros((1, 1), dtype=jnp.int32), method=embedder.encode)
        # 直接访问参数（因为 Embedder 是独立模块，没有嵌套）
        embedding_table = np.array(embedder_vars["params"]["input_embedding"])
    
    # 转换为 numpy 数组
    if isinstance(embedding_table, jax.Array):
        embedding_table = np.array(embedding_table)
    else:
        embedding_table = np.array(embedding_table)
    
    # 初始化 tokenizer
    tokenizer = PaligemmaTokenizer()
    
    # 将单词编码为 token IDs
    # SentencePiece 可能将单词拆分成多个 tokens，我们取第一个 token
    tokens1 = tokenizer._tokenizer.encode(word1, add_bos=False)
    tokens2 = tokenizer._tokenizer.encode(word2, add_bos=False)
    print(tokens1)
    print(tokens2)
    if not tokens1:
        raise ValueError(f"无法将 '{word1}' 编码为 token")
    if not tokens2:
        raise ValueError(f"无法将 '{word2}' 编码为 token")
    
    token_id1 = tokens1[0]  # 取第一个 token
    token_id2 = tokens2[0]  # 取第一个 token
    
    # 获取 token 信息
    token_piece1 = tokenizer._tokenizer.id_to_piece(token_id1)
    token_piece2 = tokenizer._tokenizer.id_to_piece(token_id2)
    
    # 获取 embedding 向量
    embedding1 = embedding_table[token_id1]
    embedding2 = embedding_table[token_id2]
    
    # 计算内积（点积）
    dot_product = np.dot(embedding1, embedding2)
    
    # 计算 L2 范数
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    # 计算余弦相似度（归一化的内积）
    cosine_similarity = dot_product / (norm1 * norm2)
    
    # 输出结果
    print(f"\n{'='*80}")
    print(f"单词: '{word1}' 和 '{word2}' 的 Embedding 分析")
    print(f"{'='*80}")
    print(f"\n{word1}:")
    print(f"  Token ID: {token_id1}")
    print(f"  Token piece: {token_piece1}")
    print(f"  Embedding 形状: {embedding1.shape}")
    print(f"  Embedding 向量 (前10维): {embedding1[:10]}")
    print(f"  L2 范数: {norm1:.6f}")
    
    print(f"\n{word2}:")
    print(f"  Token ID: {token_id2}")
    print(f"  Token piece: {token_piece2}")
    print(f"  Embedding 形状: {embedding2.shape}")
    print(f"  Embedding 向量 (前10维): {embedding2[:10]}")
    print(f"  L2 范数: {norm2:.6f}")
    
    print(f"\n内积 (点积): {dot_product:.6f}")
    print(f"余弦相似度: {cosine_similarity:.6f}")
    print(f"{'='*80}\n")
    
    return {
        "word1": word1,
        "word2": word2,
        "token_id1": int(token_id1),
        "token_id2": int(token_id2),
        "token_piece1": token_piece1,
        "token_piece2": token_piece2,
        "embedding1": embedding1,
        "embedding2": embedding2,
        "dot_product": float(dot_product),
        "cosine_similarity": float(cosine_similarity),
        "norm1": float(norm1),
        "norm2": float(norm2),
    }


def analyze_semantic_space(word1: str, word2: str, params_path: str | None = None, 
                           comparison_words: list[str] | None = None):
    """深入分析两个词在语义空间中的位置，并与其他词对比。
    
    Args:
        word1: 第一个单词（例如 "good"）
        word2: 第二个单词（例如 "bad"）
        params_path: 可选的检查点路径
        comparison_words: 用于对比的其他单词列表
    """
    # 加载 embedding 表
    if params_path:
        from openpi.models.model import restore_params
        loaded_params = restore_params(params_path, restore_type=np.ndarray)
        flat_params = traverse_util.flatten_dict(loaded_params)
        embedding_key = None
        for key in flat_params:
            key_str = "/".join(key)
            if "embedder" in key_str and "input_embedding" in key_str:
                embedding_key = key
                break
        
        if embedding_key is None:
            for key in flat_params:
                if "input_embedding" in "/".join(key):
                    embedding_key = key
                    break
        
        if embedding_key is None:
            raise ValueError("找不到 embedding 参数。")
        
        embedding_table = np.array(flat_params[embedding_key])
    else:
        config = gemma.get_config("gemma_300m")
        rng = jax.random.PRNGKey(0)
        embedder = gemma.Embedder(
            vocab_size=gemma.PALIGEMMA_VOCAB_SIZE,
            embed_dim=config.width,
        )
        # 在 Flax Linen 中获取参数的标准方式
        embedder_vars = embedder.init(rng, jnp.zeros((1, 1), dtype=jnp.int32), method=embedder.encode)
        embedding_table = np.array(embedder_vars["params"]["input_embedding"])
    
    if isinstance(embedding_table, jax.Array):
        embedding_table = np.array(embedding_table)
    else:
        embedding_table = np.array(embedding_table)
    
    tokenizer = PaligemmaTokenizer()
    
    # 获取两个词的 token IDs 和 embeddings
    tokens1 = tokenizer._tokenizer.encode(word1, add_bos=False)
    tokens2 = tokenizer._tokenizer.encode(word2, add_bos=False)
    
    if not tokens1 or not tokens2:
        raise ValueError(f"无法编码单词")
    
    token_id1 = tokens1[0]
    token_id2 = tokens2[0]
    embedding1 = embedding_table[token_id1]
    embedding2 = embedding_table[token_id2]
    
    # 计算差异向量
    diff_vector = embedding1 - embedding2
    diff_norm = np.linalg.norm(diff_vector)
    relative_diff = diff_norm / (np.linalg.norm(embedding1) + np.linalg.norm(embedding2)) * 2
    
    print(f"\n{'='*80}")
    print(f"语义空间深度分析: '{word1}' vs '{word2}'")
    print(f"{'='*80}")
    print(f"\n差异向量分析:")
    print(f"  差异向量的 L2 范数: {diff_norm:.6f}")
    print(f"  相对差异 (相对于平均范数): {relative_diff:.6f}")
    print(f"  差异向量前10维: {diff_vector[:10]}")
    
    # 计算与所有其他词的相似度，找出最相似和最不相似的词
    print(f"\n正在计算与所有词的相似度...")
    norms = np.linalg.norm(embedding_table, axis=1)
    normalized_embeddings = embedding_table / (norms[:, np.newaxis] + 1e-8)
    
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    emb1_norm = embedding1 / norm1
    emb2_norm = embedding2 / norm2
    
    # 计算与所有词的余弦相似度
    similarities1 = np.dot(normalized_embeddings, emb1_norm)
    similarities2 = np.dot(normalized_embeddings, emb2_norm)
    
    # 找出与 word1 最相似和最不相似的词（排除自己）
    top_similar_to_1 = np.argsort(similarities1)[-20:][::-1]
    top_dissimilar_to_1 = np.argsort(similarities1)[:20]
    
    print(f"\n与 '{word1}' 最相似的20个词:")
    print("-" * 80)
    for i, idx in enumerate(top_similar_to_1[:20], 1):
        if idx == token_id1:
            continue
        token_piece = tokenizer._tokenizer.id_to_piece(int(idx))
        sim = similarities1[idx]
        print(f"{i:2d}. Token ID: {idx:6d}, 相似度: {sim:.6f}, Token: {token_piece}")
    
    print(f"\n与 '{word1}' 最不相似的20个词:")
    print("-" * 80)
    for i, idx in enumerate(top_dissimilar_to_1[:20], 1):
        token_piece = tokenizer._tokenizer.id_to_piece(int(idx))
        sim = similarities1[idx]
        print(f"{i:2d}. Token ID: {idx:6d}, 相似度: {sim:.6f}, Token: {token_piece}")
    
    print(f"\n与 '{word2}' 最相似的20个词:")
    print("-" * 80)
    for i, idx in enumerate(np.argsort(similarities2)[-20:][::-1], 1):
        if idx == token_id2:
            continue
        token_piece = tokenizer._tokenizer.id_to_piece(int(idx))
        sim = similarities2[idx]
        print(f"{i:2d}. Token ID: {idx:6d}, 相似度: {sim:.6f}, Token: {token_piece}")
    
    # 与特定词对比
    if comparison_words:
        print(f"\n与特定词的相似度对比:")
        print("-" * 80)
        for comp_word in comparison_words:
            try:
                comp_tokens = tokenizer._tokenizer.encode(comp_word, add_bos=False)
                if comp_tokens:
                    comp_id = comp_tokens[0]
                    comp_emb = embedding_table[comp_id]
                    comp_norm = np.linalg.norm(comp_emb)
                    comp_emb_norm = comp_emb / comp_norm
                    
                    sim_to_1 = np.dot(emb1_norm, comp_emb_norm)
                    sim_to_2 = np.dot(emb2_norm, comp_emb_norm)
                    
                    print(f"  '{comp_word}' (ID: {comp_id}):")
                    print(f"    与 '{word1}' 的相似度: {sim_to_1:.6f}")
                    print(f"    与 '{word2}' 的相似度: {sim_to_2:.6f}")
                    print(f"    差异: {abs(sim_to_1 - sim_to_2):.6f}")
            except:
                pass
    
    # 计算 good 和 bad 之间的相似度在所有词对中的排名
    cosine_sim = np.dot(emb1_norm, emb2_norm)
    print(f"\n{'='*80}")
    print(f"总结:")
    print(f"  '{word1}' 和 '{word2}' 的余弦相似度: {cosine_sim:.6f}")
    print(f"  这个相似度非常高（接近1.0），说明两个向量在方向上几乎相同")
    print(f"  相对差异仅为: {relative_diff:.6f}")
    if relative_diff < 0.05:
        print(f"  ⚠️  警告: 相对差异非常小，模型可能确实难以区分这两个词")
    print(f"{'='*80}\n")


def tokennize_text(text: str):
    tokenizer = PaligemmaTokenizer()
    tokens = tokenizer.tokenize(text, state=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13]))
    print(tokenizer._tokenizer.id_to_piece(2))
    return tokens

def dataloader_test():
    config = _config.get_config("pi05_aloha_pen_uncap")
    data_loader = _data_loader.create_data_loader(config)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    print(batch)

if __name__ == "__main__":
    # dataloader_test()
    import sys
    # 获取检查点路径（如果提供）
    params_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # 计算 "good" 和 "bad" 的 embedding 并计算内积
    get_word_embeddings_and_dot_product("yes", "no", params_path)
    
    # 深入分析语义空间
    comparison_words = ["excellent", "terrible", "great", "awful", "nice", "horrible", 
                       "perfect", "wrong", "correct", "yes", "no", "right", "left"]
    analyze_semantic_space("good", "bad", params_path, comparison_words)
    
    # 可选：计算 embedding 范数统计
    # compute_embedding_norms(params_path)