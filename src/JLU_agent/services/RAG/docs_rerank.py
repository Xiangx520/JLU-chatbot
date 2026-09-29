# 重排序
from typing import Dict

from langchain_core.documents import Document





def min_max_normalize(scores_dict: Dict[str, float]):
    """对字典值进行 Min-Max 归一化"""
    if not scores_dict:
        return {}
    scores = list(scores_dict.values())
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return {k: 0.5 for k in scores_dict}
    return {k: (v - min_s) / (max_s - min_s) for k, v in scores_dict.items()}



# rerank策略：得分加权求和
def weighted_sum_fusion(
    results_list: list[dict[str, float]],
    weights: list[float],
    normalized: list[bool]
) -> list[tuple[str, float]]:

    """
    加权求和融合多个检索结果
    results_list: [检索器1的{doc_id: score}, 检索器2的{doc_id: score}, ...]
    weights: 对应权重，应总和为1
    normalized: 对应结果集是否需要归一化
    返回: [(doc_id, final_score), ...] 按分数降序
    """

    assert len(results_list) == len(weights)
    assert len(normalized) == len(weights)

    # 1. 收集所有文档 ID （去重）
    all_doc_ids = set()
    for results in results_list:
        all_doc_ids.update(results.keys())

    # 2. 可选：归一化每个检索器的分数
    normalized_results = []
    for i, results in enumerate(results_list):
        normalized_results.append(
            min_max_normalize(results) if normalized[i] else results
        )

    # 3. 加权求和
    final_scores = {}
    for doc_id in all_doc_ids:
        total = 0.0
        for i, norm_results in enumerate(normalized_results):
            score = norm_results.get(doc_id, 0.0)  # 未出现得0分
            total += weights[i] * score
        final_scores[doc_id] = total

    # 4. 重新排序并返回
    sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs




def rerank(
        doc1: list[tuple[Document, float]],
        doc2: list[tuple[Document, float]],
) -> list[tuple[Document, float]]:

    raw_docs = {}           # 文档集合 相同文档去重
    vector_rs = {}          # 分数集合
    bm25_rs = {}

    for doc, score in doc1:
        raw_docs[doc.id] = doc
        # Chroma 返回距离，越小越相关；取反后统一为越大越相关。
        vector_rs[doc.id] = -float(score)

    for doc, score in doc2:
        raw_docs[doc.id] = doc
        bm25_rs[doc.id] = float(score)

    # 返回结果是list[tuple[str, float]] 前者是文档id 后者是对应reranks
    ranked_results = weighted_sum_fusion([vector_rs, bm25_rs], [0.5, 0.5], [True, True])

    # 把结果处理成文档加得分的形式
    reranked_docs = []
    for doc_id, score in ranked_results:
        reranked_docs.append((raw_docs[doc_id], score))

    return reranked_docs
