"""用 RRF 合并双库结果，再用本地模型重排。"""

from functools import lru_cache

from langchain_core.documents import Document

from JLU_agent.config import chroma_config as config


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Document, float]]], k: int = 60
) -> list[Document]:
    """按各库中的名次融合文档；原始检索分数不参与计算。"""
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, (document, _) in enumerate(ranked_list, start=1):
            doc_id = document.id
            documents.setdefault(doc_id, document)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return [documents[doc_id] for doc_id in ranked_ids]


@lru_cache(maxsize=1)
def get_cross_encoder():
    """首次检索时加载模型，后续问答复用同一个实例。"""
    import torch
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        model_name_or_path=config.RERANK_MODEL,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def cross_encoder_rerank(
    query: str, documents: list[Document], top_k: int = 3
) -> list[tuple[Document, float]]:
    """让交叉编码模型对问题和候选正文打分，返回得分最高的文档。"""
    if not documents:
        return []

    pairs = [(query, document.page_content) for document in documents]
    scores = get_cross_encoder().predict(pairs)
    results = [(document, float(score)) for document, score in zip(documents, scores)]
    results.sort(key=lambda item: item[1], reverse=True)
    return results[:top_k]


def rerank(
    query: str,
    vector_results: list[tuple[Document, float]],
    bm25_results: list[tuple[Document, float]],
) -> list[tuple[Document, float]]:
    """融合两库候选，再用交叉编码分数确定最终顺序。"""
    documents = reciprocal_rank_fusion([vector_results, bm25_results])
    return cross_encoder_rerank(query, documents)
