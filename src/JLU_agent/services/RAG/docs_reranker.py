"""对 Milvus 已融合的候选使用本地交叉编码模型重排。"""

from functools import lru_cache

from langchain_core.documents import Document

from JLU_agent.config import rag_config as config


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
