"""Cross-encoder reranking — the precision layer on top of retrieval.

A bi-encoder (our embeddings) encodes query and chunk SEPARATELY then compares
vectors — fast, but it never sees them together. A cross-encoder feeds
[query, chunk] jointly through one transformer and outputs a single relevance
score. Far more accurate at judging 'does this chunk actually answer this
query', but too slow to run over the whole corpus — so we only run it over the
candidate pool that retrieval already narrowed down.

Architecture: hybrid retrieval casts a wide net (fetch ~20) -> cross-encoder
re-scores those candidates -> return the top k. This is the standard
retrieve-then-rerank pattern.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from askmydocs.config import settings
from askmydocs.retrieval import hybrid_search
from askmydocs.schema import Chunk

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        from loguru import logger
        logger.info("Loading reranker {} (first run downloads it)...",
                    settings.reranker_model)
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def rerank(query: str, chunks: list[Chunk], k: int | None = None) -> list[Chunk]:
    """Re-score candidates by true query-chunk relevance, return top k."""
    k = k or settings.top_k
    if not chunks:
        return []
    model = get_reranker()
    # The cross-encoder scores (query, passage) pairs. We score against the
    # same heading+body text we retrieved/embedded on, for consistency.
    pairs = [(query, c.embedding_text) for c in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:k]]


def pipeline_search(query: str, k: int | None = None) -> list[Chunk]:
    """Full retrieval pipeline: hybrid (wide net) -> cross-encoder rerank -> top k."""
    k = k or settings.top_k
    # Over-fetch a generous candidate pool for the reranker to sort.
    candidates = hybrid_search(query, k=settings.retrieve_n_before_rerank)
    return rerank(query, candidates, k=k)