"""Hybrid retrieval: dense (vector) + BM25 (lexical), fused with weighted RRF.

Both indexes are per-collection: dense via Chroma, BM25 built lazily from the
collection's chunks and cached per collection. This is what makes per-session
isolation real at the retrieval layer.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from askmydocs.config import settings
from askmydocs.embed import DEFAULT_COLLECTION, get_all_chunks
from askmydocs.embed import search as dense_search
from askmydocs.schema import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
_RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class _Bm25Collection:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.bm25 = BM25Okapi([_tokenize(c.embedding_text) for c in chunks]) if chunks else None

    def search(self, query: str, k: int) -> list[Chunk]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:k]]


# Registry: collection name -> its BM25 index. Built on first use per collection.
_bm25_registry: dict[str, _Bm25Collection] = {}


def _get_bm25(collection: str) -> _Bm25Collection:
    if collection not in _bm25_registry:
        _bm25_registry[collection] = _Bm25Collection(get_all_chunks(collection))
    return _bm25_registry[collection]


def invalidate_bm25(collection: str) -> None:
    """Drop a collection's cached BM25 so it rebuilds (call after adding docs)."""
    _bm25_registry.pop(collection, None)


def bm25_search(query: str, k: int | None = None,
                collection: str = DEFAULT_COLLECTION) -> list[Chunk]:
    return _get_bm25(collection).search(query, k or settings.top_k)


def hybrid_search(query: str, k: int | None = None,
                  collection: str = DEFAULT_COLLECTION) -> list[Chunk]:
    k = k or settings.top_k
    fetch = settings.retrieve_n_before_rerank

    dense_hits = [c for c, _ in dense_search(query, k=fetch, collection=collection)]
    bm25_hits = bm25_search(query, k=fetch, collection=collection)

    fused: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for hits, weight in ((dense_hits, settings.dense_weight),
                         (bm25_hits, settings.bm25_weight)):
        for rank, c in enumerate(hits, start=1):
            fused[c.chunk_id] = fused.get(c.chunk_id, 0.0) + weight / (_RRF_K + rank)
            by_id[c.chunk_id] = c

    ranked_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)
    return [by_id[cid] for cid in ranked_ids[:k]]