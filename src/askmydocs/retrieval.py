"""Hybrid retrieval: dense (vector) + BM25 (lexical), fused with Reciprocal
Rank Fusion.

Dense search matches *meaning*; BM25 matches *exact terms*. Each alone has a
blind spot (dense ranked 'encoder' chunks over 'bi-encoder'; BM25 misses
paraphrases). RRF fuses their rankings using only positions, so no score-scale
normalization is needed: a chunk's fused score is sum(1/(k_rrf + rank)) across
retrievers. Chunks both retrievers like rise to the top.

BM25 index is built in-memory from chunks.jsonl on first use — at this corpus
size (~400 chunks) that's milliseconds, so persistence would be premature.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from askmydocs.config import settings
from askmydocs.embed import search as dense_search
from askmydocs.schema import Chunk, SourceRef

_TOKEN = re.compile(r"[a-z0-9]+")
_RRF_K = 60  # standard RRF constant; dampens the weight of very top ranks


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class _Bm25Index:
    """Lazily-built in-memory BM25 over all chunks."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None

    def _ensure(self) -> None:
        if self.bm25 is not None:
            return
        path = settings.chunks_path
        chunks: list[Chunk] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                chunks.append(
                    Chunk(text=rec["text"],
                          source=SourceRef(**rec["source"]),
                          chunk_id=rec["chunk_id"])
                )
        self.chunks = chunks
        # tokenize the same text we embed (heading + body) for parity with dense
        corpus = [_tokenize(c.embedding_text) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int) -> list[Chunk]:
        self._ensure()
        assert self.bm25 is not None
        scores = self.bm25.get_scores(_tokenize(query))
        # top-k indices by score, descending
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:k]]


_bm25 = _Bm25Index()


def bm25_search(query: str, k: int | None = None) -> list[Chunk]:
    return _bm25.search(query, k or settings.top_k)


def hybrid_search(query: str, k: int | None = None) -> list[Chunk]:
    """Dense + BM25, fused with RRF. We over-fetch from each retriever (more
    than k) so fusion has enough candidates to work with, then return top-k."""
    k = k or settings.top_k
    fetch = settings.retrieve_n_before_rerank  # reuse the over-fetch setting (20)

    dense_hits = [c for c, _ in dense_search(query, k=fetch)]
    bm25_hits = bm25_search(query, k=fetch)

    # Weighted RRF: each retriever's contribution is scaled by its weight, so
    # we can trust dense more than BM25 (BM25 adds exact-term wins like q1 but
    # injects common-term noise like q8 if given equal say).
    fused: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for hits, weight in ((dense_hits, settings.dense_weight), (bm25_hits, settings.bm25_weight)):
        for rank, c in enumerate(hits, start=1):
            fused[c.chunk_id] = fused.get(c.chunk_id, 0.0) + weight / (_RRF_K + rank)
            by_id[c.chunk_id] = c

    ranked_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)
    return [by_id[cid] for cid in ranked_ids[:k]]