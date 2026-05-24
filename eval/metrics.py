"""Retrieval metrics, computed against section-level relevance labels.

A retrieved chunk 'is relevant' if its doc_id matches a labeled matcher AND
its heading contains the matcher's substring. We deliberately label at the
section level (not exact chunk_id) so the golden set survives re-chunking.
"""

from __future__ import annotations

from askmydocs.schema import Chunk


def is_relevant(chunk: Chunk, relevant: list[dict]) -> bool:
    for m in relevant:
        if chunk.source.doc_id == m["doc_id"] and \
           m["heading_contains"].lower() in chunk.source.heading.lower():
            return True
    return False


def recall_at_k(retrieved: list[Chunk], relevant: list[dict], k: int) -> float:
    """Did we surface AT LEAST ONE relevant chunk in the top k? (hit rate)
    For this golden set, 'found the right section at all' is the key question,
    so binary hit@k is the most honest headline metric."""
    return 1.0 if any(is_relevant(c, relevant) for c in retrieved[:k]) else 0.0


def mrr(retrieved: list[Chunk], relevant: list[dict]) -> float:
    """Reciprocal rank of the FIRST relevant hit. Rewards ranking the right
    chunk higher — exactly what reranking should improve."""
    for i, c in enumerate(retrieved, start=1):
        if is_relevant(c, relevant):
            return 1.0 / i
    return 0.0