"""Embedding + dense search over a Chroma collection.

Embeds chunks with a free local BGE model and stores them with citation
metadata. Collections are the unit of isolation: the default "papers" holds the
pre-ingested corpus (CLI/eval), while the API creates one collection per user
session so uploads never cross-contaminate.

BGE detail: this model is asymmetric — documents are embedded as-is, but queries
must be prefixed with an instruction. We apply that prefix only at query time.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from askmydocs.config import settings
from askmydocs.schema import Chunk, SourceRef

DEFAULT_COLLECTION = "papers"
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """The embedding model is GLOBAL — one instance, shared across all
    collections/sessions. Only the data is per-collection, not the model."""
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model {} (first run downloads it)...",
                    settings.embedding_model)
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_store(collection: str = DEFAULT_COLLECTION) -> Chroma:
    """Open (or create) a named Chroma collection. Same persist dir, separate
    collections — Chroma keeps them isolated."""
    return Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )


def _load_chunks(chunks_path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            chunks.append(
                Chunk(text=rec["text"], source=SourceRef(**rec["source"]),
                      chunk_id=rec["chunk_id"])
            )
    return chunks


def index_chunks(chunks: list[Chunk], collection: str = DEFAULT_COLLECTION) -> int:
    """Embed in-memory chunks into a collection. This is the core indexing
    primitive used by both the CLI (via build_index) and runtime uploads."""
    if not chunks:
        return 0
    store = get_store(collection)
    texts = [c.embedding_text for c in chunks]
    metadatas = [
        {"text": c.text, "chunk_id": c.chunk_id, "doc_id": c.source.doc_id,
         "heading": c.source.heading, "page": c.source.page}
        for c in chunks
    ]
    ids = [c.chunk_id for c in chunks]
    logger.info("Embedding {} chunks into '{}'...", len(chunks), collection)
    store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    logger.success("Indexed {} chunks into collection '{}'", len(chunks), collection)
    return len(chunks)


def build_index(chunks_path: Path | None = None,
                collection: str = DEFAULT_COLLECTION) -> int:
    """CLI entrypoint: read chunks.jsonl from disk and index them."""
    chunks_path = chunks_path or settings.chunks_path
    if not chunks_path.exists():
        logger.error("No chunks at {} — run `askmydocs chunk` first.", chunks_path)
        return 0
    return index_chunks(_load_chunks(chunks_path), collection=collection)


def _doc_to_chunk(m: dict) -> Chunk:
    return Chunk(
        text=m["text"],
        source=SourceRef(doc_id=m["doc_id"], heading=m["heading"], page=m["page"]),
        chunk_id=m["chunk_id"],
    )


def get_all_chunks(collection: str = DEFAULT_COLLECTION) -> list[Chunk]:
    """Pull every chunk back out of a collection (used to build per-collection
    BM25). Chroma is the single source of truth for a collection's contents."""
    store = get_store(collection)
    got = store.get(include=["metadatas"])  # all docs in the collection
    metas = got.get("metadatas") or []
    return [_doc_to_chunk(m) for m in metas]


def search(query: str, k: int | None = None,
           collection: str = DEFAULT_COLLECTION) -> list[tuple[Chunk, float]]:
    """Dense semantic search within a collection. Returns (chunk, distance)."""
    k = k or settings.top_k
    store = get_store(collection)
    results = store.similarity_search_with_score(_QUERY_PREFIX + query, k=k)
    return [(_doc_to_chunk(doc.metadata), score) for doc, score in results]


if __name__ == "__main__":
    build_index()