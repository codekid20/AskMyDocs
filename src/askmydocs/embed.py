"""Stage 3 of ingestion: chunks.jsonl -> persistent Chroma vector store.

Embeds each chunk with a free local BGE model and stores it with full citation
metadata. After this stage the corpus is searchable.

BGE detail: this model is asymmetric — documents are embedded as-is, but queries
must be prefixed with an instruction. We apply that prefix only at query time
(see `search`), which is why ingestion and search use different entrypoints.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from askmydocs.config import settings
from askmydocs.schema import Chunk, SourceRef

_COLLECTION = "papers"
# BGE query instruction — prepended to queries only, never to documents.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Module-level cache so we load the model once per process, not per call.
_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model {} (first run downloads it)...",
                    settings.embedding_model)
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},  # cosine-ready vectors
        )
    return _embeddings


def get_store() -> Chroma:
    """Open (or create) the persistent Chroma collection."""
    return Chroma(
        collection_name=_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )


def _load_chunks(chunks_path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            chunks.append(
                Chunk(
                    text=rec["text"],
                    source=SourceRef(**rec["source"]),
                    chunk_id=rec["chunk_id"],
                )
            )
    return chunks


def build_index(chunks_path: Path | None = None) -> int:
    chunks_path = chunks_path or settings.chunks_path
    if not chunks_path.exists():
        logger.error("No chunks at {} — run `askmydocs chunk` first.", chunks_path)
        return 0

    chunks = _load_chunks(chunks_path)
    store = get_store()

    # We embed `embedding_text` (heading + body) but store clean `text` plus
    # provenance in metadata, so citations resolve from real extracted fields.
    texts = [c.embedding_text for c in chunks]
    metadatas = [
        {
            "text": c.text,
            "chunk_id": c.chunk_id,
            "doc_id": c.source.doc_id,
            "heading": c.source.heading,
            "page": c.source.page,
        }
        for c in chunks
    ]
    ids = [c.chunk_id for c in chunks]

    logger.info("Embedding {} chunks...", len(chunks))
    store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    logger.success("Indexed {} chunks into {}", len(chunks), settings.chroma_dir)
    return len(chunks)


def search(query: str, k: int | None = None) -> list[tuple[Chunk, float]]:
    """Semantic search. Returns (chunk, distance) pairs, best first."""
    k = k or settings.top_k
    store = get_store()
    prefixed = _QUERY_PREFIX + query
    results = store.similarity_search_with_score(prefixed, k=k)

    out: list[tuple[Chunk, float]] = []
    for doc, score in results:
        m = doc.metadata
        chunk = Chunk(
            text=m["text"],
            source=SourceRef(doc_id=m["doc_id"], heading=m["heading"], page=m["page"]),
            chunk_id=m["chunk_id"],
        )
        out.append((chunk, score))
    return out


if __name__ == "__main__":
    build_index()