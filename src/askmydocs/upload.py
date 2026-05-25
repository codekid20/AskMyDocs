"""Runtime PDF ingestion for the API: an uploaded PDF -> indexed into a
session's collection, ready to query. Reuses the same extract + chunk + embed
pipeline as the CLI, but in-process and scoped to one collection."""

from __future__ import annotations

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from askmydocs.config import settings
from askmydocs.embed import index_chunks
from askmydocs.extract import extract_sections
from askmydocs.retrieval import invalidate_bm25
from askmydocs.schema import Chunk, SourceRef


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def ingest_pdf(pdf_path: Path, collection: str) -> int:
    """Extract -> chunk -> embed a single PDF into `collection`. Returns the
    number of chunks indexed. Safe to call repeatedly for the same session to
    add more documents."""
    sections = extract_sections(pdf_path)
    splitter = _splitter()

    chunks: list[Chunk] = []
    for s in sections:
        ref = SourceRef(doc_id=s.doc_id, heading=s.heading, page=s.page)
        for piece in splitter.split_text(s.text):
            piece = piece.strip()
            if len(piece) < settings.min_chunk_chars:
                continue
            chunks.append(Chunk(text=piece, source=ref))

    n = index_chunks(chunks, collection=collection)
    invalidate_bm25(collection)  # force BM25 rebuild to include the new chunks
    logger.success("Ingested {} ({} chunks) into '{}'", pdf_path.name, n, collection)
    return n