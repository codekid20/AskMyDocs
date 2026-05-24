"""Stage 2 of ingestion: sections.jsonl -> chunks.jsonl.

Each section is split into retrieval-sized, overlapping chunks. Crucially, every
chunk inherits its parent section's heading + page, so a chunk taken from the
middle of '1 Introduction' still cites '§1 Introduction (p.1)'. That heading
propagation is what keeps citations precise after splitting.

We split on natural boundaries (paragraphs, then sentences, then words) via
LangChain's RecursiveCharacterTextSplitter, so chunks don't break mid-word.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from askmydocs.config import settings
from askmydocs.schema import Chunk, SourceRef


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # Try these separators in order; fall back to char split as last resort.
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,  # plain char length — matches our chunk_size units
    )


def chunk_sections(
    sections_path: Path | None = None, out_path: Path | None = None
) -> int:
    sections_path = sections_path or settings.sections_path
    out_path = out_path or settings.chunks_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not sections_path.exists():
        logger.error("No sections file at {} — run `askmydocs ingest` first.", sections_path)
        return 0

    splitter = _splitter()
    total_chunks = 0

    with sections_path.open(encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            ref = SourceRef(
                doc_id=rec["doc_id"],
                heading=rec["heading"],
                page=rec["page"],
            )
            pieces = splitter.split_text(rec["text"])
            for piece in pieces:
                piece = piece.strip()
                if len(piece) < settings.min_chunk_chars:
                    continue
                chunk = Chunk(text=piece, source=ref)
                fout.write(chunk.model_dump_json() + "\n")
                total_chunks += 1

    logger.success("Wrote {} chunks to {}", total_chunks, out_path)
    return total_chunks


if __name__ == "__main__":
    chunk_sections()