"""The citation backbone: Chunk + SourceRef.

A SourceRef is a precise, deterministic pointer back into a source document.
Every field here is something we extracted ourselves during ingestion — never
anything an LLM produces. That is the core guarantee: at answer time the model
only ever references chunk IDs we hand it, and we resolve those IDs back to
these refs in our own code, so a citation physically cannot be hallucinated.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    """Where a chunk came from. Drives the human-readable citation."""

    doc_id: str                 # "RAG" — the paper
    heading: str = ""           # "2.2 Retriever: DPR" — section anchor ("" if preamble)
    page: int = 1               # 1-indexed page the chunk's section starts on

    def display(self) -> str:
        """How this citation renders to the user."""
        if self.heading:
            return f"{self.doc_id}, §{self.heading} (p.{self.page})"
        return f"{self.doc_id} (p.{self.page})"


class Chunk(BaseModel):
    """The atomic unit of retrieval: text + provenance + a stable id."""

    text: str
    source: SourceRef
    chunk_id: str = Field(default="")

    def model_post_init(self, __context) -> None:
        # Derive a stable id from provenance + content. Stable ids make
        # re-ingestion idempotent and let us pass chunk references through
        # retrieve -> rerank -> generate without collisions.
        if not self.chunk_id:
            basis = f"{self.source.doc_id}|{self.source.heading}|{self.source.page}|{self.text}"
            object.__setattr__(
                self, "chunk_id", hashlib.sha256(basis.encode()).hexdigest()[:16]
            )

    @property
    def embedding_text(self) -> str:
        """What we actually embed: heading + text. Prepending the section
        heading gives the embedder context ('this is under Retriever: DPR'),
        which measurably improves retrieval. But we cite only `text`, so the
        user can always find the cited words verbatim in the source."""
        if self.source.heading:
            return f"{self.source.heading}\n\n{self.text}"
        return self.text