"""Generation with grounded citations.

The citation guarantee: we hand the LLM numbered chunks [1], [2], ... and ask
it to answer using ONLY those chunks, citing the numbers it used. The model
never sees or emits a source name — only integers we gave it. We then resolve
those integers back to real SourceRefs in our own code. A citation therefore
cannot be hallucinated; at worst the model cites the wrong (but real) chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_groq import ChatGroq

from askmydocs.config import settings
from askmydocs.rerank import pipeline_search
from askmydocs.schema import Chunk

_SYSTEM = (
    "You are a precise research assistant answering questions about a set of "
    "technical papers. Answer ONLY using the numbered sources provided. "
    "After each claim, cite the source number(s) that support it, like [1] or "
    "[2][3]. If the sources do not contain the answer, say so plainly — do not "
    "use outside knowledge. Be concise and factual."
)


@dataclass
class Answer:
    text: str                       # the LLM's answer, with [n] markers intact
    cited: list[tuple[int, Chunk]]  # (number, chunk) for sources actually cited
    all_sources: list[Chunk]        # every chunk we offered, in order


def _build_context(chunks: list[Chunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        # The model sees the number + the citable text. It never sees doc names,
        # so it cannot invent or mangle a source — it can only reference [i].
        blocks.append(f"[{i}] {c.text}")
    return "\n\n".join(blocks)


def _llm() -> ChatGroq:
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0,  # deterministic: same question -> same answer, good for eval
    )


def answer_question(question: str, k: int | None = None,
                    collection: str = "papers") -> Answer:
    chunks = pipeline_search(question, k=k or settings.top_k, collection=collection)
    if not chunks:
        return Answer(text="No relevant sources found.", cited=[], all_sources=[])

    context = _build_context(chunks)
    prompt = f"Sources:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"

    resp = _llm().invoke([("system", _SYSTEM), ("human", prompt)])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)

    # Resolve which [n] markers the model actually used -> real chunks.
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)
                   if 1 <= int(n) <= len(chunks)})
    cited = [(n, chunks[n - 1]) for n in used]

    return Answer(text=text, cited=cited, all_sources=chunks)