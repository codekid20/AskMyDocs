# Ask My Docs — RAG over Technical Papers

A domain-specific retrieval-augmented generation system that answers questions
over a corpus of foundational NLP/RAG papers, with **citations grounded in real
source sections**, **hybrid retrieval + cross-encoder reranking**, and an
**eval-gated CI pipeline** that fails the build if retrieval quality regresses.

> Built from scratch as a study in RAG fundamentals — every component is added
> only after measuring that it beats a simpler baseline on a held-out eval set.

---

## Corpus

Five foundational papers: *Attention Is All You Need*, *BERT*, *Sentence-BERT*,
*RAG*, and *Lost in the Middle*.

The PDFs themselves are **not committed** (see `data/raw/README.md` for the list
and sources). The pre-chunked text ships in `data/processed/chunks.jsonl`, so the
vector index — and CI — can be rebuilt without the original PDFs.

---

## Results

Retrieval quality on a 10-question golden set, measured per retrieval strategy on
the **same** questions and labels:

| Strategy                       | Recall@5 | MRR       |
|--------------------------------|----------|-----------|
| Dense (vector only)            | 0.90     | 0.658     |
| Weighted hybrid (BM25 + dense) | 0.80     | **0.700** |
| Hybrid + cross-encoder rerank  | 0.80     | 0.617     |

**What the numbers say (the interesting part):** no single strategy dominates.
Hybrid wins on ranking quality (MRR) because BM25 rescues exact-term queries —
e.g. *"what is a bi-encoder?"* went from MRR 0.33 (dense ranked the right chunk
3rd) to 1.00 once BM25 matched the exact term. Cross-encoder reranking rescued a
vocabulary-mismatch case (*"lost in the middle"*) that neither dense nor BM25
found alone. But BM25 also injects noise on common-term queries, which is why
dense retains the best raw Recall@5. The shipping default is **dense-dominant
weighted hybrid + reranking** — see
[ADR-0001](docs/decisions/0001-retrieval-config.md).

> **Caveat, stated honestly:** the golden set is 10 questions, so fusion weights
> are tuned estimates, not large-sample-validated. Growing the set is the top
> item in Future Work.

---

## Architecture

```
PDFs
  │
  ▼
extract.py        PyMuPDF4LLM → cleaned, heading-tagged sections (two-column aware)
  │               boilerplate (References/Acks) filtered
  ▼
chunk.py          sections → ≤800-char overlapping chunks
  │               every chunk inherits heading + page (precise citations)
  ▼
embed.py          BGE-small embeddings → persistent Chroma store (+ provenance metadata)
  │
  ▼
retrieval.py   ┌─ dense (Chroma + BGE) ──┐
               │                          ├─ weighted RRF fusion
               └─ BM25 (rank-bm25) ──────┘
  │
  ▼
rerank.py         BGE-reranker-base cross-encoder re-scores candidate pool
  │
  ▼
generate.py       Groq (Llama-3.3-70B) answers using ONLY retrieved chunks
  │
  ▼
Answer + resolved citations  (doc, §section, page)
```

Each stage is independently runnable and writes inspectable JSONL between steps,
so you can debug any stage in isolation.

1. **Extract** (`extract.py`) — PyMuPDF4LLM parses PDFs into cleaned,
   heading-tagged sections in correct reading order (handles two-column academic
   layout). Boilerplate sections (References, Acknowledgments) are filtered.
2. **Chunk** (`chunk.py`) — sections split into ≤800-char overlapping chunks;
   every chunk inherits its section heading + page, so citations stay precise
   after splitting.
3. **Embed** (`embed.py`) — chunks embedded with `BAAI/bge-small-en-v1.5` into a
   persistent Chroma store, with full provenance in metadata.
4. **Retrieve** (`retrieval.py`) — dense + BM25, fused via weighted Reciprocal
   Rank Fusion (RRF).
5. **Rerank** (`rerank.py`) — `BAAI/bge-reranker-base` cross-encoder re-scores
   the candidate pool for precision.
6. **Generate** (`generate.py`) — Groq (Llama-3.3-70B) answers using only the
   retrieved chunks.

---

## Citations are a data-model guarantee, not a prompt trick

The LLM **never invents a citation.** Retrieved chunks are handed to the model
numbered `[1] [2] [3]…`; the model answers using only those chunks and cites the
numbers it used. We then resolve those integers back to real `SourceRef`s
(`doc, §section, page`) **in code**.

Because the model only ever emits integers we gave it, a citation cannot be
hallucinated — at worst it cites the wrong (but real) source. Grounding holds
even adversarially: asked an out-of-corpus question (e.g. *"Who won the 2024
World Series?"*), the system reports that the sources don't cover it rather than
answering from the LLM's own training knowledge.

---

## Eval-gated CI

Retrieval metrics (Recall@5, MRR) run on every push via GitHub Actions
(`.github/workflows/eval.yml`). The gate (`tests/test_eval_gate.py`) asserts
quality stays above the floors in `eval/thresholds.yaml`; if a change drops
retrieval below threshold, **the build fails.**

The gate tests retrieval only (no LLM call), so CI is deterministic, free, and
fast (~3 min cold, faster with model caching).

<!-- Replace the lines below with your screenshots:
     drag an image into a GitHub issue comment, copy the generated URL, and use it here. -->

**Gate passing:**

![CI passing](docs/img/ci-green.png)

**Gate catching a forced regression (threshold temporarily raised):**

![CI failing](docs/img/ci-red.png)

---

## Tech stack

- **Retrieval:** Chroma (vector), rank-bm25 (lexical), weighted RRF fusion
- **Models (free / local):** BGE-small embeddings, BGE-reranker-base cross-encoder
- **Generation:** Groq (Llama-3.3-70B) — free tier, swappable behind one interface
- **Framework:** LangChain (text splitting; Chroma, HuggingFace, and Groq integrations)
- **Tooling:** uv (deps + lockfile), Typer (CLI), Pydantic (schema/config), pytest, ruff, loguru
- **PDF:** PyMuPDF4LLM (Markdown extraction, two-column reading order)

---

## Usage

```bash
# 1. Install (reproducible via uv.lock)
uv sync

# 2. For generation, add your free Groq key to .env
#    GROQ_API_KEY=gsk_...

# 3. Build the index (PDFs → sections → chunks → vectors)
uv run askmydocs ingest      # PDFs in data/raw/ -> data/processed/sections.jsonl
uv run askmydocs chunk       # sections          -> data/processed/chunks.jsonl
uv run askmydocs embed       # chunks            -> Chroma vector store

# 4. Query
uv run askmydocs query "what is a bi-encoder?"     # retrieval only (ranked chunks + citations)
uv run askmydocs ask   "what is a bi-encoder?"     # grounded answer with citations

# 5. Evaluate / test
uv run python eval/run_eval.py dense               # eval a retrieval strategy: dense | hybrid | rerank
uv run python eval/run_eval.py hybrid
uv run python eval/run_eval.py rerank
uv run pytest                                      # all tests, including the CI quality gate
```

---

## Project layout

```
ask-my-docs/
├── src/askmydocs/
│   ├── schema.py        # Chunk + SourceRef — the citation backbone
│   ├── config.py        # typed settings (pydantic-settings)
│   ├── extract.py       # PDF -> cleaned, heading-tagged sections
│   ├── chunk.py         # sections -> retrieval-sized chunks
│   ├── embed.py         # chunks -> Chroma vector store + dense search
│   ├── retrieval.py     # BM25 + dense, weighted RRF fusion
│   ├── rerank.py        # cross-encoder reranking + full pipeline
│   ├── generate.py      # grounded generation via Groq
│   └── cli.py           # Typer CLI (ingest / chunk / embed / query / ask)
├── eval/
│   ├── golden.jsonl     # question -> relevant-section labels
│   ├── metrics.py       # Recall@k, MRR
│   ├── run_eval.py      # compare retrieval strategies
│   └── thresholds.yaml  # CI quality floors
├── tests/               # unit tests + the eval quality gate
├── docs/decisions/      # architecture decision records (ADRs)
├── data/
│   ├── raw/             # source PDFs (gitignored; README lists them)
│   └── processed/       # chunks.jsonl (shipped) + Chroma store (gitignored)
└── .github/workflows/   # eval-gate CI
```

---

## Design decisions

Recorded as ADRs in `docs/decisions/`. Highlights:

- **No RAG framework for the core logic** — chunking, fusion, reranking, and
  citation resolution are owned directly so the fundamentals stay legible.
  LangChain is used only for well-solved primitives (text splitting, store/model
  integrations).
- **Section-level citation labels** in the golden set — robust to re-chunking
  (changing chunk size doesn't break the eval).
- **PyMuPDF4LLM over pypdf** — correct two-column reading order on academic PDFs.
- **Dense-dominant weighted hybrid + rerank** as the shipping default — see
  [ADR-0001](docs/decisions/0001-retrieval-config.md).

---

## Future work

- Grow the golden set to ~25–30 questions; re-validate fusion weights at scale.
- Add an answer-quality eval layer (faithfulness, citation accuracy), gated separately.
- Persist the BM25 index (currently rebuilt in-memory — fine at this corpus size).
- Optional second implementation fully from scratch (no framework) for comparison.

---

## License

MIT (code). Source papers remain under their respective licenses and are not
redistributed in this repository.
