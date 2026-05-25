# Ask My Docs — RAG over Your PDFs

A domain-specific retrieval-augmented generation system that answers questions
over PDFs, with **citations grounded in real source sections**, **hybrid
retrieval + cross-encoder reranking**, an **eval-gated CI pipeline** that fails
the build if retrieval quality regresses, and a **full-stack web app** where you
upload your own documents and chat with them.

> Built from scratch as a study in RAG fundamentals — every component is added
> only after measuring that it beats a simpler baseline on a held-out eval set.

![Ask My Docs UI](docs/img/app-screenshot.png)

---

## What it does

- **Ask questions over a fixed corpus** of foundational NLP/RAG papers via a CLI, or
- **Upload your own PDFs** (a company policy, research papers, anything) through
  the web app and chat with them — *"how many leaves am I granted per year?"* —
  with every answer citing the exact source section.

Every answer is **grounded**: the model only uses retrieved chunks, cites them by
number, and those numbers resolve to real `doc, §section, page` references. It
cannot hallucinate a citation, and it refuses out-of-corpus questions rather than
answering from its own training knowledge.

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

## Web app (full-stack)

Beyond the CLI, the system ships as a web app: upload your own PDFs and chat with
them, with every answer grounded in citations.

```
React + Vite (5173)  ──HTTP──▶  FastAPI (8000)  ──▶  per-session RAG pipeline
   upload / chat UI              /session /upload /chat     (Chroma + BM25 + rerank + Groq)
```

- **FastAPI backend** (`src/askmydocs/api.py`) wraps the pipeline in three
  endpoints: create a session, upload a PDF (extract → chunk → embed into that
  session's collection), and chat (hybrid retrieval → rerank → grounded
  generation, scoped to the session).
- **Per-session isolation:** each browser session gets its own Chroma collection
  and BM25 index, so one user's uploaded documents are never retrievable by
  another's queries. (Verified: a query in session A's collection never returns
  session B's documents.)
- **Runtime ingestion** (`src/askmydocs/upload.py`): uploaded PDFs are ingested
  in-process — the same extract/chunk/embed pipeline as the CLI, scoped to a
  collection. The PDF is written to a temp file, ingested, then deleted; only the
  embedded chunks persist.
- **React + Vite frontend** (`frontend/`): drag-and-drop upload, a chat panel,
  citations rendered as footnotes under each answer, and clear/new-session controls.

### Running the web app

Two processes (two terminals):

```bash
# Terminal 1 — backend (from project root)
uv run uvicorn askmydocs.api:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm install        # first time only
npm run dev        # serves http://localhost:5173
```

Open http://localhost:5173, upload a PDF, and ask away. Interactive API docs are
at http://localhost:8000/docs.

---

## Eval-gated CI

Retrieval metrics (Recall@5, MRR) run on every push via GitHub Actions
(`.github/workflows/eval.yml`). The gate (`tests/test_eval_gate.py`) asserts
quality stays above the floors in `eval/thresholds.yaml`; if a change drops
retrieval below threshold, **the build fails.**

The gate tests retrieval only (no LLM call), so CI is deterministic, free, and
fast (~3 min cold, faster with model caching).

**Gate passing:**

![CI passing](docs/img/ci-green.png)

**Gate catching a forced regression (threshold temporarily raised):**

![CI failing](docs/img/ci-red.png)

---

## Tech stack

- **Retrieval:** Chroma (vector), rank-bm25 (lexical), weighted RRF fusion
- **Models (free / local):** BGE-small embeddings, BGE-reranker-base cross-encoder
- **Generation:** Groq (Llama-3.3-70B) — free tier, swappable behind one interface
- **Backend / API:** FastAPI + Uvicorn, per-session Chroma collections
- **Frontend:** React + Vite, framer-motion
- **Framework:** LangChain (text splitting; Chroma, HuggingFace, and Groq integrations)
- **Tooling:** uv (deps + lockfile), Typer (CLI), Pydantic (schema/config), pytest, ruff, loguru
- **PDF:** PyMuPDF4LLM (Markdown extraction, two-column reading order)

---

## Usage (CLI)

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

## Corpus

Five foundational papers ship as the default CLI corpus: *Attention Is All You
Need*, *BERT*, *Sentence-BERT*, *RAG*, and *Lost in the Middle*.

The PDFs themselves are **not committed** (see `data/raw/README.md` for the list
and sources). The pre-chunked text ships in `data/processed/chunks.jsonl`, so the
vector index — and CI — can be rebuilt without the original PDFs. The web app
lets users supply their own PDFs at runtime instead.

---

## Project layout

```
ask-my-docs/
├── src/askmydocs/
│   ├── schema.py        # Chunk + SourceRef — the citation backbone
│   ├── config.py        # typed settings (pydantic-settings)
│   ├── extract.py       # PDF -> cleaned, heading-tagged sections
│   ├── chunk.py         # sections -> retrieval-sized chunks
│   ├── embed.py         # chunks -> Chroma vector store + dense search (collection-aware)
│   ├── retrieval.py     # BM25 + dense, weighted RRF fusion (per-collection)
│   ├── rerank.py        # cross-encoder reranking + full pipeline
│   ├── generate.py      # grounded generation via Groq
│   ├── upload.py        # runtime PDF ingestion into a session collection
│   ├── api.py           # FastAPI backend (/session, /upload, /chat)
│   └── cli.py           # Typer CLI (ingest / chunk / embed / query / ask)
├── frontend/            # React + Vite web app
│   └── src/
│       ├── App.jsx      # upload + chat UI
│       └── api.js       # backend client
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
- **Per-session collections** for the web app — document isolation without auth.

---

## Future work

- Grow the golden set to ~25–30 questions; re-validate fusion weights at scale.
- Add an answer-quality eval layer (faithfulness, citation accuracy), gated separately.
- Persist the BM25 index (currently rebuilt in-memory — fine at this corpus size).
- Web app: session expiry + cleanup, a persistent session store, and authentication.
- Optional second implementation fully from scratch (no framework) for comparison.

---

## License

MIT (code). Source papers remain under their respective licenses and are not
redistributed in this repository.