# ADR 0001: Default retrieval = dense-dominant weighted hybrid + reranking

## Status
Accepted (2026-05-25)

## Context
Compared dense, hybrid (BM25+dense via RRF), and hybrid+rerank on a 10-question
golden set. Findings:
- Dense alone: R@5 0.90, MRR 0.658. Best recall.
- Weighted hybrid (dense:bm25 = 2:1): R@5 0.80, MRR 0.700. Best MRR.
- Hybrid + cross-encoder rerank: R@5 0.80, MRR 0.617.

Per-query: BM25 rescued exact-term queries (q1 "bi-encoder" 0.33->1.0).
Reranking rescued vocabulary-mismatch (q3 "lost in the middle"). BM25 noise on
common-term queries marginally hurt recall (q8 "BERT datasets").

## Decision
Default to weighted hybrid (dense_weight=2.0, bm25_weight=1.0) + reranking.
Prioritise ranking quality (MRR) since the generator sees the top-k; a correct
chunk ranked 2nd is as useful as 1st for grounding.

## Tradeoff / known limitation
- Tuned on only 10 questions — weight is a soft estimate, not validated at scale.
- BM25 can displace correct dense hits on common-term queries.
- Future: grow golden set to ~30 questions, re-tune, add a CI gate.

## Revisit when
Golden set exceeds ~25 questions, or corpus/domain changes.