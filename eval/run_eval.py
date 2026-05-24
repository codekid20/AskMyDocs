"""Run the golden set against the current retriever and report metrics.

Usage: uv run python eval/run_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))            # so `import eval` works
sys.path.insert(0, str(_ROOT / "src"))    # so `import askmydocs` works

from askmydocs.embed import search  # noqa: E402
from eval.metrics import mrr, recall_at_k  # noqa: E402
from askmydocs.embed import search as dense_search        # noqa: E402
from askmydocs.retrieval import hybrid_search              # noqa: E402
from eval.metrics import mrr, recall_at_k                  # noqa: E402

RETRIEVERS = {
    "dense": lambda q, k: [c for c, _ in dense_search(q, k=k)],
    "hybrid": lambda q, k: hybrid_search(q, k=k),
}

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"
K = 5


def load_golden() -> list[dict]:
    return [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(retriever: str = "dense") -> None:
    retrieve = RETRIEVERS[retriever]
    golden = load_golden()
    r_at_k, mrr_total, misses = 0.0, 0.0, []

    for item in golden:
        retrieved = retrieve(item["question"], K)
        r = recall_at_k(retrieved, item["relevant"], K)
        m = mrr(retrieved, item["relevant"])
        r_at_k += r
        mrr_total += m
        if r == 0.0:
            misses.append(item["id"])
        print(f"{'✓' if r else '✗'} {item['id']:4} mrr={m:.3f}  {item['question']}")

    n = len(golden)
    print("\n" + "=" * 50)
    print(f"[{retriever}]  Recall@{K}: {r_at_k / n:.3f}   MRR: {mrr_total / n:.3f}   ({n} queries)")
    if misses:
        print(f"Misses: {', '.join(misses)}")


if __name__ == "__main__":
    import sys
    retriever = sys.argv[1] if len(sys.argv) > 1 else "dense"
    main(retriever)