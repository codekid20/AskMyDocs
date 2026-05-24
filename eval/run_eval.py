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

from askmydocs.embed import search          # noqa: E402
from eval.metrics import mrr, recall_at_k    # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"
K = 5


def load_golden() -> list[dict]:
    return [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    golden = load_golden()
    r_at_k, mrr_total = 0.0, 0.0
    misses = []

    for item in golden:
        retrieved = [c for c, _ in search(item["question"], k=K)]
        r = recall_at_k(retrieved, item["relevant"], K)
        m = mrr(retrieved, item["relevant"])
        r_at_k += r
        mrr_total += m
        if r == 0.0:
            misses.append(item["id"])
        flag = "✓" if r else "✗"
        print(f"{flag} {item['id']:4} mrr={m:.3f}  {item['question']}")

    n = len(golden)
    print("\n" + "=" * 50)
    print(f"Recall@{K}: {r_at_k / n:.3f}   MRR: {mrr_total / n:.3f}   ({n} queries)")
    if misses:
        print(f"Misses (no relevant chunk in top {K}): {', '.join(misses)}")


if __name__ == "__main__":
    main()