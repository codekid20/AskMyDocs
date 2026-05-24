"""CI quality gate: fails the build if retrieval quality drops below the
thresholds in eval/thresholds.yaml. Runs the SHIPPING retriever (hybrid+rerank)
over the golden set. Retrieval-only — no Groq call — so CI is deterministic,
free, and fast.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from askmydocs.rerank import pipeline_search          # noqa: E402
from eval.metrics import mrr, recall_at_k             # noqa: E402

_THRESH = yaml.safe_load((_ROOT / "eval" / "thresholds.yaml").read_text())
_GOLDEN = [json.loads(l) for l in
           (_ROOT / "eval" / "golden.jsonl").read_text(encoding="utf-8").splitlines()
           if l.strip()]


def _run() -> tuple[float, float]:
    k = _THRESH["k"]
    r_total = m_total = 0.0
    for item in _GOLDEN:
        retrieved = pipeline_search(item["question"], k=k)
        r_total += recall_at_k(retrieved, item["relevant"], k)
        m_total += mrr(retrieved, item["relevant"])
    n = len(_GOLDEN)
    return r_total / n, m_total / n


def test_retrieval_quality_meets_threshold():
    recall, mrr_score = _run()
    print(f"\nRecall@{_THRESH['k']}={recall:.3f} (floor {_THRESH['recall_at_k']}) | "
          f"MRR={mrr_score:.3f} (floor {_THRESH['mrr']})")
    assert recall >= _THRESH["recall_at_k"], \
        f"Recall@{_THRESH['k']} {recall:.3f} below floor {_THRESH['recall_at_k']}"
    assert mrr_score >= _THRESH["mrr"], \
        f"MRR {mrr_score:.3f} below floor {_THRESH['mrr']}"