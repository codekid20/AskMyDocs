# tests/test_extract.py
from pathlib import Path

from askmydocs.extract import _normalize, extract_sections

RAG_PDF = Path("data/raw/RAG.pdf")


def test_normalize_fixes_ligatures_and_hyphenation():
    assert _normalize("ﬁne-tuned") == "fine-tuned"
    assert _normalize("knowl-\nedge") == "knowledge"
    assert _normalize("**==> picture [10 x 10] omitted <==**") == ""


def test_extract_finds_sections_with_provenance():
    secs = extract_sections(RAG_PDF)
    assert len(secs) > 10                          # the paper has many sections
    headings = [s.heading for s in secs]
    assert any("Retriever" in h for h in headings) # known section is present
    # every section carries provenance needed for a citation
    for s in secs:
        assert s.doc_id == "RAG"
        assert s.page >= 1
        assert s.text                              # no empty sections slipped through