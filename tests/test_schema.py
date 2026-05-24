from askmydocs.schema import Chunk, SourceRef


def test_chunk_id_is_deterministic():
    ref = SourceRef(doc_id="RAG", heading="2.2 Retriever: DPR", page=3)
    a = Chunk(text="DPR uses a bi-encoder.", source=ref)
    b = Chunk(text="DPR uses a bi-encoder.", source=ref)
    assert a.chunk_id == b.chunk_id
    assert len(a.chunk_id) == 16


def test_embedding_text_prepends_heading_but_text_stays_clean():
    ref = SourceRef(doc_id="RAG", heading="2.2 Retriever: DPR", page=3)
    c = Chunk(text="DPR uses a bi-encoder.", source=ref)
    assert c.embedding_text == "2.2 Retriever: DPR\n\nDPR uses a bi-encoder."
    assert c.text == "DPR uses a bi-encoder."  # citation text untouched


def test_citation_display_with_and_without_heading():
    assert SourceRef(doc_id="RAG", heading="2.2 Retriever: DPR", page=3).display() \
        == "RAG, §2.2 Retriever: DPR (p.3)"
    assert SourceRef(doc_id="RAG", page=1).display() == "RAG (p.1)"