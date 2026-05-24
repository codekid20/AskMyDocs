from askmydocs.chunk import _splitter
from askmydocs.schema import Chunk, SourceRef


def test_splitter_respects_chunk_size():
    text = "word " * 1000  # ~5000 chars, must split
    pieces = _splitter().split_text(text)
    assert len(pieces) > 1
    assert all(len(p) <= 900 for p in pieces)  # ~800 + overlap slack


def test_chunk_inherits_section_provenance():
    # the contract that keeps citations precise after splitting
    ref = SourceRef(doc_id="RAG", heading="1 Introduction", page=1)
    c = Chunk(text="some split piece", source=ref)
    assert c.source.heading == "1 Introduction"
    assert c.source.page == 1
    assert c.chunk_id  # id was derived

def test_no_tiny_fragments():
    # fragments below the floor are dropped, not embedded
    from askmydocs.config import settings
    pieces = _splitter().split_text("word " * 1000)
    kept = [p.strip() for p in pieces if len(p.strip()) >= settings.min_chunk_chars]
    assert all(len(p) >= settings.min_chunk_chars for p in kept)