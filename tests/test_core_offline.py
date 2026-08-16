"""
Simple tests for PDF extraction, chunking, and highlighting.
These do not need internet.

Run:
    python tests/test_core_offline.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_core import (
    extract_pages_from_pdf,
    chunk_text,
    build_chunks_for_documents,
    highlight_keywords,
    format_context,
    RetrievedChunk,
    Chunk,
)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_pdfs")


def test_extract_pages_from_pdf():
    pages = extract_pages_from_pdf(os.path.join(SAMPLE_DIR, "company_policy.pdf"))
    assert len(pages) == 3
    assert "Leave Policy" in pages[0]["text"]
    assert pages[0]["page"] == 1


def test_chunk_text_respects_overlap_and_size():
    text = "Sentence one. " * 100
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 220  # allow small slack for sentence-boundary snapping


def test_chunk_text_rejects_bad_overlap():
    try:
        chunk_text("hello world", chunk_size=10, chunk_overlap=10)
        assert False, "should have raised"
    except ValueError:
        pass


def test_build_chunks_for_documents_tracks_metadata():
    pages = extract_pages_from_pdf(os.path.join(SAMPLE_DIR, "product_manual.pdf"))
    docs = {"product_manual.pdf": pages}
    chunks = build_chunks_for_documents(docs, chunk_size=300, chunk_overlap=50)
    assert all(c.source == "product_manual.pdf" for c in chunks)
    assert all(c.page >= 1 for c in chunks)
    # chunk_ids should be unique and increasing
    ids = [c.chunk_id for c in chunks]
    assert ids == sorted(set(ids))


def test_highlight_keywords():
    text = "The warranty covers manufacturing defects for two years."
    out = highlight_keywords(text, "warranty defects")
    assert "**warranty**" in out
    assert "**defects**" in out


def test_format_context_includes_source_tags():
    chunk = Chunk(text="Some content", source="doc.pdf", page=2, chunk_id=0)
    retrieved = [RetrievedChunk(chunk=chunk, score=0.87)]
    ctx = format_context(retrieved)
    assert "doc.pdf" in ctx
    assert "page 2" in ctx
    assert "Some content" in ctx


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"PASS: {t.__name__}")
    print(f"\n{passed}/{len(tests)} tests passed.")
