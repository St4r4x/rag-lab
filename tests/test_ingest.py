# tests/test_ingest.py
from ingestion.ingest import chunk_document


def test_chunk_document_splits_long_text_and_attaches_metadata():
    text = "word " * 500
    chunks = chunk_document(text, source="upload", path="notes.md", url="")
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["source"] == "upload"
        assert chunk["path"] == "notes.md"
        assert chunk["url"] == ""
        assert chunk["text"]


def test_chunk_document_short_text_produces_one_chunk():
    chunks = chunk_document("Short text.", source="upload", path="short.md", url="")
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Short text."
