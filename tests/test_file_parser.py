"""Tests for the file parser."""
import tempfile
import os
from ref_engine.file_parser import FileParser, ParsedDocument


def test_parse_text_file():
    parser = FileParser()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False) as f:
        f.write("Hello world. This is a test document with enough content.")
        f.flush()
        doc = parser.parse(f.name)
    os.unlink(f.name)
    assert isinstance(doc, ParsedDocument)
    assert doc.word_count > 0
    assert doc.filename.endswith(".txt")


def test_parse_nonexistent_file():
    parser = FileParser()
    doc = parser.parse("/nonexistent/file.pdf")
    assert len(doc.parse_errors) > 0
    assert doc.word_count == 0


def test_get_chunk():
    parser = FileParser()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False) as f:
        f.write(" ".join(["word"] * 1000))
        f.flush()
        doc = parser.parse(f.name)
    os.unlink(f.name)
    chunks = doc.get_chunk(chunk_size=100)
    assert len(chunks) > 1
