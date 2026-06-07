"""Unit tests for the document_loader infrastructure (pypdf mocked)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.domains.documents.infrastructure import document_loader
from app.domains.documents.infrastructure.document_loader import (
    DocumentDecodeError,
    UnsupportedDocumentError,
    extract_text,
    is_supported,
)


class TestIsSupported:
    @pytest.mark.parametrize("name", ["a.txt", "A.TXT", "report.pdf", "X.PDF", "n.Txt"])
    def test_supported_extensions(self, name):
        assert is_supported(name) is True

    @pytest.mark.parametrize("name", ["a.docx", "a.csv", "noext", "a.txt.zip", ""])
    def test_unsupported_extensions(self, name):
        assert is_supported(name) is False


class TestExtractTxt:
    def test_decodes_utf8(self):
        text = extract_text(filename="f.txt", raw_bytes="héllo wörld".encode("utf-8"))
        assert text == "héllo wörld"

    def test_case_insensitive_extension(self):
        assert extract_text(filename="F.TXT", raw_bytes=b"hi") == "hi"

    def test_invalid_utf8_raises_decode_error(self):
        with pytest.raises(DocumentDecodeError, match="UTF-8"):
            extract_text(filename="f.txt", raw_bytes=b"\xff\xfe\x00bad")


class TestExtractUnsupported:
    def test_unsupported_extension_raises(self):
        with pytest.raises(UnsupportedDocumentError, match="Only .txt and .pdf"):
            extract_text(filename="f.docx", raw_bytes=b"data")


class TestExtractPdf:
    def test_joins_page_text_with_newlines(self):
        fake_reader = SimpleNamespace(
            pages=[
                SimpleNamespace(extract_text=lambda: "page one"),
                SimpleNamespace(extract_text=lambda: "page two"),
            ]
        )
        with patch("pypdf.PdfReader", return_value=fake_reader):
            text = extract_text(filename="doc.pdf", raw_bytes=b"%PDF-1.4 fake")
        assert text == "page one\npage two"

    def test_none_page_text_becomes_empty_string(self):
        fake_reader = SimpleNamespace(
            pages=[
                SimpleNamespace(extract_text=lambda: None),
                SimpleNamespace(extract_text=lambda: "real"),
            ]
        )
        with patch("pypdf.PdfReader", return_value=fake_reader):
            text = extract_text(filename="doc.pdf", raw_bytes=b"%PDF")
        assert text == "\nreal"

    def test_parser_failure_raises_decode_error(self):
        with patch("pypdf.PdfReader", side_effect=ValueError("corrupt")):
            with pytest.raises(DocumentDecodeError, match="Could not read the PDF"):
                extract_text(filename="doc.pdf", raw_bytes=b"not a pdf")

    def test_supported_extensions_constant(self):
        assert document_loader.SUPPORTED_EXTENSIONS == (".txt", ".pdf")
