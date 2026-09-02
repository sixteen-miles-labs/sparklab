from pathlib import Path

import pytest

from sparklab.shell import _build_parser
from sparklab.shell.documents import DocumentIndex


def test_document_index_retrieves_matching_file(tmp_path: Path):
    (tmp_path / "json.txt").write_text("JSON values include objects arrays strings and null.")
    (tmp_path / "http.txt").write_text(
        "An idempotent HTTP method has the same intended effect when repeated."
    )
    index = DocumentIndex.from_directory(tmp_path)

    matches = index.retrieve("Which HTTP methods are idempotent?", limit=1)

    assert matches[0].source == "http.txt"
    assert "[Source: http.txt, chunk 1]" in index.context("idempotent HTTP")


def test_document_index_loads_markdown(tmp_path: Path):
    (tmp_path / "policy.md").write_text("The renewal date is 2027-01-15.")

    index = DocumentIndex.from_directory(tmp_path)

    assert index.retrieve("renewal date", limit=1)[0].source == "policy.md"


def test_document_index_rejects_directory_without_documents(tmp_path: Path):
    with pytest.raises(ValueError, match=r"no \.txt or \.md documents"):
        DocumentIndex.from_directory(tmp_path)


def test_shell_parser_accepts_documents_directory(tmp_path: Path):
    parsed = _build_parser("sparklab shell").parse_args(["--documents", str(tmp_path)])

    assert parsed.documents == tmp_path
