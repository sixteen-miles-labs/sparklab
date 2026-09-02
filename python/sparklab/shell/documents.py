"""Small dependency-free document index for ``sparklab shell --documents``."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


_WORD_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]+")


def _words(text: str) -> list[str]:
    return [word.lower() for word in _WORD_RE.findall(text)]


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    number: int
    text: str


class DocumentIndex:
    """An in-memory TF-IDF index over UTF-8 ``.txt`` files."""

    def __init__(self, chunks: list[DocumentChunk]):
        if not chunks:
            raise ValueError("the document index is empty")
        self.chunks = chunks
        self._counts = [Counter(_words(chunk.text)) for chunk in chunks]
        self._document_frequency: Counter[str] = Counter()
        for counts in self._counts:
            self._document_frequency.update(counts.keys())

    @classmethod
    def from_directory(
        cls, directory: str | Path, *, chunk_words: int = 350, overlap: int = 50
    ) -> "DocumentIndex":
        path = Path(directory).expanduser()
        if not path.is_dir():
            raise ValueError(f"document directory does not exist: {path}")
        chunks: list[DocumentChunk] = []
        step = chunk_words - overlap
        documents = sorted(
            document
            for pattern in ("*.txt", "*.md")
            for document in path.glob(pattern)
        )
        for document in documents:
            tokens = document.read_text(encoding="utf-8", errors="replace").split()
            for number, start in enumerate(range(0, len(tokens), step), 1):
                text = " ".join(tokens[start : start + chunk_words])
                if text:
                    chunks.append(DocumentChunk(document.name, number, text))
        if not chunks:
            raise ValueError(f"no .txt or .md documents found in {path}")
        return cls(chunks)

    @property
    def source_count(self) -> int:
        return len({chunk.source for chunk in self.chunks})

    def retrieve(self, query: str, *, limit: int = 4) -> list[DocumentChunk]:
        query_counts = Counter(_words(query))
        if not query_counts:
            return []
        total = len(self.chunks)
        idf = {
            term: math.log((total + 1) / (self._document_frequency[term] + 1)) + 1
            for term in query_counts
        }
        query_norm = math.sqrt(
            sum((count * idf[term]) ** 2 for term, count in query_counts.items())
        )
        ranked: list[tuple[float, DocumentChunk]] = []
        for chunk, counts in zip(self.chunks, self._counts):
            dot = sum(
                query_counts[term] * counts[term] * idf[term] ** 2 for term in query_counts
            )
            chunk_norm = math.sqrt(
                sum((counts[term] * idf[term]) ** 2 for term in query_counts)
            )
            score = dot / (query_norm * chunk_norm) if chunk_norm else 0.0
            ranked.append((score, chunk))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [chunk for score, chunk in ranked[:limit] if score > 0]

    def context(self, query: str, *, limit: int = 4) -> str:
        return "\n\n".join(
            f"[Source: {chunk.source}, chunk {chunk.number}]\n{chunk.text}"
            for chunk in self.retrieve(query, limit=limit)
        )


DOCUMENT_SYSTEM_PROMPT = (
    "Answer using only the document excerpts supplied with the current question. "
    "If the answer is not present, say so. Use conversation history to understand follow-up "
    "questions. Cite factual claims as [filename, chunk N]."
)
