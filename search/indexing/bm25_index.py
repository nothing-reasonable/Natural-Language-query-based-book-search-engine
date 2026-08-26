"""BM25 index over the Bengali-analysed text of every field.

Field weighting is done by repeating a field's tokens `lexical_weight` times inside the
document (see `fields.py`). It is the cheapest way to get "a title match counts more than
a flap match" without maintaining one index per field.
"""

from __future__ import annotations

from pathlib import Path

import bm25s

from search.core import bengali
from config import Settings, settings as default_settings
from search.core.fields import lexical_tokens
from search.core.schemas import IndexedBook


class LexicalIndex:
    def __init__(self, retriever: bm25s.BM25, book_ids: list[str], doc_terms: list[set[str]]):
        self.retriever = retriever
        self.book_ids = book_ids
        # Per-document vocabulary, kept so explanations can name the words that actually
        # matched *this* book rather than the words that exist somewhere in the corpus.
        self.doc_terms = doc_terms

    # ------------------------------------------------------------------ build / load
    @classmethod
    def build(cls, records: list[IndexedBook], settings: Settings = default_settings) -> "LexicalIndex":
        corpus = [lexical_tokens(r, bengali.analyze) for r in records]
        retriever = bm25s.BM25()
        retriever.index(corpus, show_progress=False)
        index = cls(retriever, [r.book_id for r in records], [set(tokens) for tokens in corpus])
        index.save(settings.lexical_dir)
        return index

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(directory))
        (directory / "book_ids.txt").write_text("\n".join(self.book_ids), encoding="utf-8")
        (directory / "doc_terms.txt").write_text(
            "\n".join(" ".join(sorted(terms)) for terms in self.doc_terms), encoding="utf-8"
        )

    @classmethod
    def load(cls, settings: Settings = default_settings) -> "LexicalIndex":
        directory = settings.lexical_dir
        retriever = bm25s.BM25.load(str(directory))
        book_ids = (directory / "book_ids.txt").read_text(encoding="utf-8").splitlines()
        doc_terms = [
            set(line.split())
            for line in (directory / "doc_terms.txt").read_text(encoding="utf-8").splitlines()
        ]
        return cls(retriever, book_ids, doc_terms)

    # ------------------------------------------------------------------ query
    def search(self, terms: list[str], k: int = 50) -> list[tuple[str, float, list[str]]]:
        """`terms` are raw surface strings (query words plus taxonomy expansions).

        Returns (book_id, score, matched_terms).
        """
        tokens = _dedup([t for term in terms for t in bengali.analyze(term)])
        if not tokens:
            return []
        k = min(k, len(self.book_ids))
        indices, scores = self.retriever.retrieve([tokens], k=k, show_progress=False)
        results = []
        for i, score in zip(indices[0], scores[0], strict=True):
            if score <= 0:
                continue
            i = int(i)
            matched = [t for t in tokens if t in self.doc_terms[i]]
            results.append((self.book_ids[i], float(score), matched))
        return results


def _dedup(items: list[str]) -> list[str]:
    seen, out = set(), []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out
