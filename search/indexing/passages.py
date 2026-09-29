"""Stable, token-bounded catalogue passages used by dense retrieval."""

from __future__ import annotations

import hashlib
import json

from search.core.schemas import Chunk, IndexedBook


def text_fingerprint(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def enrichment_input_fingerprint(book) -> str:
    """Only fields the deterministic/LLM enricher reads participate."""
    payload = {
        name: getattr(book, name, None)
        for name in (
            "title", "author", "author_bio", "publisher", "description",
            "publish_year", "table_of_contents", "language",
        )
    }
    return _fingerprint(payload)


def record_fingerprint(record: IndexedBook) -> str:
    """Search-content identity. URL/display-only changes do not force embeddings."""
    book = record.book.model_dump(exclude={"cover_url", "source_url"})
    return _fingerprint({"book": book, "enrichment": record.enrichment.model_dump()})


def catalogue_fingerprint(records: list[IndexedBook]) -> str:
    rows = sorted((record.book_id, record_fingerprint(record)) for record in records)
    return _fingerprint(rows)


def make_passages(records: list[IndexedBook], *, tokens: int = 256,
                  overlap: int = 48) -> list[Chunk]:
    """Chunk descriptions and contents, preserving source attribution and stable ids.

    Whitespace tokens are used deliberately: reconstruction then remains a verbatim
    excerpt of catalogue text rather than stemmed/token-normalised prose.
    """
    if tokens <= 0:
        raise ValueError("passage token count must be positive")
    if overlap < 0 or overlap >= tokens:
        raise ValueError("passage overlap must be between 0 and tokens-1")

    chunks: list[Chunk] = []
    step = tokens - overlap
    for record in records:
        for source_field in ("description", "table_of_contents"):
            source = record.book.field_text(source_field).strip()
            words = source.split()
            if not words:
                continue
            for ordinal, start in enumerate(range(0, len(words), step)):
                window = words[start : start + tokens]
                if not window:
                    break
                text = " ".join(window)
                fingerprint = text_fingerprint(text)
                seed = f"{record.book_id}|{source_field}|{ordinal}|{fingerprint}"
                chunk_id = "passage:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    book_id=record.book_id,
                    text=text,
                    ordinal=ordinal,
                    source_field=source_field,
                    fingerprint=fingerprint,
                ))
                if start + tokens >= len(words):
                    break
    return chunks


def _fingerprint(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
