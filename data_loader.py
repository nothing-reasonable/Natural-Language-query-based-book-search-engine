"""Source catalogue -> `Book` records.

`COLUMN_MAP` is the only thing that needs editing when the crawler starts producing
extra columns (ISBN, table of contents, ...). `Publication Year` is already wired,
which is what gives every book a historical period without asking a model.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from search.core import bengali
from search.core.schemas import Book

# source column (case/space-insensitive) -> Book field
COLUMN_MAP: dict[str, str] = {
    "book name": "title",
    "title": "title",
    "author": "author_raw",
    "author bio": "author_bio",
    "publisher": "publisher",
    "description (flap)": "description",
    "description": "description",
    "publication year": "publish_year",
    "publish year": "publish_year",
    "year": "publish_year",
    # --- fields we do not have yet, wired up in advance ---
    "language": "language",
    "table of contents": "table_of_contents",
    "contents": "table_of_contents",
    "isbn": "isbn",
}

INT_FIELDS = {"publish_year"}


def make_book_id(title: str, author: str) -> str:
    """Stable identity: a hash of the *analysed* title and author.

    Stability matters more than it looks -- enrichment.jsonl is keyed by this, so a
    catalogue that gains a column (or fixes a spelling) keeps every record it already
    paid an LLM to produce.
    """
    seed = f"{bengali.key(title)}|{bengali.key(author)}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def load_csv(path: Path) -> list[Book]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return load_frame(frame)


def load_frame(frame: pd.DataFrame) -> list[Book]:
    mapping = {c: COLUMN_MAP[c.strip().lower()] for c in frame.columns if c.strip().lower() in COLUMN_MAP}
    unknown = [c for c in frame.columns if c.strip().lower() not in COLUMN_MAP]
    if unknown:
        print(f"[data_loader] ignoring unmapped columns: {unknown} (add them to COLUMN_MAP)")

    books: list[Book] = []
    for row in frame.to_dict(orient="records"):
        payload: dict = {}
        for column, field in mapping.items():
            value = str(row.get(column, "") or "").strip()
            if not value:
                continue
            coerced = _coerce(field, value)
            if coerced is None:
                continue
            payload[field] = coerced
        title = payload.get("title", "")
        author = payload.get("author_raw", "")
        if not title:
            continue
        payload["book_id"] = make_book_id(title, author)
        payload.setdefault("author", author)
        books.append(Book(**payload))
    return books


def _coerce(field: str, value: str):
    if field in INT_FIELDS:
        digits = "".join(ch for ch in bengali.fold_digits(value) if ch.isdigit())
        return int(digits[:4]) if len(digits) >= 4 else None
    return value
