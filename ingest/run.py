"""Offline ingestion pipeline: source file -> cleaned books -> enriched books.

Both stages are resumable and write plain JSONL, so you can inspect (or hand-edit) the
output of each step before moving on.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from config import Settings, settings as default_settings
from search.llm import LMStudio
from search.core.schemas import Book, Enrichment, EnrichmentRecord, IndexedBook
from search.core.store import append_jsonl, read_jsonl, write_jsonl
from search.indexing.passages import enrichment_input_fingerprint
from . import clean
from .enrich import Enricher
from data_loader import load_csv

log = logging.getLogger(__name__)
console = Console()


def ingest(source: Path, settings: Settings = default_settings) -> list[Book]:
    """Load -> normalise -> resolve author identities -> de-duplicate -> books.jsonl."""
    previous = list(read_jsonl(settings.books_path, Book))
    raw = load_csv(source)
    console.print(f"[dim]loaded[/] {len(raw)} rows from {source.name}")

    books = clean.clean(raw, review_path=settings.artifacts_dir / "author_merges.json")
    _preserve_ids(books, previous)
    authors = len({b.author_id for b in books})
    console.print(f"[green]cleaned[/] {len(books)} books, {authors} distinct authors")

    write_jsonl(settings.books_path, books)
    console.print(f"[dim]wrote[/] {settings.books_path}")
    return books


def enrich(settings: Settings = default_settings, *, use_llm: bool = True,
           limit: int | None = None, redo: bool = False) -> None:
    """Tag every book with subjects / periods / places / author roles. Resumable."""
    books = list(read_jsonl(settings.books_path, Book))
    if not books:
        raise FileNotFoundError(f"{settings.books_path} is empty -- run ingest first.")

    if redo and settings.enrichment_path.exists():
        settings.enrichment_path.unlink()
    stored = list(read_jsonl(settings.enrichment_path, EnrichmentRecord))
    # A row is reusable only when it proves which catalogue text produced it. Legacy
    # rows and changed descriptions are regenerated instead of retaining stale tags.
    books_by_id = {book.book_id: book for book in books}
    done = {
        r.book_id for r in stored
        if r.book_id in books_by_id
        and r.input_fingerprint
        and r.input_fingerprint == enrichment_input_fingerprint(books_by_id[r.book_id])
    }
    # Compact before continuing: deleted books, changed inputs, legacy rows and older
    # duplicate attempts are removed. Completed matching rows remain resumable.
    reusable = {
        row.book_id: row for row in stored
        if row.book_id in done
    }
    write_jsonl(settings.enrichment_path, reusable.values())
    todo = [b for b in books if b.book_id not in done]
    if limit:
        todo = todo[:limit]
    if not todo:
        console.print("[green]enrichment already complete[/]")
        return

    llm = LMStudio(settings) if use_llm else None
    if llm is not None and not llm.is_available():
        console.print("[yellow]LM Studio unreachable -- falling back to dictionary tagging only[/]")
        llm = None
    enricher = Enricher(llm=llm, taxonomy=None, use_llm=llm is not None)

    console.print(f"enriching {len(todo)} books ({len(done)} already done)")
    with Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(),
                  console=console) as progress:
        task = progress.add_task("enrich", total=len(todo))
        batch_size = max(1, settings.llm_workers * 4)
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            results = (
                llm.map_parallel(enricher.enrich, batch)
                if llm is not None
                else [enricher.enrich(b) for b in batch]
            )
            append_jsonl(
                settings.enrichment_path,
                [
                    EnrichmentRecord(
                        book_id=book.book_id,
                        enrichment=enrichment,
                        input_fingerprint=enrichment_input_fingerprint(book),
                        method="llm" if llm is not None else "dictionary",
                    )
                    for book, enrichment in zip(batch, results, strict=True)
                ],
            )
            progress.advance(task, len(batch))

    console.print(f"[green]done[/] -> {settings.enrichment_path}")


def load_indexed(settings: Settings = default_settings, *, derive_facts: bool = True) -> list[IndexedBook]:
    """Join books.jsonl with enrichment.jsonl, then fill in what can be inferred.

    The derivation step (see `derive.py`) is cheap, deterministic and additive, so it
    runs on every load rather than being baked into the stored artifact -- editing
    `data/taxonomy.yaml` takes effect on the next build with no re-enrichment.
    """
    books = list(read_jsonl(settings.books_path, Book))
    if not books:
        raise FileNotFoundError(f"{settings.books_path} is empty -- run ingest first.")
    stored = {r.book_id: r for r in read_jsonl(settings.enrichment_path, EnrichmentRecord)}
    deterministic = Enricher(llm=None, taxonomy=None, use_llm=False)
    enrichments: dict[str, Enrichment] = {}
    for book in books:
        row = stored.get(book.book_id)
        if row and row.input_fingerprint == enrichment_input_fingerprint(book):
            enrichments[book.book_id] = row.enrichment
        else:
            # Safe migration for missing/legacy provenance. This is deterministic and
            # avoids serving stale model-derived tags until `enrich` is rerun.
            enrichments[book.book_id] = deterministic.enrich(book)
    records = [IndexedBook(book=b, enrichment=enrichments[b.book_id]) for b in books]
    if derive_facts:
        import search.derive as derive

        records = derive.augment(records)
    return records


def _preserve_ids(books: list[Book], previous: list[Book]) -> None:
    """Carry stable IDs across metadata edits when the source exposes an identity key."""
    by_source = {book.source_url: book.book_id for book in previous if book.source_url}
    by_isbn = {book.isbn: book.book_id for book in previous if book.isbn}
    for book in books:
        prior = ((by_source.get(book.source_url) if book.source_url else None)
                 or (by_isbn.get(book.isbn) if book.isbn else None))
        if prior:
            book.book_id = prior
