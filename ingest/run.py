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
from . import clean
from .enrich import Enricher
from data_loader import load_csv

log = logging.getLogger(__name__)
console = Console()


def ingest(source: Path, settings: Settings = default_settings) -> list[Book]:
    """Load -> normalise -> resolve author identities -> de-duplicate -> books.jsonl."""
    raw = load_csv(source)
    console.print(f"[dim]loaded[/] {len(raw)} rows from {source.name}")

    books = clean.clean(raw, review_path=settings.artifacts_dir / "author_merges.json")
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
    done = {r.book_id for r in read_jsonl(settings.enrichment_path, EnrichmentRecord)}
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
                    EnrichmentRecord(book_id=book.book_id, enrichment=enrichment)
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
    enrichments = {r.book_id: r.enrichment for r in read_jsonl(settings.enrichment_path, EnrichmentRecord)}
    records = [IndexedBook(book=b, enrichment=enrichments.get(b.book_id, Enrichment())) for b in books]
    if derive_facts:
        import search.derive as derive

        records = derive.augment(records)
    return records
