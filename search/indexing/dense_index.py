"""Dense (semantic) index backed by LanceDB.

Stores two kinds of row in one table:

  * `kind="book"`  -- one vector per book, built from the fields marked `embed=True`
  * `kind="chunk"` -- one vector per slice of full text, for when book contents arrive

Chunk hits are rolled up to their parent book at query time, so the rest of the pipeline
never has to know whether a match came from metadata or from page 240.
"""

from __future__ import annotations

import logging
from typing import Callable

import lancedb
import numpy as np
import pyarrow as pa

from config import Settings, settings as default_settings
from search.indexing.embedding import Embedder
from search.core.fields import embedding_text
from search.core.schemas import Chunk, Filters, IndexedBook
from search.indexing.passages import make_passages, record_fingerprint

log = logging.getLogger(__name__)

TABLE = "books"
LIST_FACETS = ("genres", "subjects", "periods", "places")
UNKNOWN_YEAR = 0


def _arrow_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("book_id", pa.string()),
            pa.field("kind", pa.string()),
            pa.field("text", pa.string()),
            pa.field("source_field", pa.string()),
            pa.field("fingerprint", pa.string()),
            pa.field("author_id", pa.string()),
            pa.field("publisher", pa.string()),
            pa.field("language", pa.string()),
            pa.field("publish_year", pa.int32()),
            *[pa.field(name, pa.list_(pa.string())) for name in LIST_FACETS],
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


class VectorIndex:
    def __init__(self, table):
        self.table = table

    # ------------------------------------------------------------------ build / open
    @classmethod
    def build(cls, records: list[IndexedBook], embedder: Embedder,
              settings: Settings = default_settings,
              on_batch: Callable[[int], None] | None = None,
              resume: bool = True, flush_every: int = 256) -> "VectorIndex":
        """Embed every book and write it to LanceDB.

        Rows are flushed as they are produced and already-embedded books are skipped, so a
        run that dies an hour in (LM Studio restart, laptop asleep) resumes instead of
        starting over.
        """
        settings.vector_dir.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(settings.vector_dir)

        table = None
        done: set[str] = set()
        passages = make_passages(
            records, tokens=settings.passage_tokens, overlap=settings.passage_overlap
        )
        records_by_id = {record.book_id: record for record in records}
        expected = {
            record.book_id: record_fingerprint(record) for record in records
        } | {chunk.chunk_id: chunk.fingerprint for chunk in passages}
        if TABLE in db.table_names():
            existing = db.open_table(TABLE)
            # Dimension is necessary but not sufficient for compatibility; the manifest
            # checks model identity/prompts. Here we additionally require the v2 row
            # schema before attempting an incremental resume.
            compatible_schema = all(
                name in existing.schema.names for name in ("fingerprint", "source_field")
            )
            same_shape = resume and compatible_schema and _vector_dim(existing) == embedder.dimension
            if same_shape:
                table = existing
                stored = _stored_rows(table)
                obsolete = [row_id for row_id, fingerprint in stored.items()
                            if expected.get(row_id) != fingerprint]
                _delete_ids(table, obsolete)
                done = {row_id for row_id, fingerprint in stored.items()
                        if expected.get(row_id) == fingerprint}
            else:
                if resume:
                    log.warning("Existing vector index is incompatible -- rebuilding it.")
                db.drop_table(TABLE)

        items: list[tuple[str, str, IndexedBook, Chunk | None]] = []
        for record in records:
            if record.book_id not in done:
                items.append((record.book_id, embedding_text(record), record, None))
        for chunk in passages:
            if chunk.chunk_id not in done and chunk.book_id in records_by_id:
                items.append((chunk.chunk_id, chunk.text, records_by_id[chunk.book_id], chunk))
        log.info("embedding %d rows (%d unchanged)", len(items), len(done))
        if on_batch is not None and done:
            on_batch(len(done))

        for start in range(0, len(items), flush_every):
            batch = items[start : start + flush_every]
            texts = [item[1] for item in batch]
            vectors = embedder.embed_documents(texts, on_batch=on_batch)
            rows = [
                _book_row(record, text, vector, chunk=chunk)
                for (_, text, record, chunk), vector in zip(batch, vectors, strict=True)
            ]
            if table is None:
                table = db.create_table(TABLE, data=rows, schema=_arrow_schema(vectors.shape[1]))
            else:
                table.add(rows)

        if table is None:
            table = db.open_table(TABLE)
        return cls(table)

    @classmethod
    def open(cls, settings: Settings = default_settings) -> "VectorIndex":
        db = lancedb.connect(settings.vector_dir)
        return cls(db.open_table(TABLE))

    def add_chunks(self, chunks: list[Chunk], records_by_id: dict[str, IndexedBook],
                   embedder: Embedder) -> int:
        """Extension point for full text. Chunks inherit their book's facets."""
        if not chunks:
            return 0
        vectors = embedder.embed_documents([c.text for c in chunks])
        rows = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            record = records_by_id.get(chunk.book_id)
            if record is None:
                continue
            row = _book_row(record, chunk.text, vector, chunk=chunk)
            rows.append(row)
        self.table.add(rows)
        return len(rows)

    # ------------------------------------------------------------------ query
    def search(self, query_vector: np.ndarray, k: int = 50,
               filters: Filters | None = None) -> list[tuple[str, float, str, str]]:
        """Return the best row per book as (id, score, excerpt, source field)."""
        query = self.table.search(query_vector, vector_column_name="vector").metric("cosine")
        where = build_where(filters)
        if where:
            query = query.where(where, prefilter=True)
        # Over-fetch before book-level aggregation so a long description cannot fill the
        # candidate list with passages from one title.
        total_rows = self.table.count_rows()
        limit = min(total_rows, max(k * 10, k))
        hits = query.limit(limit).to_list()
        # Pathological long texts can still fill an over-fetch window. Grow only when
        # needed, stopping as soon as k distinct books are represented.
        while len({hit["book_id"] for hit in hits}) < min(k, total_rows) and limit < total_rows:
            limit = min(total_rows, limit * 2)
            hits = query.limit(limit).to_list()

        # Ranking uses the best row of either kind; evidence separately retains the best
        # source passage even when the book-level metadata vector scored slightly higher.
        best: dict[str, dict] = {}
        for hit in hits:
            score = 1.0 - float(hit["_distance"])
            book_id = hit["book_id"]
            entry = best.setdefault(book_id, {
                "score": -1.0, "passage_score": -1.0, "excerpt": "", "source": ""
            })
            entry["score"] = max(entry["score"], score)
            if hit.get("kind") == "passage" and score > entry["passage_score"]:
                entry.update({
                    "passage_score": score,
                    "excerpt": _excerpt(hit.get("text", ""), 500),
                    "source": hit.get("source_field", ""),
                })
        ranked = sorted(best.items(), key=lambda kv: (-kv[1]["score"], kv[0]))[:k]
        return [(book_id, data["score"], data["excerpt"], data["source"])
                for book_id, data in ranked]


# --------------------------------------------------------------------------- helpers

def _stored_rows(table) -> dict[str, str]:
    """Column-projected scan -- never pulls vectors back out of storage."""
    rows = table.search().select(["id", "fingerprint"]).limit(table.count_rows()).to_list()
    return {row["id"]: row.get("fingerprint", "") for row in rows}


def _delete_ids(table, row_ids: list[str]) -> None:
    for start in range(0, len(row_ids), 200):
        batch = row_ids[start : start + 200]
        if batch:
            table.delete("id IN (" + ", ".join(_lit(value) for value in batch) + ")")


def _vector_dim(table) -> int:
    field = table.schema.field("vector")
    return getattr(field.type, "list_size", -1)


def _book_row(record: IndexedBook, text: str, vector: np.ndarray,
              *, chunk: Chunk | None = None) -> dict:
    book, enrichment = record.book, record.enrichment
    return {
        "id": chunk.chunk_id if chunk else book.book_id,
        "book_id": book.book_id,
        "kind": "passage" if chunk else "book",
        "text": text[:4000],
        "source_field": chunk.source_field if chunk else "catalogue",
        "fingerprint": chunk.fingerprint if chunk else record_fingerprint(record),
        "author_id": book.author_id,
        "publisher": book.publisher,
        "language": book.language,
        "publish_year": book.publish_year or UNKNOWN_YEAR,
        "genres": enrichment.genres,
        "subjects": enrichment.subjects,
        "periods": enrichment.periods,
        "places": enrichment.places,
        "vector": vector.tolist(),
    }


def build_where(filters: Filters | None) -> str:
    """Translate hard filters into a LanceDB (DataFusion) SQL predicate."""
    if filters is None:
        return ""
    clauses: list[str] = []
    if filters.language:
        clauses.append(f"language = {_lit(filters.language)}")
    if filters.author_ids:
        clauses.append(f"author_id IN ({', '.join(_lit(a) for a in filters.author_ids)})")
    if filters.publishers:
        clauses.append(f"publisher IN ({', '.join(_lit(p) for p in filters.publishers)})")
    if filters.year_from is not None:
        clauses.append(f"(publish_year >= {int(filters.year_from)} AND publish_year != {UNKNOWN_YEAR})")
    if filters.year_to is not None:
        clauses.append(f"(publish_year <= {int(filters.year_to)} AND publish_year != {UNKNOWN_YEAR})")
    for name in LIST_FACETS:
        values = getattr(filters, name, None)
        if values:
            listed = ", ".join(_lit(v) for v in values)
            clauses.append(f"array_has_any({name}, [{listed}])")
    return " AND ".join(clauses)


def _lit(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _excerpt(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    head = text[:limit].rsplit(" ", 1)[0] or text[:limit]
    return head.rstrip(" ,;।") + "…"
