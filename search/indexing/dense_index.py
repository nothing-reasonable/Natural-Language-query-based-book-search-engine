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
        if TABLE in db.table_names():
            existing = db.open_table(TABLE)
            # Resuming only makes sense if the stored vectors came from the same model.
            same_model = resume and _vector_dim(existing) == embedder.dimension
            if same_model:
                table = existing
                done = _stored_book_ids(table)
            else:
                if resume:
                    log.warning("Existing index has a different vector size -- rebuilding it.")
                db.drop_table(TABLE)

        todo = [r for r in records if r.book_id not in done]
        log.info("embedding %d books (%d already done)", len(todo), len(done))
        if on_batch is not None and done:
            on_batch(len(records) - len(todo))

        for start in range(0, len(todo), flush_every):
            batch = todo[start : start + flush_every]
            texts = [embedding_text(r) for r in batch]
            vectors = embedder.embed_documents(texts, on_batch=on_batch)
            rows = [
                _book_row(record, text, vector)
                for record, text, vector in zip(batch, texts, vectors, strict=True)
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
            row = _book_row(record, chunk.text, vector)
            row["id"] = chunk.chunk_id
            row["kind"] = "chunk"
            rows.append(row)
        self.table.add(rows)
        return len(rows)

    # ------------------------------------------------------------------ query
    def search(self, query_vector: np.ndarray, k: int = 50,
               filters: Filters | None = None) -> list[tuple[str, float, str]]:
        """Returns (book_id, similarity in 0..1, matched text snippet), best first."""
        query = self.table.search(query_vector, vector_column_name="vector").metric("cosine")
        where = build_where(filters)
        if where:
            query = query.where(where, prefilter=True)
        # Over-fetch so that several chunks of one book still leave room for other books.
        hits = query.limit(k * 3).to_list()

        best: dict[str, tuple[float, str]] = {}
        for hit in hits:
            score = 1.0 - float(hit["_distance"])
            book_id = hit["book_id"]
            if score > best.get(book_id, (-1.0, ""))[0]:
                best[book_id] = (score, hit.get("text", "")[:200])
        ranked = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))[:k]
        return [(book_id, score, snippet) for book_id, (score, snippet) in ranked]


# --------------------------------------------------------------------------- helpers

def _stored_book_ids(table) -> set[str]:
    """Column-projected scan -- never pulls the vectors back out of storage."""
    rows = table.search().select(["book_id"]).limit(table.count_rows()).to_list()
    return {row["book_id"] for row in rows}


def _vector_dim(table) -> int:
    field = table.schema.field("vector")
    return getattr(field.type, "list_size", -1)


def _book_row(record: IndexedBook, text: str, vector: np.ndarray) -> dict:
    book, enrichment = record.book, record.enrichment
    return {
        "id": book.book_id,
        "book_id": book.book_id,
        "kind": "book",
        "text": text[:4000],
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
