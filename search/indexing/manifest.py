"""Versioned contract tying every search artifact to one catalogue generation."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from config import Settings
from search.core.schemas import IndexedBook
from search.indexing.passages import catalogue_fingerprint

TEXT_PREPARATION_VERSION = "catalogue-fields-v2"


class IndexCompatibilityError(RuntimeError):
    pass


class IndexManifest(BaseModel):
    schema_version: int
    generation: str
    created_at: str
    complete: bool = False
    catalogue_fingerprint: str
    record_count: int
    embedding_backend: str
    embedding_model: str
    embedding_revision: str = ""
    embedding_dimension: int | None = None
    query_prompt: str = ""
    document_prompt: str = ""
    text_preparation: str = TEXT_PREPARATION_VERSION
    passage_tokens: int = 256
    passage_overlap: int = 48
    vector_enabled: bool = True
    artifacts: list[str] = Field(default_factory=list)


def expected_manifest(records: list[IndexedBook], settings: Settings, *,
                      dimension: int | None = None,
                      vector_enabled: bool = True) -> IndexManifest:
    model = (settings.embedding_model if settings.embedding_backend == "huggingface"
             else settings.lmstudio_embedding_model)
    return IndexManifest(
        schema_version=settings.index_schema_version,
        generation=uuid.uuid4().hex,
        created_at=datetime.now(timezone.utc).isoformat(),
        complete=False,
        catalogue_fingerprint=catalogue_fingerprint(records),
        record_count=len(records),
        embedding_backend=settings.embedding_backend,
        embedding_model=model,
        embedding_revision=settings.embedding_model_revision,
        embedding_dimension=dimension,
        query_prompt=settings.embedding_query_prompt,
        document_prompt=settings.embedding_document_prompt,
        passage_tokens=settings.passage_tokens,
        passage_overlap=settings.passage_overlap,
        vector_enabled=vector_enabled,
        artifacts=["lexical", "graph"] + (["vector"] if vector_enabled else []),
    )


def publish(manifest: IndexManifest, path: Path) -> None:
    """The manifest is the commit marker and is atomically replaced only at the end."""
    path.parent.mkdir(parents=True, exist_ok=True)
    complete = manifest.model_copy(update={"complete": True})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(complete.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def load_and_validate(path: Path, records: list[IndexedBook], settings: Settings) -> IndexManifest:
    rebuild = "Run `python cli.py build-index` after ingest/enrich to rebuild one compatible generation."
    if not path.exists():
        raise IndexCompatibilityError(f"Index manifest is missing. {rebuild}")
    try:
        manifest = IndexManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise IndexCompatibilityError(f"Index manifest is unreadable. {rebuild}") from exc
    if not manifest.complete:
        raise IndexCompatibilityError(f"The previous index build was interrupted. {rebuild}")

    current = expected_manifest(records, settings, vector_enabled=manifest.vector_enabled)
    checks = {
        "schema version": manifest.schema_version == current.schema_version,
        "catalogue": manifest.catalogue_fingerprint == current.catalogue_fingerprint,
        "embedding backend": manifest.embedding_backend == current.embedding_backend,
        "embedding model": manifest.embedding_model == current.embedding_model,
        "embedding revision": manifest.embedding_revision == current.embedding_revision,
        "query prompt": manifest.query_prompt == current.query_prompt,
        "document prompt": manifest.document_prompt == current.document_prompt,
        "text preparation": manifest.text_preparation == current.text_preparation,
        "passage configuration": (
            manifest.passage_tokens == current.passage_tokens
            and manifest.passage_overlap == current.passage_overlap
        ),
    }
    incompatible = [name for name, ok in checks.items() if not ok]
    if incompatible:
        raise IndexCompatibilityError(
            "Index is stale or incompatible (" + ", ".join(incompatible) + "). " + rebuild
        )
    return manifest
