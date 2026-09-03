"""Data model for the whole pipeline.

`Book` holds catalogue facts, `Enrichment` holds everything a model inferred.
Keeping them apart means the LLM's JSON schema *is* `Enrichment` -- no mapping code.

To add a new catalogue field (publishing year, table of contents, ...):
  1. add it here,
  2. declare how it should be searched in `fields.py`,
  3. map the source column in `ingest/loader.py`.
Nothing else needs to change.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["simple", "semantic", "filtered", "personalized", "multi_hop"]


# --------------------------------------------------------------------------- catalogue

class Book(BaseModel):
    book_id: str
    title: str = ""
    author: str = ""  # canonical author name after alias resolution
    author_raw: str = ""  # name exactly as it appeared in the source
    author_id: str = ""
    author_bio: str = ""
    publisher: str = ""
    description: str = ""

    # --- optional / future fields (safe to leave empty) ---
    publish_year: int | None = None
    language: str = "bn"
    table_of_contents: str = ""
    isbn: str = ""

    # --- ranking signals ---
    popularity: float = 0.0
    available: bool = True
    metadata_quality: float = 0.0  # 0..1, share of populated fields

    def field_text(self, name: str) -> str:
        value = getattr(self, name, "")
        if value is None:
            return ""
        if isinstance(value, list):
            return " ".join(str(v) for v in value)
        return str(value)


# --------------------------------------------------------------------------- enrichment

class Enrichment(BaseModel):
    """Produced by the LLM, then canonicalised against `data/taxonomy.yaml`."""

    subjects: list[str] = Field(default_factory=list, description="বইয়ের প্রধান বিষয়")
    topics: list[str] = Field(default_factory=list, description="আলোচিত উপ-বিষয়")
    genres: list[str] = Field(default_factory=list, description="সাহিত্যের ধরন")
    periods: list[str] = Field(default_factory=list, description="বইটি যে ঐতিহাসিক কাল নিয়ে")
    places: list[str] = Field(default_factory=list, description="স্থান")
    events: list[str] = Field(default_factory=list, description="ঐতিহাসিক ঘটনা")
    persons: list[str] = Field(default_factory=list, description="উল্লেখযোগ্য ব্যক্তি")
    author_roles: list[str] = Field(default_factory=list, description="লেখকের পেশা বা ভূমিকা")
    author_periods: list[str] = Field(default_factory=list, description="লেখক যে কালে সক্রিয় ছিলেন")
    summary: str = Field(default="", description="এক বাক্যে সারসংক্ষেপ")


class EnrichmentRecord(BaseModel):
    """What `enrichment.jsonl` stores -- one line per book, resumable."""

    book_id: str
    enrichment: Enrichment


class IndexedBook(BaseModel):
    """A book plus its enrichment -- the unit every index stores."""

    book: Book
    enrichment: Enrichment = Field(default_factory=Enrichment)

    @property
    def book_id(self) -> str:
        return self.book.book_id

    def field_text(self, name: str) -> str:
        if hasattr(self.book, name):
            return self.book.field_text(name)
        value = getattr(self.enrichment, name, "")
        if isinstance(value, list):
            return " ".join(str(v) for v in value)
        return str(value or "")


class Chunk(BaseModel):
    """A slice of full text (contents / OCR / flap). Only the dense index uses these."""

    chunk_id: str
    book_id: str
    text: str
    ordinal: int = 0


# --------------------------------------------------------------------------- query plan

class Filters(BaseModel):
    """Hard constraints. Empty lists / None mean 'no constraint'."""

    authors: list[str] = Field(default_factory=list)
    # Resolved ids for the names in `authors`. Names are what a user (or the LLM) says;
    # ids are what the indexes store, and only an id can be pushed down into the vector
    # store's pre-filter. Set by the entity linker, never by the model.
    author_ids: list[str] = Field(default_factory=list)
    publishers: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    subjects: list[str] = Field(default_factory=list)
    periods: list[str] = Field(default_factory=list)
    places: list[str] = Field(default_factory=list)
    language: str | None = None
    year_from: int | None = None
    year_to: int | None = None

    def is_empty(self) -> bool:
        return self.model_dump(exclude_defaults=True) == {}


class Concepts(BaseModel):
    """Controlled-vocabulary terms spotted in the query.

    These are *soft* evidence -- they steer the graph channel and query expansion.
    Anything that should actually exclude results belongs in `Filters` instead.
    """

    subjects: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    periods: list[str] = Field(default_factory=list)
    places: list[str] = Field(default_factory=list)
    occupations: list[str] = Field(default_factory=list)

    def all_terms(self) -> list[str]:
        return [term for name in type(self).model_fields for term in getattr(self, name)]


class EntityRef(BaseModel):
    """A named thing the query actually referred to, resolved against the catalogue."""

    kind: str          # "author" | "publisher"
    name: str          # canonical display name
    entity_id: str
    score: float = 0.0
    hard: bool = False  # was it confident enough to constrain the result set?


class GraphStep(BaseModel):
    """One hop of a decomposed multi-hop question."""

    kind: Literal["find_authors", "find_books"]
    occupations: list[str] = Field(default_factory=list)
    periods: list[str] = Field(default_factory=list)
    subjects: list[str] = Field(default_factory=list)
    places: list[str] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)


class QueryPlanDraft(BaseModel):
    """The narrow slice of a plan the LLM is asked to produce.

    Deliberately smaller than `QueryPlan`: fewer fields means fewer tokens to generate,
    which is the difference between a snappy and a sluggish search on a local model.
    """

    intent: Intent = "semantic"
    keywords: list[str] = Field(default_factory=list)
    expanded_terms: list[str] = Field(default_factory=list)
    filters: Filters = Field(default_factory=Filters)
    steps: list[GraphStep] = Field(default_factory=list)


class QueryPlan(BaseModel):
    raw_query: str = ""
    normalized_query: str = ""
    intent: Intent = "semantic"
    keywords: list[str] = Field(default_factory=list)
    expanded_terms: list[str] = Field(default_factory=list)
    concepts: Concepts = Field(default_factory=Concepts)  # soft signals
    filters: Filters = Field(default_factory=Filters)  # hard constraints
    steps: list[GraphStep] = Field(default_factory=list)
    entities: list[EntityRef] = Field(default_factory=list)
    rationale: str = ""


# --------------------------------------------------------------------------- results

class Evidence(BaseModel):
    """Why a book surfaced. Explanations are rendered from these, never invented."""

    channel: str  # lexical | dense | graph | facet
    detail: str  # Bengali phrase shown to the user
    terms: list[str] = Field(default_factory=list)


class Candidate(BaseModel):
    book_id: str
    channel: str
    rank: int
    score: float
    evidence: list[Evidence] = Field(default_factory=list)


class SearchHit(BaseModel):
    book: Book
    enrichment: Enrichment
    score: float
    relevance: float  # score before personalisation
    components: dict[str, float] = Field(default_factory=dict)
    channels: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    explanation: str = ""


class RerankEntry(BaseModel):
    """One (query, passage) pair exactly as the reranker saw it, and what it gave back.

    `passage` is the reranker's *input*, not a display string: it is what `rerank._passage`
    builds, which is what the cross-encoder and the LM Studio backend both actually read.
    The one exception is the deprecated `reranker_backend="llm"` grader, which composes its
    own listing via `rerank._describe` -- close, but not identical.
    """

    book_id: str
    title: str
    fusion_rank: int  # position going in (1-based, fusion order)
    passage: str  # the text handed to the model
    score: float  # what came back, 0..1
    # Position after the signal blend but *before* personalisation, so the movement from
    # `fusion_rank` is attributable to relevance alone. None if the blend dropped it.
    final_rank: int | None = None


class RerankTrace(BaseModel):
    """Reranker I/O for one search. Populated only when tracing is explicitly asked for.

    `backend` is the reranker that *ran*, not the one configured. Every backend degrades to
    `noop` rather than raising (a missing download, a CUDA OOM, a dead server), and a noop
    still returns plausible-looking descending scores -- so the label is the only reliable
    way to tell a real reranking from fusion order wearing its clothes.
    """

    backend: str = ""
    model: str = ""
    query: str = ""  # the normalised query, which is not always what the user typed
    entries: list[RerankEntry] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    plan: QueryPlan
    hits: list[SearchHit] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)

    # --- diagnostics: what retrieval produced before the top-k truncation ---
    # Recall is a property of the *candidate* set, not of the ten books shown, so the
    # evaluator needs to see this to tell "we never found it" apart from "we found it
    # and ranked it badly" -- two problems with completely different fixes.
    candidates: list[str] = Field(default_factory=list)  # fused order, pre-rerank
    channel_hits: dict[str, list[str]] = Field(default_factory=dict)
    # The reformulations RAG-Fusion actually searched, empty for an ordinary search.
    # Worth surfacing: when multi-query retrieval helps or hurts, the reason is almost
    # always visible in the variants the rewriter chose.
    query_variants: list[str] = Field(default_factory=list)
    # Reranker input and output, when the caller asked for it. None means "not traced",
    # which is different from "traced and the reranker saw nothing".
    rerank: RerankTrace | None = None
    # Where the full stage-by-stage trace was written, "" when tracing is off.
    trace_path: str = ""
