"""Candidate generation across all channels, run in parallel.

Each channel produces `Candidate` objects carrying their own evidence, so nothing later
in the pipeline has to guess why a book showed up.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from search.core import bengali
from config import Settings, settings as default_settings
from search.indexing.embedding import Embedder
from search.indexing.kg_index import KnowledgeGraph
from search.indexing.bm25_index import LexicalIndex
from search.indexing.dense_index import VectorIndex
from search.indexing.facet_index import FacetIndex
from search.core.schemas import Candidate, Evidence, QueryPlan
from search.query.query_understanding import QueryUnderstanding

log = logging.getLogger(__name__)


class Retriever:
    def __init__(self, lexical: LexicalIndex, vector: VectorIndex | None,
                 graph: KnowledgeGraph, embedder: Embedder | None,
                 understanding: QueryUnderstanding, settings: Settings = default_settings,
                 facets: FacetIndex | None = None):
        self.lexical = lexical
        self.vector = vector
        self.graph = graph
        self.embedder = embedder
        self.understanding = understanding
        self.settings = settings
        self.facets = facets
        # One pool for the life of the engine. Building a fresh one per query costs more
        # than the channels now take to run.
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="retrieve")

    def retrieve(self, plan: QueryPlan) -> dict[str, list[Candidate]]:
        """Run every enabled channel in parallel. A channel that raises is dropped, not
        propagated: a dead index should cost recall, not the whole search."""
        available = {"lexical": self._lexical, "dense": self._dense,
                     "graph": self._graph, "facet": self._facet}
        channels = {n: fn for n, fn in available.items() if n in self.settings.enabled_channels}

        futures = {name: self._pool.submit(fn, plan) for name, fn in channels.items()}
        results = {}
        for name, future in futures.items():
            try:
                results[name] = future.result()
            except Exception as exc:  # noqa: BLE001 - a dead channel must not kill the search
                log.warning("retrieval channel %r failed: %s", name, exc)
                results[name] = []
        return {name: hits for name, hits in results.items() if hits}

    def close(self) -> None:
        """Release the worker threads. Long-lived processes that build many engines
        (the ablation runner) would otherwise accumulate a pool per engine."""
        self._pool.shutdown(wait=False)

    # ------------------------------------------------------------------ channels
    def _entity_evidence(self, plan: QueryPlan) -> list[Evidence]:
        """Why a hard-filtered result set is what it is."""
        out = []
        for entity in plan.entities:
            if not entity.hard:
                continue
            label = "লেখক" if entity.kind == "author" else "প্রকাশক"
            out.append(Evidence(channel="facet",
                                detail=f"{label} মিলেছে: {entity.name}",
                                terms=[entity.name]))
        return out

    def _facet(self, plan: QueryPlan) -> list[Candidate]:
        """Books that satisfy the hard constraints, whether or not any other channel
        happened to surface them. Without this, a constraint can only ever remove
        results, never find them."""
        if self.facets is None or plan.filters.is_empty():
            return []
        selected = self.facets.select(plan.filters)
        if not selected:
            return []
        ranked = self.facets.rank(selected, plan.concepts, self.settings.channel_top_k)

        described = ", ".join(
            e.name for e in plan.entities if e.hard
        ) or _describe_filters(plan.filters)
        return [
            Candidate(
                book_id=book_id, channel="facet", rank=rank, score=score,
                evidence=[Evidence(channel="facet",
                                   detail=f"শর্ত মিলেছে: {described}" if described else "মেটাডেটা শর্ত মিলেছে",
                                   terms=[described] if described else [])],
            )
            for rank, (book_id, score) in enumerate(ranked, start=1)
        ]

    def _lexical(self, plan: QueryPlan) -> list[Candidate]:
        terms = self.understanding.search_terms(plan)
        hits = self.lexical.search(terms, k=self.settings.channel_top_k)
        # Matching happens on stems; explanations show the words the user actually typed.
        surface = bengali.surface_forms(terms)

        candidates = []
        for rank, (book_id, score, matched) in enumerate(hits, start=1):
            words = [surface.get(m, m) for m in matched[:6]]
            candidates.append(
                Candidate(
                    book_id=book_id, channel="lexical", rank=rank, score=score,
                    evidence=[Evidence(channel="lexical",
                                       detail="শব্দ মিলেছে: " + ", ".join(words),
                                       terms=words)],
                )
            )
        return candidates

    def _dense(self, plan: QueryPlan) -> list[Candidate]:
        if self.vector is None or self.embedder is None:
            return []
        query_text = " ".join([plan.normalized_query, *plan.expanded_terms[:8]])
        vector = self.embedder.embed_query(query_text)
        hits = self.vector.search(vector, k=self.settings.channel_top_k, filters=plan.filters)
        return [
            Candidate(
                book_id=book_id, channel="dense", rank=rank, score=score,
                evidence=[Evidence(channel="dense", detail=f"অর্থগত মিল ({score:.2f})")],
            )
            for rank, (book_id, score, _snippet) in enumerate(hits, start=1)
        ]

    def _graph(self, plan: QueryPlan) -> list[Candidate]:
        term_kinds = _term_kinds(plan)
        if plan.steps:
            found = self.graph.run(plan.steps)
        else:
            c = plan.concepts
            if not (c.subjects or c.periods or c.places or c.genres):
                return []
            matches = self.graph.find_books(
                subjects=c.subjects, periods=c.periods, places=c.places, genres=c.genres
            )
            found = {
                node.split(":", 1)[1]: {"book_terms": terms, "author_terms": [], "authors": []}
                for node, terms in matches.items()
            }

        # Rank by how *specific* the matched concepts are rather than how many matched.
        # In a catalogue that is mostly about মুক্তিযুদ্ধ, matching মুক্তিযুদ্ধ says almost
        # nothing while matching ভাষা আন্দোলন says a lot. Author evidence counts double
        # because it is exactly the part a text search could never have found.
        def specificity(evidence: dict) -> float:
            return (
                sum(self._idf(t, term_kinds) for t in evidence["book_terms"])
                + 2.0 * sum(self._idf(t, term_kinds) for t in evidence["author_terms"])
            )

        scored = [(book_id, ev, specificity(ev)) for book_id, ev in found.items()]
        # A broad concept ties thousands of books at one identical score -- "ইতিহাস" is a
        # genre on 1,897 of them. Truncating that tie to `channel_top_k` by book_id is a
        # lottery: the hash is unrelated to relevance, so a well-tagged book loses its
        # graph evidence entirely because its id happens to start with `f`. Richer
        # metadata breaks the tie instead, as it does in the facet channel.
        scored = sorted(
            (item for item in scored if item[2] >= self.settings.graph_min_specificity),
            key=lambda item: (-item[2], -self._quality(item[0]), item[0]),
        )

        candidates = []
        for rank, (book_id, evidence, score) in enumerate(scored[: self.settings.channel_top_k], start=1):
            matched = evidence["book_terms"] + evidence["author_terms"]
            details = []
            if evidence["book_terms"]:
                details.append("বিষয় মিলেছে: " + ", ".join(evidence["book_terms"]))
            if evidence["author_terms"]:
                details.append("লেখকের পরিচয় মিলেছে: " + ", ".join(evidence["author_terms"]))
            candidates.append(
                Candidate(
                    book_id=book_id, channel="graph", rank=rank, score=score,
                    evidence=[Evidence(channel="graph", detail="; ".join(details), terms=matched)],
                )
            )
        return candidates

    _IDF_KINDS = ("subject", "event", "occupation", "genre", "place", "period", "topic")

    def _idf(self, term: str, term_kinds: dict[str, set[str]] | None = None) -> float:
        """How rare is this concept, in the facet the query actually asked about.

        Concept nodes are keyed by facet, and the same word can name several: `ইতিহাস` is
        a genre on 1,897 books *and* an event on 15. Scanning the facets in a fixed order
        and taking the first hit returned the event's idf (5.72) for a genre match worth
        0.94 -- a six-fold overstatement applied to every one of those 1,897 books, which
        is precisely the specificity ranking this function exists to provide, inverted.

        So restrict to the facets the term was queried under. A term queried under two
        facets is genuinely ambiguous -- the book may be linked to either node -- and
        takes the lower value, because over-crediting a common concept is the failure
        mode that actually hurts.
        """
        kinds = (term_kinds or {}).get(term) or self._IDF_KINDS
        values = [v for v in (self.graph.idf(k, term) for k in kinds) if v]
        if not values:
            return 0.0
        return min(values) if term_kinds and term in term_kinds else values[0]

    def _quality(self, book_id: str) -> float:
        record = self.facets.records.get(book_id) if self.facets is not None else None
        return record.book.metadata_quality if record else 0.0


def _term_kinds(plan: QueryPlan) -> dict[str, set[str]]:
    """Which graph facet each queried concept came from.

    `find_books`/`run` report *which* terms matched but not which facet node they matched
    through, and the caller cannot recover it from the name alone. The query plan can:
    a term only ever matches under the facet it was asked for.
    """
    # plan field -> graph node kind
    pairs = (("subjects", "subject"), ("genres", "genre"), ("periods", "period"),
             ("places", "place"), ("occupations", "occupation"), ("events", "event"))
    out: dict[str, set[str]] = {}
    sources = [plan.concepts, *plan.steps]
    for source in sources:
        for field, kind in pairs:
            for term in getattr(source, field, []) or []:
                out.setdefault(term, set()).add(kind)
    return out


def _describe_filters(filters) -> str:
    """Short human label for whatever constraint was applied."""
    parts = []
    for label, values in (("লেখক", filters.authors), ("প্রকাশক", filters.publishers),
                          ("বিষয়", filters.subjects), ("ধরন", filters.genres),
                          ("কাল", filters.periods), ("স্থান", filters.places)):
        if values:
            parts.append(f"{label}: {', '.join(values[:3])}")
    # An open-ended range has to say which end is open. Formatting a missing bound as
    # an empty string produced "প্রকাশকাল: -2100", which reads as a negative year, or as
    # the book's own publication date -- neither of which is what it meant.
    low, high = filters.year_from, filters.year_to
    if low is not None and high is not None:
        parts.append(f"প্রকাশকাল: {low}" if low == high else f"প্রকাশকাল: {low}–{high}")
    elif low is not None:
        parts.append(f"প্রকাশকাল: {low} সাল থেকে")
    elif high is not None:
        parts.append(f"প্রকাশকাল: {high} সাল পর্যন্ত")
    return "; ".join(parts)
