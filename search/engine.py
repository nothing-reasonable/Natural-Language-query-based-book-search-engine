"""The search pipeline, end to end.

    query
      -> query understanding        (classify / normalise / expand / decompose)
      -> parallel retrieval         (BM25 + dense + knowledge graph)
      -> rank fusion + hard filters
      -> semantic reranking + signal blend
      -> personalised re-scoring
      -> results + explanations
"""

from __future__ import annotations

import logging
import math
import time
from contextlib import contextmanager

from config import Settings, settings as default_settings
from search.indexing.embedding import Embedder, make_embedder
from search.indexing.kg_index import KnowledgeGraph
from search.indexing.bm25_index import LexicalIndex
from search.indexing.entity_index import EntityIndex
from search.indexing.facet_index import FacetIndex
from search.indexing.title_index import ExactTitleIndex
from search.indexing.manifest import load_and_validate
from search.ranking.profile_index import ProfileStore, Session, UserProfile
from search.indexing.dense_index import VectorIndex
from ingest.run import load_indexed
from search.llm import LMStudio
from search.core.schemas import (
    Filters, IndexedBook, RerankEntry, RerankTrace, SearchHit, SearchOptions,
    SearchResponse,
)
from search.query.taxonomy import get_taxonomy
from search.retrieval import fusion
from search.ranking import personalize
from search.retrieval import rag_fusion
from search.ranking.explanation_generator import explain, flap_quote
# `_passage` is the reranker's input builder; the trace records exactly what was fed in
# rather than a lookalike, so the two cannot drift apart.
from search.ranking.rerank import NoOpReranker, Reranker, _passage, final_scores
# rerank2 adds the `lmstudio` backend and delegates every other one to rerank.py.
from search.ranking.rerank2 import make_reranker
from search.retrieval.retrieve import Retriever
from search.trace import make_tracer, record_search
from search.query.query_understanding import QueryUnderstanding

log = logging.getLogger(__name__)


class SearchEngine:
    def __init__(self, records: list[IndexedBook], lexical: LexicalIndex,
                 vector: VectorIndex | None, graph: KnowledgeGraph,
                 llm: LMStudio | None, embedder: Embedder | None = None,
                 settings: Settings = default_settings, *, index_generation: str = ""):
        self.settings = settings
        self.records = {r.book_id: r for r in records}
        self.llm = llm
        self.profiles = ProfileStore(settings)
        self.index_generation = index_generation

        # Built from the records already in memory -- ~0.3 s, no extra artifact.
        self.entities = EntityIndex(records)
        self.understanding = QueryUnderstanding(
            llm=llm,
            taxonomy=get_taxonomy(),
            vocabulary=set(getattr(lexical.retriever, "vocab_dict", {}) or {}),
            mode=settings.llm_query_understanding,
            entities=self.entities,
            year_bounds=_year_bounds(records),
            settings=settings,
        )
        self.embedder = embedder
        self.facets = FacetIndex(records)
        self.titles = ExactTitleIndex(records)
        self.retriever = Retriever(lexical, vector, graph, embedder, self.understanding,
                                   settings, facets=self.facets, titles=self.titles)
        # The reranker is chosen by `reranker_backend`, not by whether LM Studio happens
        # to be up: the cross-encoder runs in-process, so reranking survives a dead server.
        self.reranker: Reranker = (
            make_reranker(settings, llm) if settings.use_reranker else NoOpReranker()
        )

    # ------------------------------------------------------------------ construction
    @classmethod
    def load(cls, settings: Settings = default_settings, *, use_llm: bool = True) -> "SearchEngine":
        records = load_indexed(settings)
        manifest = load_and_validate(settings.index_manifest_path, records, settings)
        lexical = LexicalIndex.load(settings)
        graph = KnowledgeGraph.load(settings)

        llm = LMStudio(settings) if use_llm else None
        if llm is not None and not llm.is_available():
            log.warning("LM Studio unavailable -- chat query planning, RAG rewrites, and "
                        "the LM Studio reranker will be skipped.")
            llm = None

        # The dense channel is independent of LM Studio: embeddings have their own
        # backend, so semantic search keeps working even with no chat model loaded.
        vector, embedder = None, None
        if manifest.vector_enabled:
            try:
                vector = VectorIndex.open(settings)
                embedder = make_embedder(settings, llm)
                if manifest.embedding_dimension not in (None, embedder.dimension):
                    raise RuntimeError(
                        f"manifest dimension {manifest.embedding_dimension} does not match "
                        f"loaded model dimension {embedder.dimension}"
                    )
            except Exception as exc:  # noqa: BLE001
                log.warning("Dense channel unavailable (%s) -- run `build-index` to create it.", exc)
                vector = None

        return cls(records, lexical, vector, graph, llm, embedder, settings,
                   index_generation=manifest.generation)

    # ------------------------------------------------------------------ search
    def search(self, query: str, *, user_id: str | None = None,
               session: Session | None = None, top_k: int | None = None,
               use_rag_fusion: bool | None = None,
               filters: Filters | None = None,
               options: SearchOptions | None = None,
               trace_rerank: bool = False,
               trace: bool | None = None) -> SearchResponse:
        """One search. `use_rag_fusion` overrides the `rag_fusion` setting for this call.

        Under RAG-Fusion the query is retrieved several times over -- once per
        reformulation -- and the rankings are fused, so `top_k` is both how many fused
        candidates are carried forward and how many results come back.

        `trace_rerank` fills `SearchResponse.rerank` with what the reranker was shown and
        what it returned. Off by default: it carries every candidate passage in full, which
        is a few hundred kilobytes of Bengali text on a normal search.

        `trace` writes the full stage-by-stage account to a text file, overriding the
        `trace_queries` setting for this call. The path comes back on the response.
        """
        options = options or SearchOptions()
        if options.trace is not None:
            trace = options.trace
        tracer = make_tracer(query, self.settings, enabled=trace)
        # The written trace includes the reranker's inputs, so asking for one implies the
        # rerank trace -- otherwise its most useful section would be empty.
        trace_rerank = trace_rerank or options.trace_rerank or tracer.enabled
        fusing = (self.settings.rag_fusion if options.rag_fusion is None
                  else options.rag_fusion)
        if use_rag_fusion is not None:  # backward-compatible CLI argument
            fusing = use_rag_fusion
        top_k = top_k or (self.settings.rag_fusion_top_n if fusing
                          else self.settings.final_top_k)
        timings: dict[str, float] = {}
        profile: UserProfile | None = self.profiles.get(user_id) if user_id else None
        variants: list[str] = []

        # The plan comes from the query the user typed even under RAG-Fusion: its filters
        # and its normalised form are what the reranker and the hard filters must honour.
        # A variant is a retrieval device, not a restatement of the user's constraints.
        with _timed(timings, "understand"):
            plan = self.understanding.analyze(
                query, personalized=profile is not None, mode=options.plan_mode
            )
            plan.filters = self._applied_filters(plan.filters, filters)
            if filters is not None:
                for entity in plan.entities:
                    if entity.kind == "author" and (filters.authors or filters.author_ids):
                        entity.hard = (
                            entity.entity_id in plan.filters.author_ids
                            or entity.name in plan.filters.authors
                        )
                    elif entity.kind == "publisher" and filters.publishers:
                        entity.hard = entity.name in plan.filters.publishers

        shortlist_size = options.shortlist_size or max(self.settings.rerank_top_k, top_k)

        if fusing:
            with _timed(timings, "retrieve"):
                fused, variants, channel_hits = rag_fusion.search(
                    query, self.understanding, self.retriever, self.llm, self.settings,
                    top_n=shortlist_size, base_plan=plan,
                )
            with _timed(timings, "fuse"):
                pre_filter = fused
                # Defensive final check; retrieval has already applied the constraints
                # before each channel's top-k truncation.
                fused = fusion.apply_filters(fused, plan.filters, self.records)
                shortlist = fused[:shortlist_size]
        else:
            with _timed(timings, "retrieve"):
                channels = self.retriever.retrieve(plan)
                channel_hits = {n: [c.book_id for c in cands] for n, cands in channels.items()}

            with _timed(timings, "fuse"):
                fused = fusion.fuse(channels, self.settings)
                pre_filter = fused
                fused = fusion.apply_filters(fused, plan.filters, self.records)
                shortlist = fused[:shortlist_size]

        rerank_fallback = ""
        rerank_backend = "fusion"
        with _timed(timings, "rerank"):
            records = [self.records[c.book_id] for c in shortlist if c.book_id in self.records]
            shortlist = [c for c in shortlist if c.book_id in self.records]
            rerank_records = [
                self._record_with_passage(record, candidate)
                for record, candidate in zip(records, shortlist, strict=True)
            ]
            should_rerank = (self.settings.use_reranker if options.rerank is None
                             else options.rerank)
            semantic: list[float] = []
            if not should_rerank:
                rerank_fallback = "reranking disabled for this request; fusion order preserved"
            elif isinstance(self.reranker, NoOpReranker):
                rerank_fallback = "reranker unavailable; fusion order preserved"
            elif rerank_records:
                try:
                    semantic = self.reranker.score(plan.normalized_query, rerank_records)
                except Exception as exc:  # noqa: BLE001
                    log.warning("reranking failed (%s); preserving fusion order", exc)
                if not _valid_scores(semantic, len(rerank_records)):
                    semantic = []
                    rerank_fallback = "reranker failed or returned invalid scores; fusion order preserved"

            if semantic:
                rerank_backend = getattr(self.reranker, "name", type(self.reranker).__name__)
                ranked = final_scores(shortlist, self.records, semantic, self.settings)
            else:
                ranked = _fusion_order(shortlist)

        # `ranked` is reassigned by personalisation below; both traces want the ordering
        # the relevance signals produced, so hold on to it here.
        blended = ranked
        rerank_trace = (self._rerank_trace(
            plan.normalized_query, rerank_records, semantic, ranked,
            backend=rerank_backend, fallback=rerank_fallback,
        ) if trace_rerank else None)

        with _timed(timings, "personalize"):
            affinity = personalize.build_affinity(profile, session, self.records, self.settings)
            ranked = personalize.apply(ranked, self.records, affinity, self.settings)

        entity_evidence = self.retriever._entity_evidence(plan)
        hits = [self._to_hit(item, entity_evidence) for item in ranked[:top_k]]

        if session is not None:
            session.record_query(query)
        if profile is not None:
            self.profiles.record(profile.user_id, "query", query)

        record_search(
            tracer, plan=plan, fusing=fusing, variants=variants,
            channel_hits=channel_hits, pre_filter=pre_filter, fused=fused,
            shortlist=shortlist, semantic=semantic, ranked=blended, hits=hits,
            timings=timings, records=self.records, rerank_trace=rerank_trace,
            personalized=profile is not None, settings=self.settings,
        )
        written = tracer.write()

        return SearchResponse(
            query=query, plan=plan, hits=hits, timings_ms=timings,
            candidates=[c.book_id for c in fused],
            channel_hits=channel_hits,
            query_variants=variants,
            rerank=rerank_trace,
            trace_path=str(written) if written else "",
            applied_filters=plan.filters,
            rerank_backend=rerank_backend,
            rerank_fallback=rerank_fallback,
            index_generation=self.index_generation,
        )

    def _rerank_trace(self, normalized_query: str, records: list[IndexedBook],
                      semantic: list[float], ranked: list[tuple], *, backend: str,
                      fallback: str = "") -> RerankTrace:
        """What the reranker read, what it returned, and where each candidate ended up.

        `records` is the shortlist in fusion order and `semantic` is the reranker's output
        for it, so `strict=True` turns any future drift between the two into an error here
        rather than into a silently mismatched trace.

        Called before personalisation, so `final_rank` is the ranking the *relevance*
        signals produced -- which is the one worth attributing to the reranker.
        """
        final_rank = {item[0].book_id: rank for rank, item in enumerate(ranked, start=1)}
        entries = [
            RerankEntry(
                book_id=record.book.book_id,
                title=record.book.title,
                fusion_rank=position,
                passage=_passage(record, self.settings),
                score=round(float(value), 4),
                final_rank=final_rank.get(record.book.book_id),
            )
            for position, (record, value) in enumerate(zip(records, semantic), start=1)
        ]
        return RerankTrace(
            backend=backend,
            model=getattr(self.reranker, "model_name", ""),
            query=normalized_query,
            entries=entries,
            fallback=fallback,
        )

    # ------------------------------------------------------------------ helpers
    def _to_hit(self, item: tuple, entity_evidence: list | None = None) -> SearchHit:
        candidate, score, relevance, components, matched = item
        record = self.records[candidate.book_id]
        evidence = (entity_evidence or []) + candidate.evidence + personalize.evidence_for(matched)
        excerpt, source = _best_excerpt(candidate)
        if not excerpt:
            excerpt = flap_quote(record, evidence)
            source = "description" if excerpt else ""
        return SearchHit(
            book=record.book,
            enrichment=record.enrichment,
            score=round(score, 4),
            relevance=round(relevance, 4),
            components={k: round(v, 4) for k, v in components.items()},
            channels=sorted(set(candidate.channels)),
            evidence=evidence,
            explanation=explain(record, evidence, components),
            match_excerpt=excerpt,
            match_source=source,
        )

    def _applied_filters(self, inferred: Filters, explicit: Filters | None) -> Filters:
        """Explicit fields replace inferred values for that field; all others intersect."""
        if explicit is None or explicit.is_empty():
            return inferred
        applied = inferred.model_copy(deep=True)
        tax = get_taxonomy()
        for field in ("publishers", "subjects", "genres", "periods", "places"):
            values = list(getattr(explicit, field))
            if values:
                if field == "publishers":
                    values = self.facets.resolve_publishers(values)
                if field in ("subjects", "genres", "periods", "places"):
                    values = tax.canonicalize_all(values, field)
                setattr(applied, field, values)
        if explicit.authors or explicit.author_ids:
            names, ids = self.entities.resolve_authors(explicit.authors)
            applied.authors = names
            applied.author_ids = list(dict.fromkeys([*explicit.author_ids, *ids]))
        if explicit.language is not None:
            applied.language = explicit.language
        if explicit.year_from is not None or explicit.year_to is not None:
            applied.year_from = explicit.year_from
            applied.year_to = explicit.year_to
        return applied

    @staticmethod
    def _record_with_passage(record: IndexedBook, candidate) -> IndexedBook:
        excerpt, source = _best_excerpt(candidate)
        if not excerpt or source not in {"description", "table_of_contents"}:
            return record
        copied = record.model_copy(deep=True)
        copied.book.description = excerpt
        copied.book.table_of_contents = ""
        return copied


def _year_bounds(records: list[IndexedBook]) -> tuple[int, int] | None:
    """The publication years the catalogue actually spans."""
    years = [r.book.publish_year for r in records if r.book.publish_year]
    return (min(years), max(years)) if years else None


def _valid_scores(values: list[float], expected: int) -> bool:
    if len(values) != expected:
        return False
    try:
        return all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


def _fusion_order(shortlist) -> list[tuple]:
    """Keep candidate order exactly; no fabricated semantic score enters the blend."""
    return [
        (candidate, candidate.fusion_score, {"fusion": candidate.fusion_score})
        for candidate in shortlist
    ]


def _best_excerpt(candidate) -> tuple[str, str]:
    for evidence in candidate.evidence:
        if evidence.excerpt and evidence.source_field in {"description", "table_of_contents"}:
            return evidence.excerpt, evidence.source_field
    return "", ""


@contextmanager
def _timed(target: dict[str, float], label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        target[label] = round((time.perf_counter() - start) * 1000, 1)
