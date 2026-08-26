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
import time
from contextlib import contextmanager

from config import Settings, settings as default_settings
from search.indexing.embedding import Embedder, make_embedder
from search.indexing.kg_index import KnowledgeGraph
from search.indexing.bm25_index import LexicalIndex
from search.indexing.entity_index import EntityIndex
from search.indexing.facet_index import FacetIndex
from search.ranking.profile_index import ProfileStore, Session, UserProfile
from search.indexing.dense_index import VectorIndex
from ingest.run import load_indexed
from search.llm import LMStudio
from search.core.schemas import IndexedBook, SearchHit, SearchResponse
from search.query.taxonomy import get_taxonomy
from search.retrieval import fusion
from search.ranking import personalize
from search.retrieval import rag_fusion
from search.ranking.explanation_generator import explain
from search.ranking.rerank import NoOpReranker, Reranker, final_scores
# rerank2 adds the `lmstudio` backend and delegates every other one to rerank.py.
from search.ranking.rerank2 import make_reranker
from search.retrieval.retrieve import Retriever
from search.query.query_understanding import QueryUnderstanding

log = logging.getLogger(__name__)


class SearchEngine:
    def __init__(self, records: list[IndexedBook], lexical: LexicalIndex,
                 vector: VectorIndex | None, graph: KnowledgeGraph,
                 llm: LMStudio | None, embedder: Embedder | None = None,
                 settings: Settings = default_settings):
        self.settings = settings
        self.records = {r.book_id: r for r in records}
        self.llm = llm
        self.profiles = ProfileStore(settings)

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
        self.retriever = Retriever(lexical, vector, graph, embedder, self.understanding,
                                   settings, facets=self.facets)
        # The reranker is chosen by `reranker_backend`, not by whether LM Studio happens
        # to be up: the cross-encoder runs in-process, so reranking survives a dead server.
        self.reranker: Reranker = (
            make_reranker(settings, llm) if settings.use_reranker else NoOpReranker()
        )

    # ------------------------------------------------------------------ construction
    @classmethod
    def load(cls, settings: Settings = default_settings, *, use_llm: bool = True) -> "SearchEngine":
        records = load_indexed(settings)
        lexical = LexicalIndex.load(settings)
        graph = KnowledgeGraph.load(settings)

        llm = LMStudio(settings) if use_llm else None
        if llm is not None and not llm.is_available():
            log.warning("LM Studio unavailable -- dense retrieval, reranking and query "
                        "understanding will be skipped.")
            llm = None

        # The dense channel is independent of LM Studio: embeddings have their own
        # backend, so semantic search keeps working even with no chat model loaded.
        vector, embedder = None, None
        try:
            vector = VectorIndex.open(settings)
            embedder = make_embedder(settings, llm)
        except Exception as exc:  # noqa: BLE001
            log.warning("Dense channel unavailable (%s) -- run `build-index` to create it.", exc)
            vector = None

        return cls(records, lexical, vector, graph, llm, embedder, settings)

    # ------------------------------------------------------------------ search
    def search(self, query: str, *, user_id: str | None = None,
               session: Session | None = None, top_k: int | None = None,
               use_rag_fusion: bool | None = None) -> SearchResponse:
        """One search. `use_rag_fusion` overrides the `rag_fusion` setting for this call.

        Under RAG-Fusion the query is retrieved several times over -- once per
        reformulation -- and the rankings are fused, so `top_k` is both how many fused
        candidates are carried forward and how many results come back.
        """
        fusing = self.settings.rag_fusion if use_rag_fusion is None else use_rag_fusion
        top_k = top_k or (self.settings.rag_fusion_top_n if fusing
                          else self.settings.final_top_k)
        timings: dict[str, float] = {}
        profile: UserProfile | None = self.profiles.get(user_id) if user_id else None
        variants: list[str] = []

        # The plan comes from the query the user typed even under RAG-Fusion: its filters
        # and its normalised form are what the reranker and the hard filters must honour.
        # A variant is a retrieval device, not a restatement of the user's constraints.
        with _timed(timings, "understand"):
            plan = self.understanding.analyze(query, personalized=profile is not None)

        if fusing:
            with _timed(timings, "retrieve"):
                fused, variants, channel_hits = rag_fusion.search(
                    query, self.understanding, self.retriever, self.llm, self.settings,
                    top_n=max(self.settings.rerank_top_k, top_k),
                )
            with _timed(timings, "fuse"):
                fused = fusion.apply_filters(fused, plan.filters, self.records)
                shortlist = fused[: max(self.settings.rerank_top_k, top_k)]
        else:
            with _timed(timings, "retrieve"):
                channels = self.retriever.retrieve(plan)
                channel_hits = {n: [c.book_id for c in cands] for n, cands in channels.items()}

            with _timed(timings, "fuse"):
                fused = fusion.fuse(channels, self.settings)
                fused = fusion.apply_filters(fused, plan.filters, self.records)
                shortlist = fused[: max(self.settings.rerank_top_k, top_k)]

        with _timed(timings, "rerank"):
            records = [self.records[c.book_id] for c in shortlist if c.book_id in self.records]
            shortlist = [c for c in shortlist if c.book_id in self.records]
            semantic = self.reranker.score(plan.normalized_query, records) if records else []
            ranked = final_scores(shortlist, self.records, semantic, self.settings)

        with _timed(timings, "personalize"):
            affinity = personalize.build_affinity(profile, session, self.records, self.settings)
            ranked = personalize.apply(ranked, self.records, affinity, self.settings)

        entity_evidence = self.retriever._entity_evidence(plan)
        hits = [self._to_hit(item, entity_evidence) for item in ranked[:top_k]]

        if session is not None:
            session.record_query(query)
        if profile is not None:
            self.profiles.record(profile.user_id, "query", query)

        return SearchResponse(
            query=query, plan=plan, hits=hits, timings_ms=timings,
            candidates=[c.book_id for c in fused],
            channel_hits=channel_hits,
            query_variants=variants,
        )

    # ------------------------------------------------------------------ helpers
    def _to_hit(self, item: tuple, entity_evidence: list | None = None) -> SearchHit:
        candidate, score, relevance, components, matched = item
        record = self.records[candidate.book_id]
        evidence = (entity_evidence or []) + candidate.evidence + personalize.evidence_for(matched)
        return SearchHit(
            book=record.book,
            enrichment=record.enrichment,
            score=round(score, 4),
            relevance=round(relevance, 4),
            components={k: round(v, 4) for k, v in components.items()},
            channels=sorted(set(candidate.channels)),
            evidence=evidence,
            explanation=explain(record, evidence, components),
        )


def _year_bounds(records: list[IndexedBook]) -> tuple[int, int] | None:
    """The publication years the catalogue actually spans."""
    years = [r.book.publish_year for r in records if r.book.publish_year]
    return (min(years), max(years)) if years else None


@contextmanager
def _timed(target: dict[str, float], label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        target[label] = round((time.perf_counter() - start) * 1000, 1)
