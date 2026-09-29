"""Acceptance coverage for PLAN.md. Prepared here; execute on the target PC."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import Settings
from search.core.schemas import Book, Candidate, Enrichment, EnrichmentRecord, Filters, IndexedBook, QueryPlan
from search.core.store import write_jsonl
from search.engine import SearchEngine, _fusion_order, _valid_scores
from search.indexing.entity_index import EntityIndex
from search.indexing.facet_index import FacetIndex
from search.indexing.manifest import IndexCompatibilityError, expected_manifest, load_and_validate, publish
from search.indexing.passages import enrichment_input_fingerprint, make_passages
from search.indexing.title_index import ExactTitleIndex
from search.query.query_understanding import QueryUnderstanding
from search.retrieval import rag_fusion
from search.retrieval.retrieve import Retriever
from web import _pages, _parse_filters, _positive_int


def record(book_id: str, title: str, author: str, **kwargs) -> IndexedBook:
    enrichment = Enrichment(**{k: v for k, v in kwargs.items() if k in Enrichment.model_fields})
    book = Book(book_id=book_id, title=title, author=author,
                author_raw=author, author_id=f"author-{author}",
                **{k: v for k, v in kwargs.items() if k in Book.model_fields})
    return IndexedBook(book=book, enrichment=enrichment)


@pytest.fixture
def catalogue():
    return [
        record("b1", "একাত্তরের দিনগুলি", "জাহানারা ইমাম", publish_year=1986,
               subjects=["মুক্তিযুদ্ধ"], publisher="সন্ধানী"),
        record("b2", "দেয়াল", "হুমায়ূন আহমেদ", publish_year=2013,
               subjects=["রাজনীতি"], publisher="অন্যপ্রকাশ"),
        record("b3", "হুমায়ূন আহমেদের জীবন", "অন্য লেখক", publish_year=2020,
               subjects=["জীবনী"], publisher="সময়"),
    ]


def test_exact_title_lookup_survives_normalisation(catalogue):
    titles = ExactTitleIndex(catalogue)
    assert titles.find("‘একাত্তরের দিনগুলি’ নামের বই") == ["b1"]
    assert titles.find("একাত্তরের দিনগুলি") == ["b1"]


def test_person_as_subject_is_not_an_author_filter(catalogue):
    understanding = QueryUnderstanding(
        llm=None, entities=EntityIndex(catalogue), mode="never"
    )
    plan = understanding.analyze("হুমায়ূন আহমেদ সম্পর্কে লেখা বই")
    assert not plan.filters.authors
    assert any(entity.name == "হুমায়ূন আহমেদ" and not entity.hard for entity in plan.entities)


def test_english_publication_expression_is_not_a_historical_subject_date():
    assert QueryUnderstanding._year_filter("books published in the 1990s", []) == (1990, 1999)
    assert QueryUnderstanding._year_filter(
        "books published between 1990 and 2000", [1990, 2000]
    ) == (1990, 2000)
    assert QueryUnderstanding._year_filter("books published after 2000", [2000]) == (2000, None)
    assert QueryUnderstanding._year_filter("books about the 1971 war", [1971]) is None
    assert QueryUnderstanding._year_filter("books written about the 1971 war", [1971]) is None


def test_user_filter_overrides_same_inferred_field_and_keeps_other_constraints(catalogue):
    engine = object.__new__(SearchEngine)
    engine.entities = EntityIndex(catalogue)
    engine.facets = FacetIndex(catalogue)
    inferred = Filters(authors=["জাহানারা ইমাম"], author_ids=["old"], year_from=1980)
    applied = engine._applied_filters(inferred, Filters(authors=["হুমায়ূন আহমেদ"]))
    assert applied.authors == ["হুমায়ূন আহমেদ"]
    assert applied.author_ids == ["author-হুমায়ূন আহমেদ"]
    assert applied.year_from == 1980


class _LexicalSpy:
    def __init__(self):
        self.allowed = None

    def search(self, terms, k, allowed_ids=None):
        self.allowed = allowed_ids
        return [("b1", 1.0, ["মুক্তিযুদ্ধ"])] if "b1" in (allowed_ids or set()) else []


def test_filters_reach_lexical_channel_before_its_top_k(catalogue):
    lexical = _LexicalSpy()
    understanding = QueryUnderstanding(llm=None, mode="never")
    retriever = Retriever(
        lexical=lexical, vector=None, graph=None, embedder=None,
        understanding=understanding, facets=FacetIndex(catalogue),
        settings=Settings(enabled_channels=["lexical"], channel_top_k=1),
    )
    plan = QueryPlan(raw_query="মুক্তিযুদ্ধ", normalized_query="মুক্তিযুদ্ধ",
                     filters=Filters(authors=["জাহানারা ইমাম"]))
    retriever._lexical(plan)
    assert lexical.allowed == {"b1"}


def test_rag_fusion_variants_keep_original_filters(monkeypatch):
    class Understanding:
        def analyze(self, text, mode="never"):
            return QueryPlan(raw_query=text, normalized_query=text)

    class Retrieval:
        seen = []

        def retrieve(self, plan):
            self.seen.append(plan.filters.model_copy(deep=True))
            return {"lexical": [Candidate(book_id="b1", channel="lexical", rank=1, score=1)]}

    monkeypatch.setattr(rag_fusion, "generate_variants", lambda llm, query, count: ["variant"])
    retrieval = Retrieval()
    base = QueryPlan(raw_query="original", normalized_query="original",
                     filters=Filters(authors=["লেখক"]))
    rag_fusion.search("original", Understanding(), retrieval, None,
                      Settings(enabled_channels=["lexical"]), base_plan=base)
    assert len(retrieval.seen) == 2
    assert all(filters.authors == ["লেখক"] for filters in retrieval.seen)


def test_passages_are_bounded_overlapping_stable_and_attributed():
    words = [f"শব্দ{i}" for i in range(600)]
    records = [record("long", "দীর্ঘ বই", "লেখক", description=" ".join(words))]
    first = make_passages(records, tokens=256, overlap=48)
    second = make_passages(records, tokens=256, overlap=48)
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert max(len(chunk.text.split()) for chunk in first) <= 256
    assert first[0].text.split()[-48:] == first[1].text.split()[:48]
    assert {chunk.source_field for chunk in first} == {"description"}


def test_changed_description_invalidates_enrichment_provenance():
    book = Book(book_id="b", title="t", description="old")
    before = enrichment_input_fingerprint(book)
    book.description = "new"
    assert enrichment_input_fingerprint(book) != before


def test_legacy_enrichment_without_provenance_is_not_reused(tmp_path):
    from ingest.run import load_indexed

    config = Settings(artifacts_dir=tmp_path)
    book = Book(book_id="b", title="মুক্তিযুদ্ধের দলিল", description="মুক্তিযুদ্ধের ইতিহাস")
    write_jsonl(config.books_path, [book])
    write_jsonl(config.enrichment_path, [
        EnrichmentRecord(book_id="b", enrichment=Enrichment(subjects=["stale-model-tag"]))
    ])
    loaded = load_indexed(config, derive_facts=False)
    assert "stale-model-tag" not in loaded[0].enrichment.subjects


def test_same_dimension_different_model_is_incompatible(tmp_path, catalogue):
    old = Settings(artifacts_dir=tmp_path, embedding_model="model-a")
    manifest = expected_manifest(catalogue, old, dimension=768)
    publish(manifest, old.index_manifest_path)
    new = Settings(artifacts_dir=tmp_path, embedding_model="model-b")
    with pytest.raises(IndexCompatibilityError, match="embedding model"):
        load_and_validate(new.index_manifest_path, catalogue, new)


def test_reranker_scores_must_align_and_be_finite():
    assert _valid_scores([.2, .8], 2)
    assert not _valid_scores([.2], 2)
    assert not _valid_scores([np.nan, .8], 2)


def test_reranker_failure_order_contains_no_semantic_component():
    from search.retrieval.fusion import Fused

    candidates = [Fused(book_id="first", fusion_score=1), Fused(book_id="second", fusion_score=.5)]
    fallback = _fusion_order(candidates)
    assert [row[0].book_id for row in fallback] == ["first", "second"]
    assert all("semantic" not in row[2] for row in fallback)


def test_unavailable_reranker_degrades_to_noop(monkeypatch):
    import search.ranking.rerank as module

    monkeypatch.setattr(module, "CrossEncoderReranker",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))
    assert module.make_reranker(Settings()).name == "noop"


@pytest.mark.parametrize("raw", ["bad", [], {"year_from": True}, {"unknown": "x"}])
def test_malformed_filters_are_rejected(raw):
    with pytest.raises(ValueError):
        _parse_filters(raw)


def test_pagination_rejects_boolean_and_out_of_range_values():
    with pytest.raises(ValueError):
        _positive_int(True, "page", maximum=10)
    with pytest.raises(ValueError):
        _positive_int(11, "page", maximum=10)
    assert _pages(48, 12) == 4
