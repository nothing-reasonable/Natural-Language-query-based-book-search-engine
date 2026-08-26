"""Tests for the parts that must stay correct: Bengali analysis, the controlled
vocabulary, author identity, rank fusion, graph traversal and personalisation.

None of them need LM Studio -- run with `pytest` from the project root.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from search.core import bengali
from config import Settings
from search.indexing.embedding import LMStudioEmbedder, make_embedder
from search.indexing.kg_index import KnowledgeGraph
from search.ranking.profile_index import Session, UserProfile
from ingest.clean import author_key, clean, dedupe_books
from search.indexing.dense_index import build_where
from search.core.schemas import (
    Book, Candidate, Enrichment, Filters, GraphStep, IndexedBook, QueryPlan,
)
from search.retrieval import fusion
from search.ranking import personalize
from search.retrieval.retrieve import Retriever
from search.query.query_understanding import QueryUnderstanding
from search.query.taxonomy import get_taxonomy


# --------------------------------------------------------------------------- Bengali

@pytest.mark.parametrize(
    "word",
    ["মুক্তিযুদ্ধ", "একাত্তর", "ইতিহাস", "কিশোর", "গল্প", "বাংলাদেশ"],
)
def test_stem_is_stable_across_inflections(word):
    """Every inflected form of a word must land on the same stem, or nothing matches."""
    stem = bengali.stem(word)
    for suffix in ("ের", "রা", "গুলো", "দের", "ে", "র"):
        assert bengali.stem(word + suffix) == stem, f"{word}+{suffix}"


def test_stem_is_idempotent():
    for word in ["একাত্তরের", "মুক্তিযুদ্ধের", "কিশোরদের", "গল্পগুলো"]:
        once = bengali.stem(word)
        assert bengali.stem(once) == once


def test_bengali_digits_fold_to_ascii():
    assert bengali.analyze("১৯৭১ সাল") == bengali.analyze("1971 সাল")


def test_stopwords_and_punctuation_are_dropped():
    tokens = bengali.analyze("এই বইটি সম্পর্কে কিছু কথা।")
    assert "এই" not in tokens and "।" not in " ".join(tokens)


def test_surface_forms_map_stems_back_to_typed_words():
    mapping = bengali.surface_forms(["একাত্তরের গল্প"])
    assert mapping[bengali.stem("একাত্তরের")] == "একাত্তরের"


# --------------------------------------------------------------------------- taxonomy

@pytest.mark.parametrize(
    "surface,canonical",
    [
        ("একাত্তর", "মুক্তিযুদ্ধ"),
        ("১৯৭১", "মুক্তিযুদ্ধ"),
        ("liberation war", "মুক্তিযুদ্ধ"),
        ("প্রাক্তন কনসাল", "কূটনীতিক"),
        ("ex consulate", "কূটনীতিক"),
        ("স্মৃতিচারণ", "স্মৃতিকথা"),
        ("পূর্ব পাকিস্তান", "বাংলাদেশ"),
    ],
)
def test_taxonomy_canonicalises_aliases(surface, canonical):
    assert get_taxonomy().canonicalize(surface) == canonical


def test_taxonomy_expansion_includes_aliases():
    expanded = get_taxonomy().expand("১৯৭১")
    assert "মুক্তিযুদ্ধ" in expanded and "একাত্তর" in expanded


def test_period_lookup_by_year():
    tax = get_taxonomy()
    assert tax.period_for_year(1965) == "পাকিস্তান আমল"
    assert tax.period_for_year(1980) == "স্বাধীন বাংলাদেশ"


# --------------------------------------------------------------------------- ingestion

def test_author_key_ignores_honorifics():
    assert author_key("ড. আনিসুজ্জামান") == author_key("অধ্যাপক আনিসুজ্জামান")
    assert author_key("মোঃ রেজাউল করিম") == author_key("মো. রেজাউল করিম")


def _book(book_id: str, title: str, author: str, **kw) -> Book:
    return Book(book_id=book_id, title=title, author=author, author_raw=author, **kw)


def test_clean_merges_the_same_work_and_keeps_the_richest_copy():
    books = [
        _book("1", "মুক্তিযুদ্ধের ইতিহাস", "ড. করিম", description="সংক্ষিপ্ত"),
        _book("2", "মুক্তিযুদ্ধের ইতিহাস", "করিম", description="অনেক লম্বা বিবরণ যা বেশি তথ্যবহুল"),
    ]
    merged = clean(books)
    assert len(merged) == 1
    assert merged[0].description.startswith("অনেক লম্বা")


def test_different_authors_are_not_merged():
    books = [
        _book("1", "মুক্তিযুদ্ধের গল্প", "রফিকুর রশীদ"),
        _book("2", "মুক্তিযুদ্ধের গল্প", "সেলিনা হোসেন"),
    ]
    assert len(clean(books)) == 2


def test_dedupe_is_a_no_op_for_distinct_titles():
    books = [_book("1", "ক", "লেখক এক"), _book("2", "খ", "লেখক দুই")]
    assert len(dedupe_books(books)) == 2


# --------------------------------------------------------------------------- fusion

def _candidate(book_id: str, channel: str, rank: int) -> Candidate:
    return Candidate(book_id=book_id, channel=channel, rank=rank, score=1.0 / rank)


def test_rrf_rewards_agreement_between_channels():
    channels = {
        "lexical": [_candidate("a", "lexical", 1), _candidate("b", "lexical", 2)],
        "dense": [_candidate("b", "dense", 1), _candidate("c", "dense", 2)],
    }
    ranked = fusion.fuse(channels)
    assert ranked[0].book_id == "b"  # only book found by both channels
    assert ranked[0].fusion_score == pytest.approx(1.0)  # normalised to the top score


def test_hard_filters_exclude_non_matching_books():
    record = IndexedBook(
        book=_book("a", "শিরোনাম", "লেখক", publish_year=1990),
        enrichment=Enrichment(genres=["ইতিহাস"]),
    )
    fused = fusion.fuse({"lexical": [_candidate("a", "lexical", 1)]})
    assert fusion.apply_filters(fused, Filters(genres=["ইতিহাস"]), {"a": record})
    assert not fusion.apply_filters(fused, Filters(genres=["কবিতা"]), {"a": record})
    assert not fusion.apply_filters(fused, Filters(year_from=2000), {"a": record})


# --------------------------------------------------------------------------- knowledge graph

@pytest.fixture
def graph_records() -> list[IndexedBook]:
    return [
        IndexedBook(
            book=_book("b1", "প্রবাসে মুক্তিযুদ্ধ", "রাষ্ট্রদূত ক", author_id="a1"),
            enrichment=Enrichment(subjects=["মুক্তিযুদ্ধ"], author_roles=["কূটনীতিক"],
                                  author_periods=["পাকিস্তান আমল"]),
        ),
        IndexedBook(
            book=_book("b2", "কবিতা সমগ্র", "কবি খ", author_id="a2"),
            enrichment=Enrichment(subjects=["সামাজিক ইতিহাস"], genres=["কবিতা"],
                                  author_roles=["লেখক"]),
        ),
    ]


def test_multi_hop_traversal(graph_records, tmp_path):
    kg = KnowledgeGraph()
    for record in graph_records:
        kg._add_record(record)

    found = kg.run([
        GraphStep(kind="find_authors", occupations=["কূটনীতিক"], periods=["পাকিস্তান আমল"]),
        GraphStep(kind="find_books", subjects=["মুক্তিযুদ্ধ"]),
    ])
    assert set(found) == {"b1"}
    assert found["b1"]["author_terms"] == ["কূটনীতিক", "পাকিস্তান আমল"]


def test_graph_idf_prefers_rare_concepts(graph_records):
    kg = KnowledgeGraph()
    for record in graph_records:
        kg._add_record(record)
    # কবিতা is on one of two books, মুক্তিযুদ্ধ on the other -- both rare, both informative.
    assert kg.idf("genre", "কবিতা") > 0
    assert kg.idf("subject", "নেই-এমন-বিষয়") == 0


def test_graph_round_trips_through_json(graph_records, tmp_path):
    kg = KnowledgeGraph()
    for record in graph_records:
        kg._add_record(record)
    path = tmp_path / "graph.json"
    kg.save(path)

    from config import Settings

    reloaded = KnowledgeGraph.load(Settings(artifacts_dir=tmp_path))
    assert reloaded.stats()["book"] == kg.stats()["book"]


# --------------------------------------------------------------------------- personalisation

def _records() -> dict[str, IndexedBook]:
    return {
        "kids": IndexedBook(book=_book("kids", "কিশোর গল্প", "ক"),
                            enrichment=Enrichment(genres=["শিশু কিশোর"], subjects=["মুক্তিযুদ্ধ"])),
        "academic": IndexedBook(book=_book("academic", "গবেষণা", "খ"),
                                enrichment=Enrichment(genres=["গবেষণা"], subjects=["মুক্তিযুদ্ধ"])),
    }


def test_affinity_reflects_explicit_preferences():
    profile = UserProfile(user_id="u", genres=["শিশু কিশোর"])
    affinity = personalize.build_affinity(profile, None, _records())
    assert affinity["শিশু কিশোর"] > 0


def test_session_intent_outweighs_long_term_taste():
    profile = UserProfile(user_id="u", genres=["শিশু কিশোর"])
    session = Session(clicks=["academic"])
    affinity = personalize.build_affinity(profile, session, _records())
    assert affinity["গবেষণা"] > affinity["শিশু কিশোর"]


def test_personalisation_cannot_overturn_a_large_relevance_gap():
    """A strongly relevant book must not be buried by taste -- the boost is capped."""
    records = _records()
    ranked = [
        (fusion.Fused(book_id="academic", fusion_score=1.0), 0.90, {}),
        (fusion.Fused(book_id="kids", fusion_score=0.5), 0.50, {}),
    ]
    affinity = personalize.build_affinity(UserProfile(user_id="u", genres=["শিশু কিশোর"]), None, records)
    adjusted = personalize.apply(ranked, records, affinity)
    assert adjusted[0][0].book_id == "academic"


def test_personalisation_breaks_a_close_tie():
    records = _records()
    ranked = [
        (fusion.Fused(book_id="academic", fusion_score=1.0), 0.80, {}),
        (fusion.Fused(book_id="kids", fusion_score=1.0), 0.79, {}),
    ]
    affinity = personalize.build_affinity(UserProfile(user_id="u", genres=["শিশু কিশোর"]), None, records)
    adjusted = personalize.apply(ranked, records, affinity)
    assert adjusted[0][0].book_id == "kids"
    assert adjusted[0][2] == 0.79  # relevance is preserved alongside the adjusted score


# --------------------------------------------------------------------------- embeddings

class _StubEmbedder:
    """Records how it was called, so the query/document asymmetry can be asserted."""

    name = "stub"

    def __init__(self, dimension: int = 4):
        self._dimension = dimension
        self.documents: list[str] = []
        self.queries: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts, on_batch=None):
        self.documents.extend(texts)
        return np.zeros((len(texts), self._dimension), dtype="float32")

    def embed_query(self, text):
        self.queries.append(text)
        return np.zeros(self._dimension, dtype="float32")


class _StubVectorIndex:
    def __init__(self):
        self.filters = "unset"

    def search(self, query_vector, k=50, filters=None):
        self.filters = filters
        return [("b1", 0.9, "একাত্তরের দিনগুলি")]


def _retriever(embedder, vector):
    understanding = QueryUnderstanding(llm=None, taxonomy=get_taxonomy(), vocabulary=set(),
                                       mode="never")
    return Retriever(lexical=None, vector=vector, graph=None,
                     embedder=embedder, understanding=understanding)


def test_dense_channel_encodes_the_query_as_a_query():
    """Asymmetric models need the instruction prefix on queries only -- if the retriever
    ever routed the query through `embed_documents`, recall would quietly collapse."""
    embedder, vector = _StubEmbedder(), _StubVectorIndex()
    plan = QueryPlan(raw_query="মুক্তিযুদ্ধের বই", normalized_query="মুক্তিযুদ্ধের বই",
                     expanded_terms=["একাত্তর"])

    candidates = _retriever(embedder, vector)._dense(plan)

    assert embedder.queries == ["মুক্তিযুদ্ধের বই একাত্তর"]
    assert embedder.documents == []
    assert [c.book_id for c in candidates] == ["b1"]
    assert candidates[0].channel == "dense"


def test_dense_channel_is_skipped_when_there_is_no_embedder():
    assert _retriever(None, _StubVectorIndex())._dense(QueryPlan(raw_query="x", normalized_query="x")) == []


def test_dense_channel_passes_hard_filters_to_the_vector_store():
    vector = _StubVectorIndex()
    plan = QueryPlan(raw_query="x", normalized_query="x", filters=Filters(genres=["উপন্যাস"]))
    _retriever(_StubEmbedder(), vector)._dense(plan)
    assert vector.filters.genres == ["উপন্যাস"]


@pytest.mark.parametrize(
    "filters, expected",
    [
        (Filters(), ""),
        (Filters(language="bn"), "language = 'bn'"),
        (Filters(year_from=1971), "(publish_year >= 1971 AND publish_year != 0)"),
        (Filters(genres=["উপন্যাস"]), "array_has_any(genres, ['উপন্যাস'])"),
    ],
)
def test_vector_filters_become_sql(filters, expected):
    assert build_where(filters) == expected


def test_vector_filter_literals_are_escaped():
    assert build_where(Filters(publishers=["O'Reilly"])) == "publisher IN ('O''Reilly')"


class _StubLMStudio:
    embedding_model = "some-embedder"

    def __init__(self):
        self.calls: list[str] = []

    def embed_one(self, text):
        self.calls.append(text)
        return np.zeros(768, dtype="float32")

    def embed(self, texts, on_batch=None):
        return np.zeros((len(texts), 768), dtype="float32")


def test_lmstudio_backend_is_selected_by_configuration():
    llm = _StubLMStudio()
    embedder = make_embedder(Settings(embedding_backend="lmstudio"), llm)
    assert isinstance(embedder, LMStudioEmbedder)
    assert embedder.dimension == 768
    assert embedder.dimension == 768  # probed once, then cached
    assert len(llm.calls) == 1
