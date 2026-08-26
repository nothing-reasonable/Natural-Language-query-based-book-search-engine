"""Tests for the stages added to complete the design: entity linking, the metadata
channel, derived facts, and the evaluation metrics.

None of them need LM Studio, an index, or the catalogue -- they build tiny fixtures, so
they run in milliseconds and fail for one reason each.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from search.core import bengali  # noqa: E402
from search.derive import augment  # noqa: E402
from search.indexing.entity_index import EntityIndex  # noqa: E402
from search.indexing.facet_index import FacetIndex  # noqa: E402
from ingest.clean import strip_boilerplate  # noqa: E402
from search.core.schemas import Book, Concepts, Enrichment, Filters, IndexedBook  # noqa: E402
from search.query.taxonomy import get_taxonomy  # noqa: E402


def record(book_id, title, author, author_id="", **kw):
    enrichment = Enrichment(**{k: v for k, v in kw.items() if k in Enrichment.model_fields})
    book = Book(book_id=book_id, title=title, author=author,
                author_id=author_id or author.replace(" ", ""),
                **{k: v for k, v in kw.items() if k in Book.model_fields})
    return IndexedBook(book=book, enrichment=enrichment)


@pytest.fixture
def catalogue():
    return [
        record("b1", "মুক্তিযুদ্ধের উপন্যাসসমগ্র", "হুমায়ূন আহমেদ", publish_year=1995,
               subjects=["মুক্তিযুদ্ধ"], publisher="অন্যপ্রকাশ"),
        record("b2", "শঙ্খনীল কারাগার", "হুমায়ূন আহমেদ", publish_year=1973,
               subjects=["সামাজিক ইতিহাস"], publisher="অন্যপ্রকাশ"),
        record("b3", "মুক্তিযুদ্ধ ১৯৭১", "মুনতাসীর মামুন", publish_year=2000,
               subjects=["মুক্তিযুদ্ধ"], publisher="সময়"),
        record("b4", "একাত্তরের দিনগুলি", "জাহানারা ইমাম", publish_year=1986,
               subjects=["মুক্তিযুদ্ধ"], publisher="সন্ধানী"),
    ]


# --------------------------------------------------------------------------- entities

def test_full_name_becomes_a_hard_filter(catalogue):
    matches = EntityIndex(catalogue).find("হুমায়ূন আহমেদ এর মুক্তিযুদ্ধের বই")
    assert [m.name for m in matches] == ["হুমায়ূন আহমেদ"]
    assert matches[0].hard, "a full-name match is what a hard author filter exists for"


def test_inflected_name_still_matches(catalogue):
    """"আহমেদের" stems to "আহম" while "আহমেদ" stems to itself -- the matcher has to
    survive the stemmer's known 1.8% mis-stem rate."""
    assert [m.name for m in EntityIndex(catalogue).find("হুমায়ূন আহমেদের বই")] == ["হুমায়ূন আহমেদ"]


def test_shared_surname_is_not_an_author_match(catalogue):
    """Guessing a person from a surname is worse than not filtering at all."""
    assert EntityIndex(catalogue).find("আহমেদ এর বই") == []


def test_topic_only_query_matches_no_entity(catalogue):
    assert EntityIndex(catalogue).find("মুক্তিযুদ্ধের বই") == []


def test_publisher_is_recognised(catalogue):
    matches = EntityIndex(catalogue).find("অন্যপ্রকাশ থেকে প্রকাশিত বই")
    assert [(m.kind, m.name) for m in matches] == [("publisher", "অন্যপ্রকাশ")]


# --------------------------------------------------------------------------- facets

def test_facet_channel_finds_every_book_the_filter_allows(catalogue):
    """The point of the channel: constraints must *retrieve*, not merely prune."""
    facets = FacetIndex(catalogue)
    selected = facets.select(Filters(author_ids=["হুমায়ূনআহমেদ"]))
    assert selected == {"b1", "b2"}


def test_facet_ranking_puts_the_asked_for_topic_first(catalogue):
    facets = FacetIndex(catalogue)
    selected = facets.select(Filters(author_ids=["হুমায়ূনআহমেদ"]))
    ranked = facets.rank(selected, Concepts(subjects=["মুক্তিযুদ্ধ"]), limit=10)
    assert ranked[0][0] == "b1"


def test_year_range_selects_inclusively(catalogue):
    facets = FacetIndex(catalogue)
    assert facets.select(Filters(year_from=1973, year_to=1986)) == {"b2", "b4"}


def test_no_filter_means_no_constraint(catalogue):
    assert FacetIndex(catalogue).select(Filters()) is None


def test_constraint_groups_intersect(catalogue):
    facets = FacetIndex(catalogue)
    both = facets.select(Filters(author_ids=["হুমায়ূনআহমেদ"], subjects=["মুক্তিযুদ্ধ"]))
    assert both == {"b1"}


# --------------------------------------------------------------------------- derivation

def test_author_periods_come_from_publication_years(catalogue):
    augment(catalogue, get_taxonomy())
    humayun = [r for r in catalogue if r.book_id == "b2"][0]
    # 1973 and 1995 -- both after independence.
    assert "স্বাধীন বাংলাদেশ" in humayun.enrichment.author_periods


def test_publication_year_does_not_become_a_subject_period(catalogue):
    """A book printed in 1995 is not therefore *about* independent Bangladesh.
    Conflating 'written in' with 'about' would wreck period filtering."""
    before = list(catalogue[0].enrichment.periods)
    augment(catalogue, get_taxonomy())
    assert catalogue[0].enrichment.periods == before


def test_derivation_is_idempotent(catalogue):
    tax = get_taxonomy()
    augment(catalogue, tax)
    once = [list(r.enrichment.author_periods) for r in catalogue]
    augment(catalogue, tax)
    assert [list(r.enrichment.author_periods) for r in catalogue] == once


# --------------------------------------------------------------------------- determinism

def test_fusion_breaks_ties_by_book_id():
    """Equal scores must not be ordered by set/dict iteration: Python randomises string
    hashing per process, which moved nDCG@10 by 0.027 between identical evaluation runs."""
    from search.retrieval.fusion import fuse
    from search.core.schemas import Candidate

    def run(order):
        channels = {"lexical": [Candidate(book_id=b, channel="lexical", rank=1, score=1.0)
                                for b in order]}
        return [f.book_id for f in fuse(channels)]

    # Same scores and ranks, different arrival order -> same output order.
    assert run(["zeta", "alpha", "mid"]) == run(["mid", "zeta", "alpha"]) == \
           sorted(["zeta", "alpha", "mid"])


def test_facet_ranking_is_stable_for_equal_scores(catalogue):
    from search.core.schemas import Concepts

    facets = FacetIndex(catalogue)
    ids = {r.book_id for r in catalogue}
    first = facets.rank(ids, Concepts(), limit=10)
    assert first == facets.rank(set(reversed(list(ids))), Concepts(), limit=10)


# --------------------------------------------------------------------------- graph idf

def test_idf_cache_returns_the_same_values_as_recomputing():
    """The cache made the graph channel 56x faster; it must not change a single score."""
    from search.indexing.kg_index import KnowledgeGraph

    kg = KnowledgeGraph()
    for i, subjects in enumerate([["মুক্তিযুদ্ধ"], ["মুক্তিযুদ্ধ"], ["মুক্তিযুদ্ধ"], ["ভাষা আন্দোলন"]]):
        kg._add_record(record(f"b{i}", f"t{i}", f"a{i}", subjects=subjects))

    fresh = {}
    for name in ("মুক্তিযুদ্ধ", "ভাষা আন্দোলন", "অনুপস্থিত"):
        kg._idf_cache.clear()
        fresh[name] = kg.idf("subject", name)
    cached = {name: kg.idf("subject", name) for name in fresh}          # now warm
    again = {name: kg.idf("subject", name) for name in fresh}
    assert fresh == cached == again
    # And the ranking property the cache exists to serve still holds.
    assert kg.idf("subject", "ভাষা আন্দোলন") > kg.idf("subject", "মুক্তিযুদ্ধ")


def test_mutating_the_graph_invalidates_the_idf_cache():
    """A stale idf would silently mis-rank every later query."""
    from search.indexing.kg_index import KnowledgeGraph

    kg = KnowledgeGraph()
    kg._add_record(record("b0", "t0", "a0", subjects=["মুক্তিযুদ্ধ"]))
    before = kg.idf("subject", "মুক্তিযুদ্ধ")
    assert kg._idf_cache, "should be warm"

    # Books *without* the subject, so its document frequency stays put while the corpus
    # grows -- that is what makes the concept more informative, and the number move.
    for i in range(1, 6):
        kg._add_record(record(f"b{i}", f"t{i}", f"a{i}", subjects=["রান্না"]))
    assert kg._idf_cache == {}, "adding a record must drop the cached answers"
    assert kg.idf("subject", "মুক্তিযুদ্ধ") > before


# --------------------------------------------------------------------------- boilerplate

def test_pure_advertisement_leaves_nothing():
    ad = ("মো. নুরুল ইসলাম এর বীরাঙ্গনা সখিনা অরিজিনাল বইটি সংগ্রহ করুন রকমারি ডট কম থেকে। "
          "বই হাতে পেয়ে মূল্য পরিশোধের সুবিধাসহ অফারভেদে উপভোগ করুন ফ্রি শিপিং এবং সর্বোচ্চ ছাড়!")
    assert strip_boilerplate(ad) == ""


def test_a_real_blurb_survives_a_trailing_advert():
    text = "এটি মুক্তিযুদ্ধের একটি দলিল। অরিজিনাল বইটি সংগ্রহ করুন রকমারি ডট কম থেকে।"
    assert strip_boilerplate(text) == "এটি মুক্তিযুদ্ধের একটি দলিল।"


# --------------------------------------------------------------------------- time spans

@pytest.mark.parametrize(
    "query,expected",
    [
        # A named span plus an authorship or publication verb is a real constraint.
        ("একবিংশ শতাব্দীতে লেখা মুক্তিযুদ্ধের বই", (2001, 2100)),
        ("বিংশ শতাব্দীতে প্রকাশিত বই", (1901, 2000)),
        ("নব্বইয়ের দশকে লেখা উপন্যাস", (1990, 1999)),
        ("আশির দশকে প্রকাশিত কবিতা", (1980, 1989)),
        # A span without a verb is a subject: 19th-century Bengal, not 19th-century printing.
        ("ঊনবিংশ শতাব্দীর বাংলা সাহিত্য", None),
        # A verb without a span constrains nothing.
        ("মুক্তিযুদ্ধ নিয়ে লেখা বই", None),
    ],
)
def test_named_time_spans_constrain_only_when_the_query_says_so(query, expected):
    from search.query.query_understanding import _named_time_span, QueryUnderstanding

    assert QueryUnderstanding._year_filter(query, []) == expected
    if expected:
        assert _named_time_span(query) == expected


def test_most_specific_time_span_wins():
    """"বিংশ" is a substring of "একবিংশ"; the longer phrase has to claim the match."""
    from search.query.query_understanding import _named_time_span

    assert _named_time_span("একবিংশ শতাব্দীতে লেখা") == (2001, 2100)


def test_a_named_span_outranks_a_bare_year_in_the_same_query():
    """"একবিংশ শতাব্দীতে লেখা ১৯৭১-এর বই" is a modern book about 1971."""
    from search.query.query_understanding import QueryUnderstanding

    assert QueryUnderstanding._year_filter(
        "একবিংশ শতাব্দীতে লেখা 1971 এর বই", [1971]
    ) == (2001, 2100)


# --------------------------------------------------------------------------- periods

def test_period_ranges_do_not_overlap():
    """Overlapping ranges made period assignment depend on YAML ordering: 1947 and 1971
    -- 599 books, 12% of the catalogue -- were dated by dict order rather than history."""
    assert get_taxonomy().overlapping_periods() == []


@pytest.mark.parametrize(
    "year,period",
    [
        (1946, "ব্রিটিশ আমল"),
        (1947, "পাকিস্তান আমল"),   # Partition
        (1971, "পাকিস্তান আমল"),   # the war; matches `মুক্তিযুদ্ধ: period:` in the taxonomy
        (1972, "স্বাধীন বাংলাদেশ"),
    ],
)
def test_boundary_years_are_assigned_by_history(year, period):
    assert get_taxonomy().period_for_year(year) == period


# --------------------------------------------------------------------------- llm filters

def test_model_invented_year_bound_cannot_become_a_filter():
    """Regression: the model answered "মুক্তিযুদ্ধের বই" with year_to=2100 — the sentinel
    end of স্বাধীন বাংলাদেশ in taxonomy.yaml. It matched all 4,880 books while announcing
    itself in the explanation as a publication-date match."""
    from search.query.query_understanding import QueryUnderstanding
    from search.core.schemas import Concepts, Filters, QueryPlan

    qu = QueryUnderstanding(llm=None, year_bounds=(1800, 2026))
    rules = QueryPlan(normalized_query="মুক্তিযুদ্ধের বই",
                      concepts=Concepts(subjects=["মুক্তিযুদ্ধ"]))
    grounded = qu._ground_filters(
        rules, Filters(subjects=["মুক্তিযুদ্ধ"], periods=["স্বাধীন বাংলাদেশ"], year_to=2100)
    )
    assert grounded.year_to is None and grounded.year_from is None
    assert grounded.is_empty(), "no part of an inferred description may constrain results"


def test_ungrounded_facets_are_demoted_not_discarded():
    """They still steer ranking; they just lose the power to exclude."""
    from search.query.query_understanding import QueryUnderstanding
    from search.core.schemas import Concepts, Filters, QueryPlan

    rules = QueryPlan(concepts=Concepts(subjects=["মুক্তিযুদ্ধ"]))
    QueryUnderstanding(llm=None)._ground_filters(rules, Filters(periods=["স্বাধীন বাংলাদেশ"]))
    assert "স্বাধীন বাংলাদেশ" in rules.concepts.periods


def test_constraints_the_query_actually_states_survive():
    """The guard must not throw away real constraints — a year the rules read from
    "১৯৭১ সালে প্রকাশিত", and an author resolved against the catalogue."""
    from search.query.query_understanding import QueryUnderstanding
    from search.core.schemas import Filters, QueryPlan

    rules = QueryPlan(filters=Filters(year_from=1971, year_to=1971,
                                      authors=["হুমায়ূন আহমেদ"], author_ids=["e5cc"]))
    grounded = QueryUnderstanding(llm=None)._ground_filters(rules, Filters(year_to=2100))
    assert grounded.year_from == 1971 and grounded.year_to == 1971
    assert grounded.authors == ["হুমায়ূন আহমেদ"] and grounded.author_ids == ["e5cc"]


def test_open_ended_year_range_reads_as_a_range():
    """"প্রকাশকাল: -2100" read as a negative year, or as the book's own publication date."""
    from search.retrieval.retrieve import _describe_filters
    from search.core.schemas import Filters

    assert _describe_filters(Filters(year_to=2100)) == "প্রকাশকাল: 2100 সাল পর্যন্ত"
    assert _describe_filters(Filters(year_from=1971)) == "প্রকাশকাল: 1971 সাল থেকে"
    assert _describe_filters(Filters(year_from=1960, year_to=1970)) == "প্রকাশকাল: 1960–1970"
    assert _describe_filters(Filters(year_from=1971, year_to=1971)) == "প্রকাশকাল: 1971"


# --------------------------------------------------------------------------- llm year spans

def _ground(query, proposed, bounds=(1800, 2026), rule_filters=None):
    from search.query.query_understanding import QueryUnderstanding
    from search.core.schemas import QueryPlan

    qu = QueryUnderstanding(llm=None, year_bounds=bounds)
    rules = QueryPlan(normalized_query=query, filters=rule_filters or Filters())
    return qu._ground_filters(rules, proposed)


def test_model_year_span_refused_when_the_query_mentions_no_date():
    """The original bug: "মুক্তিযুদ্ধের বই" asks nothing about dates, so a proposed
    year bound has nothing in the query to stand on."""
    assert _ground("মুক্তিযুদ্ধের বই", Filters(year_to=2100)).is_empty()


def test_model_year_span_refused_when_it_narrows_nothing():
    """A sentinel spanning the whole catalogue is a no-op wearing a filter's clothes."""
    assert _ground("লেখা মুক্তিযুদ্ধের বই", Filters(year_to=2100)).is_empty()


def test_model_year_span_accepted_when_asked_for_and_narrowing():
    """The model earns its place on phrasings `_TIME_EXPRESSIONS` does not list."""
    out = _ground("গত দুই দশকে লেখা বই", Filters(year_from=2005, year_to=2026))
    assert (out.year_from, out.year_to) == (2005, 2026)


def test_one_sided_model_span_is_clamped_to_the_catalogue():
    out = _ground("সাম্প্রতিক সময়ে প্রকাশিত বই", Filters(year_from=2015))
    assert (out.year_from, out.year_to) == (2015, 2026)


def test_rules_win_over_the_model_on_years():
    out = _ground("একবিংশ শতাব্দীতে লেখা", Filters(year_from=1900),
                  rule_filters=Filters(year_from=2001, year_to=2100))
    assert (out.year_from, out.year_to) == (2001, 2100)


def test_merge_runs_without_an_llm_present():
    """Regression: `_merge` was a staticmethod calling `self`, so every LLM-assisted query
    raised NameError, was swallowed by the fallback, and silently degraded to rules-only
    while still returning plausible results."""
    from search.query.query_understanding import QueryUnderstanding
    from search.core.schemas import QueryPlan, QueryPlanDraft

    qu = QueryUnderstanding(llm=None, year_bounds=(1800, 2026))
    merged = qu._merge(
        QueryPlan(normalized_query="গত দুই দশকে লেখা বই", intent="filtered"),
        QueryPlanDraft(intent="filtered", keywords=["বই"],
                       filters=Filters(year_from=2005, year_to=2026)),
    )
    assert (merged.filters.year_from, merged.filters.year_to) == (2005, 2026)


# --------------------------------------------------------------------------- llm mode

def test_llm_mode_never_and_always_ignore_the_gate():
    """`always` exists precisely so it does not depend on what the rules found."""
    from search.query.query_understanding import QueryUnderstanding
    from search.core.schemas import EntityRef, QueryPlan

    llm = object()  # never called; only its presence is checked
    resolved = QueryPlan(entities=[EntityRef(kind="author", name="x", entity_id="1", hard=True)])
    bare = QueryPlan()

    assert QueryUnderstanding(llm=llm, mode="never")._should_call_llm(bare) is False
    assert QueryUnderstanding(llm=llm, mode="always")._should_call_llm(resolved) is True


def test_llm_mode_auto_calls_only_when_the_rules_found_nothing():
    from search.query.query_understanding import QueryUnderstanding
    from search.core.schemas import Concepts, EntityRef, QueryPlan

    auto = QueryUnderstanding(llm=object(), mode="auto")
    assert auto._should_call_llm(QueryPlan()) is True, "free text is what the model is for"
    assert auto._should_call_llm(
        QueryPlan(entities=[EntityRef(kind="author", name="x", entity_id="1", hard=True)])
    ) is False
    assert auto._should_call_llm(QueryPlan(concepts=Concepts(subjects=["মুক্তিযুদ্ধ"]))) is False


def test_no_llm_client_disables_every_mode():
    """`--no-llm` must win over `--force-plan`; there is no model to force."""
    from search.query.query_understanding import QueryUnderstanding

    assert QueryUnderstanding(llm=None, mode="always").mode == "never"


# --------------------------------------------------------------------------- metrics

def test_metrics_match_worked_examples():
    import metrics

    ranked, gold = ["a", "b", "c", "d", "e"], {"a", "c"}
    assert metrics.precision_at_k(ranked, gold, 5) == pytest.approx(0.4)
    assert metrics.recall_at_k(ranked, gold, 10) == pytest.approx(1.0)
    assert metrics.reciprocal_rank(ranked, gold) == pytest.approx(1.0)
    assert metrics.reciprocal_rank(["x", "a"], gold) == pytest.approx(0.5)


def test_ndcg_is_normalised_by_the_achievable_ideal():
    """A query with 2 correct answers must be able to score 1.0, or queries with big
    gold sets would look artificially better than queries with small ones."""
    import metrics

    assert metrics.ndcg_at_k(["a", "c", "z"], {"a", "c"}, 10) == pytest.approx(1.0)
    assert metrics.ndcg_at_k(["z", "y", "a"], {"a"}, 10) < 1.0


def test_empty_gold_never_divides_by_zero():
    import metrics

    assert metrics.ndcg_at_k(["a"], set(), 10) == 0.0
    assert metrics.recall_at_k(["a"], set(), 10) == 0.0


# --------------------------------------------------------------------------- graph facets

def _graph_retriever(records):
    """A Retriever with only the graph channel wired -- enough to score it."""
    from search.indexing.kg_index import KnowledgeGraph
    from search.retrieval.retrieve import Retriever

    kg = KnowledgeGraph()
    for r in records:
        kg._add_record(r)
    return Retriever(lexical=None, vector=None, graph=kg, embedder=None,
                     understanding=None, facets=FacetIndex(records))


def test_idf_uses_the_facet_the_query_asked_about():
    """`ইতিহাস` names a genre on most of the catalogue *and* a rare event.

    Scanning facets in a fixed order and taking the first hit charged every genre match
    the event's rarity -- a six-fold overstatement of exactly the signal the graph
    channel ranks by, applied to a third of the catalogue.
    """
    from search.core.schemas import QueryPlan

    records = [record(f"b{i}", f"t{i}", f"a{i}", genres=["ইতিহাস"]) for i in range(20)]
    records.append(record("rare", "tr", "ar", events=["ইতিহাস"]))
    retriever = _graph_retriever(records)

    kinds = {"ইতিহাস": {"genre"}}
    assert retriever._idf("ইতিহাস", kinds) == pytest.approx(
        retriever.graph.idf("genre", "ইতিহাস"))
    # The event node is far rarer; charging its idf to a genre match is the bug.
    assert retriever.graph.idf("event", "ইতিহাস") > retriever._idf("ইতিহাস", kinds)

    from search.retrieval.retrieve import _term_kinds
    plan = QueryPlan(raw_query="ইতিহাসের বই", normalized_query="ইতিহাস")
    plan.concepts.genres = ["ইতিহাস"]
    assert _term_kinds(plan) == {"ইতিহাস": {"genre"}}


def test_term_kinds_reads_multi_hop_steps_too():
    from search.core.schemas import GraphStep, QueryPlan

    plan = QueryPlan(raw_query="q", normalized_query="q")
    plan.steps = [GraphStep(kind="find_authors", occupations=["কূটনীতিক"]),
                  GraphStep(kind="find_books", subjects=["মুক্তিযুদ্ধ"])]
    from search.retrieval.retrieve import _term_kinds
    assert _term_kinds(plan) == {"কূটনীতিক": {"occupation"}, "মুক্তিযুদ্ধ": {"subject"}}


def test_broad_concept_ties_are_broken_by_metadata_not_book_id():
    """A concept on thousands of books ties them all at one score. Truncating that tie
    to `channel_top_k` by book_id drops well-tagged books for the sole reason that their
    id starts with a late hex digit."""
    from search.core.schemas import QueryPlan

    records = [
        record("ffff", "well tagged", "a1", genres=["ইতিহাস"], metadata_quality=0.9),
        record("0000", "sparse", "a2", genres=["ইতিহাস"], metadata_quality=0.1),
    ]
    retriever = _graph_retriever(records)
    retriever.settings = retriever.settings.model_copy(
        update={"channel_top_k": 1, "graph_min_specificity": 0.0})

    plan = QueryPlan(raw_query="ইতিহাস", normalized_query="ইতিহাস")
    plan.concepts.genres = ["ইতিহাস"]
    got = retriever._graph(plan)
    assert [c.book_id for c in got] == ["ffff"], "the richer record should win the tie"
