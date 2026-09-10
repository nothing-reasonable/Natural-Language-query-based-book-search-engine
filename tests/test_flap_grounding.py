"""Every result has to be grounded in the two fields the catalogue always has: the book's
title and its `Description (Flap)`.

Both source CSVs populate those for 100% of their rows, and the flap is the only field
that says what a *particular* book contains. It was nevertheless losing to text a model
invented -- `enrichment.summary` replaced it in the reranker, and inferred `subjects` tags
outweighed it in BM25 -- so these tests pin the ordering down.

Like `test_new_stages.py`, none of them need LM Studio, an index or the catalogue.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from ingest.clean import QUALITY_FIELDS, _quality  # noqa: E402
from search.core import bengali  # noqa: E402
from search.core.fields import EMBED_FIELDS, TEXT_FIELDS, embedding_text, lexical_tokens  # noqa: E402
from search.core.schemas import Book, Enrichment, Evidence, IndexedBook  # noqa: E402
from search.ranking.explanation_generator import explain, flap_quote  # noqa: E402
from search.ranking.rerank import _blurb, _describe, _passage  # noqa: E402

FLAP = ("চা বাগানের শ্রমিকদের জীবন নিয়ে লেখা এগারোটি গল্পের সংকলন। "
        "সিলেটের মালনিছড়া ও লাক্কাতুরা বাগানের দিনমজুরদের কথা এখানে উঠে এসেছে।")


def make(**kw) -> IndexedBook:
    """A record with a real flap, plus whatever the test wants to override."""
    enrichment = Enrichment(**{k: v for k, v in kw.items() if k in Enrichment.model_fields})
    fields = {k: v for k, v in kw.items() if k in Book.model_fields}
    book = Book(**{
        "book_id": "b1",
        "title": "চা বাগানের গল্প",
        "author": "সাদাত হোসাইন",
        "description": FLAP,
        "publish_year": 2015,
        **fields,
    })
    return IndexedBook(book=book, enrichment=enrichment)


# --------------------------------------------------------------------------- the blurb

def test_flap_beats_an_llm_summary():
    """The regression this whole change exists for: wherever enrichment had produced a
    summary, the reranker never saw the flap at all."""
    record = make(summary="একটি গল্পের বই।")
    assert _blurb(record) == FLAP
    assert FLAP[:40] in _passage(record)
    assert "একটি গল্পের বই।" not in _passage(record)


def test_summary_is_used_only_when_there_is_no_flap():
    """~23% of rows lose their flap to `strip_boilerplate`; for those the summary is the
    only description there is, so it must not be discarded outright."""
    record = make(description="", summary="একটি গল্পের বই।")
    assert _blurb(record) == "একটি গল্পের বই।"
    assert "একটি গল্পের বই।" in _passage(record)


def test_no_blurb_at_all_is_not_an_error():
    record = make(description="", summary="")
    assert _blurb(record) == ""
    passage = _passage(record)
    assert "ফ্ল্যাপ" not in passage
    assert "চা বাগানের গল্প" in passage  # the title still identifies it


# --------------------------------------------------------------------------- passage shape

def test_title_and_flap_come_before_the_inferred_tags():
    """Field order decides what survives truncation: the cross-encoder cuts the tail, so
    inferred tags at the bottom cost a genre label rather than the flap."""
    passage = _passage(make(subjects=["শ্রমজীবী"], genres=["গল্প"],
                            author_roles=["ঔপন্যাসিক"]))
    assert passage.index("শিরোনাম:") < passage.index("ফ্ল্যাপ")
    for tag_line in ("বিষয়:", "ধরন:", "লেখকের ভূমিকা:"):
        assert passage.index("ফ্ল্যাপ") < passage.index(tag_line), \
            f"{tag_line} must sit below the flap, not above it"


def test_flap_is_not_truncated_below_the_configured_budget():
    long_flap = "চা বাগানের গল্প। " * 200  # ~3,400 chars, well past any budget
    passage = _passage(make(description=long_flap))
    kept = passage.split("ফ্ল্যাপ): ", 1)[1]
    assert len(kept) == settings.rerank_flap_chars


def test_flap_budget_is_configurable():
    tight = settings.model_copy(update={"rerank_flap_chars": 30})
    passage = _passage(make(), tight)
    assert len(passage.split("ফ্ল্যাপ): ", 1)[1]) == 30


def test_listwise_grader_is_grounded_the_same_way():
    """`_describe` feeds the deprecated `reranker_backend="llm"`. It shared the summary
    bug, and a backend that can be switched on must not be quietly ungrounded."""
    block = _describe(0, make(summary="একটি গল্পের বই।", subjects=["শ্রমজীবী"]))
    assert FLAP[:40] in block
    assert "একটি গল্পের বই।" not in block
    assert block.index("ফ্ল্যাপ") < block.index("বিষয়:")


# --------------------------------------------------------------------------- field weights

def _weight(name: str) -> int:
    return next(f.lexical_weight for f in TEXT_FIELDS if f.name == name)


def test_no_inferred_field_outweighs_the_flap():
    """A `subjects` tag at weight 3 against the flap at 1 meant BM25 ranked a model's
    guess about a book above what the book says about itself."""
    inferred = ("subjects", "topics", "genres", "periods", "events", "places",
                "persons", "author_roles")
    flap = _weight("description")
    assert flap >= 3
    for name in inferred:
        assert _weight(name) <= flap, f"{name} must not outweigh the flap"


def test_title_is_still_the_strongest_field():
    assert _weight("title") > _weight("description")


def test_flap_tokens_are_repeated_more_than_a_subject_tags():
    """Field weighting is token repetition, so the count in the document *is* the weight.

    One word per field, both put through the same stemmer, so the comparison cannot turn
    on a stemming difference between the two sides.
    """
    record = make(description="শ্রমিকদের কথা", subjects=["ঔপন্যাসিক"])
    tokens = lexical_tokens(record, bengali.analyze)
    flap_stem = bengali.analyze("শ্রমিকদের")[0]
    tag_stem = bengali.analyze("ঔপন্যাসিক")[0]
    assert tokens.count(flap_stem) > tokens.count(tag_stem) > 0


def test_author_bio_reaches_neither_index():
    """It is about the author, not the book, and identical across their whole
    bibliography -- so it can only blur their books together."""
    assert _weight("author_bio") == 0
    assert "author_bio" not in [f.name for f in EMBED_FIELDS]

    bio = "লেখক একজন প্রখ্যাত ঔপন্যাসিক ও চিত্রনাট্যকার।"
    record = make(author_bio=bio)
    assert bengali.analyze("চিত্রনাট্যকার")[0] not in lexical_tokens(record, bengali.analyze)
    assert "ঔপন্যাসিক" not in embedding_text(record)


def test_embedding_text_leads_with_title_author_flap():
    """`embedding_max_tokens` truncates the tail, and the flap used to sit eleventh."""
    record = make(subjects=["শ্রমজীবী"], genres=["গল্প"], topics=["সিলেট"])
    labels = [line.split(":", 1)[0] for line in embedding_text(record).splitlines()]
    assert labels[:3] == ["শিরোনাম", "লেখক", "ফ্ল্যাপ"]


# --------------------------------------------------------------------------- quality

def test_metadata_quality_can_reach_one():
    """`table_of_contents` was in the list and no CSV supplies it, so every book was
    capped at 6/7 and the signal could not discriminate at the top."""
    assert "table_of_contents" not in QUALITY_FIELDS
    full = Book(book_id="b", title="t", author="a", author_bio="bio",
                publisher="p", description="d", publish_year=2000)
    assert _quality(full) == 1.0


def test_a_stripped_flap_costs_quality():
    """The flap's presence is what `metadata_quality` is there to notice."""
    with_flap = Book(book_id="b", title="t", author="a", author_bio="bio",
                     publisher="p", description="d", publish_year=2000)
    without = with_flap.model_copy(update={"description": ""})
    assert _quality(without) < _quality(with_flap)


# --------------------------------------------------------------------------- explanations

def test_explanation_quotes_the_flap_sentence_that_matched():
    matched = [Evidence(channel="lexical", detail="শব্দ মিলেছে: বাগান",
                        terms=["বাগানের"])]
    line = explain(make(subjects=["শ্রমজীবী"]), matched, {})
    assert "ফ্ল্যাপে:" in line
    assert "চা বাগানের শ্রমিকদের" in line
    # The book's own words come before the inferred tag.
    assert line.index("ফ্ল্যাপে:") < line.index("বিষয়ে চিহ্নিত")


def test_quote_is_matched_on_stems_not_surfaces():
    """The query says "মুক্তিযোদ্ধাদের", the flap says "মুক্তিযোদ্ধা"; comparing surfaces
    would find nothing."""
    record = make(description="এটি মুক্তিযোদ্ধাদের কথা বলা একটি বই। আরও কিছু লেখা এখানে আছে।")
    quote = flap_quote(record, [Evidence(channel="lexical", detail="",
                                         terms=["মুক্তিযোদ্ধা"])])
    assert "মুক্তিযোদ্ধাদের কথা" in quote


def test_no_matching_sentence_means_no_quotation():
    """A quote unrelated to the query is not evidence, and showing it first would imply
    it is."""
    assert flap_quote(make(), [Evidence(channel="lexical", detail="",
                                        terms=["রান্না"])]) == ""
    assert flap_quote(make(), []) == ""


def test_quotation_needs_a_flap():
    assert flap_quote(make(description=""), [Evidence(channel="lexical", detail="",
                                                      terms=["বাগান"])]) == ""


def test_long_quotation_is_clipped_and_marked():
    long_flap = "চা বাগানের " + "শ্রমিকদের জীবন " * 60 + "শেষ।"
    quote = flap_quote(make(description=long_flap),
                       [Evidence(channel="lexical", detail="", terms=["বাগান"])])
    assert quote.endswith("…")
    assert len(quote) <= 221
