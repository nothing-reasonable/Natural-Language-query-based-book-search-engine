"""Facts the catalogue already implies, computed at load time instead of asked of a model.

Enrichment from an LLM is expensive and, on this catalogue, thin: 85% of books came back
with no period, 89% of authors with no active period, and the multi-hop queries the
design is built around ("authors who were diplomats *during the Pakistan era*") need
exactly those fields.

A lot of that is derivable. Publication years are known for every book, the taxonomy maps
years to periods, and dictionary tagging is free. Doing it here rather than in
`ingest/enrich.py` has two advantages: it costs nothing to re-run, and it never touches
`enrichment.jsonl`, so the hours of model output already stored there stay untouched.

One distinction is deliberate and worth stating, because conflating the two would quietly
wreck precision:

  * `Enrichment.periods` is what a book is *about*. A 1958 book is not therefore about
    the Pakistan era, so publication year is **not** written here.
  * `Enrichment.author_periods` is when the author was *active*. Publishing in 1958 does
    mean exactly that, so it is written there.
"""

from __future__ import annotations

from collections import defaultdict

from search.core.schemas import IndexedBook
from search.query.taxonomy import Taxonomy, get_taxonomy


def augment(records: list[IndexedBook], taxonomy: Taxonomy | None = None) -> list[IndexedBook]:
    """Fill in what can be inferred. Additive and idempotent -- existing values win."""
    tax = taxonomy or get_taxonomy()
    _derive_author_periods(records, tax)
    _dictionary_backfill(records, tax)
    return records


# --------------------------------------------------------------------------- authors

def _derive_author_periods(records: list[IndexedBook], tax: Taxonomy) -> None:
    """When was this author active? Every year they published is direct evidence.

    Computed per author across their whole bibliography, so a writer with one 1965 book
    and one 1974 book is correctly active in both the Pakistan era and after it.
    """
    years_by_author: dict[str, set[int]] = defaultdict(set)
    for record in records:
        if record.book.author_id and record.book.publish_year:
            years_by_author[record.book.author_id].add(record.book.publish_year)

    periods_by_author: dict[str, list[str]] = {}
    for author_id, years in years_by_author.items():
        periods = []
        for year in sorted(years):
            period = tax.period_for_year(year)
            if period and period not in periods:
                periods.append(period)
        periods_by_author[author_id] = periods

    for record in records:
        derived = periods_by_author.get(record.book.author_id, [])
        if derived:
            record.enrichment.author_periods = _merge(record.enrichment.author_periods, derived)


# --------------------------------------------------------------------------- books

# Which enrichment field each taxonomy facet feeds, and where to look for it.
_BACKFILL = (
    ("subjects", "subjects", ("title", "description", "table_of_contents")),
    ("genres", "genres", ("title", "description")),
    ("periods", "periods", ("title", "description")),
    ("places", "places", ("title", "description")),
)


def _dictionary_backfill(records: list[IndexedBook], tax: Taxonomy) -> None:
    """Controlled-vocabulary spotting over the text, for everything the model missed.

    Deterministic and high precision: a concept is only added when one of its curated
    surface forms literally occurs. This is what makes expanding `data/taxonomy.yaml`
    pay off immediately, with no re-enrichment run.
    """
    # Author biographies repeat verbatim across every book by the same person; tagging
    # one of them 36 times is the bulk of the work in a naive pass.
    bio_cache: dict[str, list[str]] = {}

    for record in records:
        book = record.book
        about = " ".join(filter(None, (book.title, book.description, book.table_of_contents)))
        found = tax.find_all_in_text(about)
        for facet, field, _sources in _BACKFILL:
            values = found.get(facet) or []
            if values:
                setattr(record.enrichment, field,
                        _merge(getattr(record.enrichment, field), values))

        # Author roles are stated in the biography, not the blurb.
        if book.author_bio:
            key = book.author_id or book.author
            roles = bio_cache.get(key)
            if roles is None:
                roles = tax.find_in_text(book.author_bio, "occupations")
                bio_cache[key] = roles
            if roles:
                record.enrichment.author_roles = _merge(record.enrichment.author_roles, roles)


def _merge(existing: list[str], extra: list[str]) -> list[str]:
    seen = set(existing)
    return existing + [item for item in extra if item not in seen and not seen.add(item)]
