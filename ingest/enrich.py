"""Entity extraction, subject tagging and author-profile enrichment.

Two sources, combined:

  * **dictionary tagging** -- exact matches against `data/taxonomy.yaml`. Free, deterministic,
    and high precision. This alone is enough to build a usable index.
  * **the local LLM** -- fills in what the dictionary cannot see (implicit topics, the
    author's occupation buried in a biography, the historical period of the subject matter).

Everything the LLM returns is snapped back onto the controlled vocabulary, so downstream
code only ever sees canonical Bengali labels.
"""

from __future__ import annotations

import logging

from search.llm import LMStudio
from search.core.schemas import Book, Enrichment
from search.query.taxonomy import Taxonomy, get_taxonomy

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """তুমি একজন অভিজ্ঞ গ্রন্থাগার বিশেষজ্ঞ। তোমার কাজ বাংলা বইয়ের মেটাডেটা থেকে
কাঠামোবদ্ধ তথ্য বের করা।

নিয়ম:
- সব উত্তর বাংলায় দাও।
- যেখানে সম্ভব নিচের নিয়ন্ত্রিত তালিকা থেকে শব্দ বেছে নাও; তালিকায় না থাকলে সংক্ষিপ্ত বাংলা পরিভাষা লেখো।
- অনুমান করো না; যে তথ্য লেখায় নেই তা বাদ দাও, খালি তালিকা দেওয়া গ্রহণযোগ্য।
- প্রতিটি তালিকায় সর্বোচ্চ ৫টি আইটেম।
- শুধুমাত্র JSON ফেরত দাও।

নিয়ন্ত্রিত তালিকা:
বিষয় (subjects): {subjects}
ধরন (genres): {genres}
কাল (periods): {periods}
লেখকের ভূমিকা (author_roles): {occupations}"""

USER_TEMPLATE = """শিরোনাম: {title}
লেখক: {author}
প্রকাশক: {publisher}
প্রকাশকাল: {year}

লেখক পরিচিতি:
{author_bio}

বইয়ের বিবরণ:
{description}

{extra}"""


class Enricher:
    def __init__(self, llm: LMStudio | None = None, taxonomy: Taxonomy | None = None,
                 use_llm: bool = True):
        self.llm = llm
        self.taxonomy = taxonomy or get_taxonomy()
        self.use_llm = use_llm and llm is not None

    # ------------------------------------------------------------------ public
    def enrich(self, book: Book) -> Enrichment:
        base = self._from_dictionary(book)
        if not self.use_llm:
            return base
        try:
            predicted = self._from_llm(book)
        except Exception as exc:  # noqa: BLE001 - never let one book kill the run
            log.warning("LLM enrichment failed for %s: %s", book.book_id, exc)
            return base
        return self._merge(base, self._canonicalize(predicted))

    # ------------------------------------------------------------------ sources
    def _from_dictionary(self, book: Book) -> Enrichment:
        """Concept spotting on the fields where each facet is actually likely to appear."""
        about = " ".join([book.title, book.description, book.table_of_contents])
        bio = book.author_bio
        tax = self.taxonomy
        return Enrichment(
            subjects=tax.find_in_text(about, "subjects"),
            genres=tax.find_in_text(book.title + " " + book.description, "genres"),
            periods=_dedup(
                tax.find_in_text(about, "periods")
                + [p for p in [tax.period_for_year(book.publish_year)] if p]
            ),
            places=tax.find_in_text(about, "places"),
            author_roles=tax.find_in_text(bio, "occupations"),
            author_periods=tax.find_in_text(bio, "periods"),
        )

    def _from_llm(self, book: Book) -> Enrichment:
        tax = self.taxonomy
        system = SYSTEM_PROMPT.format(
            subjects=", ".join(tax.names("subjects")),
            genres=", ".join(tax.names("genres")),
            periods=", ".join(tax.names("periods")),
            occupations=", ".join(tax.names("occupations")),
        )
        extra = (
            f"সূচিপত্র:\n{book.table_of_contents[:1500]}" if book.table_of_contents else ""
        )
        user = USER_TEMPLATE.format(
            title=book.title,
            author=book.author,
            publisher=book.publisher or "-",
            year=book.publish_year or "-",
            author_bio=book.author_bio[:2000] or "-",
            description=book.description[:3000] or "-",
            extra=extra,
        )
        return self.llm.structured(system, user, Enrichment)

    # ------------------------------------------------------------------ post-processing
    def _canonicalize(self, e: Enrichment) -> Enrichment:
        tax = self.taxonomy
        return Enrichment(
            subjects=tax.canonicalize_all(e.subjects, "subjects"),
            topics=tax.canonicalize_all(e.topics),
            genres=tax.canonicalize_all(e.genres, "genres"),
            periods=tax.canonicalize_all(e.periods, "periods"),
            places=tax.canonicalize_all(e.places, "places"),
            events=tax.canonicalize_all(e.events),
            persons=[p.strip() for p in e.persons if p.strip()],
            author_roles=tax.canonicalize_all(e.author_roles, "occupations"),
            author_periods=tax.canonicalize_all(e.author_periods, "periods"),
            summary=e.summary.strip(),
        )

    @staticmethod
    def _merge(a: Enrichment, b: Enrichment) -> Enrichment:
        data = {}
        for name in Enrichment.model_fields:
            va, vb = getattr(a, name), getattr(b, name)
            data[name] = _dedup(list(va) + list(vb)) if isinstance(va, list) else (va or vb)
        return Enrichment(**data)


def _dedup(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out
