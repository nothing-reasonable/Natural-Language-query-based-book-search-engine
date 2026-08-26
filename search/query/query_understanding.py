"""Query understanding: classify, normalise, expand, and decompose.

Runs in three passes so the engine keeps working when the LLM is not:

  1. **rules**  -- taxonomy lookups, year detection, typo repair against the index vocabulary
  2. **LLM**    -- intent, implicit filters, and multi-hop decomposition
  3. **merge**  -- everything the LLM produced is snapped onto the controlled vocabulary
"""

from __future__ import annotations

import logging
import re

from rapidfuzz import process as fuzzy

from search.core import bengali
from search.indexing.entity_index import EntityIndex
from search.llm import LMStudio
from config import Settings, settings as default_settings
from search.core.schemas import Concepts, EntityRef, Filters, GraphStep, QueryPlan, QueryPlanDraft
from search.query.taxonomy import Taxonomy, get_taxonomy

log = logging.getLogger(__name__)

_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_TYPO_THRESHOLD = 88

# A bare year in a query is usually the *subject* ("১৯৭১" means the war, not the print
# run). It only constrains publication when the query says so.
#
# The hint list includes authorship verbs as well as publication ones: "একবিংশ শতাব্দীতে
# লেখা" asks when the book was written, and publication year is the only date this
# catalogue records -- an approximation, but the one the user means.
_PUBLICATION_HINTS = (
    "প্রকাশিত", "প্রকাশকাল", "প্রকাশনার", "ছাপা", "সালে প্রকাশ", "প্রকাশের",
    "লেখা", "রচিত", "লিখিত", "লেখেন",
    "published", "written",
)
_RANGE_HINTS = ("থেকে", "পর্যন্ত", "মধ্যে", "সাল থেকে")

# Named spans of time, as inclusive publication-year ranges.
#
# These are deliberately *not* taxonomy periods. "পাকিস্তান আমল" describes what a book is
# about; "একবিংশ শতাব্দী" here describes when it was printed, and the two must not be
# confused -- a 2010 book about the Pakistan era belongs to both, for different reasons.
#
# Keys are matched on the analysed form, so inflections ("শতাব্দীতে", "দশকে") resolve
# without being listed.
_TIME_EXPRESSIONS: tuple[tuple[str, int, int], ...] = (
    ("সপ্তদশ শতাব্দী", 1601, 1700),
    ("অষ্টাদশ শতাব্দী", 1701, 1800),
    ("ঊনবিংশ শতাব্দী", 1801, 1900),
    ("উনবিংশ শতাব্দী", 1801, 1900),
    ("বিংশ শতাব্দী", 1901, 2000),
    ("একবিংশ শতাব্দী", 2001, 2100),
    ("সপ্তদশ শতক", 1601, 1700),
    ("অষ্টাদশ শতক", 1701, 1800),
    ("ঊনবিংশ শতক", 1801, 1900),
    ("বিংশ শতক", 1901, 2000),
    ("একবিংশ শতক", 2001, 2100),
    ("ত্রিশের দশক", 1930, 1939),
    ("চল্লিশের দশক", 1940, 1949),
    ("পঞ্চাশের দশক", 1950, 1959),
    ("ষাটের দশক", 1960, 1969),
    ("সত্তরের দশক", 1970, 1979),
    ("আশির দশক", 1980, 1989),
    ("নব্বইয়ের দশক", 1990, 1999),
)

SYSTEM_PROMPT = """তুমি একটি বাংলা বই-অনুসন্ধান ইঞ্জিনের কোয়েরি বিশ্লেষক।
ব্যবহারকারীর প্রশ্ন পড়ে একটি অনুসন্ধান পরিকল্পনা (JSON) তৈরি করো।

intent-এর অর্থ:
- simple: নির্দিষ্ট বই বা লেখকের নাম ধরে খোঁজা
- semantic: বিষয়ভিত্তিক বা ভাবগত খোঁজা
- filtered: সাল, ভাষা, ধরন বা প্রকাশকের মতো স্পষ্ট শর্ত আছে
- personalized: "আমার পছন্দের", "আমার জন্য" ধরনের ব্যক্তিগত চাহিদা
- multi_hop: উত্তর পেতে দুই বা তার বেশি ধাপ লাগে, যেমন আগে লেখক খুঁজে তারপর তাঁদের বই

multi_hop হলে steps-এ ধাপগুলো লেখো:
- {{"kind": "find_authors", "occupations": [...], "periods": [...]}}
- {{"kind": "find_books", "subjects": [...], "periods": [...], "places": [...]}}

নিয়ম:
- keywords: প্রশ্নের মূল বাংলা শব্দ।
- expanded_terms: সমার্থক ও সম্পর্কিত পরিভাষা (যেমন "একাত্তর" -> "মুক্তিযুদ্ধ")।
- filters-এ কেবল স্পষ্টভাবে বলা শর্ত রাখো, অনুমান করো না।
- যেখানে সম্ভব নিয়ন্ত্রিত তালিকার শব্দ ব্যবহার করো।
- শুধু JSON ফেরত দাও।

নিয়ন্ত্রিত তালিকা:
বিষয়: {subjects}
ধরন: {genres}
কাল: {periods}
পেশা: {occupations}"""


class QueryUnderstanding:
    def __init__(self, llm: LMStudio | None = None, taxonomy: Taxonomy | None = None,
                 vocabulary: set[str] | None = None, mode: str = "never",
                 entities: EntityIndex | None = None,
                 year_bounds: tuple[int, int] | None = None,
                 settings: Settings = default_settings):
        self.settings = settings
        self.llm = llm
        self.taxonomy = taxonomy or get_taxonomy()
        self.vocabulary = vocabulary or set()
        # No client means no model, whatever the configuration asked for.
        self.mode = mode if llm is not None else "never"
        self.entities = entities
        # The catalogue's actual publication range, used to reject year filters that do
        # not narrow anything. Without it, "everything up to 2100" looks like a constraint.
        self.year_bounds = year_bounds

    # ------------------------------------------------------------------ public
    def analyze(self, query: str, *, personalized: bool = False) -> QueryPlan:
        plan = self._rule_based(query)
        if self._should_call_llm(plan):
            try:
                draft = self.llm.structured(self._system(), query, QueryPlanDraft, max_tokens=700)
                plan = self._merge(plan, draft)
            except Exception as exc:  # noqa: BLE001
                log.warning("LLM query understanding failed, using rules only: %s", exc)

        plan.raw_query = query
        plan.normalized_query = plan.normalized_query or bengali.normalize(query)
        plan = self._canonicalize(plan)
        if personalized and plan.intent == "semantic" and _wants_personalization(query):
            plan.intent = "personalized"
        return plan

    def _should_call_llm(self, plan: QueryPlan) -> bool:
        """Whether to spend ~25 s asking the model to read a query the rules already read.

        "always" skips the gate entirely -- that is the point of it.
        """
        if self.mode == "never":
            return False
        if self.mode == "always":
            return True
        return self._needs_llm(plan)

    @staticmethod
    def _needs_llm(plan: QueryPlan) -> bool:
        """Is there anything left for a model to work out?

        The rule pass resolves author and publisher names against the real catalogue,
        spots controlled-vocabulary concepts, reads publication-year constraints and
        detects the occupation+period shape of a multi-hop question. When it has found
        any of that, the query is already understood, and calling a local thinking model
        costs ~25 seconds to produce something measurably worse -- on
        "হুমায়ূন আহমেদের মুক্তিযুদ্ধের বই" the rules give intent=filtered with the author
        resolved, while the model returns intent=multi_hop and four English paraphrases.

        So the model is kept for what rules genuinely cannot do: free text with no
        recognised name and no vocabulary hit.
        """
        if plan.entities or plan.steps:
            return False
        if not plan.filters.is_empty():
            return False
        if plan.concepts.all_terms():
            return False
        return True

    # ------------------------------------------------------------------ pass 1: rules
    def _rule_based(self, query: str) -> QueryPlan:
        """Dictionary spotting only. It never sets hard filters -- a concept found by
        string matching is a hint, not a constraint, and over-filtering hides good books."""
        normalized = bengali.normalize(query)
        digits = bengali.fold_digits(normalized)
        tax = self.taxonomy

        concepts = Concepts(
            subjects=tax.find_in_text(digits, "subjects"),
            genres=tax.find_in_text(digits, "genres"),
            periods=tax.find_in_text(digits, "periods"),
            places=tax.find_in_text(digits, "places"),
            occupations=tax.find_in_text(digits, "occupations"),
        )
        years = [int(y) for y in _YEAR.findall(digits)]
        for year in years:
            period = tax.period_for_year(year)
            if period and period not in concepts.periods:
                concepts.periods.append(period)

        named = self._link_entities(normalized)

        multi_hop = bool(concepts.occupations) and bool(concepts.periods or concepts.subjects)
        intent = "multi_hop" if multi_hop else "semantic"
        steps: list[GraphStep] = []
        if intent == "multi_hop":
            steps = [
                GraphStep(kind="find_authors", occupations=concepts.occupations, periods=concepts.periods),
                GraphStep(kind="find_books", subjects=concepts.subjects, places=concepts.places),
            ]

        year_filter = self._year_filter(normalized, years)

        plan = QueryPlan(
            raw_query=query,
            normalized_query=normalized,
            intent=intent,
            keywords=self._repair(bengali.analyze(normalized)),
            # Left empty on purpose: _canonicalize builds the expansion from the
            # keywords, capped per keyword. Filling it here with the uncapped
            # alias dump would put back exactly what the cap exists to prevent.
            expanded_terms=[],
            concepts=concepts,
            steps=steps,
            entities=named,
        )
        self._apply_entities(plan)
        if year_filter:
            plan.filters.year_from, plan.filters.year_to = year_filter
            if plan.intent == "semantic":
                plan.intent = "filtered"
        return plan

    @staticmethod
    def _year_filter(normalized: str, years: list[int]) -> tuple[int, int] | None:
        """Only treat a date as a publication constraint when the query says so.

        "১৯৭১ সালের বই" is about the war; "১৯৭১ সালে প্রকাশিত বই" is about the print date.
        The same ambiguity applies to named spans: "ঊনবিংশ শতাব্দীর বাংলা" is a subject,
        while "ঊনবিংশ শতাব্দীতে লেখা" is a constraint. Guessing wrong empties the result
        set for the most common topic in the catalogue, so both forms require an explicit
        authorship or publication word.
        """
        if not any(hint in normalized for hint in _PUBLICATION_HINTS):
            return None

        # A named span wins over a bare year: "একবিংশ শতাব্দীতে লেখা ১৯৭১-এর বই" is a
        # 21st-century book about 1971, and the year in it is the subject.
        span = _named_time_span(normalized)
        if span is not None:
            return span
        if not years:
            return None
        if len(years) >= 2 and any(hint in normalized for hint in _RANGE_HINTS):
            return min(years), max(years)
        return years[0], years[0]

    # ------------------------------------------------------------------ entity linking
    def _link_entities(self, normalized: str) -> list[EntityRef]:
        if self.entities is None:
            return []
        return [
            EntityRef(kind=m.kind, name=m.name, entity_id=m.entity_id,
                      score=m.score, hard=m.hard)
            for m in self.entities.find(normalized)
        ]

    @staticmethod
    def _apply_entities(plan: QueryPlan) -> None:
        """Turn confident name matches into hard constraints.

        This is the difference between answering "books by হুমায়ূন আহমেদ about the war" and
        answering "books about the war, by anyone, that happen to mention him". Only
        `hard` matches constrain; a weaker one stays on the plan as evidence and lets
        ranking decide.
        """
        authors = [e for e in plan.entities if e.kind == "author" and e.hard]
        publishers = [e for e in plan.entities if e.kind == "publisher" and e.hard]
        if authors:
            plan.filters.authors = [e.name for e in authors]
            plan.filters.author_ids = [e.entity_id for e in authors if e.entity_id]
        if publishers:
            plan.filters.publishers = [e.name for e in publishers]
        if (authors or publishers) and plan.intent == "semantic":
            plan.intent = "filtered"

    def _repair(self, tokens: list[str]) -> list[str]:
        """Map out-of-vocabulary tokens onto the nearest indexed term (typo tolerance)."""
        if not self.vocabulary:
            return tokens
        vocab = list(self.vocabulary)
        repaired = []
        for token in tokens:
            if token in self.vocabulary or len(token) < 4:
                repaired.append(token)
                continue
            match = fuzzy.extractOne(token, vocab, score_cutoff=_TYPO_THRESHOLD)
            repaired.append(match[0] if match else token)
        return repaired

    # ------------------------------------------------------------------ pass 2/3
    def _system(self) -> str:
        tax = self.taxonomy
        return SYSTEM_PROMPT.format(
            subjects=", ".join(tax.names("subjects")),
            genres=", ".join(tax.names("genres")),
            periods=", ".join(tax.names("periods")),
            occupations=", ".join(tax.names("occupations")),
        )

    def _merge(self, rules: QueryPlan, draft: QueryPlanDraft) -> QueryPlan:
        """Rules keep the concepts (they are grounded in the taxonomy); the LLM contributes
        intent, hard filters and the multi-hop decomposition."""
        merged = rules.model_copy(deep=True)
        merged.intent = draft.intent or rules.intent
        merged.keywords = _dedup(rules.keywords + draft.keywords)
        merged.expanded_terms = _dedup(rules.expanded_terms + draft.expanded_terms)

        # Two guards against a small local model over-reaching. Both failure modes --
        # inventing a graph traversal for a plain topical query, or inventing a hard
        # filter that empties the result set -- are worse than ignoring the model.
        if merged.intent == "multi_hop":
            merged.steps = draft.steps or rules.steps
        if merged.intent in ("filtered", "multi_hop"):
            merged.filters = self._ground_filters(rules, draft.filters)
        return merged

    def _ground_filters(self, rules: QueryPlan, proposed: Filters) -> Filters:
        """Accept a model-proposed constraint only where the query actually states it.

        A hard filter removes books permanently, so the bar for creating one is that the
        user asked for it -- not that a model inferred it. The distinction is easy to lose:
        asked about "মুক্তিযুদ্ধের বই", the local model returns
        `periods=[স্বাধীন বাংলাদেশ]` and `year_to=2100`, which is a reasonable *description*
        of the topic and a terrible *constraint*. The year bound alone matched all 4,880
        books while announcing itself in the explanation as a publication-date match, and
        the period bound silently dropped every book tagged with a different era.

        So each field is grounded against evidence that already exists:

          * authors / publishers -- from the entity linker, which matched real catalogue
            names, so the model never contributes them at all;
          * years -- from the rule pass, which requires both a year and a publication word
            in the query ("১৯৭১ সালে প্রকাশিত"), and whose regex cannot even produce 2100;
          * subjects / genres / periods / places -- kept only where the taxonomy spotted
            the same term in the query text.

        Nothing is thrown away: an ungrounded term is demoted to `concepts`, where it still
        steers ranking. It just loses the power to exclude.
        """
        grounded = Filters()

        # Resolved against the catalogue; the model has no say here.
        grounded.authors = list(rules.filters.authors)
        grounded.author_ids = list(rules.filters.author_ids)
        grounded.publishers = list(rules.filters.publishers)

        # Years: the rules first, since they parse the common phrasings exactly. The model
        # is allowed to fill the gap for wordings `_TIME_EXPRESSIONS` does not list ("গত
        # দুই দশকে লেখা"), but only under the same gate the rules use -- the query has to
        # contain an authorship or publication word. That gate is what the original bug
        # failed: "মুক্তিযুদ্ধের বই" mentions no date at all, so a proposed year_to=2100 has
        # nothing in the query to stand on and is refused.
        grounded.year_from = rules.filters.year_from
        grounded.year_to = rules.filters.year_to
        if grounded.year_from is None and grounded.year_to is None:
            asks_about_dates = any(h in rules.normalized_query for h in _PUBLICATION_HINTS)
            if asks_about_dates:
                span = self._usable_year_span(proposed.year_from, proposed.year_to)
                if span is not None:
                    grounded.year_from, grounded.year_to = span

        # Topical facets never become hard filters here, even when the query does contain
        # the term. Two reasons, and the second is the decisive one:
        #
        #   * the design document is explicit that a concept found by matching text is a
        #     hint, not an instruction;
        #   * enrichment is incomplete -- 32% of this catalogue has no `subjects` at all --
        #     so filtering on a subject silently discards relevant books that simply were
        #     never tagged. "মুক্তিযুদ্ধের বই" would drop war books whose enrichment is thin,
        #     which is the opposite of what the user asked for.
        #
        # They are kept as soft concepts instead, where they steer the graph channel and
        # query expansion without excluding anything. An explicit facet control in a UI
        # should set `Filters` directly; that is a stated constraint, not an inferred one.
        for facet in ("subjects", "genres", "periods", "places"):
            existing = getattr(rules.concepts, facet, [])
            for value in getattr(proposed, facet, []) or []:
                if value not in existing:
                    existing.append(value)

        # `language` is never inferred: the catalogue is effectively monolingual, so a
        # guessed value can only remove results. It belongs to an explicit control.
        grounded.language = rules.filters.language
        return grounded

    def _usable_year_span(self, low: int | None, high: int | None) -> tuple[int, int] | None:
        """Clamp a proposed range to the catalogue, and reject it if it narrows nothing.

        Models reach for sentinels: `year_to=2100` with no lower bound reads as a filter
        and behaves as a no-op, matching every book while telling the user a publication
        date matched. Once clamped to what the catalogue actually holds, a range that
        still spans the whole thing is recognised as the no-op it is.
        """
        if low is None and high is None:
            return None
        if self.year_bounds is None:
            # Nothing to compare against -- require an explicit two-sided range.
            return (low, high) if low is not None and high is not None and low <= high else None

        floor, ceiling = self.year_bounds
        low = floor if low is None else max(low, floor)
        high = ceiling if high is None else min(high, ceiling)
        if low > high:
            return None
        if low <= floor and high >= ceiling:
            return None  # covers the catalogue; not a constraint
        return low, high

    def _canonicalize(self, plan: QueryPlan) -> QueryPlan:
        tax = self.taxonomy
        for facet in ("subjects", "genres", "periods", "places", "occupations"):
            setattr(plan.concepts, facet, tax.canonicalize_all(getattr(plan.concepts, facet), facet))
            if hasattr(plan.filters, facet):
                setattr(plan.filters, facet, tax.canonicalize_all(getattr(plan.filters, facet), facet))

        for step in plan.steps:
            step.occupations = tax.canonicalize_all(step.occupations, "occupations")
            step.periods = tax.canonicalize_all(step.periods, "periods")
            step.subjects = tax.canonicalize_all(step.subjects, "subjects")
            step.places = tax.canonicalize_all(step.places, "places")

        # Expansion is driven by the *keywords the user typed*, a few terms each, rather
        # than by whichever single concept the whole query happened to collapse onto.
        # Collapsing lost the query: "প্রাচীন বাংলার স্থাপত্যধারা ও মসজিদ" resolved to the
        # concept প্রাচীন ইতিহাস and expanded to its aliases, so it became
        # indistinguishable from three other queries about Greece, Sindhu and archaeology
        # -- the words that made it a question about mosques were dropped on the floor.
        limit = self.settings.expansion_per_keyword
        keyword_terms = tax.expand_terms(plan.keywords, limit, self.vocabulary)
        concept_terms = tax.expand_terms(plan.concepts.all_terms(), limit, self.vocabulary)
        step_terms = tax.expand_terms(
            [t for s in plan.steps for t in s.subjects + s.periods], limit, self.vocabulary
        )
        plan.expanded_terms = _dedup(
            plan.expanded_terms + keyword_terms + concept_terms + step_terms
        )
        return plan

    # ------------------------------------------------------------------ helper
    def search_terms(self, plan: QueryPlan) -> list[str]:
        """Everything worth throwing at the lexical index."""
        return _dedup([plan.normalized_query, *plan.keywords, *plan.expanded_terms])


_PERSONAL_HINTS = ("আমার", "আমাকে", "আমি পছন্দ", "আমার জন্য", "সুপারিশ", "রেকমেন্ড")


def _wants_personalization(query: str) -> bool:
    return any(hint in query for hint in _PERSONAL_HINTS)


def _dedup(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        item = (item or "").strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _named_time_span(normalized: str) -> tuple[int, int] | None:
    """Resolve "একবিংশ শতাব্দী", "নব্বইয়ের দশক" and friends to a year range.

    Matched on analysed tokens so inflected forms need no separate entry. The most
    specific expression wins, so "বিংশ" inside "একবিংশ" cannot claim the match.
    """
    tokens = set(bengali.analyze(normalized))
    if not tokens:
        return None
    best: tuple[int, tuple[int, int]] | None = None
    for phrase, start, end in _TIME_EXPRESSIONS:
        phrase_tokens = set(bengali.analyze(phrase))
        if phrase_tokens and phrase_tokens <= tokens:
            if best is None or len(phrase_tokens) > best[0]:
                best = (len(phrase_tokens), (start, end))
    return best[1] if best else None
