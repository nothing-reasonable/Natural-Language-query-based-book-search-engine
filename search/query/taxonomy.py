"""Loads `data/taxonomy.yaml` and turns free text into canonical concepts.

Two jobs:
  * canonicalise  -- "একাত্তর", "১৯৭১", "liberation war"  ->  "মুক্তিযুদ্ধ"
  * expand        -- "মুক্তিযুদ্ধ" -> {"মুক্তিযুদ্ধ", "একাত্তর", "১৯৭১", ...} for query expansion
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from search.core import bengali
from config import settings

FACETS = ("subjects", "periods", "genres", "occupations", "places")


@dataclass
class Concept:
    name: str
    facet: str
    aliases: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    period: str | None = None
    start: int | None = None
    end: int | None = None
    # Whether this concept may *contribute* its aliases to query expansion. Recognising a
    # term and broadcasting its synonyms are different jobs, and a few concepts are good
    # at the first and actively harmful at the second: "বাংলাদেশ" must still absorb "পূর্ব
    # পাকিস্তান" when tagging a book, but a query about ancient architecture that merely
    # contains the word "বাংলার" should not be rewritten to include "পূর্ব পাকিস্তান".
    # Set `expand: false` in taxonomy.yaml for concepts that broad. Tagging is unaffected.
    expandable: bool = True

    @property
    def surface_forms(self) -> list[str]:
        return [self.name, *self.aliases]


class Taxonomy:
    def __init__(self, path: Path):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        self.concepts: dict[str, Concept] = {}
        self._lookup: dict[str, str] = {}  # analysed alias key -> canonical name
        # (alias key, its token set, canonical name) -- built once, scanned per lookup.
        self._alias_tokens: list[tuple[str, frozenset[str], str]] = []

        for facet in FACETS:
            for name, body in (raw.get(facet) or {}).items():
                body = body or {}
                concept = Concept(
                    name=name,
                    facet=facet,
                    aliases=list(body.get("aliases") or []),
                    related=list(body.get("related") or []),
                    period=body.get("period"),
                    start=body.get("start"),
                    end=body.get("end"),
                    expandable=bool(body.get("expand", True)),
                )
                self.concepts[name] = concept
                for surface in concept.surface_forms:
                    k = bengali.key(surface)
                    if k:
                        self._lookup.setdefault(k, name)

        for alias_key, name in sorted(self._lookup.items()):
            tokens = frozenset(alias_key.split())
            if tokens:
                self._alias_tokens.append((alias_key, tokens, name))

    # ------------------------------------------------------------------ lookups
    def canonicalize(self, term: str) -> str | None:
        """Exact (analysed) match first, then a containment fallback for phrases.

        When several aliases are contained in the term, the most specific one wins --
        "পূর্ব পাকিস্তান" must not collapse to whichever of "পাকিস্তান"/"পূর্ব পাকিস্তান"
        a dict happened to yield first.
        """
        if not term:
            return None
        k = bengali.key(term)
        if k in self._lookup:
            return self._lookup[k]
        tokens = set(bengali.analyze(term))
        if not tokens:
            return None
        matches = [
            (len(alias_key.split()), alias_key, name)
            for alias_key, name in self._lookup.items()
            if alias_key and set(alias_key.split()) <= tokens
        ]
        return max(matches)[2] if matches else None

    def canonicalize_all(self, terms: list[str], facet: str | None = None) -> list[str]:
        """Map a list of raw terms to canonical names, keeping unknown terms as-is."""
        out: list[str] = []
        for term in terms:
            term = bengali.normalize(term)
            if not term:
                continue
            name = self.canonicalize(term)
            if name and facet and self.concepts[name].facet != facet:
                name = None
            out.append(name or term)
        return _dedup(out)

    def expand(self, term: str) -> list[str]:
        """Canonical name + all its aliases + related concepts. Used for query expansion.

        A concept marked `expand: false` contributes only the term the user actually
        typed. Those concepts are the ones broad enough to match almost any query --
        "বাংলা" inside "প্রাচীন বাংলার স্থাপত্য", or the genre "ইতিহাস", which 2,034 of
        the 5,285 books carry. Broadcasting their synonyms does not widen the search
        toward the topic, it drowns it: the four distinct queries about Greece, Sindhu,
        archaeology and mosques all expanded to the same six generic terms, and one about
        colonial policing came back rewritten to include "পূর্ব পাকিস্তান".
        """
        name = self.canonicalize(term)
        if not name:
            return [term]
        concept = self.concepts[name]
        if not concept.expandable:
            return [term]
        forms = list(concept.surface_forms)
        for rel in concept.related:
            related = self.concepts.get(rel)
            if related is None:
                forms.append(rel)
            elif related.expandable:
                forms.extend(related.surface_forms)
        if concept.period:
            forms.append(concept.period)
        return _dedup(forms)

    def expand_term(self, term: str, limit: int = 4,
                    vocabulary: set[str] | None = None) -> list[str]:
        """At most `limit` expansion terms for ONE query keyword.

        `expand` returns a concept's entire alias list, which is the right answer for
        tagging and the wrong one for a query: "মুক্তিযুদ্ধ" carries eleven aliases, so a
        four-word question came back rewritten into eighteen terms, most of them
        restating the one concept the query already named. Retrieval then scored the
        restatement rather than the question.

        Two rules decide which few survive:

        * a term that does not occur in the lexical vocabulary cannot match any book, so
          it is dead weight -- "freedom fight" and "muktijuddho" are in the taxonomy for
          recognising user input, not for searching this Bengali catalogue;
        * among the rest, prefer multi-word aliases. They are the specific ones
          ("শরণার্থী শিবির" over "refugee"), and specificity is the whole point of
          expanding at all.
        """
        name = self.canonicalize(term)
        if not name:
            return []
        concept = self.concepts[name]
        if not concept.expandable:
            return []

        seen = {bengali.key(term)}
        candidates: list[str] = []
        for form in self._expansion_forms(concept):
            k = bengali.key(form)
            if k and k not in seen:
                seen.add(k)
                candidates.append(form)

        def rank(form: str) -> tuple[int, int]:
            in_vocab = 0 if _in_vocabulary(form, vocabulary) else 1
            return (in_vocab, -len(bengali.analyze(form)))

        return sorted(candidates, key=rank)[:max(0, limit)]

    def _expansion_forms(self, concept: Concept) -> list[str]:
        forms = list(concept.surface_forms)
        for rel in concept.related:
            related = self.concepts.get(rel)
            if related is None:
                forms.append(rel)
            elif related.expandable:
                forms.extend(related.surface_forms)
        return forms

    def expand_terms(self, terms: list[str], limit: int = 4,
                     vocabulary: set[str] | None = None) -> list[str]:
        """Per-keyword expansion, capped. Order follows the keywords the user typed."""
        out: list[str] = []
        for term in terms:
            out.extend(self.expand_term(term, limit, vocabulary))
        return _dedup(out)

    def expand_many(self, terms: list[str]) -> list[str]:
        out: list[str] = []
        for t in terms:
            out.extend(self.expand(t))
        return _dedup(out)

    def facet_of(self, name: str) -> str | None:
        c = self.concepts.get(name)
        return c.facet if c else None

    def period_for_year(self, year: int | None) -> str | None:
        """Which historical period a year falls in.

        The ranges in `taxonomy.yaml` are meant to be disjoint. If an edit ever makes two
        of them overlap, the narrowest match wins rather than whichever the YAML happened
        to list first -- ordering is invisible at the call site, and silently mis-dating
        12% of the catalogue is a hard bug to notice from the outside.
        """
        if year is None:
            return None
        matches = [
            c for c in self.concepts.values()
            if c.facet == "periods" and c.start is not None and c.end is not None
            and c.start <= year <= c.end
        ]
        if not matches:
            return None
        return min(matches, key=lambda c: (c.end - c.start, c.name)).name

    def overlapping_periods(self) -> list[tuple[str, str]]:
        """Period pairs whose ranges intersect. Empty is the healthy state."""
        ranges = [
            c for c in self.concepts.values()
            if c.facet == "periods" and c.start is not None and c.end is not None
        ]
        clashes = []
        for i, a in enumerate(ranges):
            for b in ranges[i + 1:]:
                if a.start <= b.end and b.start <= a.end:
                    clashes.append(tuple(sorted((a.name, b.name))))
        return sorted(set(clashes))

    def names(self, facet: str) -> list[str]:
        return [c.name for c in self.concepts.values() if c.facet == facet]

    def find_in_text(self, text: str, facet: str | None = None) -> list[str]:
        """Dictionary tagging: which canonical concepts appear in this text?"""
        tokens = set(bengali.analyze(text))
        return self.find_in_tokens(tokens, facet)

    def find_in_tokens(self, tokens: set[str], facet: str | None = None) -> list[str]:
        """Same, for callers that already analysed the text.

        Tagging one book across five facets used to analyse its text five times, which
        is most of the cost of a full backfill over the catalogue.
        """
        if not tokens:
            return []
        hits = []
        for alias_key, alias_tokens, name in self._alias_tokens:
            if facet and self.concepts[name].facet != facet:
                continue
            if alias_tokens <= tokens:
                hits.append(name)
        return _dedup(hits)

    def find_all_in_text(self, text: str) -> dict[str, list[str]]:
        """Every facet at once, analysing the text exactly once."""
        tokens = set(bengali.analyze(text))
        found: dict[str, list[str]] = {f: [] for f in FACETS}
        if not tokens:
            return found
        for _alias_key, alias_tokens, name in self._alias_tokens:
            if alias_tokens <= tokens:
                facet = self.concepts[name].facet
                if name not in found[facet]:
                    found[facet].append(name)
        return found


def _dedup(items: list[str]) -> list[str]:
    seen, out = set(), []
    for i in items:
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out


@functools.lru_cache(maxsize=1)
def get_taxonomy() -> Taxonomy:
    return Taxonomy(settings.taxonomy_path)


def _in_vocabulary(form: str, vocabulary: set[str] | None) -> bool:
    """True when every stem of `form` is a term the lexical index actually holds."""
    if not vocabulary:
        return True  # no vocabulary to check against: do not penalise anything
    stems = bengali.analyze(form)
    return bool(stems) and all(stem in vocabulary for stem in stems)
