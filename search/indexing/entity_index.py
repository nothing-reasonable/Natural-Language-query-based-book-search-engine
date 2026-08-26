"""Recognising the *named things* in a query: authors and publishers.

Half the queries this catalogue gets name a person, and until the pipeline can tell that
"হুমায়ূন আহমেদ এর মুক্তিযুদ্ধের বই" contains an author rather than three more topic words, it
answers a different question than the one asked -- returning other people's books on the
same subject, which is exactly what the retrieval baseline does today.

Matching runs over `ingest.clean.author_key`: honorifics dropped, tokens stemmed and
sorted. That makes "ড. আনিসুজ্জামান", "অধ্যাপক আনিসুজ্জামান" and "আনিসুজ্জামান" one key, and
makes word order irrelevant.

The hard part is not finding matches, it is *not* finding them: "আহমেদ" is a surname
shared by dozens of authors, and turning a query into a hard filter on the wrong person
is worse than not filtering at all. Hence the rules in `_match`:

  * a match must cover the candidate's whole name, not just part of it
  * single-token matches never produce a hard filter, however good the score
  * an ambiguous token (many authors share it) is discarded rather than guessed at
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from search.core import bengali
from ingest.clean import author_key
from search.core.schemas import IndexedBook

# Score at or above which a fuzzy match is treated as certain enough to filter on.
HARD_THRESHOLD = 92
# Score at or above which a match is kept as a soft ranking signal only.
SOFT_THRESHOLD = 82
# Longest name (in tokens) we try to spot inside a query.
MAX_SPAN = 5


@dataclass(frozen=True)
class EntityMatch:
    kind: str            # "author" | "publisher"
    name: str            # canonical display name
    entity_id: str       # author_id, or the normalised publisher name
    score: float         # 0..100
    span: tuple[int, ...]  # indices of the query tokens it consumed
    hard: bool           # confident enough to exclude everything else?

    @property
    def tokens(self) -> int:
        return len(self.span)


class EntityIndex:
    """Name lookup built from the catalogue itself, so it can only ever match a real
    author or publisher -- there is nothing here for a model to hallucinate."""

    def __init__(self, records: list[IndexedBook]):
        self.author_by_key: dict[str, tuple[str, str]] = {}   # key -> (author_id, display)
        self.author_counts: dict[str, int] = defaultdict(int)
        self.publisher_by_key: dict[str, str] = {}            # key -> display
        self.publisher_counts: dict[str, int] = defaultdict(int)
        # How many distinct people share a single name token ("আহমেদ" -> many).
        self.token_owners: dict[str, set[str]] = defaultdict(set)

        for record in records:
            book = record.book
            if book.author:
                key = author_key(book.author)
                if key:
                    self.author_by_key.setdefault(key, (book.author_id, book.author))
                    self.author_counts[key] += 1
                    for token in key.split():
                        self.token_owners[token].add(key)
            if book.publisher:
                key = bengali.key(book.publisher)
                if key:
                    self.publisher_by_key.setdefault(key, book.publisher)
                    self.publisher_counts[key] += 1

        self._author_keys = list(self.author_by_key)
        self._publisher_keys = list(self.publisher_by_key)

    # ------------------------------------------------------------------ public
    def find(self, query: str) -> list[EntityMatch]:
        """Best non-overlapping author/publisher mentions in the query, longest first."""
        tokens = bengali.analyze(query)
        if not tokens:
            return []

        found: list[EntityMatch] = []
        for kind in ("author", "publisher"):
            found.extend(self._scan(tokens, kind))

        # A longer, higher-scoring match wins the tokens it consumed.
        found.sort(key=lambda m: (m.tokens, m.score), reverse=True)
        chosen: list[EntityMatch] = []
        taken: set[int] = set()
        for match in found:
            if taken.isdisjoint(match.span):
                chosen.append(match)
                taken.update(match.span)
        return chosen

    # ------------------------------------------------------------------ internals
    def _scan(self, tokens: list[str], kind: str) -> list[EntityMatch]:
        table = self.author_by_key if kind == "author" else self.publisher_by_key
        keys = self._author_keys if kind == "author" else self._publisher_keys
        out = []
        # Longest spans first: "হুমায়ূন আহমেদ" should beat the "আহমেদ" inside it.
        for width in range(min(MAX_SPAN, len(tokens)), 0, -1):
            for start in range(len(tokens) - width + 1):
                span = tuple(range(start, start + width))
                window = tokens[start : start + width]
                match = self._match(window, span, table, keys, kind)
                if match is not None:
                    out.append(match)
        return out

    def _match(self, window: list[str], span: tuple[int, ...], table: dict,
               keys: list[str], kind: str) -> EntityMatch | None:
        probe = " ".join(sorted(set(window)))
        if not probe:
            return None

        exact = table.get(probe)
        if exact is not None:
            # A single token is enough to filter on when exactly one person owns it
            # ("আনিসুজ্জামান"), and never enough when many do ("আহমেদ").
            confident = len(window) >= 2 or not self._ambiguous(window[0], kind)
            return self._build(kind, probe, exact, 100.0, span, hard=confident)

        # Fuzzy, but only against names of a similar length -- token_sort_ratio would
        # happily give "রফিক" a good score against "রফিকুজ্জামান হুমায়ুন".
        candidate = process.extractOne(
            probe, keys, scorer=fuzz.token_sort_ratio, score_cutoff=SOFT_THRESHOLD
        )
        if candidate is None:
            return None
        key, score, _ = candidate
        if not self._covers(probe, key):
            return None
        if len(window) == 1 and self._ambiguous(window[0], kind):
            return None
        hard = score >= HARD_THRESHOLD and (len(window) >= 2 or kind == "publisher")
        return self._build(kind, key, table[key], float(score), span, hard=hard)

    @staticmethod
    def _covers(probe: str, key: str) -> bool:
        """The query has to account for most of the name, not one word of it.

        Without this, "আহমেদ এর বই" fuzzy-matches "হুমায়ূন আহমেদ" strongly enough to filter
        on, and the search silently answers a question nobody asked.

        Tokens are compared loosely rather than with `==`. The stemmer is lexicon-free
        and mis-stems about 1.8% of inflected words (measured over this catalogue --
        every alternative rule tried was worse, so the residue is accepted rather than
        fixed). Author surnames are over-represented in that residue because so many end
        in দ: "আহমেদের" stems to "আহম" while "আহমেদ" stems to itself. Exact token equality
        would throw away the match; prefix agreement keeps it.
        """
        probe_tokens, key_tokens = set(probe.split()), set(key.split())
        if not key_tokens:
            return False
        covered = sum(1 for k in key_tokens if any(_same_word(p, k) for p in probe_tokens))
        return covered / len(key_tokens) >= 0.6

    def _ambiguous(self, token: str, kind: str) -> bool:
        """A single token that many different people share identifies nobody."""
        if kind != "author":
            return False
        return len(self.token_owners.get(token, ())) > 1

    def _build(self, kind: str, key: str, value, score: float,
               span: tuple[int, ...], hard: bool) -> EntityMatch:
        if kind == "author":
            entity_id, display = value
        else:
            entity_id, display = key, value
        return EntityMatch(kind=kind, name=display, entity_id=entity_id,
                           score=score, span=span, hard=hard)


def _same_word(a: str, b: str) -> bool:
    """Equal, or one is a prefix of the other -- which is what a mis-stemmed pair
    looks like ("আহম" vs "আহমেদ")."""
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return len(shorter) >= 3 and longer.startswith(shorter)
