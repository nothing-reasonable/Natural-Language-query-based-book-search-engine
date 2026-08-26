"""Personalised re-scoring, applied *after* relevance has been decided.

Two guardrails, both straight out of the design:

  * the adjustment is a bounded multiplicative nudge (`personalization_strength`), so a
    strongly relevant book can never be pushed out by taste;
  * session intent outweighs long-term history (`session_weight`), so someone who normally
    reads light fiction still gets academic history when that is what they are searching for.
"""

from __future__ import annotations

from collections import Counter

from config import Settings, settings as default_settings
from search.ranking.profile_index import Session, UserProfile
from search.core.schemas import Evidence, IndexedBook

# Which enrichment facets carry taste signal, and how strongly.
FACET_WEIGHTS = {"subjects": 1.0, "genres": 1.0, "topics": 0.6, "periods": 0.4, "places": 0.3}


def build_affinity(profile: UserProfile | None, session: Session | None,
                   records: dict[str, IndexedBook],
                   settings: Settings = default_settings) -> dict[str, float]:
    """concept -> affinity in -1..1, blending explicit prefs, history and this session."""
    long_term = Counter()
    if profile is not None:
        for genre in profile.genres:
            long_term[genre] += 2.0
        for subject in profile.subjects:
            long_term[subject] += 2.0
        for author in profile.authors:
            long_term[f"author::{author}"] += 2.0
        _add_book_signals(long_term, profile.interacted_books(), records)

    short_term = Counter()
    if session is not None:
        _add_book_signals(short_term, session.interacted_books(), records)

    w = settings.session_weight
    combined = Counter()
    for concept, value in long_term.items():
        combined[concept] += (1.0 - w) * value
    for concept, value in short_term.items():
        combined[concept] += w * value

    peak = max((abs(v) for v in combined.values()), default=0.0)
    if peak == 0.0:
        return {}
    return {concept: value / peak for concept, value in combined.items()}


def _add_book_signals(target: Counter, interactions: Counter,
                      records: dict[str, IndexedBook]) -> None:
    for book_id, strength in interactions.items():
        record = records.get(book_id)
        if record is None:
            continue
        for facet, weight in FACET_WEIGHTS.items():
            for value in getattr(record.enrichment, facet, []):
                target[value] += strength * weight
        if record.book.author:
            target[f"author::{record.book.author}"] += strength * 0.8


def score_for(record: IndexedBook, affinity: dict[str, float]) -> tuple[float, list[str]]:
    """Affinity of one book in -1..1, plus the concepts that drove it."""
    if not affinity:
        return 0.0, []
    total, matched = 0.0, []
    for facet, weight in FACET_WEIGHTS.items():
        for value in getattr(record.enrichment, facet, []):
            if value in affinity:
                total += affinity[value] * weight
                matched.append(value)
    author_key = f"author::{record.book.author}"
    if author_key in affinity:
        total += affinity[author_key]
        matched.append(record.book.author)

    denominator = sum(FACET_WEIGHTS.values()) + 1.0
    return max(-1.0, min(1.0, total / denominator)), matched


def apply(ranked: list[tuple], records: dict[str, IndexedBook],
          affinity: dict[str, float], settings: Settings = default_settings) -> list[tuple]:
    """`ranked` is [(candidate, relevance, components)]. Returns the same shape, re-sorted."""
    if not affinity:
        return [(c, r, r, comp, []) for c, r, comp in ranked]

    strength = settings.personalization_strength
    adjusted = []
    for candidate, relevance, components in ranked:
        record = records.get(candidate.book_id)
        boost, matched = score_for(record, affinity) if record else (0.0, [])
        score = relevance * (1.0 + strength * boost)
        components = {**components, "personalization": strength * boost}
        adjusted.append((candidate, score, relevance, components, matched))

    adjusted.sort(key=lambda item: (-item[1], item[0].book_id))
    return adjusted


def evidence_for(matched: list[str]) -> list[Evidence]:
    if not matched:
        return []
    return [
        Evidence(
            channel="profile",
            detail="আপনার পছন্দের সাথে মিল: " + ", ".join(matched[:4]),
            terms=matched[:4],
        )
    ]
