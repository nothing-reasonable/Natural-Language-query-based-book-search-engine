"""Reciprocal Rank Fusion + hard filtering.

RRF merges rankings without needing the channels' scores to be on the same scale, which
is exactly the situation here (BM25 scores, cosine similarities and graph match counts).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from config import Settings, settings as default_settings
from search.core.schemas import Candidate, Evidence, Filters, IndexedBook


class Fused(BaseModel):
    book_id: str
    fusion_score: float
    channels: list[str] = Field(default_factory=list)
    ranks: dict[str, int] = Field(default_factory=dict)
    scores: dict[str, float] = Field(default_factory=dict)  # raw per-channel scores
    evidence: list[Evidence] = Field(default_factory=list)


def fuse(channels: dict[str, list[Candidate]], settings: Settings = default_settings) -> list[Fused]:
    """Merge the channels' rankings.

    Equal scores are ordered by `book_id` rather than left to chance. Candidates arrive
    from sets and dicts keyed by strings, and Python randomises string hashing per
    process, so ties came out in a different order on every run -- moving nDCG@10 by up
    to 0.027 between identical runs of the evaluation set, which is larger than most of
    the changes being measured against it. Every ranking step in the pipeline breaks ties
    the same way for the same reason.
    """
    k = settings.rrf_k
    weights = settings.channel_weights

    merged: dict[str, Fused] = {}
    for channel, candidates in channels.items():
        weight = weights.get(channel, 1.0)
        for candidate in candidates:
            entry = merged.setdefault(candidate.book_id, Fused(book_id=candidate.book_id, fusion_score=0.0))
            entry.fusion_score += weight / (k + candidate.rank)
            entry.channels.append(channel)
            entry.ranks[channel] = candidate.rank
            entry.scores[channel] = candidate.score
            entry.evidence.extend(candidate.evidence)

    ranked = sorted(merged.values(), key=lambda f: (-f.fusion_score, f.book_id))
    return _normalize(ranked)


def apply_filters(fused: list[Fused], filters: Filters,
                  records: dict[str, IndexedBook]) -> list[Fused]:
    """Drop anything that violates a hard constraint.

    The dense channel already pre-filters inside LanceDB; this catches candidates that
    arrived through the lexical or graph channels.
    """
    if filters.is_empty():
        return fused
    return [f for f in fused if f.book_id in records and _matches(records[f.book_id], filters)]


def _matches(record: IndexedBook, f: Filters) -> bool:
    book, enrichment = record.book, record.enrichment
    checks = [
        (f.author_ids, [book.author_id]),
        (f.authors, [book.author]),
        (f.publishers, [book.publisher]),
        (f.genres, enrichment.genres),
        (f.subjects, enrichment.subjects),
        (f.periods, enrichment.periods),
        (f.places, enrichment.places),
    ]
    for wanted, actual in checks:
        if wanted and not set(wanted) & set(actual):
            return False
    if f.language and book.language != f.language:
        return False
    if f.year_from is not None and (book.publish_year or 0) < f.year_from:
        return False
    if f.year_to is not None and (book.publish_year or 9999) > f.year_to:
        return False
    return True


def _normalize(ranked: list[Fused]) -> list[Fused]:
    """Scale fusion scores into 0..1 so they can be blended with the other signals."""
    if not ranked:
        return ranked
    top = ranked[0].fusion_score or 1.0
    for item in ranked:
        item.fusion_score = item.fusion_score / top
    return ranked
