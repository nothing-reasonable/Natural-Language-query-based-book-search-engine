"""RAG-Fusion: retrieve for several reformulations of one query, then fuse the rankings.

A single query is one point in embedding space, and a book that says the same thing in
different words can sit far from it. Query expansion at the *term* level (taxonomy.py)
widens the lexical channel but leaves the dense channel with exactly one vector to search
from. RAG-Fusion attacks that directly: ask the model for a handful of ways to say the
same question, retrieve for each independently, and let books that surface for several
phrasings rise.

    query
      -> N reformulations               (LLM, or rule-based when no LLM is available)
      -> retrieve each independently    (all four channels, per variant)
      -> per-variant rank fusion        (fusion.fuse -- the existing weighted RRF)
      -> cross-variant fusion           (RRF again: sum of 1/(k + rank) over variants)
      -> top N

Why RRF twice rather than averaging scores: the variants' score distributions are not
comparable -- one phrasing may hit the lexical index hard and another only the dense
channel -- and RRF only needs the *order* each variant produced. That is the same reason
`fusion.py` uses it across channels, applied one level up.

The original query is fused in alongside its variants and weighted above them
(`rag_fusion_original_weight`). It is the only phrasing the user actually chose; the
others are guesses about what they meant, and a guess should not outvote the question.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from search.core import bengali
from search.retrieval import fusion
from config import Settings, settings as default_settings
from search.retrieval.fusion import Fused
from search.llm import LMStudio
from search.query.query_understanding import QueryUnderstanding
from search.retrieval.retrieve import Retriever

log = logging.getLogger(__name__)

VARIANT_SYSTEM = """তুমি একটি বাংলা বই-অনুসন্ধান ইঞ্জিনের কোয়েরি রিরাইটার।

ব্যবহারকারীর প্রশ্নটিকে ভিন্নভাবে লেখা কয়েকটি রূপ তৈরি করো, যাতে একই জিনিস খুঁজলেও
আলাদা শব্দ ব্যবহার করা হয়।

কঠোর নিয়ম:
১। প্রতিটি রূপ অবশ্যই মূল প্রশ্নের একই বিষয় সম্পর্কে হতে হবে। বিষয় বদলানো যাবে না।
২। মূল প্রশ্নের মূল শব্দগুলো (বিষয়, ব্যক্তি, স্থান, কাল) ধরে রাখতে হবে — শুধু বাক্যগঠন,
   সমার্থক শব্দ বা দৃষ্টিভঙ্গি বদলাবে।
৩। প্রশ্নটি যত সুনির্দিষ্ট, রূপগুলোও তত সুনির্দিষ্ট হবে। বেশি সাধারণ করে ফেলা যাবে না —
   "মুক্তিযুদ্ধের বই" বা "ইতিহাসের বই" এর মতো ঢালাও রূপ লিখবে না।
৪। প্রতিটি রূপ বাংলায়, একটি ছোট বাক্য বা পদগুচ্ছ।
৫। কোনো ব্যাখ্যা নয়, শুধু JSON।

উদাহরণ — প্রশ্ন: "জেলা পর্যায়ের মুক্তিযুদ্ধের দলিল"
{"variants": ["জেলাভিত্তিক মুক্তিযুদ্ধের ইতিহাস ও নথিপত্র",
              "স্থানীয় পর্যায়ে একাত্তরের ঘটনার প্রামাণ্য দলিল",
              "বিভিন্ন জেলার মুক্তিযুদ্ধ সংক্রান্ত দলিলপত্র"]}"""


class _Variants(BaseModel):
    variants: list[str] = Field(default_factory=list)


def generate_variants(llm: LMStudio | None, query: str, count: int) -> list[str]:
    """`count` reformulations of `query`, never including the original.

    Returns [] rather than raising: RAG-Fusion degrades to an ordinary single-query
    search when the model is unavailable or answers with nothing usable.
    """
    if llm is None or count <= 0:
        return []
    try:
        drafted = llm.structured(VARIANT_SYSTEM, f"প্রশ্ন: {query}", _Variants, max_tokens=400)
    except Exception as exc:  # noqa: BLE001 - a rewriter is an optimisation, not a dependency
        log.warning("query variant generation failed (%s) -- searching the original only", exc)
        return []
    return _usable(drafted.variants, query, count)


def _usable(drafted: list[str], query: str, count: int) -> list[str]:
    """Keep variants that are new, non-empty, and still about the original question.

    The last check is the one that matters. A small model asked for "different phrasings"
    will happily return the generic parent topic -- "মুক্তিযুদ্ধের বই" for a question about
    district-level war documents -- and a variant broader than the query actively hurts:
    it retrieves the whole subject and its ranks then vote against the specific books the
    user asked for. A variant must keep at least one content word of the original.
    """
    anchors = {t for t in bengali.analyze(query) if len(t) > 1}
    seen = {bengali.key(query)}
    kept: list[str] = []
    for variant in drafted:
        variant = (variant or "").strip()
        key = bengali.key(variant)
        if not key or key in seen:
            continue
        if anchors and not (set(bengali.analyze(variant)) & anchors):
            log.debug("dropped off-topic variant: %s", variant)
            continue
        seen.add(key)
        kept.append(variant)
        if len(kept) >= count:
            break
    return kept


def search(query: str, understanding: QueryUnderstanding, retriever: Retriever,
           llm: LMStudio | None, settings: Settings = default_settings,
           top_n: int | None = None) -> tuple[list[Fused], list[str], dict]:
    """Multi-query retrieval + cross-variant RRF.

    Returns (fused candidates, the queries actually used, per-channel hits) so the caller
    can rerank, explain, and report exactly which phrasings were searched.
    """
    top_n = top_n or settings.rag_fusion_top_n
    variants = generate_variants(llm, query, settings.rag_fusion_variants)
    queries = [query, *variants]

    per_query: list[tuple[float, list[Fused]]] = []
    channel_hits: dict[str, list[str]] = {}
    for index, text in enumerate(queries):
        try:
            plan = understanding.analyze(text)
            channels = retriever.retrieve(plan)
        except Exception as exc:  # noqa: BLE001 - one bad variant must not sink the search
            log.warning("retrieval failed for variant %r: %s", text, exc)
            continue
        for name, candidates in channels.items():
            channel_hits.setdefault(name, [])
            channel_hits[name].extend(c.book_id for c in candidates)
        weight = settings.rag_fusion_original_weight if index == 0 else 1.0
        per_query.append((weight, fusion.fuse(channels, settings)))

    if not per_query:
        return [], queries, channel_hits
    return _reciprocal_rank_fusion(per_query, settings)[:top_n], queries, channel_hits


def _reciprocal_rank_fusion(per_query: list[tuple[float, list[Fused]]],
                            settings: Settings) -> list[Fused]:
    """Sum weight/(k + rank) for each book across the variants that returned it.

    Ties break on book_id, for the reason `fusion.fuse` documents: Python randomises
    string hashing per process, so leaving equal scores to dict order moved nDCG@10
    between identical runs by more than most of the changes being measured.
    """
    k = settings.rrf_k
    merged: dict[str, Fused] = {}
    for weight, ranked in per_query:
        for rank, candidate in enumerate(ranked, start=1):
            entry = merged.get(candidate.book_id)
            if entry is None:
                # Copy so the per-variant lists are not mutated underneath the caller.
                entry = candidate.model_copy(deep=True)
                entry.fusion_score = 0.0
                merged[candidate.book_id] = entry
            else:
                entry.channels.extend(candidate.channels)
                entry.evidence.extend(candidate.evidence)
                for name, value in candidate.scores.items():
                    entry.scores[name] = max(entry.scores.get(name, 0.0), value)
                for name, value in candidate.ranks.items():
                    entry.ranks[name] = min(entry.ranks.get(name, value), value)
            entry.fusion_score += weight / (k + rank)

    ranked = sorted(merged.values(), key=lambda f: (-f.fusion_score, f.book_id))
    for entry in ranked:
        entry.channels = sorted(set(entry.channels))
    return _normalize(ranked)


def _normalize(ranked: list[Fused]) -> list[Fused]:
    """Scale fusion scores to 0..1 so the downstream blend sees the same range it does
    for an ordinary search -- `final_scores` weights `fusion` at 25% and would otherwise
    be comparing RRF sums over a different number of queries."""
    top = ranked[0].fusion_score if ranked else 0.0
    if top > 0:
        for entry in ranked:
            entry.fusion_score /= top
    return ranked
