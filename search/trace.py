"""A written account of one search, for reading afterwards.

The pipeline has eight stages and each one can quietly undo the last: an expansion term
that drowns the question, a hard filter that removes the right book before reranking ever
sees it, a reranker that degraded to fusion order without saying so. From the outside all
of those look the same -- a result list that is slightly wrong -- so the only way to tell
them apart is to see what each stage handed to the next.

That is what this writes: one plain-text file per query, stages in the order they ran,
with the *inputs and outputs* of each rather than a summary of them. It is a debugging
artifact, not a log -- verbose on purpose, and safe to delete.

Nothing here may break a search. Tracing is an observation of the pipeline, not a part
of it, so every failure below degrades to a warning and an untraced query.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from config import Settings, settings as default_settings

log = logging.getLogger(__name__)

WIDTH = 78


# --------------------------------------------------------------------------- tracers

class NullTracer:
    """What a search gets when tracing is off. Every call is a no-op.

    A null object rather than `if tracer is not None` at nine call sites: the engine
    should not have to remember that tracing is optional.
    """

    enabled = False
    path: Path | None = None

    def section(self, title: str, lines: Iterable[str] = ()) -> None:
        pass

    def write(self) -> Path | None:
        return None


class QueryTracer:
    """Collects sections during a search, writes them out at the end.

    Sections are buffered rather than appended to the file as they happen: a crash
    mid-search should not leave a half-written trace that reads like a complete one.
    """

    enabled = True

    def __init__(self, query: str, settings: Settings = default_settings):
        self.query = query
        self.settings = settings
        self.started = datetime.now()
        self._sections: list[str] = []

    @property
    def path(self) -> Path:
        # Timestamp first so the directory sorts chronologically; a hash of the query
        # keeps two searches in the same second apart. The query itself stays out of the
        # filename -- Bengali text in a path is portable in theory and a nuisance in
        # practice -- and is written in full as the first line of the file instead.
        stamp = self.started.strftime("%Y%m%d-%H%M%S")
        digest = hashlib.sha1(self.query.encode("utf-8")).hexdigest()[:6]
        return self.settings.trace_dir / f"{stamp}-{digest}.txt"

    def section(self, title: str, lines: Iterable[str] = ()) -> None:
        number = len(self._sections) + 1
        block = [f"{number}. {title}", "-" * WIDTH]
        block.extend(str(line) for line in lines)
        self._sections.append("\n".join(block))

    def write(self) -> Path | None:
        header = [
            "=" * WIDTH,
            f"SEARCH TRACE  {self.started.isoformat(timespec='seconds')}",
            "=" * WIDTH,
            f"query: {self.query}",
        ]
        body = "\n".join(header) + "\n\n" + "\n\n".join(self._sections) + "\n"
        try:
            path = self.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
            return path
        except OSError as exc:  # noqa: BLE001 - a trace is never worth failing a search for
            log.warning("could not write query trace: %s", exc)
            return None


def make_tracer(query: str, settings: Settings = default_settings, *,
                enabled: bool | None = None):
    """A real tracer or a null one. `enabled` overrides the `trace_queries` setting."""
    on = settings.trace_queries if enabled is None else enabled
    return QueryTracer(query, settings) if on else NullTracer()


# --------------------------------------------------------------------------- layout

def kv(label: str, value) -> str:
    return f"{label:<28} {value}"


def bullets(items: Sequence, empty: str = "(none)") -> list[str]:
    return [f"  - {item}" for item in items] or [f"  {empty}"]


def table(headers: Sequence[str], rows: Sequence[Sequence], empty: str = "(none)") -> list[str]:
    """A fixed-width table. Bengali renders at one column per code point in most editors,
    which is close enough for a debugging artifact -- alignment may drift by a few
    characters on conjuncts, and that costs nothing here."""
    if not rows:
        return [f"  {empty}"]
    cells = [[str(c) for c in row] for row in rows]
    widths = [max(len(str(headers[i])), max(len(r[i]) for r in cells))
              for i in range(len(headers))]
    out = ["  " + "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))]
    out.append("  " + "  ".join("-" * w for w in widths))
    out.extend("  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(row)) for row in cells)
    return out


def _title(records: dict, book_id: str, width: int = 48) -> str:
    record = records.get(book_id)
    if record is None:
        return f"<{book_id}>"
    text = f"{record.book.title} — {record.book.author}"
    return text if len(text) <= width else text[: width - 1] + "…"


# --------------------------------------------------------------------------- composition

def record_search(tracer, *, plan, fusing: bool, variants: list[str],
                  channel_hits: dict[str, list[str]], pre_filter: list, fused: list,
                  shortlist: list, semantic: list[float], ranked: list[tuple],
                  hits: list, timings: dict, records: dict, rerank_trace,
                  personalized: bool, settings: Settings = default_settings) -> None:
    """Write every stage of one search into `tracer`, in the order they ran."""
    if not tracer.enabled:
        return
    try:
        _compose(tracer, plan=plan, fusing=fusing, variants=variants,
                 channel_hits=channel_hits, pre_filter=pre_filter, fused=fused,
                 shortlist=shortlist, semantic=semantic, ranked=ranked, hits=hits,
                 timings=timings, records=records, rerank_trace=rerank_trace,
                 personalized=personalized, settings=settings)
    except Exception as exc:  # noqa: BLE001 - see the module docstring
        log.warning("query trace could not be composed (%s) -- search is unaffected", exc)
        tracer.section("TRACE INCOMPLETE", [f"  composition failed: {exc!r}"])


def _compose(tracer, *, plan, fusing, variants, channel_hits, pre_filter, fused,
             shortlist, semantic, ranked, hits, timings, records, rerank_trace,
             personalized, settings) -> None:

    # ---------------------------------------------------------------- 1. understanding
    consulted = timings.get("understand", 0) > 1000
    tracer.section("QUERY UNDERSTANDING", [
        kv("raw query", plan.raw_query),
        kv("normalized query", plan.normalized_query),
        kv("intent", plan.intent),
        kv("llm_query_understanding", f"{settings.llm_query_understanding}"
                                      f" ({'LLM consulted' if consulted else 'rules only'})"),
        "",
        "keywords -- the content words lifted from the query:",
        *bullets(plan.keywords),
        "",
        "entities -- names resolved against the catalogue ('hard' must match):",
        *bullets([f"{e.name}  [{e.kind}{', hard' if e.hard else ''}]" for e in plan.entities]),
        "",
        "concepts -- soft taxonomy signals, used for scoring not filtering:",
        *bullets(plan.concepts.all_terms()),
        "",
        "filters -- hard constraints; a book failing these is removed entirely:",
        *bullets([f"{k} = {v}" for k, v in
                  plan.filters.model_dump(exclude_defaults=True).items()]),
        "",
        "graph steps -- traversals over the author/subject knowledge graph:",
        *bullets([s.model_dump(exclude_defaults=True) for s in plan.steps]),
    ])

    # ---------------------------------------------------------------- 2. expansion
    tracer.section("TERM EXPANSION", [
        "Extra search terms added from data/taxonomy.yaml, so that a book indexed under",
        "a synonym of what the user typed can still be found. Each keyword contributes at",
        f"most {settings.expansion_per_keyword} terms (expansion_per_keyword), and only terms that actually occur",
        "in the lexical vocabulary survive -- a term no book contains cannot match one.",
        "",
        kv("keywords in", ", ".join(plan.keywords) or "(none)"),
        kv("expanded terms out", len(plan.expanded_terms)),
        *bullets(plan.expanded_terms),
        "",
        "Used by: the lexical (BM25) query, and the dense query text, which is built from",
        "the normalized query plus the first 8 expansion terms.",
    ])

    # ---------------------------------------------------------------- 3. rag-fusion
    if fusing:
        tracer.section("RAG-FUSION QUERY VARIANTS", [
            "The query was rewritten several times and each rewrite retrieved separately;",
            "the rankings are then fused. Filters come from the original query only.",
            "",
            *bullets(variants),
        ])

    # ---------------------------------------------------------------- 4. retrieval
    channel_lines: list[str] = [
        "Each channel retrieves independently. Overlap between them is the point: a book",
        "found by several channels outranks one found by a single channel.",
        "",
        kv("enabled channels", ", ".join(settings.enabled_channels)),
        kv("channel weights", settings.channel_weights),
        "",
    ]
    for name, ids in channel_hits.items():
        channel_lines.append(f"{name} — {len(ids)} candidates")
        channel_lines.extend(f"    {i:>2}. {_title(records, bid)}"
                             for i, bid in enumerate(ids[:10], start=1))
        if len(ids) > 10:
            channel_lines.append(f"       … {len(ids) - 10} more")
        channel_lines.append("")
    tracer.section("RETRIEVAL CHANNELS", channel_lines)

    # ---------------------------------------------------------------- 5. fusion
    tracer.section("RANK FUSION", [
        f"Reciprocal-rank fusion (rrf_k={settings.rrf_k}) over the channels above.",
        "",
        *table(["#", "rrf", "channels", "book"],
               [[i, f"{f.fusion_score:.4f}", ",".join(f.channels), _title(records, f.book_id)]
                for i, f in enumerate(pre_filter[:25], start=1)]),
    ])

    # ---------------------------------------------------------------- 6. filters
    removed = {f.book_id for f in pre_filter} - {f.book_id for f in fused}
    tracer.section("HARD FILTERS", [
        "Applied after fusion. This is where a correct book most often disappears: if it",
        "is listed below, no later stage can bring it back.",
        "",
        kv("filters", plan.filters.model_dump(exclude_defaults=True) or "(none)"),
        kv("candidates before", len(pre_filter)),
        kv("candidates after", len(fused)),
        kv("removed", len(removed)),
        "",
        *bullets([_title(records, bid) for bid in list(removed)[:20]], empty="(nothing removed)"),
    ])

    # ---------------------------------------------------------------- 7. rerank
    if rerank_trace is None or not rerank_trace.entries:
        tracer.section("RERANKING", ["  no candidates reached stage 2"])
    else:
        warning = ([
            "*** The reranker is NOT running. `noop` returns fusion order as a linear",
            "*** ramp, so the scores below carry no information. Look for a",
            "*** 'cross-encoder unavailable' warning at startup.",
            "",
        ] if rerank_trace.backend == "noop" else [])
        tracer.section("RERANKING", [
            *warning,
            "The cross-encoder reads the query and each book together and scores the pair.",
            f"Its score is {settings.score_weights.get('semantic', 0):.0%} of the final ranking.",
            "",
            kv("backend", rerank_trace.backend),
            kv("model", rerank_trace.model or "—"),
            kv("query as fed to reranker", rerank_trace.query),
            kv("candidates scored", f"{len(rerank_trace.entries)} (rerank_top_k={settings.rerank_top_k})"),
            "",
            *table(["in", "score", "out", "book"],
                   [[e.fusion_rank, f"{e.score:.4f}", e.final_rank or "-",
                     _title(records, e.book_id)]
                    for e in sorted(rerank_trace.entries, key=lambda e: -e.score)]),
            "",
            "--- what the reranker actually read ---",
            "",
            *_passage_blocks(rerank_trace.entries, settings.trace_passage_chars),
        ])

    # ---------------------------------------------------------------- 8. blend
    tracer.section("FINAL SCORE BLEND", [
        "Every signal, weighted. The reranker score is one term among five.",
        "",
        kv("weights", settings.score_weights),
        kv("unavailable_penalty", settings.unavailable_penalty),
        "",
        *table(["#", "score"] + list(settings.score_weights) + ["book"],
               [[i, f"{score:.4f}"]
                + [f"{components.get(name, 0.0):.3f}" for name in settings.score_weights]
                + [_title(records, candidate.book_id)]
                for i, (candidate, score, components) in enumerate(ranked[:20], start=1)]),
    ])

    # ---------------------------------------------------------------- 9. personalisation
    tracer.section("PERSONALISATION", [
        kv("profile applied", "yes" if personalized else "no"),
        kv("personalization_strength", settings.personalization_strength),
        "",
        "`relevance` is the score before personalisation, `score` after. A gap between the",
        "two columns is the profile moving a book; identical columns mean it changed nothing.",
        "",
        *table(["#", "relevance", "score", "book"],
               [[i, f"{h.relevance:.4f}", f"{h.score:.4f}", _title(records, h.book.book_id)]
                for i, h in enumerate(hits, start=1)]),
    ])

    # ---------------------------------------------------------------- 10. results
    tracer.section("RESULTS RETURNED", [
        kv("hits", len(hits)),
        "",
        *[line for i, h in enumerate(hits, start=1)
          for line in (f"{i}. {h.book.title} — {h.book.author}",
                       f"   channels: {', '.join(h.channels) or '-'}",
                       f"   {h.explanation}",
                       "")],
    ])

    # ---------------------------------------------------------------- 11. timings
    total = sum(timings.values())
    tracer.section("TIMINGS", [
        *table(["stage", "ms", "share"],
               [[name, f"{ms:.0f}", f"{(ms / total if total else 0):.0%}"]
                for name, ms in timings.items()]),
        "",
        kv("total (ms)", f"{total:.0f}"),
    ])


def _passage_blocks(entries, limit: int) -> list[str]:
    out: list[str] = []
    for entry in sorted(entries, key=lambda e: -e.score):
        passage = entry.passage if limit <= 0 else entry.passage[:limit]
        truncated = limit > 0 and len(entry.passage) > limit
        out.append(f"[in #{entry.fusion_rank}]  score {entry.score:.4f}")
        out.extend(f"    {line}" for line in passage.splitlines())
        if truncated:
            out.append(f"    … (+{len(entry.passage) - limit} chars, "
                       f"raise trace_passage_chars to see all)")
        out.append("")
    return out
