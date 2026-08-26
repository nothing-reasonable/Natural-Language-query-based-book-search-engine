"""End-to-end smoke test: build nothing, load everything, answer real questions.

`pytest tests/` covers the pieces in isolation with fixtures. This runs the assembled
pipeline against the actual indexes and asserts the behaviours that were broken before
the rewrite -- the ones worth failing a build over.

    python test_search.py            # run the checks
    python test_search.py --show     # ... and print the results
"""

from __future__ import annotations

import argparse
import sys
import time

from config import settings
from search.engine import SearchEngine

# Each case: a query, and something that must be true of the answer.
CHECKS = [
    {
        "query": "হুমায়ূন আহমেদ এর মুক্তিযুদ্ধের বই",
        "why": "a named author must constrain the results, not merely influence them",
        "assert": lambda hits: all(h.book.author == "হুমায়ূন আহমেদ" for h in hits),
        "detail": lambda hits: f"{sum(1 for h in hits if h.book.author == 'হুমায়ূন আহমেদ')}/{len(hits)} by the right author",
    },
    {
        "query": "পাকিস্তান আমলে কূটনীতিক ছিলেন এমন লেখকদের মুক্তিযুদ্ধের বই",
        "why": "the two-hop question from the design document must traverse the graph",
        "assert": lambda hits: any("graph" in h.channels for h in hits),
        "detail": lambda hits: f"channels seen: {sorted({c for h in hits for c in h.channels})}",
    },
    {
        "query": "একাত্তরের কিশোরদের গল্প",
        "why": "a purely semantic query must still return children's war books",
        "assert": lambda hits: sum(
            1 for h in hits
            if any(w in h.book.title for w in ("ছোটদের", "কিশোর", "শিশু"))
        ) >= 3,
        "detail": lambda hits: f"{sum(1 for h in hits if any(w in h.book.title for w in ('ছোটদের', 'কিশোর', 'শিশু')))}/10 look like children's books",
    },
    {
        "query": "অন্যপ্রকাশ থেকে প্রকাশিত বই",
        "why": "a publisher name must be recognised and applied",
        "assert": lambda hits: all(h.book.publisher == "অন্যপ্রকাশ" for h in hits),
        "detail": lambda hits: f"{sum(1 for h in hits if h.book.publisher == 'অন্যপ্রকাশ')}/{len(hits)} from the right publisher",
    },
    {
        "query": "মুক্তিযুদ্ধের বই",
        "why": "a bare topical query must not be filtered by a misfired entity match",
        "assert": lambda hits: len(hits) >= 5,
        "detail": lambda hits: f"{len(hits)} results",
    },
    {
        "query": "একাত্তরের কিশোরদের গল্প",
        "why": "every result must carry an explanation traceable to recorded evidence",
        "assert": lambda hits: all(h.explanation.strip() and h.evidence for h in hits),
        "detail": lambda hits: f"{sum(1 for h in hits if h.explanation.strip() and h.evidence)}/{len(hits)} explained",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="print the ranked results too")
    parser.add_argument("query", nargs="*", help="run one ad-hoc query instead")
    args = parser.parse_args()

    print("loading engine ...", flush=True)
    started = time.perf_counter()
    engine = SearchEngine.load(settings, use_llm=False)
    print(f"ready in {time.perf_counter() - started:.1f}s\n")

    if args.query:
        _run(engine, " ".join(args.query), show=True)
        return 0

    failures = 0
    for check in CHECKS:
        hits, elapsed = _run(engine, check["query"], show=args.show)
        ok = bool(hits) and check["assert"](hits)
        failures += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {check['why']}")
        print(f"         {check['detail'](hits)}  ({elapsed * 1000:.0f} ms)\n")

    total = len(CHECKS)
    print(f"{total - failures}/{total} checks passed")
    return 1 if failures else 0


def _run(engine: SearchEngine, query: str, show: bool):
    started = time.perf_counter()
    response = engine.search(query)
    elapsed = time.perf_counter() - started
    print(f"— {query}   [intent={response.plan.intent}]")
    if show:
        for rank, hit in enumerate(response.hits, start=1):
            print(f"    {rank:2}. {hit.book.title[:56]:58} | {hit.book.author[:24]}")
            print(f"        {hit.explanation[:160]}")
    return response.hits, elapsed


if __name__ == "__main__":
    sys.exit(main())
