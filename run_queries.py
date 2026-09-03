"""Run every query in `queries.txt` and write each result to its own text file.

    python run_queries.py

Equivalent to running this for every line of the file:

    python cli.py search "<query>" --top-k 100 --force-plan --verbose \
                                  --show-rerank --show-passages

with two differences that matter for a batch of this size:

  * **The engine is loaded once.** `cli.search` builds a `SearchEngine` per invocation,
    which means downloading nothing but loading the cross-encoder and the embedder --
    tens of seconds -- sixteen times over. The first query here pays that cost and the
    rest reuse the instance, because the flags (and therefore the settings) are identical
    across the batch.
  * **Output is captured, not just printed.** `cli.console` is already a
    `Console(record=True)`, so the exact text the CLI renders is exported per query
    rather than reconstructed by this script. What lands in the file is what you would
    have seen on screen.

Each query still writes its own stage-by-stage trace to `artifacts/query_traces/`
(see search/trace.py); the path appears in the output file.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Bengali text through a Windows console is cp1252 by default, which raises rather than
# mangles. Reconfigure before importing `cli`, since that builds its Console on import.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # already utf-8, or not a real stream
        pass

import cli  # noqa: E402  - must follow the reconfigure above
from search.engine import SearchEngine  # noqa: E402

ROOT = Path(__file__).resolve().parent

# "1। ", "১২। ", "3. ", "4) " -- the numbering in queries.txt is a label, not part of the
# question, and leaving it in would put a stray token into every lexical query.
NUMBERING = re.compile(r"^[0-9০-৯]+\s*[।.):\]-]\s*")

# Fixed for the whole batch, and the reason one engine can serve every query.
FLAGS = dict(
    user="",
    top=15,
    rag_fusion=False,
    no_llm=False,
    no_rerank=False,
    plan=False,
    force_plan=True,
    no_plan=False,
    verbose=True,
    show_rerank=True,
    show_passages=True,
    no_trace=False,
)


def read_queries(path: Path) -> list[tuple[str, str]]:
    """(original line, query with its numbering stripped) for every non-empty line."""
    text = path.read_text(encoding="utf-8-sig")  # tolerate a BOM
    out: list[tuple[str, str]] = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        out.append((raw, NUMBERING.sub("", raw).strip()))
    return out


def install_engine_cache() -> None:
    """Make `cli.search` reuse one engine per distinct settings object.

    Deliberately *not* a pre-built engine handed to the CLI: letting the first call build
    it means the settings come from `cli.search`'s own `--flag` handling, so this script
    cannot drift out of step with how the CLI configures a search.
    """
    real_load = SearchEngine.load
    cache: dict[tuple[str, bool], SearchEngine] = {}

    class CachedEngine:
        @staticmethod
        def load(settings, *, use_llm: bool = True) -> SearchEngine:
            key = (settings.model_dump_json(), use_llm)
            if key not in cache:
                print("[runner] loading the engine (first query only)...", file=sys.stderr)
                cache[key] = real_load(settings, use_llm=use_llm)
            return cache[key]

    cli.SearchEngine = CachedEngine  # cli only ever calls SearchEngine.load


def run_one(query: str, top_k: int) -> tuple[str, float, bool]:
    """Run one query through the real CLI command. Returns (rendered text, seconds, ok)."""
    cli.console.export_text(clear=True)  # drop anything buffered by a previous query
    started = time.perf_counter()
    ok = True
    try:
        cli.search(query=query, top_k=top_k, **FLAGS)
    except Exception:  # noqa: BLE001 - one bad query must not end the batch
        ok = False
        cli.console.print("[red]this query failed[/]")
        cli.console.print(traceback.format_exc())
    return cli.console.export_text(clear=True), time.perf_counter() - started, ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=Path, default=ROOT / "queries.txt")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--width", type=int, default=160,
                        help="render width; tables are wide with --show-rerank")
    parser.add_argument("--start", type=int, default=1,
                        help="resume from this query number (1-based)")
    args = parser.parse_args()

    if not args.queries.exists():
        print(f"no such file: {args.queries}", file=sys.stderr)
        return 1

    queries = read_queries(args.queries)
    if not queries:
        print(f"{args.queries} has no queries in it", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cli.console.width = args.width
    install_engine_cache()

    print(f"[runner] {len(queries)} queries -> {args.out_dir}", file=sys.stderr)
    print(f"[runner] flags: --top-k {args.top_k} --force-plan --verbose "
          f"--show-rerank --show-passages", file=sys.stderr)

    index: list[str] = []
    failures = 0
    for number, (raw, query) in enumerate(queries, start=1):
        if number < args.start:
            continue
        print(f"[runner] {number}/{len(queries)}  {query}", file=sys.stderr)
        try:
            body, seconds, ok = run_one(query, args.top_k)
        except KeyboardInterrupt:
            print("\n[runner] interrupted -- files written so far are complete",
                  file=sys.stderr)
            break

        failures += not ok
        destination = args.out_dir / f"output_{number}.txt"
        header = [
            "=" * args.width,
            f"QUERY {number} of {len(queries)}",
            "=" * args.width,
            f"line in queries.txt : {raw}",
            f"query as searched   : {query}",
            f"flags               : --top-k {args.top_k} --force-plan --verbose "
            f"--show-rerank --show-passages",
            f"run at              : {datetime.now().isoformat(timespec='seconds')}",
            f"elapsed             : {seconds:.1f}s",
            f"status              : {'ok' if ok else 'FAILED'}",
            "=" * args.width,
            "",
        ]
        destination.write_text("\n".join(header) + body, encoding="utf-8")
        index.append(f"output_{number}.txt  [{'ok' if ok else 'FAILED'}] "
                     f"{seconds:6.1f}s  {query}")
        print(f"[runner]   -> {destination.name}  ({seconds:.1f}s)", file=sys.stderr)

    (args.out_dir / "index.txt").write_text(
        f"generated {datetime.now().isoformat(timespec='seconds')}\n"
        f"source: {args.queries}\n\n" + "\n".join(index) + "\n",
        encoding="utf-8",
    )
    print(f"[runner] done — {len(index)} files, {failures} failed", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
