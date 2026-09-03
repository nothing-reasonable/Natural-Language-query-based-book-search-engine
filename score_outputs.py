"""Score the search results in `outputs/` against the judgments in `judgments.yaml`.

    python score_outputs.py

The metric is **precision up to the last relevant hit**. For a ranked list judged

    correct correct correct correct  wrong wrong wrong  correct  wrong wrong ...

the last correct book sits at rank 8, so eight books are *considered* and the score is
5/8. Everything after the last correct hit is ignored: the list is only held responsible
for the stretch it was still producing good answers in.

That definition has one consequence worth stating plainly, because it shows up in this
very run. The denominator is chosen by where the last relevant book happens to sit, so a
query that returns two relevant books at ranks 1-2 and nothing else in the remaining
ninety-eight scores a perfect 1.0, while a query that finds fifty relevant books but
whose last one sits at rank 100 scores 0.5. The metric measures *how clean the list is
before it gives up*, not how much of the catalogue it found. Read it next to `returned`
and `considered`, never on its own -- and see `--depth` to cap it at a fixed cutoff,
which turns it into ordinary precision@k and makes queries comparable to each other.

Two files come out: `scores.csv` for analysis, and `scores.html`, where clicking a
correct or incorrect count opens the list of books that count was counting.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent

# Result panels as the CLI renders them: "╭─ 12. TITLE — AUTHOR ─────╮"
PANEL = re.compile(r"^╭─+\s*(\d+)\.\s+(.*?)\s*─+╮\s*$")
HEADER_QUERY = re.compile(r"^query as searched\s*:\s*(.*)$")


def parse_output(path: Path) -> tuple[str, list[tuple[int, str]]]:
    """(query, [(rank, "title — author"), ...]) from one output_N.txt."""
    lines = path.read_text(encoding="utf-8").splitlines()
    query = next((HEADER_QUERY.match(l).group(1).strip()
                  for l in lines[:20] if HEADER_QUERY.match(l)), "")
    results: list[tuple[int, str]] = []
    for line in lines:
        match = PANEL.match(line)
        if match:
            results.append((int(match.group(1)), match.group(2)))
    results.sort(key=lambda r: r[0])
    return query, results


def score_one(results: list[tuple[int, str]], relevant: set[int], depth: int = 0) -> dict:
    """Precision up to the last relevant hit (or up to `depth`, when given)."""
    ranks = {rank for rank, _ in results}
    unknown = sorted(relevant - ranks)  # judged a rank the file does not have
    relevant = relevant & ranks
    if depth:
        relevant = {r for r in relevant if r <= depth}

    if not relevant:
        # No relevant hit at all: nothing to "consider up to", so the score is 0 over
        # the whole list rather than an undefined 0/0.
        considered = depth or len(results)
        return {"returned": len(results), "considered": considered, "correct": 0,
                "incorrect": considered, "score": 0.0, "unknown_ranks": unknown}

    considered = depth or max(relevant)
    correct = len(relevant)
    return {"returned": len(results), "considered": considered, "correct": correct,
            "incorrect": considered - correct, "score": correct / considered,
            "unknown_ranks": unknown}


# --------------------------------------------------------------------------- report

CSS = """
  :root {
    --bg:#fbfbfa; --fg:#1b1b1a; --dim:#71716c; --line:#e3e3df;
    --ok:#1a7f4b; --ok-bg:#e8f5ee; --bad:#b03030; --bad-bg:#fbecec; --card:#fff;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg:#17181a; --fg:#e8e8e4; --dim:#95958e; --line:#2e3033;
      --ok:#5fce93; --ok-bg:#16301f; --bad:#f08a8a; --bad-bg:#331a1a; --card:#1e2022;
    }
  }
  * { box-sizing:border-box; }
  body {
    margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
    font:15px/1.55 "Segoe UI","Nirmala UI","Noto Sans Bengali",system-ui,sans-serif;
  }
  main { max-width:1060px; margin:0 auto; }
  h1 { font-size:1.35rem; margin:0 0 .35rem; letter-spacing:-.01em; }
  .sub { color:var(--dim); margin:0 0 1.4rem; font-size:.9rem; }
  .totals { display:flex; gap:.7rem; flex-wrap:wrap; margin:0 0 1.4rem; }
  .totals div { background:var(--card); border:1px solid var(--line);
                border-radius:8px; padding:.55rem .9rem; min-width:148px; }
  .totals b { display:block; font-size:1.3rem; font-variant-numeric:tabular-nums; }
  .totals span { color:var(--dim); font-size:.72rem; text-transform:uppercase;
                 letter-spacing:.07em; }
  .hint { color:var(--dim); font-size:.85rem; margin:0 0 .7rem; }
  table { width:100%; border-collapse:collapse; background:var(--card);
          border:1px solid var(--line); border-radius:8px; overflow:hidden; }
  th { text-align:right; font-size:.7rem; text-transform:uppercase; color:var(--dim);
       letter-spacing:.07em; padding:.6rem .7rem; border-bottom:1px solid var(--line); }
  th.query { text-align:left; }
  td { padding:.45rem .7rem; border-bottom:1px solid var(--line); }
  .num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
  .dim { color:var(--dim); }
  .query { width:100%; }
  .score { position:relative; font-weight:600; padding-bottom:.62rem; }
  .bar { position:absolute; left:.5rem; right:.5rem; bottom:.3rem; height:3px;
         border-radius:2px;
         background:linear-gradient(to right, var(--ok) var(--pct),
                                    var(--line) var(--pct)); }
  .count { font:inherit; font-variant-numeric:tabular-nums; cursor:pointer;
           border:1px solid transparent; border-radius:5px; padding:.1rem .5rem;
           min-width:2.7rem; background:transparent; text-decoration:underline;
           text-underline-offset:3px; text-decoration-style:dotted; }
  .count.ok { color:var(--ok); }
  .count.bad { color:var(--bad); }
  .count:hover { text-decoration:none; }
  .count.ok:hover, .count.ok[aria-expanded="true"] {
    background:var(--ok-bg); border-color:var(--ok); text-decoration:none; }
  .count.bad:hover, .count.bad[aria-expanded="true"] {
    background:var(--bad-bg); border-color:var(--bad); text-decoration:none; }
  tr.detail > td { background:var(--bg); padding:1rem 1.25rem 1.2rem; }
  tr.detail h3 { margin:0 0 .55rem; font-size:.88rem; font-weight:600; }
  h3.ok { color:var(--ok); }
  h3.bad { color:var(--bad); }
  ol.books { margin:0; padding:0; list-style:none; columns:2; column-gap:2rem; }
  @media (max-width:720px) { ol.books { columns:1; } }
  ol.books li { break-inside:avoid; padding:.15rem 0; font-size:.88rem; }
  .rank { display:inline-block; min-width:2.8rem; color:var(--dim);
          font-variant-numeric:tabular-nums; font-size:.8rem; }
  .empty { color:var(--dim); margin:0; }
  footer { color:var(--dim); font-size:.82rem; margin-top:1.4rem; }
"""

# One panel open at a time: with sixteen queries the table otherwise scrolls away from
# whatever you were comparing against. Buttons rather than `<a href="#...">` so that
# clicking does not push a fragment onto the history stack -- a reader comparing
# queries clicks a lot, and every click would become a back-button step.
SCRIPT = """
  document.querySelectorAll('.count').forEach(function (button) {
    button.setAttribute('aria-expanded', 'false');
    button.addEventListener('click', function () {
      var panel = document.getElementById(button.dataset.target);
      var opening = panel.hidden;
      document.querySelectorAll('tr.detail').forEach(function (p) { p.hidden = true; });
      document.querySelectorAll('.count').forEach(function (b) {
        b.setAttribute('aria-expanded', 'false');
      });
      panel.hidden = !opening;
      button.setAttribute('aria-expanded', String(opening));
    });
  });
"""


def _book_list(items: list[tuple[int, str]], kind: str) -> str:
    if not items:
        return '<p class="empty">none</p>'
    entries = "".join(
        f'<li><span class="rank">#{rank}</span>{html.escape(book)}</li>'
        for rank, book in items
    )
    return f'<ol class="books {kind}">{entries}</ol>'


def write_html(rows: list[dict], details: dict, path: Path, depth: int) -> None:
    """A report where the correct/incorrect counts open the books behind them.

    Self-contained -- no CDN, no build step -- so it opens from the filesystem by
    double-click and keeps working when the folder is copied or emailed.
    """
    mean = sum(r["score"] for r in rows) / len(rows)
    pooled_correct = sum(r["correct"] for r in rows)
    pooled_considered = sum(r["considered"] for r in rows)
    cutoff = f"precision@{depth}" if depth else "precision up to the last correct hit"

    body = []
    for row in rows:
        number = row["no"]
        query = html.escape(str(row["query"]))
        # A wrong-heavy list should read as wrong at a glance, before any digit is.
        bar = round(row["score"] * 100)
        body.append(
            f'<tr class="row">'
            f'<td class="num dim">{number}</td>'
            f'<td class="query">{query}</td>'
            f'<td class="num dim">{row["returned"]}</td>'
            f'<td class="num">{row["considered"]}</td>'
            f'<td class="num"><button class="count ok" data-target="d{number}-ok">'
            f'{row["correct"]}</button></td>'
            f'<td class="num"><button class="count bad" data-target="d{number}-bad">'
            f'{row["incorrect"]}</button></td>'
            f'<td class="num score"><span class="bar" style="--pct:{bar}%"></span>'
            f'{row["score"]:.3f}</td>'
            f'</tr>'
            f'<tr class="detail" id="d{number}-ok" hidden><td colspan="7">'
            f'<h3 class="ok">{row["correct"]} correct — {query}</h3>'
            f'{_book_list(details[number]["correct"], "ok")}</td></tr>'
            f'<tr class="detail" id="d{number}-bad" hidden><td colspan="7">'
            f'<h3 class="bad">{row["incorrect"]} incorrect — {query}</h3>'
            f'{_book_list(details[number]["incorrect"], "bad")}</td></tr>'
        )

    page = f"""<!doctype html>
<html lang="bn">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Search quality — {len(rows)} queries</title>
<style>{CSS}</style>
</head>
<body>
<main>
  <h1>Search quality — {len(rows)} queries</h1>
  <p class="sub">{cutoff}. Judgments from <code>judgments.yaml</code>.</p>

  <div class="totals">
    <div><span>mean score</span><b>{mean:.3f}</b></div>
    <div><span>pooled</span><b>{pooled_correct / pooled_considered:.3f}</b></div>
    <div><span>books judged</span><b>{pooled_considered}</b></div>
  </div>

  <p class="hint">Click a <b>correct</b> or <b>incorrect</b> count to see the books behind it.</p>

  <table>
    <thead><tr>
      <th>no</th><th class="query">query</th><th>returned</th><th>considered</th>
      <th>correct</th><th>incorrect</th><th>score</th>
    </tr></thead>
    <tbody>{"".join(body)}</tbody>
  </table>

  <footer>Generated by score_outputs.py — re-run it after editing judgments.yaml.</footer>
</main>
<script>{SCRIPT}</script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")


# --------------------------------------------------------------------------- driver

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--judgments", type=Path, default=ROOT / "judgments.yaml")
    parser.add_argument("--csv", type=Path, default=None,
                        help="where to write the summary (default: <outputs>/scores.csv)")
    parser.add_argument("--html", type=Path, default=None,
                        help="clickable report (default: alongside the csv)")
    parser.add_argument("--no-html", action="store_true", help="write only the csv")
    parser.add_argument("--audit", action="store_true",
                        help="also write per-book correct/incorrect rows, so the "
                             "judgments can be checked and corrected")
    parser.add_argument("--depth", type=int, default=0,
                        help="score precision@DEPTH instead of up to the last correct "
                             "hit; makes queries comparable to one another")
    args = parser.parse_args()

    if not args.judgments.exists():
        print(f"no judgments file: {args.judgments}", file=sys.stderr)
        return 1
    judgments = yaml.safe_load(args.judgments.read_text(encoding="utf-8")) or {}
    summary_path = args.csv or (args.outputs / "scores.csv")

    rows, audit_rows, details = [], [], {}
    for number in sorted(judgments, key=int):
        entry = judgments[number] or {}
        path = args.outputs / f"output_{number}.txt"
        if not path.exists():
            print(f"[warn] missing {path.name}, skipping", file=sys.stderr)
            continue

        query, results = parse_output(path)
        query = query or entry.get("query", "")
        relevant = set(entry.get("relevant") or [])
        stats = score_one(results, relevant, args.depth)

        if stats["unknown_ranks"]:
            print(f"[warn] query {number}: judged ranks not present in the output file: "
                  f"{stats['unknown_ranks']}", file=sys.stderr)

        # The books behind the two counts. Split at the same cut-off the score uses, so
        # what the report shows is exactly what the number counted.
        details[number] = {
            "correct": [(r, b) for r, b in results
                        if r <= stats["considered"] and r in relevant],
            "incorrect": [(r, b) for r, b in results
                          if r <= stats["considered"] and r not in relevant],
        }

        rows.append({
            "no": number,
            "query": query,
            "returned": stats["returned"],
            "considered": stats["considered"],
            "correct": stats["correct"],
            "incorrect": stats["incorrect"],
            "score": round(stats["score"], 3),
        })

        if args.audit:
            for rank, book in results:
                if rank > stats["considered"]:
                    break
                audit_rows.append({"no": number, "query": query, "rank": rank,
                                   "book": book, "correct": int(rank in relevant)})

    if not rows:
        print("nothing scored", file=sys.stderr)
        return 1

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if args.audit:
        audit_path = summary_path.with_name(summary_path.stem + "_audit.csv")
        with audit_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
            writer.writeheader()
            writer.writerows(audit_rows)
        print(f"per-book audit  -> {audit_path}")

    html_path = None
    if not args.no_html:
        html_path = args.html or summary_path.with_suffix(".html")
        write_html(rows, details, html_path, args.depth)

    width = max(len(r["query"]) for r in rows)
    print(f"\n{'no':>3}  {'query':<{width}}  {'ret':>4}  {'cons':>5}  "
          f"{'corr':>5}  {'inc':>5}  {'score':>6}")
    print("-" * (3 + width + 34))
    for row in rows:
        print(f"{row['no']:>3}  {row['query']:<{width}}  {row['returned']:>4}  "
              f"{row['considered']:>5}  {row['correct']:>5}  {row['incorrect']:>5}  "
              f"{row['score']:>6.3f}")

    mean = sum(r["score"] for r in rows) / len(rows)
    pooled_correct = sum(r["correct"] for r in rows)
    pooled_considered = sum(r["considered"] for r in rows)
    print("-" * (3 + width + 34))
    print(f"mean score over {len(rows)} queries : {mean:.3f}")
    print(f"pooled ({pooled_correct}/{pooled_considered})".ljust(30)
          + f": {pooled_correct / pooled_considered:.3f}")
    print(f"\nsummary  -> {summary_path}")
    if html_path:
        print(f"clickable -> {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
