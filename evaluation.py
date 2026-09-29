"""Stable-ID evaluation and legacy-judgment migration for the target PC.

This file is intentionally not part of normal startup. `run` loads models and executes
searches; use it only on the evaluation/GPU machine described in README.md.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from config import settings
from ingest.run import load_indexed
from search.core import bengali
from search.core.schemas import Filters, SearchOptions
from search.core.store import read_jsonl
from search.engine import SearchEngine
from search.indexing.manifest import IndexManifest
from search.indexing.passages import catalogue_fingerprint

ROOT = Path(__file__).resolve().parent


def migrate(outputs: Path, rank_judgments: Path, out: Path) -> int:
    """Convert only unambiguous legacy rank judgments, preserving both source files."""
    from score_outputs import parse_output

    source = yaml.safe_load(rank_judgments.read_text(encoding="utf-8")) or {}
    records = load_indexed(settings)
    labels: dict[str, list[str]] = defaultdict(list)
    for record in records:
        label = f"{record.book.title} — {record.book.author}"
        labels[_label_key(label)].append(record.book_id)

    queries: dict[str, dict] = {}
    unresolved: list[dict] = []
    for number, judgment in sorted(source.items(), key=lambda item: int(item[0])):
        number = int(number)
        path = outputs / f"output_{number}.txt"
        query_id = f"legacy-{number}"
        if not path.exists():
            unresolved.append({"query_id": query_id, "reason": "result file missing"})
            continue
        query, ranked = parse_output(path)
        by_rank = dict(ranked)
        resolved: list[str] = []
        for rank in sorted(set(judgment.get("relevant") or [])):
            label = by_rank.get(rank)
            if label is None:
                unresolved.append({
                    "query_id": query_id, "rank": rank, "reason": "rank absent from result file"
                })
                continue
            matches = labels.get(_label_key(label), [])
            if len(matches) == 1:
                resolved.append(matches[0])
            else:
                unresolved.append({
                    "query_id": query_id, "rank": rank, "label": label,
                    "reason": "catalogue mapping missing" if not matches else "catalogue mapping ambiguous",
                    "candidate_book_ids": matches,
                })
        queries[query_id] = {
            "query": query,
            "relevant_book_ids": list(dict.fromkeys(resolved)),
            "source": str(path.name),
        }

    payload = {
        "version": 1,
        "catalogue_fingerprint": catalogue_fingerprint(records),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "queries": queries,
        "unresolved": unresolved,
    }
    _write_json(out, payload)
    print(f"wrote {out}: {len(queries)} queries, {len(unresolved)} unresolved mappings")
    return 0 if not unresolved else 2


def run(queries_path: Path, out: Path) -> int:
    fixture = json.loads(queries_path.read_text(encoding="utf-8"))
    queries = fixture.get("queries") or []
    if not queries:
        raise ValueError("query fixture is empty")

    engine = SearchEngine.load(settings)
    options = SearchOptions(shortlist_size=60, trace=False, trace_rerank=False)
    # Warm model kernels/caches before collecting latency. This result is discarded.
    first = queries[0]
    engine.search(first["query"], top_k=60, filters=_filters(first.get("filters")), options=options)

    rows = []
    for item in queries:
        started = time.perf_counter()
        response = engine.search(
            item["query"], top_k=60, filters=_filters(item.get("filters")), options=options
        )
        elapsed = (time.perf_counter() - started) * 1000
        rows.append({
            "id": item["id"],
            "category": item.get("category", "uncategorized"),
            "query": item["query"],
            "filters": response.applied_filters.model_dump(exclude_defaults=True),
            "hits": [hit.book.book_id for hit in response.hits],
            "candidates": response.candidates[:60],
            "latency_ms": round(elapsed, 2),
            "rerank_backend": response.rerank_backend,
            "rerank_fallback": response.rerank_fallback,
        })
        print(f"{item['id']}: {elapsed:.0f} ms")

    manifest = IndexManifest.model_validate_json(
        settings.index_manifest_path.read_text(encoding="utf-8")
    )
    _write_json(out, {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "catalogue_fingerprint": manifest.catalogue_fingerprint,
        "index_generation": manifest.generation,
        "warm_queries": True,
        "queries": rows,
    })
    print(f"wrote {out}")
    return 0


def score(results_path: Path, judgments_path: Path, out: Path,
          baseline_path: Path | None = None) -> int:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    judgments = json.loads(judgments_path.read_text(encoding="utf-8"))
    expected = judgments.get("catalogue_fingerprint")
    if expected and results.get("catalogue_fingerprint") != expected:
        raise ValueError("results and judgments use different catalogue fingerprints")

    report = _report(results, judgments)
    payload = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": str(results_path),
        "judgments": str(judgments_path),
        "metrics": report,
        "recall_label": "pooled recall (judgments are incomplete)",
        "ranking_gains_verified": False,
    }
    if baseline_path is not None:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline.get("catalogue_fingerprint") != results.get("catalogue_fingerprint"):
            raise ValueError("baseline and updated results must use the same catalogue")
        baseline_report = _report(baseline, judgments)
        payload["baseline"] = baseline_report
        payload["delta"] = _metric_delta(report, baseline_report)
        payload["ranking_gains_verified"] = True
    _write_json(out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _report(results: dict, judgments: dict) -> dict:
    gold_rows = judgments.get("queries") or {}
    groups: dict[str, list[dict]] = defaultdict(list)
    judged = 0
    for row in results.get("queries") or []:
        gold = _gold(gold_rows.get(row["id"]))
        if gold:
            judged += 1
        measured = {
            "ndcg_at_10": _ndcg(row.get("hits", []), gold, 10) if gold else None,
            "mrr_at_10": _mrr(row.get("hits", []), gold, 10) if gold else None,
            "candidate_recall_at_60": _recall(row.get("candidates", []), gold, 60) if gold else None,
            "latency_ms": float(row.get("latency_ms", 0)),
        }
        groups["overall"].append(measured)
        groups[row.get("category", "uncategorized")].append(measured)

    output = {name: _aggregate(rows) for name, rows in groups.items()}
    total = len(results.get("queries") or [])
    output["judgment_coverage"] = round(judged / total, 6) if total else 0.0
    output["judged_queries"] = judged
    output["total_queries"] = total
    return output


def _aggregate(rows: list[dict]) -> dict:
    def mean(name: str):
        values = [row[name] for row in rows if row[name] is not None]
        return round(statistics.fmean(values), 6) if values else None

    latency = sorted(row["latency_ms"] for row in rows)
    return {
        "queries": len(rows),
        "ndcg_at_10": mean("ndcg_at_10"),
        "mrr_at_10": mean("mrr_at_10"),
        "candidate_pooled_recall_at_60": mean("candidate_recall_at_60"),
        "warm_latency_median_ms": round(statistics.median(latency), 2) if latency else None,
        "warm_latency_p95_ms": round(_percentile(latency, .95), 2) if latency else None,
    }


def _metric_delta(updated: dict, baseline: dict) -> dict:
    output = {}
    for group in set(updated) & set(baseline):
        if not isinstance(updated[group], dict) or not isinstance(baseline[group], dict):
            continue
        output[group] = {}
        for name in set(updated[group]) & set(baseline[group]):
            a, b = updated[group][name], baseline[group][name]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                output[group][name] = round(a - b, 6)
    return output


def _filters(raw: dict | None) -> Filters:
    raw = raw or {}
    aliases = {"author": "authors", "publisher": "publishers",
               "subject": "subjects", "genre": "genres"}
    data = {}
    for key, value in raw.items():
        target = aliases.get(key, key)
        data[target] = [value] if target in aliases.values() and isinstance(value, str) else value
    return Filters(**data)


def _gold(value) -> set[str]:
    if isinstance(value, list):
        return set(value)
    if isinstance(value, dict):
        return set(value.get("relevant_book_ids") or [])
    return set()


def _ndcg(ranked: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank, book_id in enumerate(ranked[:k], 1)
              if book_id in gold)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(gold)) + 1))
    return dcg / ideal if ideal else 0.0


def _mrr(ranked: list[str], gold: set[str], k: int) -> float:
    return next((1.0 / rank for rank, book_id in enumerate(ranked[:k], 1)
                 if book_id in gold), 0.0)


def _recall(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = (len(values) - 1) * quantile
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (index - low)


def _label_key(value: str) -> str:
    return bengali.normalize(value).casefold()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    migration = commands.add_parser("migrate")
    migration.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    migration.add_argument("--rank-judgments", type=Path, default=ROOT / "judgments.yaml")
    migration.add_argument("--out", type=Path, default=ROOT / "eval" / "judgments_book_ids.json")
    runner = commands.add_parser("run")
    runner.add_argument("--queries", type=Path, default=ROOT / "eval" / "queries.json")
    runner.add_argument("--out", type=Path, default=ROOT / "eval" / "updated-results.json")
    scorer = commands.add_parser("score")
    scorer.add_argument("--results", type=Path, required=True)
    scorer.add_argument("--judgments", type=Path, default=ROOT / "eval" / "judgments_book_ids.json")
    scorer.add_argument("--baseline", type=Path)
    scorer.add_argument("--out", type=Path, default=ROOT / "eval" / "report.json")
    args = parser.parse_args()
    if args.command == "migrate":
        return migrate(args.outputs, args.rank_judgments, args.out)
    if args.command == "run":
        return run(args.queries, args.out)
    return score(args.results, args.judgments, args.out, args.baseline)


if __name__ == "__main__":
    raise SystemExit(main())
