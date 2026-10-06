"""Evidence-based reranker evaluation. Only `run` imports the search/model stack.

prepare -> run (search PC) -> pool -> assess the JSON sheet -> score (offline).
See RERANKER_EVALUATION.md for the protocol, decisions, and annotation instructions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DIR = ROOT / "eval" / "reranker"
NUMBERING = re.compile(r"^[0-9০-৯]+\s*[।.):\]-]\s*")
METADATA_FIELDS = (
    "title", "author", "author_raw", "author_bio", "publisher", "description",
    "publish_year", "table_of_contents", "language", "isbn", "source_url",
)
SYSTEMS = ("fusion", "without_reranker", "reranked")
RUBRIC = {
    "0": "Irrelevant, or documented evidence contradicts an explicit requirement.",
    "1": "Weak/tangential connection; the main intent is not substantially addressed.",
    "2": "Substantial match to the intent, supported by catalogue evidence.",
    "3": "Direct, strong match to the intent and explicit requirements, supported by evidence.",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def digest(payload) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_queries(path: Path) -> dict:
    rows, seen = [], {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        query = NUMBERING.sub("", raw).strip()
        if not query:
            raise ValueError(f"empty numbered query at line {line_number}")
        query_id = f"q{len(rows) + 1:03d}"
        rows.append({"id": query_id, "line": line_number, "raw": raw, "query": query,
                     "duplicate_of": seen.get(query)})
        seen.setdefault(query, query_id)
    if not rows:
        raise ValueError("query file is empty")
    return {"version": 1, "source_name": path.name,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "query_count": len(rows), "unique_query_count": len(seen), "queries": rows}


def prepare(queries_path: Path, out_dir: Path) -> None:
    if any((out_dir / name).exists() for name in ("queries.json", "protocol.json")):
        raise ValueError("prepared files already exist; choose a new --out-dir to preserve decisions")
    manifest = read_queries(queries_path)
    ambiguous = {"জুলাইয়ের আন্দোলন", "পঞ্চপাণ্ডব ক্রিকেটাদের গল্প",
                 "হুমায়ূন আহমেদের আশির দশকে লেখা উপন্যাস"}
    protocol = {
        "version": 1, "query_manifest_digest": digest(manifest),
        "assessment_mode": None,  # assistant_final | assistant_draft
        "duplicate_policy": None,  # all | unique
        "include_bm25": None,
        "k": 10, "candidate_depth": 60, "pool_depth": 20,
        "relevant_threshold": 2, "pool_seed": 55,
        "bootstrap_samples": 10000, "bootstrap_seed": 20261006,
        "query_intents": {r["id"]: None if r["query"] in ambiguous else r["query"]
                          for r in manifest["queries"]},
        "query_guidance": {},
        "categories": {}, "excluded_queries": {}, "rubric": RUBRIC,
        "evidence_policy": "Only exported catalogue metadata. No prior knowledge, generated "
                           "enrichment, ranking scores, explanations, or external facts.",
    }
    write_json(out_dir / "queries.json", manifest)
    write_json(out_dir / "protocol.json", protocol)
    print(f"Prepared {len(manifest['queries'])} rows ({manifest['unique_query_count']} unique).")
    print(f"Resolve null decisions in {out_dir / 'protocol.json'} before pooling/scoring.")


def validate_protocol(manifest: dict, protocol: dict, *, require_decisions: bool = True) -> None:
    if protocol.get("query_manifest_digest") != digest(manifest):
        raise ValueError("protocol belongs to a different query manifest")
    if protocol.get("k") != 10 or protocol.get("relevant_threshold") != 2:
        raise ValueError("this protocol uses k=10 and relevance threshold=2")
    depth, pool = protocol.get("candidate_depth", 0), protocol.get("pool_depth", 0)
    if type(depth) is not int or type(pool) is not int or not (10 <= pool <= depth <= 100):
        raise ValueError("require 10 <= pool_depth <= candidate_depth <= 100")
    if not isinstance(protocol.get("pool_seed"), int):
        raise ValueError("pool_seed must be an integer")
    if not isinstance(protocol.get("bootstrap_samples"), int) or protocol["bootstrap_samples"] < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not isinstance(protocol.get("bootstrap_seed"), int):
        raise ValueError("bootstrap_seed must be an integer")
    ids = {r["id"] for r in manifest["queries"]}
    if protocol.get("pending_query_replacements"):
        raise ValueError("finalize query replacements and update the snapshot before running: "
                         + ", ".join(sorted(protocol["pending_query_replacements"])))
    excluded = protocol.get("excluded_queries", {})
    if set(excluded) - ids or any(not str(v).strip() for v in excluded.values()):
        raise ValueError("excluded_queries must map existing IDs to explicit reasons")
    if set(protocol.get("categories", {})) - ids:
        raise ValueError("unknown query ID in categories")
    if set(protocol.get("query_guidance", {})) - ids:
        raise ValueError("unknown query ID in query_guidance")
    if require_decisions:
        missing = []
        if protocol.get("assessment_mode") not in {"assistant_final", "assistant_draft"}:
            missing.append("assessment_mode")
        if protocol.get("duplicate_policy") not in {"all", "unique"}:
            missing.append("duplicate_policy")
        if type(protocol.get("include_bm25")) is not bool:
            missing.append("include_bm25")
        for query_id in ids - set(excluded):
            if not str(protocol.get("query_intents", {}).get(query_id) or "").strip():
                missing.append(f"query_intents.{query_id}")
        if missing:
            raise ValueError("unresolved decisions: " + ", ".join(sorted(missing)))


def run_searches(manifest_path: Path, protocol_path: Path, queries_path: Path, out: Path) -> int:
    """GPU/search PC only. One retrieval per row; all paired rankings share candidates."""
    manifest, protocol = read_json(manifest_path), read_json(protocol_path)
    validate_protocol(manifest, protocol, require_decisions=False)
    if type(protocol.get("include_bm25")) is not bool:
        raise ValueError("resolve include_bm25 in protocol.json before running searches")
    if read_queries(queries_path) != manifest:
        raise ValueError("queries.txt changed since prepare; prepare a new evaluation directory")
    if out.exists():
        raise ValueError("run output already exists; choose a new --out (do not overwrite evidence)")

    # These imports deliberately remain inside this command. Offline commands need stdlib only.
    from config import settings
    from search.core.schemas import SearchOptions
    from search.engine import SearchEngine
    from search.indexing.manifest import IndexManifest

    engine = SearchEngine.load(settings)
    index = IndexManifest.model_validate_json(settings.index_manifest_path.read_text(encoding="utf-8"))
    depth = protocol["candidate_depth"]
    options = SearchOptions(shortlist_size=depth, trace=False, trace_rerank=False,
                            compare_rerank=True, rerank=True)
    # Warm kernels/caches once. The discarded call is not counted in quality or latency.
    engine.search(manifest["queries"][0]["query"], top_k=depth, options=options)
    llm = getattr(engine, "llm", None)
    loaded_models = (llm.loaded_models() if llm is not None else [])
    loaded_models = [{k: model[k] for k in ("id", "type", "state") if k in model}
                     for model in loaded_models]
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True, timeout=5)
        code_revision = revision.stdout.strip() if revision.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        code_revision = ""
    systems = list(SYSTEMS) + (["bm25"] if protocol["include_bm25"] else [])
    bundle = {
        "version": 1, "created_at": timestamp(), "status": "incomplete",
        "query_manifest": manifest, "catalogue_fingerprint": index.catalogue_fingerprint,
        "index_generation": index.generation, "candidate_depth": depth,
        "systems": systems, "include_bm25": protocol["include_bm25"],
        "warm_queries": True, "personalization": False,
        "settings": settings.model_dump(mode="json", exclude={"lmstudio_api_key", "lmstudio_base_url"}),
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "code_revision": code_revision,
        "dense_channel_available": getattr(engine, "vector", None) is not None,
        "loaded_lmstudio_models": loaded_models,
        "effective_reranker_model": (getattr(engine.reranker, "model_name", "")
                                     or getattr(engine.reranker, "model", "")),
        "index_manifest": index.model_dump(mode="json"), "queries": [], "books": {},
    }
    write_json(out, bundle)
    failures = 0
    for query in manifest["queries"]:
        row = {"id": query["id"], "query": query["query"]}
        started = time.perf_counter()
        try:
            response = engine.search(query["query"], top_k=depth, options=options)
            elapsed = (time.perf_counter() - started) * 1000
            rankings = {
                "fusion": response.candidates[:depth],
                "without_reranker": response.rerank_ablation,
                "reranked": [h.book.book_id for h in response.hits],
            }
            # Independent raw-query BM25 reference, with the same inferred hard filters.
            # It is outside the paired shortlist experiment and timed separately.
            bm25_ms = None
            if protocol["include_bm25"]:
                bm25_started = time.perf_counter()
                allowed = engine.facets.select(response.applied_filters)
                rankings["bm25"] = [b[0] for b in engine.lexical.search(
                    [query["query"]], k=depth, allowed_ids=allowed
                )]
                bm25_ms = (time.perf_counter() - bm25_started) * 1000
            for book_id in {b for ranked in rankings.values() for b in ranked}:
                book = engine.records[book_id].book.model_dump(mode="json")
                bundle["books"][book_id] = {f: book.get(f) for f in METADATA_FIELDS}
            row.update(status="ok", rankings=rankings, latency_ms=round(elapsed, 3),
                       bm25_retrieval_ms=bm25_ms, timings_ms=response.timings_ms,
                       candidates=response.candidates[:depth],
                       filters=response.applied_filters.model_dump(mode="json"),
                       rerank_backend=response.rerank_backend, rerank_fallback=response.rerank_fallback)
        except Exception as exc:
            failures += 1
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        bundle["queries"].append(row)
        write_json(out, bundle)  # completed rows survive interruption
        print(f"{row['id']}: {row['status']}", flush=True)
    bundle["status"] = "failed" if failures else "complete"
    write_json(out, bundle)
    print(f"Exported {len(bundle['queries'])} rows; {failures} failures -> {out}")
    return 1 if failures else 0


def validate_bundle(bundle: dict, protocol: dict) -> None:
    manifest = bundle["query_manifest"]
    validate_protocol(manifest, protocol)
    if bundle.get("status") != "complete":
        raise ValueError("run is incomplete/failed; collect a complete fresh run")
    if not bundle.get("catalogue_fingerprint") or not bundle.get("index_generation"):
        raise ValueError("export must identify its catalogue and index generation")
    if bundle.get("candidate_depth") != protocol["candidate_depth"]:
        raise ValueError("candidate depth changed; collect a new run")
    systems = list(SYSTEMS) + (["bm25"] if protocol["include_bm25"] else [])
    if bundle.get("systems") != systems or bundle.get("include_bm25") != protocol["include_bm25"]:
        raise ValueError("baseline decision changed; collect a new run")
    expected = {r["id"]: r["query"] for r in manifest["queries"]}
    rows = bundle.get("queries", [])
    if len(rows) != len(expected) or {r["id"] for r in rows} != set(expected):
        raise ValueError("run must contain each manifest query exactly once")
    for row in rows:
        if row.get("status") != "ok" or row["query"] != expected[row["id"]]:
            raise ValueError(f"failed or mismatched query: {row['id']}")
        rankings = row.get("rankings", {})
        if set(rankings) != set(systems):
            raise ValueError(f"missing/unexpected system at {row['id']}")
        for name, ranked in rankings.items():
            if len(ranked) != len(set(ranked)) or len(ranked) > protocol["candidate_depth"]:
                raise ValueError(f"duplicate or excess book IDs: {row['id']}/{name}")
            if set(ranked) - set(bundle.get("books", {})):
                raise ValueError(f"missing catalogue metadata: {row['id']}/{name}")
        if any(set(rankings[name]) != set(rankings["fusion"]) for name in SYSTEMS):
            raise ValueError(f"paired rankings have different candidates at {row['id']}")
        if row.get("candidates") != rankings["fusion"]:
            raise ValueError(f"candidate list differs from fusion order at {row['id']}")
        if any(set(book) - set(METADATA_FIELDS) for book in bundle["books"].values()):
            raise ValueError("export contains non-catalogue fields in judging metadata")


def make_pool(bundle: dict, protocol: dict) -> dict:
    validate_bundle(bundle, protocol)
    rng, rows = random.Random(protocol["pool_seed"]), []
    manifest_rows = {r["id"]: r for r in bundle["query_manifest"]["queries"]}
    for row in bundle["queries"]:
        if row["id"] in protocol.get("excluded_queries", {}):
            continue
        ids = sorted({b for ranked in row["rankings"].values() for b in ranked[:protocol["pool_depth"]]})
        rng.shuffle(ids)
        rows.append({
            "id": row["id"], "query": row["query"],
            "intent": protocol["query_intents"][row["id"]],
            "guidance": protocol.get("query_guidance", {}).get(row["id"], ""),
            "duplicate_of": manifest_rows[row["id"]].get("duplicate_of"),
            "books": [{"book_id": b, "metadata": bundle["books"][b],
                       "judgment": {"status": "pending", "grade": None, "reason": "",
                                    "evidence": [], "assessor": "", "assessor_type": "",
                                    "human_reviewed": False, "reviewer": ""}} for b in ids],
        })
    return {"version": 1, "run_digest": digest(bundle), "protocol_digest": digest(protocol),
            "catalogue_fingerprint": bundle["catalogue_fingerprint"],
            "rubric": RUBRIC, "evidence_policy": protocol["evidence_policy"], "queries": rows}


def write_pool(run_path: Path, protocol_path: Path, out: Path) -> None:
    if out.exists():
        raise ValueError("assessment sheet exists; choose a new --out to preserve judgments")
    pool = make_pool(read_json(run_path), read_json(protocol_path))
    write_json(out, pool)
    count = sum(len(r["books"]) for r in pool["queries"])
    print(f"Wrote {count} shuffled query-book pairs -> {out}")


def check_assessment(bundle: dict, protocol: dict, assessment: dict) -> tuple[dict, list[dict]]:
    """Reject stale evidence and invalid grades; leave pending pairs genuinely unjudged."""
    expected = make_pool(bundle, protocol)
    for key in ("run_digest", "protocol_digest", "catalogue_fingerprint", "rubric", "evidence_policy"):
        if assessment.get(key) != expected[key]:
            raise ValueError(f"assessment has stale/altered {key}; rebuild pool after a protocol change")
    reference = {r["id"]: r for r in expected["queries"]}
    actual = assessment.get("queries", [])
    if len(actual) != len(reference) or {r["id"] for r in actual} != set(reference):
        raise ValueError("assessment queries differ from the pool")
    grades, progress, duplicate_labels = {}, [], {}
    for row in actual:
        query_id, ref = row["id"], reference[row["id"]]
        if any(row.get(k) != ref.get(k) for k in ("query", "intent", "guidance", "duplicate_of")):
            raise ValueError(f"query intent/text changed at {query_id}")
        books = row.get("books", [])
        by_id = {b["book_id"]: b for b in ref["books"]}
        if len(books) != len(by_id) or {b["book_id"] for b in books} != set(by_id):
            raise ValueError(f"assessment pool changed at {query_id}")
        labels, counts = {}, Counter()
        for item in books:
            book_id, judgment = item["book_id"], item.get("judgment", {})
            if item.get("metadata") != by_id[book_id]["metadata"]:
                raise ValueError(f"catalogue evidence changed at {query_id}/{book_id}")
            status, grade = judgment.get("status"), judgment.get("grade")
            if status not in {"pending", "graded", "unjudgeable"}:
                raise ValueError(f"invalid judgment status: {query_id}/{book_id}")
            if status != "graded":
                if grade is not None:
                    raise ValueError("pending/unjudgeable pairs must have grade=null")
                if status == "unjudgeable" and not str(judgment.get("reason", "")).strip():
                    raise ValueError("unjudgeable pairs require a reason")
                counts[status] += 1
                continue
            if type(grade) is not int or grade not in range(4):
                raise ValueError(f"grade must be an integer 0-3: {query_id}/{book_id}")
            if not judgment.get("reason", "").strip() or not judgment.get("assessor", "").strip():
                raise ValueError("graded pairs require a reason and named assessor")
            if judgment.get("assessor_type") not in {"human", "assistant"}:
                raise ValueError("assessor_type must be human or assistant")
            evidence = judgment.get("evidence", [])
            if not evidence:
                raise ValueError(f"graded pair needs catalogue evidence: {query_id}/{book_id}")
            for excerpt in evidence:
                field, quote = excerpt.get("field"), excerpt.get("quote")
                source = item["metadata"].get(field)
                if field not in METADATA_FIELDS or not isinstance(quote, str) or not quote.strip():
                    raise ValueError("evidence must specify a catalogue field and a nonempty quote")
                if source is None or quote not in str(source):
                    raise ValueError(f"evidence quote is absent from {query_id}/{book_id}/{field}")
            reviewed = judgment.get("human_reviewed") is True
            if reviewed and not judgment.get("reviewer", "").strip():
                raise ValueError("human_reviewed requires a named reviewer")
            if (protocol["assessment_mode"] == "assistant_draft"
                    and judgment["assessor_type"] == "assistant" and not reviewed):
                counts["awaiting_human_review"] += 1
                continue
            identity = (row["query"], book_id)
            if identity in duplicate_labels and duplicate_labels[identity] != grade:
                raise ValueError("duplicate query/book pairs must have consistent grades")
            duplicate_labels[identity] = grade
            labels[book_id] = grade
            counts["graded"] += 1
        complete = len(labels) == len(books)
        grades[query_id] = labels if complete else None
        progress.append({"id": query_id, "pairs": len(books), "complete": complete, **counts})
    return grades, progress


def ndcg(ranked: list[str], grades: dict[str, int], k: int = 10) -> float:
    actual = sum((2 ** grades.get(b, 0) - 1) / math.log2(i + 1)
                 for i, b in enumerate(ranked[:k], 1))
    ideal = sum((2 ** grade - 1) / math.log2(i + 1)
                for i, grade in enumerate(sorted(grades.values(), reverse=True)[:k], 1))
    return actual / ideal if ideal else 0.0


def query_metrics(ranked: list[str], candidates: list[str], grades: dict[str, int],
                  k: int, depth: int) -> dict:
    relevant = {b for b, grade in grades.items() if grade >= 2}
    return {
        "ndcg_at_10": ndcg(ranked, grades, k),
        "precision_at_10": len(set(ranked[:k]) & relevant) / k,
        "mrr_at_10": next((1 / i for i, b in enumerate(ranked[:k], 1) if b in relevant), 0.0),
        f"candidate_pooled_recall_at_{depth}": (
            len(set(candidates[:depth]) & relevant) / len(relevant) if relevant else None),
    }


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def mean_metrics(rows: list[dict]) -> dict:
    keys = {key for row in rows for key in row}
    return {key: statistics.fmean(values) if values else None
            for key in sorted(keys)
            for values in [[row[key] for row in rows if row.get(key) is not None]]}


def bootstrap(deltas: list[float], samples: int, seed: int) -> dict:
    if not deltas:
        return {"queries": 0, "mean_delta": None, "ci_95": None}
    rng = random.Random(seed)
    estimates = [statistics.fmean(rng.choices(deltas, k=len(deltas))) for _ in range(samples)]
    return {"queries": len(deltas), "mean_delta": statistics.fmean(deltas),
            "ci_95": [percentile(estimates, .025), percentile(estimates, .975)],
            "wins": sum(d > 1e-12 for d in deltas), "ties": sum(abs(d) <= 1e-12 for d in deltas),
            "losses": sum(d < -1e-12 for d in deltas), "resamples": samples, "seed": seed}


def report(bundle: dict, protocol: dict, assessment: dict, *, allow_partial: bool = False) -> dict:
    grades, progress = check_assessment(bundle, protocol, assessment)
    incomplete = [r["id"] for r in progress if not r["complete"]]
    if incomplete and not allow_partial:
        raise ValueError("no final metrics: unfinished/unjudgeable pairs in " + ", ".join(incomplete)
                         + "; use check to inspect progress, or --allow-partial for a provisional report")
    per_query, no_relevant = [], []
    for row in bundle["queries"]:
        labels = grades.get(row["id"])
        if labels is None:  # also excludes explicitly omitted queries
            continue
        if not any(grade >= 2 for grade in labels.values()):
            no_relevant.append(row["id"])
        per_query.append({
            "id": row["id"], "query": row["query"],
            "category": protocol.get("categories", {}).get(row["id"], "uncategorized"),
            "judged_pairs": len(labels), "known_relevant_books": sum(g >= 2 for g in labels.values()),
            "systems": {name: query_metrics(ranked, ranked if name == "bm25" else row["candidates"], labels,
                                            protocol["k"], protocol["candidate_depth"])
                        for name, ranked in row["rankings"].items()},
            "rerank_backend": row.get("rerank_backend", ""),
            "rerank_fallback": row.get("rerank_fallback", ""),
            "latency_ms": row.get("latency_ms"), "timings_ms": row.get("timings_ms", {}),
        })
    units = defaultdict(list)
    # Unique policy gives each distinct query equal weight, averaging its repeated runs.
    for row in per_query:
        units[row["query"] if protocol["duplicate_policy"] == "unique" else row["id"]].append(row)
    unit_metrics = [{name: mean_metrics([r["systems"][name] for r in rows])
                     for name in bundle["systems"]} for rows in units.values()]
    overall = {name: mean_metrics([u[name] for u in unit_metrics]) for name in bundle["systems"]}
    comparisons = {}
    for baseline in ("without_reranker", "fusion"):
        comparisons[f"reranked_vs_{baseline}"] = bootstrap(
            [u["reranked"]["ndcg_at_10"] - u[baseline]["ndcg_at_10"] for u in unit_metrics],
            protocol["bootstrap_samples"], protocol["bootstrap_seed"],
        )
    by_category = {}
    for category in sorted({r["category"] for r in per_query}):
        category_units = defaultdict(list)
        for row in per_query:
            if row["category"] == category:
                category_units[row["query"] if protocol["duplicate_policy"] == "unique" else row["id"]].append(row)
        by_category[category] = {"evaluation_units": len(category_units), "systems": {
            name: mean_metrics([mean_metrics([r["systems"][name] for r in rows])
                                for rows in category_units.values()]) for name in bundle["systems"]}}
    successful = [r for r in bundle["queries"] if r["status"] == "ok"]
    latency = [float(r["latency_ms"]) for r in successful if r.get("latency_ms") is not None]
    rerank_ms = [float(r["timings_ms"]["rerank"]) for r in successful if "rerank" in r.get("timings_ms", {})]
    fallbacks = [r["id"] for r in successful if r.get("rerank_fallback") or r.get("rerank_backend") == "fusion"]
    counts = Counter()
    for row in assessment["queries"]:
        for item in row["books"]:
            judgment = item["judgment"]
            if judgment.get("status") == "graded":
                counts[judgment.get("assessor_type", "unknown")] += 1
                if judgment.get("human_reviewed") is True:
                    counts["human_reviewed"] += 1
    total = len(bundle["query_manifest"]["queries"])
    positive = comparisons["reranked_vs_without_reranker"].get("ci_95")
    return {
        "version": 1, "created_at": timestamp(), "status": "provisional" if incomplete else "complete",
        "run_digest": digest(bundle), "protocol_digest": digest(protocol),
        "assessment_digest": digest(assessment), "catalogue_fingerprint": bundle["catalogue_fingerprint"],
        "assessment_mode": protocol["assessment_mode"], "assessor_counts": dict(counts),
        "query_rows": total, "scored_query_rows": len(per_query), "evaluation_units": len(units),
        "duplicate_policy": protocol["duplicate_policy"], "excluded_queries": protocol.get("excluded_queries", {}),
        "judgment_coverage": len(per_query) / (total - len(protocol.get("excluded_queries", {})))
                             if total > len(protocol.get("excluded_queries", {})) else 0.0,
        "unfinished_queries": incomplete, "progress": progress, "no_known_relevant_queries": no_relevant,
        "overall": overall, "by_category": by_category, "comparisons": comparisons,
        "latency": {"warm_queries": len(latency), "warm_median_ms": percentile(latency, .5),
                    "warm_p95_ms": percentile(latency, .95), "rerank_stage_median_ms": percentile(rerank_ms, .5),
                    "rerank_stage_p95_ms": percentile(rerank_ms, .95)},
        "reranker_fallback_queries": fallbacks,
        "positive_mean_gain_ci_excludes_zero": bool(not incomplete and not fallbacks and positive and positive[0] > 0),
        "per_query": per_query,
        "limitations": [
            "Assistant grades are model-assisted judgments, not independent human ground truth.",
            "Metrics describe catalogue-supported relevance, not verified full-book content.",
            "Recall and ideal DCG use an incomplete judgment pool, not the entire catalogue.",
            "Without an independent retrieval baseline, candidate recall is necessarily 1 for known relevant pooled books.",
            "Paired rankings share one search; baseline end-to-end latency is not independently measured.",
            "Zero ideal DCG gives NDCG=0; no grade>=2 gives undefined pooled recall (null).",
            "The 30-query set is exploratory if it was used to develop/tune the engine.",
        ],
    }


def write_report(payload: dict, out: Path) -> None:
    write_json(out, payload)
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "query", "system", "ndcg_at_10",
                                                   "precision_at_10", "mrr_at_10", "candidate_pooled_recall"])
        writer.writeheader()
        for row in payload["per_query"]:
            for name, metrics in row["systems"].items():
                recall = next(v for k, v in metrics.items() if k.startswith("candidate_pooled_recall"))
                writer.writerow({"query_id": row["id"], "query": row["query"], "system": name,
                                 **{k: metrics[k] for k in ("ndcg_at_10", "precision_at_10", "mrr_at_10")},
                                 "candidate_pooled_recall": recall})
    overall, delta = payload["overall"], payload["comparisons"]["reranked_vs_without_reranker"]
    lines = ["# Reranker evaluation results", "", f"Status: **{payload['status']}**.", "",
             f"Assessment: `{payload['assessment_mode']}`; assessor counts: `{payload['assessor_counts']}`.", "",
             f"Scored {payload['scored_query_rows']}/{payload['query_rows']} rows; "
             f"{payload['evaluation_units']} evaluation units; duplicate policy: `{payload['duplicate_policy']}`.", "",
             "| System | NDCG@10 | Precision@10 | MRR@10 |", "|---|---:|---:|---:|"]
    for name, metrics in overall.items():
        vals = [f"{metrics[k]:.4f}" if metrics.get(k) is not None else "n/a"
                for k in ("ndcg_at_10", "precision_at_10", "mrr_at_10")]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    lines += ["", f"Paired mean NDCG gain over `without_reranker`: {delta['mean_delta']}; "
              f"95% bootstrap interval: {delta['ci_95']}.", "",
              f"Reranker fallback rows: {payload['reranker_fallback_queries']}.", "",
              "Limitations:", ""]
    lines.extend(f"- {limitation}" for limitation in payload["limitations"])
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}, {csv_path}, and {out.with_suffix('.md')}")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="snapshot queries and create a decision protocol (offline)")
    prep.add_argument("--queries", type=Path, default=ROOT / "queries.txt")
    prep.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    runner = commands.add_parser("run", help="collect paired rankings and catalogue metadata (search PC only)")
    runner.add_argument("--manifest", type=Path, default=DEFAULT_DIR / "queries.json")
    runner.add_argument("--protocol", type=Path, default=DEFAULT_DIR / "protocol.json")
    runner.add_argument("--queries", type=Path, default=ROOT / "queries.txt")
    runner.add_argument("--out", type=Path, default=DEFAULT_DIR / "run.json")
    pooler = commands.add_parser("pool", help="create a blinded evidence sheet; never assign grades")
    checker = commands.add_parser("check", help="validate evidence and display assessment progress")
    scorer = commands.add_parser("score", help="compute graded metrics from completed judgments (offline)")
    for command in (pooler, checker, scorer):
        command.add_argument("--run", type=Path, default=DEFAULT_DIR / "run.json")
        command.add_argument("--protocol", type=Path, default=DEFAULT_DIR / "protocol.json")
    pooler.add_argument("--out", type=Path, default=DEFAULT_DIR / "assessment.json")
    for command in (checker, scorer):
        command.add_argument("--assessment", type=Path, default=DEFAULT_DIR / "assessment.json")
    scorer.add_argument("--out", type=Path, default=DEFAULT_DIR / "report.json")
    scorer.add_argument("--allow-partial", action="store_true", help="explicitly produce a provisional subset report")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args.queries, args.out_dir)
        elif args.command == "run":
            return run_searches(args.manifest, args.protocol, args.queries, args.out)
        elif args.command == "pool":
            write_pool(args.run, args.protocol, args.out)
        else:
            bundle, protocol, assessment = read_json(args.run), read_json(args.protocol), read_json(args.assessment)
            if args.command == "check":
                _, progress = check_assessment(bundle, protocol, assessment)
                print(json.dumps(progress, ensure_ascii=False, indent=2))
                return 0 if all(r["complete"] for r in progress) else 2
            write_report(report(bundle, protocol, assessment, allow_partial=args.allow_partial), args.out)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
