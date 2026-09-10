"""Score search outputs using local LM Studio LLM judgments and compute NDCG@k.

Features:
  1. Parses results from `outputs/output_N.txt`.
  2. Enriches each book with its full metadata from `books_metadata_cleaned.csv`
     (Description/Flap, Author Bio, Publisher, Publication Year).
  3. Uses a local LM Studio model to grade relevance on a 0-3 scale:
       0 = irrelevant
       1 = tangentially related
       2 = relevant
       3 = excellent match
  4. Writes judgments and scores incrementally to `judgements.yaml`.
  5. Computes graded NDCG@k (e.g. k=1, 3, 5, 10, 20, 50, 100) for all queries and
     saves results to `outputs/ndcg_scores.csv`.

Usage:
    # Run full judging using LM Studio and compute NDCG@k:
    python score_ndcg.py

    # Specify custom LM Studio model or API base:
    python score_ndcg.py --base-url http://localhost:1234/v1 --model my-model

    # Score only specific queries (e.g. query 1 and 2):
    python score_ndcg.py --queries 1,2

    # Skip calling LLM and recalculate NDCG@k from existing judgements.yaml:
    python score_ndcg.py --skip-llm

    # Dry-run / test without a running LM Studio server:
    python score_ndcg.py --mock --queries 1,2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import httpx
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Ensure Windows console does not crash on Bengali characters
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from search.core import bengali
except ImportError:
    # Minimal fallback normalizer if running outside full repo environment
    class _BengaliFallback:
        @staticmethod
        def normalize(text: str) -> str:
            return " ".join(str(text or "").split()).strip()

    bengali = _BengaliFallback()  # type: ignore

# Result panels as CLI renders them: "╭─ 12. TITLE — AUTHOR ─────╮"
PANEL_PATTERN = re.compile(r"^╭─+\s*(\d+)\.\s+(.*?)\s*─+╮\s*$")
HEADER_QUERY_PATTERN = re.compile(r"^query as searched\s*:\s*(.*)$")
JSON_BLOCK_PATTERN = re.compile(r"\[\s*\{.*\}\s*\]", re.DOTALL)

LABEL_MAP = {
    0: "irrelevant",
    1: "tangentially related",
    2: "relevant",
    3: "excellent match",
}

SYSTEM_PROMPT = """\
You are an expert relevance evaluator for a Bengali book search engine.
Your task is to evaluate the relevance of search candidate books against a search query.

Relevance Scale (0 to 3):
  0 = Irrelevant: The book has nothing to do with the query topic, or only mentions terms superficially/out of context.
  1 = Tangentially related: The book touches upon related themes or broader context, but is not specifically about the query.
  2 = Relevant: The book directly addresses the query topic or theme.
  3 = Excellent match: The book is a direct, outstanding, high-priority match that perfectly answers the query.

Evaluation Rules:
  - Form matters: If the query specifies 'কবিতা' (poetry), poetry collections are relevant, not prose criticism. If it asks for 'উপন্যাস' (novel) or 'কিশোর উপন্যাস' (juvenile fiction), history or textbooks are not novels.
  - Specificity matters: If the query asks for a specific topic (e.g., 'নারী মুক্তিযোদ্ধাদের গল্প'), general 1971 war collections without significant focus on women freedom fighters should receive lower scores (0 or 1).
  - Geographic/Historical scope matters: If a specific location or event is queried (e.g., 'খুলনা বিভাগে মুক্তিযুদ্ধ', 'নীল চাষিদের বিদ্রোহ', 'জুলাইয়ের আন্দোলন'), books about other regions or different historical events score 0.

Output Requirement:
You MUST return ONLY valid JSON (either a JSON array or a JSON object with a "results" key).
Do NOT include conversational greetings, conclusions, or remarks.
Format example:
[
  {"rank": 1, "score": 3, "reason": "Direct focus on freedom fighters"},
  {"rank": 2, "score": 0, "reason": "Not related"}
]
"""


# --------------------------------------------------------------------------- Data Parsing & Matching

def parse_output_file(path: Path) -> tuple[str, list[tuple[int, str, str]]]:
    """Extracts (query, [(rank, title, author), ...]) from output_N.txt."""
    lines = path.read_text(encoding="utf-8").splitlines()
    query = ""
    for line in lines[:25]:
        m = HEADER_QUERY_PATTERN.match(line)
        if m:
            query = m.group(1).strip()
            break

    results: list[tuple[int, str, str]] = []
    for line in lines:
        match = PANEL_PATTERN.match(line)
        if match:
            rank = int(match.group(1))
            panel_text = match.group(2).strip()
            if " — " in panel_text:
                title, _, author = panel_text.partition(" — ")
            elif " - " in panel_text:
                title, _, author = panel_text.partition(" - ")
            else:
                title, author = panel_text, ""
            results.append((rank, title.strip(), author.strip()))

    results.sort(key=lambda r: r[0])
    return query, results


def load_metadata_catalog(csv_path: Path) -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, dict[str, str]]]:
    """Loads books_metadata_cleaned.csv into normalized lookup tables."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    pair_lookup: dict[tuple[str, str], dict[str, str]] = {}
    title_lookup: dict[str, dict[str, str]] = {}

    for _, row in df.iterrows():
        book_data = {
            "title": str(row.get("Book Name", "") or "").strip(),
            "author": str(row.get("Author", "") or "").strip(),
            "author_bio": str(row.get("Author Bio", "") or "").strip(),
            "publisher": str(row.get("Publisher", "") or "").strip(),
            "description": str(row.get("Description (Flap)", "") or "").strip(),
            "publication_year": str(row.get("Publication Year", "") or "").strip(),
        }
        t_norm = bengali.normalize(book_data["title"]).casefold()
        a_norm = bengali.normalize(book_data["author"]).casefold()

        if t_norm:
            pair_lookup[(t_norm, a_norm)] = book_data
            title_lookup.setdefault(t_norm, book_data)

    return pair_lookup, title_lookup


def get_book_metadata(
    title: str,
    author: str,
    pair_lookup: dict[tuple[str, str], dict[str, str]],
    title_lookup: dict[str, dict[str, str]],
) -> dict[str, str]:
    """Finds the enriched metadata for a given title and author."""
    t_norm = bengali.normalize(title).casefold()
    a_norm = bengali.normalize(author).casefold()

    match = pair_lookup.get((t_norm, a_norm))
    if not match:
        match = title_lookup.get(t_norm)

    if match:
        return match

    return {
        "title": title,
        "author": author,
        "author_bio": "",
        "publisher": "",
        "description": "",
        "publication_year": "",
    }


# --------------------------------------------------------------------------- LM Studio Client

class LMStudioScorer:
    def __init__(self, base_url: str = "http://localhost:1234/v1", model: str = "", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self.model = model

    def resolve_model(self) -> str:
        """Finds the loaded or first available model in LM Studio."""
        if self.model:
            return self.model
        try:
            resp = self.client.get("/models")
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                raise RuntimeError("No models found in LM Studio /v1/models")
            self.model = data[0]["id"]
            return self.model
        except Exception as exc:
            raise ConnectionError(
                f"Cannot connect to LM Studio at {self.base_url}. "
                "Ensure LM Studio is running with local server started. Error: " + str(exc)
            ) from exc

    def score_batch(
        self,
        query: str,
        books_batch: list[dict[str, Any]],
        max_retries: int = 3,
    ) -> list[dict[str, Any]]:
        """Scores a batch of books using the LM Studio model."""
        model_id = self.resolve_model()

        # Build user message with book details
        items_text = []
        for b in books_batch:
            rank = b["rank"]
            title = b["title"]
            author = b["author"]
            publisher = b.get("publisher", "")
            year = b.get("publication_year", "")
            bio = b.get("author_bio", "")
            desc = b.get("description", "")

            # Trim very long descriptions to ~1200 chars to be safe on token budget
            if len(desc) > 1200:
                desc = desc[:1200] + "... [truncated]"
            if len(bio) > 400:
                bio = bio[:400] + "... [truncated]"

            item_str = (
                f"[Rank {rank}]\n"
                f"Title: {title}\n"
                f"Author: {author}\n"
                f"Publisher: {publisher} | Year: {year}\n"
                f"Author Bio: {bio or 'N/A'}\n"
                f"Description: {desc or 'N/A'}\n"
            )
            items_text.append(item_str)

        user_content = (
            f"Search Query: \"{query}\"\n\n"
            f"Evaluate the following {len(books_batch)} candidate books. "
            f"Assign each book a relevance score from 0 to 3.\n\n"
            + "\n".join(items_text)
            + "\n\nOutput JSON array format: [{\"rank\": <rank>, \"score\": <0-3>, \"reason\": \"<explanation>\"}]"
        )

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                resp = self.client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                response_json = resp.json()
                content = response_json["choices"][0]["message"]["content"]
                parsed = self._extract_json_array(content)
                if parsed is not None and len(parsed) > 0:
                    return parsed
                last_error = f"Model output could not be parsed as JSON: {repr(content[:250])}"
            except Exception as e:
                last_error = str(e)
                time.sleep(1.0 + attempt)

        print(f"[warn] LLM scoring batch failed after {max_retries} attempts: {last_error}", file=sys.stderr)
        # Fallback: return default 0 scores
        return [{"rank": b["rank"], "score": 0, "reason": "Evaluation failed/fallback"} for b in books_batch]

    @staticmethod
    def _extract_json_array(text: str) -> list[dict[str, Any]] | None:
        """Robustly extracts JSON array or list of evaluations from raw model response text.
        Handles thinking tags (<think>...</think>), markdown code fences, wrapper keys,
        trailing commas, dictionary outputs, and regex fallbacks.
        """
        if not text or not text.strip():
            return None

        # 1. Strip reasoning / thinking tags from thinking models (e.g. DeepSeek-R1)
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<thought>.*?</thought>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = cleaned.strip()

        # 2. Collect candidate JSON substrings
        candidates = []

        # Check for markdown code fences (```json ... ``` or ``` ... ```)
        fences = re.findall(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        for f in fences:
            candidates.append(f.strip())

        candidates.append(cleaned)

        # Search for outer bracketed blocks
        for c in list(candidates):
            array_match = re.search(r"\[.*\]", c, flags=re.DOTALL)
            if array_match:
                candidates.append(array_match.group(0).strip())
            obj_match = re.search(r"\{.*\}", c, flags=re.DOTALL)
            if obj_match:
                candidates.append(obj_match.group(0).strip())

        # 3. Try parsing each candidate
        for cand in candidates:
            # Strip trailing commas that invalidate standard json.loads
            cand_no_trailing = re.sub(r",\s*([\]}])", r"\1", cand)
            for raw in (cand, cand_no_trailing):
                try:
                    val = json.loads(raw)
                    if isinstance(val, list):
                        return val
                    if isinstance(val, dict):
                        # Check wrapper keys like {"results": [...]}, {"books": [...]}, etc.
                        for k in ("results", "books", "evaluations", "scores", "items", "data", "judgments"):
                            if k in val and isinstance(val[k], list):
                                return val[k]
                        # Check rank-keyed dict: {"1": 3, "2": 2} or {"1": {"score": 3, ...}}
                        out = []
                        for k, v in val.items():
                            clean_k = "".join(ch for ch in str(k) if ch.isdigit())
                            if clean_k:
                                if isinstance(v, dict):
                                    out.append({
                                        "rank": int(clean_k),
                                        "score": v.get("score", 0),
                                        "reason": v.get("reason", ""),
                                    })
                                elif isinstance(v, (int, float, str)):
                                    clean_v = "".join(ch for ch in str(v) if ch.isdigit())
                                    out.append({
                                        "rank": int(clean_k),
                                        "score": int(clean_v) if clean_v else 0,
                                        "reason": "",
                                    })
                        if out:
                            return out
                except Exception:
                    continue

        # 4. Regex fallback: extract individual rank and score pairs
        pattern1 = re.compile(r'\"?rank\"?\s*:\s*(\d+).*?\"?score\"?\s*:\s*([0-3])', re.DOTALL | re.IGNORECASE)
        matches1 = list(pattern1.finditer(cleaned))
        if matches1:
            return [{"rank": int(m.group(1)), "score": int(m.group(2)), "reason": "regex extracted"} for m in matches1]

        pattern2 = re.compile(r"(?:rank|#)\s*(\d+)[:\s\-]+(?:score[:\s=]*)?([0-3])", re.IGNORECASE)
        matches2 = list(pattern2.finditer(cleaned))
        if matches2:
            return [{"rank": int(m.group(1)), "score": int(m.group(2)), "reason": "regex extracted"} for m in matches2]

        return None


# --------------------------------------------------------------------------- Mock Scorer for Testing

class MockScorer:
    """Mock scorer to test pipeline execution without needing a running LM Studio server."""
    def __init__(self):
        self.model = "mock-evaluator"

    def score_batch(self, query: str, books_batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        q_tokens = set(bengali.normalize(query).split())
        for b in books_batch:
            rank = b["rank"]
            title = b["title"]
            desc = b.get("description", "")
            t_tokens = set(bengali.normalize(title).split())
            d_tokens = set(bengali.normalize(desc).split())

            # Heuristic mock score
            overlap_title = len(q_tokens & t_tokens)
            overlap_desc = len(q_tokens & d_tokens)

            if overlap_title >= 2 or (overlap_title >= 1 and rank <= 5):
                score = 3
                reason = "Strong query term match in title (mock)"
            elif overlap_title >= 1 or overlap_desc >= 2:
                score = 2
                reason = "Relevant topic keywords in description (mock)"
            elif overlap_desc >= 1 or rank <= 15:
                score = 1
                reason = "Tangential theme overlap (mock)"
            else:
                score = 0
                reason = "No direct topic match (mock)"

            results.append({"rank": rank, "score": score, "reason": reason})
        return results


# --------------------------------------------------------------------------- NDCG Calculation

def dcg_at_k(scores: Sequence[int | float], k: int) -> float:
    """Computes Discounted Cumulative Gain at rank k using standard graded formula:
    DCG@k = sum_{i=1}^k (2^{rel_i} - 1) / log2(i + 1)
    """
    dcg = 0.0
    for i, rel in enumerate(scores[:k], start=1):
        if rel > 0:
            dcg += (math.pow(2, rel) - 1.0) / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_scores: Sequence[int | float], all_judged_scores: Sequence[int | float], k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain at rank k.
    IDCG is derived by sorting all judged scores in descending order.
    """
    dcg = dcg_at_k(ranked_scores, k)
    ideal_scores = sorted(all_judged_scores, reverse=True)
    idcg = dcg_at_k(ideal_scores, k)
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


# --------------------------------------------------------------------------- Driver & CLI

def score_query_results(
    query_num: int,
    query_text: str,
    results: list[tuple[int, str, str]],
    pair_lookup: dict,
    title_lookup: dict,
    scorer: LMStudioScorer | MockScorer,
    batch_size: int = 10,
) -> dict[str, Any]:
    """Evaluates all books for a single query."""
    enriched_books: list[dict[str, Any]] = []
    for rank, title, author in results:
        meta = get_book_metadata(title, author, pair_lookup, title_lookup)
        enriched_books.append({
            "rank": rank,
            "title": meta["title"] or title,
            "author": meta["author"] or author,
            "publisher": meta.get("publisher", ""),
            "publication_year": meta.get("publication_year", ""),
            "author_bio": meta.get("author_bio", ""),
            "description": meta.get("description", ""),
        })

    # Run scoring in batches
    scores_by_rank: dict[int, int] = {}
    reasons_by_rank: dict[int, str] = {}

    total = len(enriched_books)
    for start_idx in range(0, total, batch_size):
        batch = enriched_books[start_idx : start_idx + batch_size]
        batch_scores = scorer.score_batch(query_text, batch)
        for item in batch_scores:
            try:
                r = int(item.get("rank", 0))
                s = int(item.get("score", 0))
                s = max(0, min(3, s))  # clamp to [0, 3]
                reason = str(item.get("reason", "")).strip()
                scores_by_rank[r] = s
                reasons_by_rank[r] = reason
            except (ValueError, TypeError):
                continue

    # Ensure every rank has a score
    ranked_book_records = []
    for b in enriched_books:
        r = b["rank"]
        s = scores_by_rank.get(r, 0)
        reason = reasons_by_rank.get(r, "No reason provided")
        ranked_book_records.append({
            "rank": r,
            "title": b["title"],
            "author": b["author"],
            "score": s,
            "label": LABEL_MAP.get(s, "unknown"),
            "reason": reason,
        })

    # Format entry
    scores_dict = {b["rank"]: b["score"] for b in ranked_book_records}
    relevant_scores_dict = {b["rank"]: b["score"] for b in ranked_book_records if b["score"] >= 1}
    relevant_ranks = [b["rank"] for b in ranked_book_records if b["score"] >= 1]

    return {
        "query": query_text,
        "scores": scores_dict,
        "relevant_scores": relevant_scores_dict,
        "relevant": relevant_ranks,
        "books": ranked_book_records,
    }


def parse_k_values(k_arg: str) -> list[int]:
    vals = []
    for item in k_arg.split(","):
        item = item.strip()
        if item.isdigit():
            vals.append(int(item))
    return sorted(vals) or [1, 3, 5, 10, 20, 50, 100]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs",
                        help="Directory containing output_N.txt files (default: outputs/)")
    parser.add_argument("--metadata", type=Path, default=ROOT / "books_metadata_cleaned.csv",
                        help="Path to books_metadata_cleaned.csv (default: books_metadata_cleaned.csv)")
    parser.add_argument("--judgments", type=Path, default=ROOT / "judgements.yaml",
                        help="YAML file to save/load judgments (default: judgements.yaml)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="CSV file to save NDCG scores (default: <outputs>/ndcg_scores.csv)")
    parser.add_argument("--base-url", type=str, default="http://localhost:1234/v1",
                        help="LM Studio API base URL (default: http://localhost:1234/v1)")
    parser.add_argument("--model", type=str, default="",
                        help="LM Studio model ID (default: auto-detect loaded model)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of candidate books to evaluate per LLM prompt (default: 10)")
    parser.add_argument("--k", type=str, default="1,3,5,10,20,50,100",
                        help="Comma-separated k values for NDCG@k (default: 1,3,5,10,20,50,100)")
    parser.add_argument("--queries", type=str, default="",
                        help="Comma-separated query numbers to score, e.g. '1,2,5' (default: all)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM scoring and compute NDCG directly from existing judgements.yaml")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock scorer for local testing without running LM Studio")
    parser.add_argument("--force", action="store_true",
                        help="Re-score queries even if they already exist in judgements.yaml")

    args = parser.parse_args()

    csv_out_path = args.csv or (args.outputs / "ndcg_scores.csv")
    k_list = parse_k_values(args.k)

    # Filter target queries if requested
    target_queries = None
    if args.queries:
        target_queries = {int(q.strip()) for q in args.queries.split(",") if q.strip().isdigit()}

    # Load existing judgments if present
    judgments_data: dict[int, Any] = {}
    if args.judgments.exists():
        try:
            loaded = yaml.safe_load(args.judgments.read_text(encoding="utf-8")) or {}
            judgments_data = {int(k): v for k, v in loaded.items() if str(k).isdigit()}
            print(f"Loaded existing judgments for {len(judgments_data)} queries from {args.judgments}")
        except Exception as e:
            print(f"[warn] Failed to read existing {args.judgments}: {e}", file=sys.stderr)

    # If not skipping LLM, prepare scorer and metadata
    scorer: LMStudioScorer | MockScorer | None = None
    pair_lookup: dict = {}
    title_lookup: dict = {}

    if not args.skip_llm:
        print(f"Loading metadata catalog from {args.metadata}...")
        pair_lookup, title_lookup = load_metadata_catalog(args.metadata)
        print(f"Catalog indexed: {len(pair_lookup)} (title, author) entries, {len(title_lookup)} unique titles.")

        if args.mock:
            print("[info] Running in MOCK mode: simulating relevance judgments.")
            scorer = MockScorer()
        else:
            print(f"Connecting to LM Studio at {args.base_url}...")
            scorer = LMStudioScorer(base_url=args.base_url, model=args.model)
            try:
                active_model = scorer.resolve_model()
                print(f"Connected to LM Studio successfully. Using model: {active_model}")
            except Exception as e:
                print(f"\n[ERROR] {e}", file=sys.stderr)
                print("Tip: Start LM Studio and load a model, or run with --mock for dry testing.", file=sys.stderr)
                return 1

    # Find output files
    output_files = sorted(
        args.outputs.glob("output_*.txt"),
        key=lambda p: int(p.stem.split("_")[-1]) if p.stem.split("_")[-1].isdigit() else 9999,
    )

    if not output_files:
        print(f"[error] No output_N.txt files found in {args.outputs}", file=sys.stderr)
        return 1

    print(f"Found {len(output_files)} search output files in {args.outputs}")

    # Process queries
    for p in output_files:
        num_str = p.stem.split("_")[-1]
        if not num_str.isdigit():
            continue
        q_num = int(num_str)

        if target_queries and q_num not in target_queries:
            continue

        if not args.force and q_num in judgments_data and "scores" in judgments_data[q_num]:
            # Already judged
            continue

        if args.skip_llm:
            if q_num not in judgments_data:
                print(f"[warn] Query {q_num} has no judgments in {args.judgments} and --skip-llm is active.", file=sys.stderr)
            continue

        query_text, results = parse_output_file(p)
        print(f"\nEvaluating Query {q_num}: \"{query_text}\" ({len(results)} books)...")

        entry = score_query_results(
            query_num=q_num,
            query_text=query_text,
            results=results,
            pair_lookup=pair_lookup,
            title_lookup=title_lookup,
            scorer=scorer,  # type: ignore
            batch_size=args.batch_size,
        )

        judgments_data[q_num] = entry

        # Incremental save to judgements.yaml
        args.judgments.parent.mkdir(parents=True, exist_ok=True)
        # Format for clean YAML output
        yaml_ready = {int(k): judgments_data[k] for k in sorted(judgments_data.keys())}
        args.judgments.write_text(
            yaml.safe_dump(yaml_ready, allow_unicode=True, sort_keys=True, width=120),
            encoding="utf-8",
        )
        print(f"Saved judgments for query {q_num} to {args.judgments}")

    # ----------------------------------------------------------------------- NDCG Calculation
    print("\nCalculating NDCG@k metrics for evaluated queries...")
    ndcg_rows = []

    for p in output_files:
        num_str = p.stem.split("_")[-1]
        if not num_str.isdigit():
            continue
        q_num = int(num_str)

        if target_queries and q_num not in target_queries:
            continue

        if q_num not in judgments_data:
            continue

        entry = judgments_data[q_num]
        query_text, results = parse_output_file(p)
        query_text = query_text or entry.get("query", f"Query {q_num}")

        # Extract ranked scores
        scores_map = entry.get("scores", {})
        # If scores_map is string-keyed due to yaml reload, cast to int
        scores_map = {int(k): int(v) for k, v in scores_map.items()}

        ranked_scores = [scores_map.get(rank, 0) for rank, _, _ in results]
        all_judged_scores = list(scores_map.values()) if scores_map else ranked_scores

        row: dict[str, Any] = {
            "query_no": q_num,
            "query": query_text,
            "returned": len(results),
            "relevant": sum(1 for s in ranked_scores if s >= 1),
            "excellent": sum(1 for s in ranked_scores if s == 3),
        }

        for k in k_list:
            score = ndcg_at_k(ranked_scores, all_judged_scores, k)
            row[f"ndcg@{k}"] = round(score, 4)

        ndcg_rows.append(row)

    if not ndcg_rows:
        print("[error] No evaluated queries to score.", file=sys.stderr)
        return 1

    # Sort rows by query number
    ndcg_rows.sort(key=lambda r: r["query_no"])

    # Compute macro-average MEAN row
    mean_row: dict[str, Any] = {
        "query_no": "MEAN",
        "query": "Average across all evaluated queries",
        "returned": round(sum(r["returned"] for r in ndcg_rows) / len(ndcg_rows), 1),
        "relevant": round(sum(r["relevant"] for r in ndcg_rows) / len(ndcg_rows), 1),
        "excellent": round(sum(r["excellent"] for r in ndcg_rows) / len(ndcg_rows), 1),
    }
    for k in k_list:
        mean_row[f"ndcg@{k}"] = round(sum(r[f"ndcg@{k}"] for r in ndcg_rows) / len(ndcg_rows), 4)

    all_rows = ndcg_rows + [mean_row]

    # Save to CSV
    csv_out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with csv_out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n[OK] Successfully wrote NDCG@k scores to: {csv_out_path}")

    # Display console table
    k_headers = [f"n@{k}" for k in k_list]
    print("\n" + "=" * 95)
    print(f"{'No':>4}  {'Query':<30}  {'Rel':>4}  {'Exc':>4}  " + "  ".join(f"{h:>7}" for h in k_headers))
    print("-" * 95)
    for r in ndcg_rows:
        q_display = r['query'][:28] + ".." if len(r['query']) > 30 else r['query']
        k_vals_str = "  ".join(f"{r[f'ndcg@{k}']:>7.3f}" for k in k_list)
        print(f"{r['query_no']:>4}  {q_display:<30}  {r['relevant']:>4}  {r['excellent']:>4}  {k_vals_str}")

    print("-" * 95)
    mean_k_str = "  ".join(f"{mean_row[f'ndcg@{k}']:>7.3f}" for k in k_list)
    print(f"{'MEAN':>4}  {'Average across all queries':<30}  {mean_row['relevant']:>4.1f}  {mean_row['excellent']:>4.1f}  {mean_k_str}")
    print("=" * 95 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
