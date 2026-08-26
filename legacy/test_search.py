"""
test_search.py — End-to-end test harness for the full search pipeline.

Integrates:
  - Step 2.1: BM25 Lexical Index
  - Step 2.2: Dense Vector Index
  - Step 2.4: KG Index/Query Layer
  - Step 2.5: Query Understanding
  - Step 2.11: Explanation Generation (via Gemma-4)

Usage:
    python test_search.py                     # Run with default query
    python test_search.py "your query here"   # Run with a custom query
"""

import sys
import os
import time

from data_loader import load_books, get_book_display
from bm25_index import BM25Index
from dense_index import DenseIndex
from kg_index import KnowledgeGraph
from query_understanding import understand_query, QueryIntent
from explanation_generator import generate_comparison_explanation

# Cache paths
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".index_cache")
BM25_CACHE = os.path.join(CACHE_DIR, "bm25_index.pkl")
DENSE_CACHE = CACHE_DIR
KG_CACHE = os.path.join(CACHE_DIR, "kg_index.pkl")

# Single default test query — exercises KG (author) + semantic + BM25
DEFAULT_QUERY = "হুমায়ূন আহমেদের মুক্তিযুদ্ধের বই"

TOP_K = 5  # Results per retriever


def build_or_load_bm25(books) -> BM25Index:
    """Build or load the BM25 index."""
    index = BM25Index()
    if os.path.exists(BM25_CACHE):
        try:
            index.load(BM25_CACHE)
            return index
        except Exception as e:
            print(f"[test] Failed to load BM25 cache: {e}. Rebuilding...")
    index.build(books)
    os.makedirs(CACHE_DIR, exist_ok=True)
    index.save(BM25_CACHE)
    return index


def build_or_load_dense(books) -> DenseIndex:
    """Build or load the Dense index."""
    index = DenseIndex()
    if not index.load(books, DENSE_CACHE):
        index.build(books)
        index.save(DENSE_CACHE)
    return index


def build_or_load_kg(books) -> KnowledgeGraph:
    """Build or load the KG."""
    kg = KnowledgeGraph()
    if os.path.exists(KG_CACHE):
        try:
            kg.load(KG_CACHE)
            return kg
        except Exception as e:
            print(f"[test] Failed to load KG cache: {e}. Rebuilding...")
    kg.build(books)
    os.makedirs(CACHE_DIR, exist_ok=True)
    kg.save(KG_CACHE)
    return kg


def print_results(results, method_name: str):
    """Print search results with reasoning."""
    if not results:
        print(f"  No results found.")
        return
    for i, (book, score, reasoning) in enumerate(results):
        print(f"\n  #{i+1}")
        print(get_book_display(book))
        print(f"     🎯 {reasoning}")


def run_query(query: str, bm25_index: BM25Index, dense_index: DenseIndex,
              kg: KnowledgeGraph, generate_explanations: bool = True):
    """Run a query through the full pipeline."""
    print(f"\n{'='*80}")
    print(f"🔍 Query: {query}")
    print(f"{'='*80}")

    # ── Step 2.5: Query Understanding ──────────────────────────────────
    print(f"\n--- 🧠 Step 2.5: Query Understanding ---")
    start = time.time()
    intent = understand_query(query)
    qu_time = time.time() - start
    print(f"  ⏱️  Classification time: {qu_time:.2f}s")
    print(intent)

    # ── Retrieval based on routing ─────────────────────────────────────
    all_results = {}  # channel_name → results list

    if "bm25" in intent.retrieval_channels:
        print(f"\n--- 📚 BM25 (Lexical) Results ---")
        start = time.time()
        # Use normalized query if available, otherwise original
        search_query = intent.normalized_query or query
        bm25_results = bm25_index.search(search_query, top_k=TOP_K)
        bm25_time = time.time() - start
        print(f"  ⏱️  Search time: {bm25_time*1000:.1f}ms | Found: {len(bm25_results)} results")
        print_results(bm25_results, "BM25")
        all_results["bm25"] = bm25_results

    if "dense" in intent.retrieval_channels:
        print(f"\n--- 🧠 Dense (Semantic) Results ---")
        start = time.time()
        search_query = intent.normalized_query or query
        dense_results = dense_index.search(search_query, top_k=TOP_K)
        dense_time = time.time() - start
        print(f"  ⏱️  Search time: {dense_time*1000:.1f}ms | Found: {len(dense_results)} results")
        print_results(dense_results, "Dense")
        all_results["dense"] = dense_results

    if "kg" in intent.retrieval_channels:
        print(f"\n--- 🔗 KG (Knowledge Graph) Results ---")
        start = time.time()
        kg_results = kg.search(query, entities=intent.entities, top_k=TOP_K)
        kg_time = time.time() - start
        print(f"  ⏱️  Search time: {kg_time*1000:.1f}ms | Found: {len(kg_results)} results")
        print_results(kg_results, "KG")
        all_results["kg"] = kg_results

    # ── Overlap Analysis ───────────────────────────────────────────────
    print(f"\n--- 📊 Cross-Channel Overlap Analysis ---")
    channel_ids = {}
    for channel, results in all_results.items():
        channel_ids[channel] = {r[0]["book_id"] for r in results}

    channels = list(channel_ids.keys())
    for i in range(len(channels)):
        for j in range(i + 1, len(channels)):
            c1, c2 = channels[i], channels[j]
            overlap = channel_ids[c1] & channel_ids[c2]
            only_c1 = channel_ids[c1] - channel_ids[c2]
            only_c2 = channel_ids[c2] - channel_ids[c1]
            print(f"  {c1} ∩ {c2}: {len(overlap)} shared | "
                  f"{c1} only: {len(only_c1)} | {c2} only: {len(only_c2)}")

    # ── LLM Explanation (single call) ──────────────────────────────────
    if generate_explanations and len(all_results) >= 2:
        print(f"\n--- 🤖 Gemma-4 Comparative Analysis ---")

        # Pick two main channels for comparison
        bm25_res = all_results.get("bm25", [])
        # Prefer dense for comparison, fall back to KG
        other_res = all_results.get("dense", all_results.get("kg", []))

        if bm25_res and other_res:
            comparison = generate_comparison_explanation(
                query=query,
                bm25_results=bm25_res,
                dense_results=other_res,
                top_n=3,
            )
            print(comparison)

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n--- 📋 Pipeline Summary ---")
    print(f"  Query type: {intent.query_type}")
    print(f"  Channels used: {', '.join(intent.retrieval_channels)}")
    total_unique = len(set().union(*channel_ids.values())) if channel_ids else 0
    print(f"  Total unique results: {total_unique}")


def main():
    """Main entry point."""
    print("=" * 80)
    print("  📖 Book Search Engine — Full Pipeline Test")
    print("     Steps: 2.1 (BM25) + 2.2 (Dense) + 2.4 (KG) + 2.5 (QU) + 2.11 (Explain)")
    print("=" * 80)

    # Determine query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"  Custom query: {query}")
    else:
        query = DEFAULT_QUERY
        print(f"  Default query: {query}")

    # Load data
    print("\n--- Loading Data ---")
    books = load_books()

    # Build/load all indices
    print("\n--- Building/Loading BM25 Index ---")
    bm25_index = build_or_load_bm25(books)

    print("\n--- Building/Loading Dense Index ---")
    dense_index = build_or_load_dense(books)

    print("\n--- Building/Loading Knowledge Graph ---")
    kg = build_or_load_kg(books)

    # Print KG stats
    print("\n--- KG Stats ---")
    print("  Top 5 Authors:")
    for author, count in kg.get_author_stats(5):
        print(f"    {author}: {count} books")
    print("  Top 5 Publishers:")
    for pub, count in kg.get_publisher_stats(5):
        print(f"    {pub}: {count} books")

    # Run the query through the full pipeline
    run_query(
        query=query,
        bm25_index=bm25_index,
        dense_index=dense_index,
        kg=kg,
        generate_explanations=True,
    )

    print(f"\n{'='*80}")
    print("  ✅ Test complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
