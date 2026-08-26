"""
kg_index.py — Knowledge Graph Index/Query Layer (Step 2.4).

Builds a lightweight in-memory Knowledge Graph from the book catalogue
CSV metadata. Extracts entities (Book, Author, Publisher) and their
relationships (WRITTEN_BY, PUBLISHED_BY) to support structured lookups.

Uses fuzzy string matching (difflib) for entity resolution, handling
Bangla name variations.

The search() method returns results in the same (book, score, reasoning)
format as BM25Index and DenseIndex for easy integration.
"""

import os
import pickle
import time
from collections import defaultdict
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Optional, Set

from data_loader import load_books


def _normalize_name(name: str) -> str:
    """Normalize a name for matching: strip, lowercase, collapse whitespace."""
    if not name:
        return ""
    return " ".join(name.strip().lower().split())


def _fuzzy_score(query: str, candidate: str) -> float:
    """
    Compute fuzzy similarity between two strings.
    Returns a score in [0.0, 1.0].
    """
    if not query or not candidate:
        return 0.0
    return SequenceMatcher(None, query, candidate).ratio()


class KnowledgeGraph:
    """
    In-memory Knowledge Graph over book catalogue metadata.

    Entities:
        - Book: identified by book_id
        - Author: identified by normalized name
        - Publisher: identified by normalized name

    Relationships:
        - WRITTEN_BY: Book → Author
        - PUBLISHED_BY: Book → Publisher

    Supports structured queries over these relationships with
    fuzzy matching for entity resolution.
    """

    def __init__(self):
        # Entity stores
        self.books: List[Dict[str, str]] = []
        self.book_by_id: Dict[int, Dict[str, str]] = {}

        # Relationship indexes (normalized name → set of book_ids)
        self.author_to_books: Dict[str, Set[int]] = defaultdict(set)
        self.publisher_to_books: Dict[str, Set[int]] = defaultdict(set)

        # Reverse indexes (book_id → entities)
        self.book_to_author: Dict[int, str] = {}
        self.book_to_publisher: Dict[int, str] = {}

        # Original name mapping (normalized → original display name)
        self.author_display: Dict[str, str] = {}
        self.publisher_display: Dict[str, str] = {}

        self._is_built = False

    def build(self, books: List[Dict[str, str]]) -> None:
        """
        Build the knowledge graph from book metadata.

        Extracts Author and Publisher entities, builds relationship
        indexes for fast structured lookups.
        """
        print("[kg_index] Building Knowledge Graph...")
        start = time.time()

        self.books = books

        for book in books:
            book_id = book["book_id"]
            self.book_by_id[book_id] = book

            # Author relationship
            author = book.get("author", "")
            if author:
                norm_author = _normalize_name(author)
                self.author_to_books[norm_author].add(book_id)
                self.book_to_author[book_id] = norm_author
                # Keep the first (or longest) original display name
                if norm_author not in self.author_display or len(author) > len(self.author_display[norm_author]):
                    self.author_display[norm_author] = author

            # Publisher relationship
            publisher = book.get("publisher", "")
            if publisher:
                norm_publisher = _normalize_name(publisher)
                self.publisher_to_books[norm_publisher].add(book_id)
                self.book_to_publisher[book_id] = norm_publisher
                if norm_publisher not in self.publisher_display or len(publisher) > len(self.publisher_display[norm_publisher]):
                    self.publisher_display[norm_publisher] = publisher

        self._is_built = True
        elapsed = time.time() - start
        print(f"[kg_index] KG built in {elapsed:.2f}s")
        print(f"[kg_index]   Books: {len(self.books)}")
        print(f"[kg_index]   Unique authors: {len(self.author_to_books)}")
        print(f"[kg_index]   Unique publishers: {len(self.publisher_to_books)}")

    # ─── Structured Query Methods ───────────────────────────────────────

    def find_books_by_author(self, author_query: str, threshold: float = 0.6,
                             top_k: int = 10) -> List[Tuple[Dict[str, str], float, str]]:
        """
        Find books by a given author using fuzzy name matching.

        Args:
            author_query: Author name to search for.
            threshold: Minimum fuzzy match score (0.0-1.0).
            top_k: Max number of results.

        Returns:
            List of (book_dict, score, reasoning) tuples.
        """
        norm_query = _normalize_name(author_query)
        if not norm_query:
            return []

        # Find matching authors
        author_matches = []
        for norm_author in self.author_to_books:
            score = _fuzzy_score(norm_query, norm_author)
            if score >= threshold:
                author_matches.append((norm_author, score))

        # Sort by match score
        author_matches.sort(key=lambda x: x[1], reverse=True)

        # Collect books from matched authors
        results = []
        seen_books = set()
        for norm_author, author_score in author_matches:
            display_author = self.author_display.get(norm_author, norm_author)
            for book_id in self.author_to_books[norm_author]:
                if book_id not in seen_books:
                    seen_books.add(book_id)
                    book = self.book_by_id[book_id]
                    reasoning = (
                        f"[KG/Author] Author match: '{display_author}' "
                        f"(similarity: {author_score:.2f}) | "
                        f"Relationship: WRITTEN_BY"
                    )
                    results.append((book, author_score, reasoning))

            if len(results) >= top_k:
                break

        return results[:top_k]

    def find_books_by_publisher(self, publisher_query: str, threshold: float = 0.6,
                                top_k: int = 10) -> List[Tuple[Dict[str, str], float, str]]:
        """
        Find books by a given publisher using fuzzy name matching.

        Args:
            publisher_query: Publisher name to search for.
            threshold: Minimum fuzzy match score.
            top_k: Max number of results.

        Returns:
            List of (book_dict, score, reasoning) tuples.
        """
        norm_query = _normalize_name(publisher_query)
        if not norm_query:
            return []

        publisher_matches = []
        for norm_pub in self.publisher_to_books:
            score = _fuzzy_score(norm_query, norm_pub)
            if score >= threshold:
                publisher_matches.append((norm_pub, score))

        publisher_matches.sort(key=lambda x: x[1], reverse=True)

        results = []
        seen_books = set()
        for norm_pub, pub_score in publisher_matches:
            display_pub = self.publisher_display.get(norm_pub, norm_pub)
            for book_id in self.publisher_to_books[norm_pub]:
                if book_id not in seen_books:
                    seen_books.add(book_id)
                    book = self.book_by_id[book_id]
                    reasoning = (
                        f"[KG/Publisher] Publisher match: '{display_pub}' "
                        f"(similarity: {pub_score:.2f}) | "
                        f"Relationship: PUBLISHED_BY"
                    )
                    results.append((book, pub_score, reasoning))

            if len(results) >= top_k:
                break

        return results[:top_k]

    def find_related_books(self, book_id: int, top_k: int = 10) -> List[Tuple[Dict[str, str], float, str]]:
        """
        Find books related to a given book (same author or publisher).

        Args:
            book_id: The book to find related books for.
            top_k: Max number of results.

        Returns:
            List of (book_dict, score, reasoning) tuples.
        """
        if book_id not in self.book_by_id:
            return []

        source_book = self.book_by_id[book_id]
        results = []
        seen = {book_id}  # Exclude the source book

        # Same author books (score 1.0 — exact relationship)
        author = self.book_to_author.get(book_id)
        if author:
            display_author = self.author_display.get(author, author)
            for related_id in self.author_to_books.get(author, set()):
                if related_id not in seen:
                    seen.add(related_id)
                    book = self.book_by_id[related_id]
                    reasoning = (
                        f"[KG/Related] Same author: '{display_author}' | "
                        f"Relationship: ALSO_WROTE"
                    )
                    results.append((book, 1.0, reasoning))

        # Same publisher books (score 0.5 — weaker relationship)
        publisher = self.book_to_publisher.get(book_id)
        if publisher:
            display_pub = self.publisher_display.get(publisher, publisher)
            for related_id in self.publisher_to_books.get(publisher, set()):
                if related_id not in seen:
                    seen.add(related_id)
                    book = self.book_by_id[related_id]
                    reasoning = (
                        f"[KG/Related] Same publisher: '{display_pub}' | "
                        f"Relationship: ALSO_PUBLISHED"
                    )
                    results.append((book, 0.5, reasoning))

        # Sort: same-author first, then same-publisher
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search(self, query: str, entities: Optional[Dict] = None,
               top_k: int = 10) -> List[Tuple[Dict[str, str], float, str]]:
        """
        General KG search — tries author and publisher lookups.

        If entities are provided by the query understanding module,
        uses them for targeted lookups. Otherwise, tries fuzzy matching
        the raw query against both author and publisher names.

        Args:
            query: The search query.
            entities: Optional dict with 'author' and/or 'publisher' keys
                      extracted by query understanding.
            top_k: Max number of results.

        Returns:
            List of (book_dict, score, reasoning) tuples.
        """
        if not self._is_built:
            raise RuntimeError("KG not built. Call build() first.")

        results = []
        seen_books = set()

        # If entities are provided, do targeted lookups
        if entities:
            if entities.get("author"):
                author_results = self.find_books_by_author(
                    entities["author"], threshold=0.5, top_k=top_k
                )
                for book, score, reasoning in author_results:
                    if book["book_id"] not in seen_books:
                        seen_books.add(book["book_id"])
                        results.append((book, score, reasoning))

            if entities.get("publisher"):
                pub_results = self.find_books_by_publisher(
                    entities["publisher"], threshold=0.5, top_k=top_k
                )
                for book, score, reasoning in pub_results:
                    if book["book_id"] not in seen_books:
                        seen_books.add(book["book_id"])
                        results.append((book, score, reasoning))
        else:
            # Try the raw query as both author and publisher search
            author_results = self.find_books_by_author(
                query, threshold=0.5, top_k=top_k
            )
            for book, score, reasoning in author_results:
                if book["book_id"] not in seen_books:
                    seen_books.add(book["book_id"])
                    results.append((book, score, reasoning))

            pub_results = self.find_books_by_publisher(
                query, threshold=0.5, top_k=top_k
            )
            for book, score, reasoning in pub_results:
                if book["book_id"] not in seen_books:
                    seen_books.add(book["book_id"])
                    results.append((book, score, reasoning))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ─── Stats ──────────────────────────────────────────────────────────

    def get_author_stats(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get authors with the most books."""
        stats = [
            (self.author_display.get(a, a), len(books))
            for a, books in self.author_to_books.items()
        ]
        stats.sort(key=lambda x: x[1], reverse=True)
        return stats[:top_n]

    def get_publisher_stats(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get publishers with the most books."""
        stats = [
            (self.publisher_display.get(p, p), len(books))
            for p, books in self.publisher_to_books.items()
        ]
        stats.sort(key=lambda x: x[1], reverse=True)
        return stats[:top_n]

    # ─── Persistence ────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the KG to disk."""
        data = {
            "books": self.books,
            "book_by_id": dict(self.book_by_id),
            "author_to_books": {k: list(v) for k, v in self.author_to_books.items()},
            "publisher_to_books": {k: list(v) for k, v in self.publisher_to_books.items()},
            "book_to_author": self.book_to_author,
            "book_to_publisher": self.book_to_publisher,
            "author_display": self.author_display,
            "publisher_display": self.publisher_display,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[kg_index] KG saved to {path}")

    def load(self, path: str) -> None:
        """Load the KG from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.books = data["books"]
        self.book_by_id = data["book_by_id"]
        self.author_to_books = defaultdict(set, {k: set(v) for k, v in data["author_to_books"].items()})
        self.publisher_to_books = defaultdict(set, {k: set(v) for k, v in data["publisher_to_books"].items()})
        self.book_to_author = data["book_to_author"]
        self.book_to_publisher = data["book_to_publisher"]
        self.author_display = data["author_display"]
        self.publisher_display = data["publisher_display"]
        self._is_built = True
        print(f"[kg_index] KG loaded from {path} ({len(self.books)} books)")


if __name__ == "__main__":
    from data_loader import get_book_display

    books = load_books()
    kg = KnowledgeGraph()
    kg.build(books)

    # Print stats
    print("\n--- Top 10 Authors by Book Count ---")
    for author, count in kg.get_author_stats(10):
        print(f"  {author}: {count} books")

    print("\n--- Top 10 Publishers by Book Count ---")
    for pub, count in kg.get_publisher_stats(10):
        print(f"  {pub}: {count} books")

    # Test author search
    print(f"\n{'='*80}")
    print("🔍 KG Author Search: হুমায়ূন আহমেদ")
    print(f"{'='*80}")
    results = kg.find_books_by_author("হুমায়ূন আহমেদ", top_k=5)
    for i, (book, score, reasoning) in enumerate(results):
        print(f"\n  #{i+1}")
        print(get_book_display(book))
        print(f"     🎯 {reasoning}")

    # Test related books
    if results:
        first_book_id = results[0][0]["book_id"]
        print(f"\n{'='*80}")
        print(f"🔍 KG Related Books for: {results[0][0]['book_name']}")
        print(f"{'='*80}")
        related = kg.find_related_books(first_book_id, top_k=5)
        for i, (book, score, reasoning) in enumerate(related):
            print(f"\n  #{i+1}")
            print(get_book_display(book))
            print(f"     🎯 {reasoning}")
