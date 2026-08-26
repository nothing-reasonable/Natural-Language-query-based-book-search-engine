"""
bm25_index.py — BM25 Lexical Index for book search (Step 2.1).

Uses rank_bm25.BM25Okapi with the BanglaBERT tokenizer
(csebuetnlp/banglabert) for proper subword tokenization of Bangla text.
Falls back to regex-based tokenization if the model is unavailable.

The index operates over the combined search_text field
(book_name + author + description).
"""

import os
import re
import pickle
import time
from typing import List, Dict, Tuple, Optional

from rank_bm25 import BM25Okapi

# Try to import BanglaBERT tokenizer for advanced Bangla tokenization
try:
    from transformers import AutoTokenizer
    _bangla_tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
    BANGLA_TOKENIZER_AVAILABLE = True
    print("[bm25_index] Using BanglaBERT (csebuetnlp/banglabert) tokenizer.")
except Exception:
    BANGLA_TOKENIZER_AVAILABLE = False
    print("[bm25_index] BanglaBERT tokenizer not available. Using regex-based tokenizer.")

# Regex pattern for fallback tokenization:
# Matches Bangla word characters, Latin word characters, or digits
_BANGLA_TOKEN_PATTERN = re.compile(
    r'[\u0980-\u09FF]+|[a-zA-Z]+|[0-9]+'
)

# Common Bangla stop words to filter out
BANGLA_STOP_WORDS = {
    'এ', 'এই', 'এক', 'একটি', 'একটু', 'একবার', 'এটা', 'এটি', 'এত',
    'এতে', 'এবং', 'এমন', 'এর', 'এস', 'ও', 'ওই', 'ওর', 'কই', 'কখনো',
    'কত', 'কবে', 'কয়েক', 'কর', 'করতে', 'করা', 'করার', 'করি', 'করিয়া',
    'করে', 'করেছে', 'করেন', 'কাজ', 'কাজে', 'কারণ', 'কি', 'কিংবা',
    'কিছু', 'কিন্তু', 'কে', 'কেউ', 'কেন', 'কোন', 'কোনো', 'ক্ষেত্রে',
    'গিয়ে', 'গুলি', 'গুলো', 'গেছে', 'গেল', 'গেলে', 'চেয়ে', 'ছিল',
    'ছিলেন', 'জন', 'জন্য', 'জানা', 'জানে', 'তখন', 'তত', 'তবে', 'তবু',
    'তা', 'তাই', 'তাকে', 'তাতে', 'তাদের', 'তার', 'তারা', 'তারপর',
    'তাহলে', 'তাহা', 'তাহাতে', 'তিনি', 'তুমি', 'তো', 'তোমার', 'থাকা',
    'থাকে', 'থেকে', 'দিকে', 'দিন', 'দিয়ে', 'দিয়েছে', 'দুই', 'দ্বারা',
    'ধরে', 'নয়', 'নাই', 'নানা', 'নিজ', 'নিজে', 'নিয়ে', 'নেই', 'পক্ষে',
    'পর', 'পরে', 'পাওয়া', 'প্রতি', 'প্রভৃতি', 'প্রায়', 'বলে', 'বসে',
    'বা', 'বাদে', 'বার', 'বিষয়', 'বিভিন্ন', 'বে', 'ব্যবহার', 'মত',
    'মতো', 'মধ্যে', 'মনে', 'মাঝে', 'যখন', 'যত', 'যদি', 'যদিও', 'যা',
    'যাওয়া', 'যায়', 'যার', 'যারা', 'যে', 'যেন', 'যেমন', 'রকম',
    'রয়েছে', 'রাখা', 'লাগে', 'সঙ্গে', 'সব', 'সবার', 'সমস্ত', 'সম্পর্কে',
    'সহ', 'সাথে', 'সেই', 'সেখানে', 'সে', 'সব', 'হইতে', 'হচ্ছে', 'হতে',
    'হন', 'হবে', 'হয়', 'হয়ে', 'হয়েছে', 'হলে', 'হলো', 'হিসেবে',
    # BanglaBERT special tokens to filter
    '[UNK]', '[CLS]', '[SEP]', '[PAD]', '[MASK]',
    # Common English stop words that may appear in the data
    'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
}


def tokenize_bangla(text: str, use_advanced: bool = True) -> List[str]:
    """
    Tokenize Bangla text using BanglaBERT tokenizer or fallback regex.

    BanglaBERT (csebuetnlp/banglabert) uses a WordPiece tokenizer trained
    on Bangla text, providing proper subword segmentation. Subword tokens
    starting with '##' are merged back with their parent token for BM25,
    since BM25 works best with whole-word tokens.

    Args:
        text: Input text (Bangla/mixed).
        use_advanced: If True, use BanglaBERT tokenizer when available.

    Returns:
        List of tokens with stop words removed.
    """
    if not text:
        return []

    if use_advanced and BANGLA_TOKENIZER_AVAILABLE:
        # Use BanglaBERT WordPiece tokenizer
        try:
            raw_tokens = _bangla_tokenizer.tokenize(text)
            # Merge subword tokens (## prefix) back into whole words
            # BM25 benefits from whole-word matching rather than subword
            tokens = []
            for token in raw_tokens:
                if token.startswith("##") and tokens:
                    # Merge subword with previous token
                    tokens[-1] = tokens[-1] + token[2:]
                else:
                    tokens.append(token)
        except Exception:
            # Fall back to regex on any error
            tokens = _BANGLA_TOKEN_PATTERN.findall(text)
    else:
        # Regex-based tokenization
        tokens = _BANGLA_TOKEN_PATTERN.findall(text)

    # Lowercase (for Latin characters) and filter stop words
    tokens = [t.lower() for t in tokens if t.lower() not in BANGLA_STOP_WORDS]
    return tokens


class BM25Index:
    """
    BM25 lexical search index over books.

    Builds a BM25Okapi index on tokenized search_text of each book.
    Supports search, save, and load operations.
    """

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.books: List[Dict[str, str]] = []
        self.tokenized_corpus: List[List[str]] = []
        self._is_built = False

    def build(self, books: List[Dict[str, str]]) -> None:
        """
        Build the BM25 index from a list of book dicts.

        Args:
            books: List of book dicts (must have 'search_text' field).
        """
        print("[bm25_index] Building BM25 index...")
        start = time.time()

        self.books = books
        self.tokenized_corpus = [
            tokenize_bangla(book["search_text"]) for book in books
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._is_built = True

        elapsed = time.time() - start
        print(f"[bm25_index] Index built over {len(books)} books in {elapsed:.2f}s")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict[str, str], float, str]]:
        """
        Search the BM25 index.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of (book_dict, score, reasoning) tuples, sorted by score descending.
            reasoning: string explaining why the BM25 matched this result.
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build() first.")

        query_tokens = tokenize_bangla(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices (sorted by score descending)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include positive-scoring results
                book = self.books[idx]
                score = float(scores[idx])

                # Build retrieval reasoning
                reasoning = self._build_reasoning(query_tokens, idx, score)
                results.append((book, score, reasoning))

        return results

    def _build_reasoning(self, query_tokens: List[str], doc_idx: int, score: float) -> str:
        """
        Build a reasoning explanation for why a document matched the query.

        Identifies which query terms were found in the document and where.
        """
        doc_tokens = self.tokenized_corpus[doc_idx]
        book = self.books[doc_idx]
        doc_tokens_set = set(doc_tokens)

        # Find matched terms
        matched_terms = [t for t in query_tokens if t in doc_tokens_set]
        unmatched_terms = [t for t in query_tokens if t not in doc_tokens_set]

        # Count term frequencies
        term_freqs = {}
        for term in matched_terms:
            freq = doc_tokens.count(term)
            term_freqs[term] = freq

        # Determine which fields contain matches
        matched_fields = []
        for field_name, field_key in [("Book Name", "book_name"), ("Author", "author"), ("Description", "description")]:
            field_text = book.get(field_key, "").lower()
            for term in matched_terms:
                if term in field_text:
                    matched_fields.append(field_name)
                    break

        # Compose reasoning
        parts = []
        parts.append(f"[BM25] Score: {score:.4f}")

        if matched_terms:
            term_info = ", ".join([f"'{t}' (×{term_freqs[t]})" for t in matched_terms])
            parts.append(f"Matched terms: {term_info}")

        if matched_fields:
            parts.append(f"Matched in: {', '.join(matched_fields)}")

        if unmatched_terms:
            parts.append(f"Unmatched terms: {', '.join(unmatched_terms)}")

        return " | ".join(parts)

    def save(self, path: str) -> None:
        """Save the index to disk."""
        data = {
            "books": self.books,
            "tokenized_corpus": self.tokenized_corpus,
            "bm25": self.bm25,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[bm25_index] Index saved to {path}")

    def load(self, path: str) -> None:
        """Load the index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.books = data["books"]
        self.tokenized_corpus = data["tokenized_corpus"]
        self.bm25 = data["bm25"]
        self._is_built = True
        print(f"[bm25_index] Index loaded from {path} ({len(self.books)} books)")


if __name__ == "__main__":
    from data_loader import load_books, get_book_display

    books = load_books()
    index = BM25Index()
    index.build(books)

    # Test queries
    test_queries = [
        "মুক্তিযুদ্ধ",
        "বাংলাদেশের স্বাধীনতা",
        "একাত্তর",
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"🔍 BM25 Query: {query}")
        print(f"{'='*80}")
        results = index.search(query, top_k=5)
        for i, (book, score, reasoning) in enumerate(results):
            print(f"\n  #{i+1}")
            print(get_book_display(book))
            print(f"     🎯 {reasoning}")
