"""
dense_index.py — Dense Vector Index for book search (Step 2.2).

Uses sentence-transformers to encode book text into dense vectors,
and FAISS IndexFlatIP (cosine similarity via L2-normalized vectors)
for fast nearest-neighbor retrieval.

Embedding model: paraphrase-multilingual-MiniLM-L12-v2
  - Supports 50+ languages including Bangla
  - 384-dimensional embeddings
  - Good balance of quality and speed for ~5K documents
"""

import os
import time
import json
import numpy as np
from typing import List, Dict, Tuple, Optional

import faiss

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("[dense_index] WARNING: sentence-transformers not installed. "
          "Install with: pip install sentence-transformers")


# Default embedding model — multilingual, supports Bangla
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Directory for cached indices
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".index_cache")


class DenseIndex:
    """
    Dense vector search index using sentence-transformers + FAISS.

    Encodes book text into dense vectors and builds a FAISS index
    for fast cosine-similarity search.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """
        Initialize the dense index.

        Args:
            model_name: HuggingFace model name for sentence embeddings.
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.books: List[Dict[str, str]] = []
        self.embeddings: Optional[np.ndarray] = None
        self._is_built = False

    def _load_model(self) -> None:
        """Load the sentence-transformer model."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            )
        if self.model is None:
            print(f"[dense_index] Loading embedding model: {self.model_name}")
            start = time.time()
            self.model = SentenceTransformer(self.model_name)
            elapsed = time.time() - start
            print(f"[dense_index] Model loaded in {elapsed:.2f}s")

    def build(self, books: List[Dict[str, str]], batch_size: int = 64,
              show_progress: bool = True) -> None:
        """
        Build the FAISS index from book texts.

        Encodes all book search_text fields into dense vectors,
        L2-normalizes them, and builds a FAISS IndexFlatIP index
        (inner product on normalized vectors = cosine similarity).

        Args:
            books: List of book dicts (must have 'search_text' field).
            batch_size: Batch size for encoding.
            show_progress: Show progress bar during encoding.
        """
        self._load_model()

        print(f"[dense_index] Encoding {len(books)} books...")
        start = time.time()

        self.books = books
        texts = [book["search_text"] for book in books]

        # Encode all texts
        self.embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize for cosine similarity
        )

        encoding_time = time.time() - start
        print(f"[dense_index] Encoded {len(books)} books in {encoding_time:.2f}s")
        print(f"[dense_index] Embedding shape: {self.embeddings.shape}")

        # Build FAISS index (inner product = cosine sim on normalized vectors)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))

        self._is_built = True
        print(f"[dense_index] FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict[str, str], float, str]]:
        """
        Search the dense index with a query.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of (book_dict, similarity_score, reasoning) tuples,
            sorted by similarity descending.
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build() first.")

        self._load_model()

        # Encode the query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # Search FAISS
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score > 0:  # Valid results only
                book = self.books[idx]
                reasoning = self._build_reasoning(query, book, float(score))
                results.append((book, float(score), reasoning))

        return results

    def _build_reasoning(self, query: str, book: Dict[str, str], score: float) -> str:
        """
        Build a reasoning explanation for why a document matched semantically.

        Provides the semantic similarity score and identifies which parts
        of the book metadata are most relevant.
        """
        parts = []
        parts.append(f"[Dense/Semantic] Cosine similarity: {score:.4f}")

        # Identify which fields likely contributed to the match
        query_lower = query.lower()
        relevant_fields = []

        # Simple heuristic: check if query terms appear in fields
        # (for semantic matches, the actual relevance comes from embeddings,
        # but we note lexical overlaps as additional signal)
        for field_name, field_key in [("Book Name", "book_name"), ("Author", "author"), ("Description", "description")]:
            field_text = book.get(field_key, "").lower()
            if field_text:
                # Check if any query word appears in the field
                query_words = query_lower.split()
                overlapping = [w for w in query_words if w in field_text]
                if overlapping:
                    relevant_fields.append(f"{field_name} (lexical overlap: {', '.join(overlapping)})")

        if relevant_fields:
            parts.append(f"Field relevance: {'; '.join(relevant_fields)}")
        else:
            parts.append("Match type: Pure semantic similarity (no lexical overlap)")

        # Score interpretation
        if score >= 0.7:
            parts.append("Confidence: Very High")
        elif score >= 0.5:
            parts.append("Confidence: High")
        elif score >= 0.3:
            parts.append("Confidence: Moderate")
        else:
            parts.append("Confidence: Low")

        return " | ".join(parts)

    def save(self, cache_dir: str = CACHE_DIR) -> None:
        """
        Save the FAISS index and metadata to disk.

        Args:
            cache_dir: Directory to save cache files.
        """
        os.makedirs(cache_dir, exist_ok=True)

        # Save FAISS index
        faiss_path = os.path.join(cache_dir, "dense_index.faiss")
        faiss.write_index(self.index, faiss_path)

        # Save embeddings
        emb_path = os.path.join(cache_dir, "embeddings.npy")
        np.save(emb_path, self.embeddings)

        # Save book IDs (to verify alignment on load)
        meta_path = os.path.join(cache_dir, "dense_meta.json")
        meta = {
            "model_name": self.model_name,
            "num_books": len(self.books),
            "embedding_dim": int(self.embeddings.shape[1]),
            "book_ids": [b["book_id"] for b in self.books],
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        print(f"[dense_index] Index saved to {cache_dir}")

    def load(self, books: List[Dict[str, str]], cache_dir: str = CACHE_DIR) -> bool:
        """
        Load a cached FAISS index from disk.

        Args:
            books: The book list (must match the cached index).
            cache_dir: Directory with cached files.

        Returns:
            True if loaded successfully, False if cache not found or invalid.
        """
        faiss_path = os.path.join(cache_dir, "dense_index.faiss")
        emb_path = os.path.join(cache_dir, "embeddings.npy")
        meta_path = os.path.join(cache_dir, "dense_meta.json")

        if not all(os.path.exists(p) for p in [faiss_path, emb_path, meta_path]):
            print("[dense_index] No cached index found.")
            return False

        # Verify metadata matches
        with open(meta_path, "r") as f:
            meta = json.load(f)

        if meta["num_books"] != len(books) or meta["model_name"] != self.model_name:
            print("[dense_index] Cached index doesn't match current data/model. Rebuilding.")
            return False

        # Load FAISS index and embeddings
        self.index = faiss.read_index(faiss_path)
        self.embeddings = np.load(emb_path)
        self.books = books
        self._is_built = True

        print(f"[dense_index] Loaded cached index: {self.index.ntotal} vectors")
        return True


if __name__ == "__main__":
    from data_loader import load_books, get_book_display

    books = load_books()
    index = DenseIndex()

    # Try loading from cache first
    if not index.load(books):
        index.build(books)
        index.save()

    # Test queries
    test_queries = [
        "মুক্তিযুদ্ধ",
        "বাংলাদেশের স্বাধীনতা",
        "একাত্তর",
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"🔍 Dense Search Query: {query}")
        print(f"{'='*80}")
        results = index.search(query, top_k=5)
        for i, (book, score, reasoning) in enumerate(results):
            print(f"\n  #{i+1}")
            print(get_book_display(book))
            print(f"     🎯 {reasoning}")
