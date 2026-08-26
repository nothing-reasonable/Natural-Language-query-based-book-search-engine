"""
data_loader.py — Book catalogue data loading and preprocessing.

Loads books_metadata_cleaned.csv into a list of book dicts, each with:
  - book_name, author, author_bio, publisher, description
  - search_text: combined field for indexing (book_name + author + description)
  - book_id: integer index for cross-referencing with indices
"""

import csv
import os
from typing import List, Dict, Optional

# CSV column mapping
CSV_COLUMNS = ["Book Name", "Author", "Author Bio", "Publisher", "Description (Flap)"]

# Internal field names
FIELD_MAP = {
    "Book Name": "book_name",
    "Author": "author",
    "Author Bio": "author_bio",
    "Publisher": "publisher",
    "Description (Flap)": "description",
}

DEFAULT_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "books_metadata_cleaned.csv")


def _clean_field(value: Optional[str]) -> str:
    """Strip whitespace and return empty string for None/NaN."""
    if value is None:
        return ""
    value = str(value).strip()
    if value.lower() in ("nan", "none", ""):
        return ""
    return value


def build_search_text(book: Dict[str, str]) -> str:
    """
    Build the combined searchable text from a book dict.
    Concatenates book_name, author, and description (separated by spaces).
    """
    parts = []
    for field in ("book_name", "author", "description"):
        text = book.get(field, "")
        if text:
            parts.append(text)
    return " ".join(parts)


def load_books(csv_path: str = DEFAULT_CSV_PATH) -> List[Dict[str, str]]:
    """
    Load books from CSV into a list of dicts.

    Each dict has keys:
        book_id (int), book_name, author, author_bio, publisher, description, search_text

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of book dicts.
    """
    books = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            book = {"book_id": idx}
            for csv_col, field_name in FIELD_MAP.items():
                book[field_name] = _clean_field(row.get(csv_col, ""))
            book["search_text"] = build_search_text(book)
            books.append(book)

    print(f"[data_loader] Loaded {len(books)} books from {csv_path}")
    return books


def get_book_display(book: Dict[str, str]) -> str:
    """Format a book for display."""
    lines = []
    lines.append(f"  📖 {book.get('book_name', 'Unknown')}")
    lines.append(f"     ✍️  Author: {book.get('author', 'Unknown')}")
    lines.append(f"     🏢 Publisher: {book.get('publisher', 'Unknown')}")
    desc = book.get("description", "")
    if desc:
        # Truncate long descriptions
        if len(desc) > 200:
            desc = desc[:200] + "..."
        lines.append(f"     📝 Description: {desc}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    books = load_books()
    print(f"\nFirst 3 books:")
    for book in books[:3]:
        print(get_book_display(book))
        print()
