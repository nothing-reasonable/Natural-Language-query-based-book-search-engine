"""The single place that decides how each field participates in search.

Adding `publish_year` or `table_of_contents` to the search stack is one line here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextField:
    name: str  # attribute on Book or Enrichment
    label: str  # Bengali label used in explanations
    lexical_weight: int = 1  # how many times the field's tokens are repeated in the BM25 doc (0 = skip)
    embed: bool = False  # include in the text we embed
    embed_chars: int = 1200  # truncation budget inside the embedding text


# Order is not cosmetic: it is the order of the embedding text, and `embedding_max_tokens`
# truncates the tail. The three fields that come from the catalogue itself and identify a
# particular book -- title, author, flap -- go first, so they are inside the budget on
# every book; the model-inferred facets follow and are what a tight budget costs.
#
# The flap used to sit eleventh, behind ten inferred fields, which is how the one field
# populated for 100% of both source CSVs ended up as the least likely to be embedded.
#
# `lexical_weight` follows the same principle: no inferred field may outweigh the book's
# own text. A `subjects` tag at weight 3 beating the flap at weight 1 meant BM25 scored a
# model's guess about a book above what the book says about itself.
TEXT_FIELDS: list[TextField] = [
    TextField("title", "শিরোনাম", lexical_weight=5, embed=True, embed_chars=300),
    TextField("author", "লেখক", lexical_weight=3, embed=True, embed_chars=200),
    TextField("description", "ফ্ল্যাপ", lexical_weight=3, embed=True, embed_chars=1200),
    TextField("subjects", "বিষয়", lexical_weight=2, embed=True, embed_chars=300),
    TextField("topics", "প্রসঙ্গ", lexical_weight=1, embed=True, embed_chars=300),
    TextField("genres", "ধরন", lexical_weight=1, embed=True, embed_chars=200),
    TextField("periods", "কাল", lexical_weight=1, embed=True, embed_chars=200),
    TextField("events", "ঘটনা", lexical_weight=1, embed=True, embed_chars=200),
    TextField("places", "স্থান", lexical_weight=1, embed=True, embed_chars=200),
    TextField("persons", "ব্যক্তি", lexical_weight=1, embed=True, embed_chars=200),
    TextField("author_roles", "লেখকের ভূমিকা", lexical_weight=1, embed=True, embed_chars=200),
    TextField("publisher", "প্রকাশক", lexical_weight=1, embed=False),
    # About the *author*, not the book, and byte-identical across every book that author
    # wrote. Embedded, it pulled a whole bibliography onto one point in the vector space;
    # in BM25 it made all of an author's books match a query about their biography. Left
    # out of both. `derive.py` still reads `book.author_bio` directly to spot author
    # roles, so multi-hop author queries are unaffected.
    TextField("author_bio", "লেখক পরিচিতি", lexical_weight=0, embed=False),
    TextField("table_of_contents", "সূচিপত্র", lexical_weight=1, embed=True, embed_chars=800),
]

LEXICAL_FIELDS = [f for f in TEXT_FIELDS if f.lexical_weight > 0]
EMBED_FIELDS = [f for f in TEXT_FIELDS if f.embed]

# Columns kept in the vector store for pre-filtering. Scalars and list-of-string only.
FACET_FIELDS: list[str] = [
    "author_id",
    "publisher",
    "language",
    "publish_year",
    "genres",
    "subjects",
    "periods",
    "places",
]


def embedding_text(record) -> str:
    """`record` is an IndexedBook. Produces the string handed to the embedding model."""
    parts = []
    for f in EMBED_FIELDS:
        text = record.field_text(f.name).strip()
        if text:
            parts.append(f"{f.label}: {text[: f.embed_chars]}")
    return "\n".join(parts)


def lexical_tokens(record, analyze) -> list[str]:
    """Field weighting via token repetition -- simple, and equivalent to a weighted
    field-sum BM25 for our purposes. `analyze` is bengali.analyze."""
    tokens: list[str] = []
    for f in LEXICAL_FIELDS:
        text = record.field_text(f.name)
        if not text:
            continue
        field_tokens = analyze(text)
        tokens.extend(field_tokens * f.lexical_weight)
    return tokens
