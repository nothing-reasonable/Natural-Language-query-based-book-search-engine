"""Normalisation, de-duplication and author alias resolution.

Author names in the catalogue carry honorifics and spelling drift
("ড. আনিসুজ্জামান", "অধ্যাপক আনিসুজ্জামান", "মোঃ" vs "মো."). Identity resolution runs in
three passes, most trustworthy first:

  1. curated overrides from `data/author_aliases.yaml`
  2. exact match on the honorific-stripped, analysed key  (safe, does most of the work)
  3. conservative fuzzy merge, with every decision written out for human review
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from rapidfuzz import fuzz, process

from search.core import bengali
from config import settings
from search.core.schemas import Book

# Titles / honorifics that must not take part in identity matching.
HONORIFICS: frozenset[str] = frozenset(
    """
    ড ডঃ ডক্টর ডা অধ্যাপক প্রফেসর শিক্ষক গবেষক লেখক কবি সম্পাদক অনুবাদক সংকলক
    মাওলানা হযরত মুফতি শায়খ আল্লামা মুহাদ্দিস মুফাক্কিরে হাফেজ ইমাম পীর
    ইঞ্জিনিয়ার স্থপতি অ্যাডভোকেট ব্যারিস্টার বিচারপতি
    লেফটেন্যান্ট মেজর কর্নেল ব্রিগেডিয়ার জেনারেল ক্যাপ্টেন কমান্ডার সিপাহি
    বীর মুক্তিযোদ্ধা শহীদ প্রয়াত জনাব
    রহ রাহ রহমাতুল্লাহি আলাইহি দাঃবাঃ বিইএম পিএসসি এমপি
    dr prof professor mr mrs md
    """.split()
)

# Storefront boilerplate. Roughly half of this catalogue was scraped from a retail site
# whose "description" is an advertisement template -- "<author> এর <title> অরিজিনাল বইটি
# সংগ্রহ করুন রকমারি ডট কম থেকে। ... ফ্রি শিপিং এবং সর্বোচ্চ ছাড়!" -- and nothing else.
#
# Left in, it does real damage: those 2,300 books end up with near-identical description
# text, so their embeddings collapse towards each other and the dense channel starts
# retrieving on shipping terms. It also makes `metadata_quality` claim a description that
# carries no information about the book.
#
# Matching is per sentence, so a genuine blurb that happens to end with a sales line keeps
# its real content.
BOILERPLATE_MARKERS: tuple[str, ...] = (
    "রকমারি", "ফ্রি শিপিং", "মূল্য পরিশোধ", "অফারভেদ", "অরিজিনাল বই",
    "সর্বোচ্চ ছাড়", "সংগ্রহ করুন",
    "rokomari", "free shipping", "cash on delivery", "eligible purchase",
    "extra offer", "original book",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[।!?])\s*")


def strip_boilerplate(text: str) -> str:
    """Drop advertising sentences, keep everything that says something about the book."""
    if not text:
        return ""
    lowered_markers = BOILERPLATE_MARKERS
    kept = []
    for sentence in _SENTENCE_SPLIT.split(text):
        probe = sentence.lower()
        if any(marker in probe for marker in lowered_markers):
            continue
        if sentence.strip():
            kept.append(sentence.strip())
    return " ".join(kept).strip()


# Fields that count towards `metadata_quality`.
QUALITY_FIELDS = ("title", "author", "author_bio", "publisher", "description",
                  "publish_year", "table_of_contents")

ALIAS_FILE = Path(__file__).resolve().parent.parent / "data" / "author_aliases.yaml"


# --------------------------------------------------------------------------- normalisation

def normalize_book(book: Book) -> Book:
    data = book.model_dump()
    for field, value in data.items():
        if isinstance(value, str):
            data[field] = bengali.normalize(value)
    # Strip the storefront copy *before* scoring quality, so a book whose only
    # "description" was an advert is correctly recorded as having none.
    for field in ("description", "author_bio"):
        data[field] = strip_boilerplate(data.get(field, ""))
    book = Book(**data)
    book.metadata_quality = _quality(book)
    return book


def _quality(book: Book) -> float:
    filled = sum(1 for f in QUALITY_FIELDS if getattr(book, f, None) not in (None, "", 0))
    return round(filled / len(QUALITY_FIELDS), 3)


# --------------------------------------------------------------------------- author identity

def author_key(name: str) -> str:
    """Order-insensitive identity key: honorifics dropped, tokens stemmed and sorted."""
    tokens = [t for t in bengali.analyze(name) if t not in HONORIFICS]
    return " ".join(sorted(set(tokens)))


def _load_manual_aliases() -> dict[str, str]:
    """alias key -> canonical display name, from the curated YAML file."""
    if not ALIAS_FILE.exists():
        return {}
    raw = yaml.safe_load(ALIAS_FILE.read_text(encoding="utf-8")) or {}
    mapping: dict[str, str] = {}
    for canonical, aliases in raw.items():
        for surface in [canonical, *(aliases or [])]:
            key = author_key(surface)
            if key:
                mapping[key] = canonical
    return mapping


def _identity_key(raw: str, manual: dict[str, str]) -> str:
    """Key a name is filed under: the curated canonical name if there is one, else itself."""
    key = author_key(raw)
    canonical = manual.get(key)
    return author_key(canonical) if canonical else key


def resolve_authors(books: list[Book], threshold: int | None = None,
                    review_path: Path | None = None) -> dict[str, str]:
    """Assign `author_id` and canonical `author` in place. Returns id -> canonical name."""
    threshold = settings.author_alias_threshold if threshold is None else threshold
    manual = _load_manual_aliases()

    surfaces_by_key: dict[str, Counter] = defaultdict(Counter)
    for book in books:
        raw = bengali.normalize(book.author_raw or book.author)
        surfaces_by_key[_identity_key(raw, manual)][raw] += 1

    keys = [k for k in surfaces_by_key if k]
    clusters, fuzzy_merges = _cluster(keys, threshold)

    key_to_id: dict[str, str] = {}
    id_to_name: dict[str, str] = {}
    for members in clusters:
        surfaces: Counter = Counter()
        for key in members:
            surfaces.update(surfaces_by_key[key])
        curated = next((manual[m] for m in members if m in manual), None)
        canonical = curated or surfaces.most_common(1)[0][0]
        author_id = hashlib.sha1(min(members).encode("utf-8")).hexdigest()[:12]
        id_to_name[author_id] = canonical
        for key in members:
            key_to_id[key] = author_id

    for book in books:
        raw = bengali.normalize(book.author_raw or book.author)
        author_id = key_to_id.get(_identity_key(raw, manual), "")
        book.author_id = author_id
        book.author = id_to_name.get(author_id, raw)

    if review_path is not None and fuzzy_merges:
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(
            json.dumps(fuzzy_merges, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return id_to_name


def _cluster(keys: list[str], threshold: int) -> tuple[list[list[str]], list[dict]]:
    """Union-find over rapidfuzz token-sort similarity. Also returns an audit trail."""
    # Single short tokens are too ambiguous to merge on ("রফিক" matches half the catalogue).
    mergeable = sorted(k for k in keys if len(k) >= 8)
    singles = [[k] for k in keys if len(k) < 8]
    if len(mergeable) < 2:
        return singles + [[k] for k in mergeable], []

    parent = {k: k for k in mergeable}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    audit: list[dict] = []
    matrix = process.cdist(mergeable, mergeable, scorer=fuzz.token_sort_ratio, workers=-1)
    for i, key in enumerate(mergeable):
        for j in range(i + 1, len(mergeable)):
            score = matrix[i][j]
            if score >= threshold:
                a, b = find(key), find(mergeable[j])
                if a != b:
                    parent[b] = a
                    audit.append({"a": key, "b": mergeable[j], "score": float(score)})

    grouped: dict[str, list[str]] = defaultdict(list)
    for key in mergeable:
        grouped[find(key)].append(key)
    return list(grouped.values()) + singles, audit


# --------------------------------------------------------------------------- de-duplication

def dedupe_books(books: list[Book]) -> list[Book]:
    """Same work by the same person = one record; the richest version wins."""
    merged: dict[tuple[str, str], Book] = {}
    for book in books:
        key = (bengali.key(book.title), book.author_id or bengali.key(book.author))
        current = merged.get(key)
        merged[key] = book if current is None else _merge(current, book)
    return list(merged.values())


def _merge(a: Book, b: Book) -> Book:
    winner, loser = (a, b) if a.metadata_quality >= b.metadata_quality else (b, a)
    data = winner.model_dump()
    for field, value in loser.model_dump().items():
        if data.get(field) in (None, "", 0) and value not in (None, "", 0):
            data[field] = value
        elif field in ("description", "author_bio") and len(str(value)) > len(str(data.get(field, ""))):
            data[field] = value  # keep the longest free-text version
    book = Book(**data)
    book.metadata_quality = _quality(book)
    return book


def clean(books: list[Book], review_path: Path | None = None) -> list[Book]:
    books = [normalize_book(b) for b in books]
    resolve_authors(books, review_path=review_path)
    return dedupe_books(books)
