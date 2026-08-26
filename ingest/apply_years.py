"""Write the Publication Year column of books_cleaned.csv from every source we have.

Two sources, in priority order:

  1. Rokomari's `Edition` field (ingest/rokomari_year.py), when the page it came from was
     confirmed to be this book. "1st Published, 2020" is a real publication date.
  2. The flap copy itself (ingest/publication_year.py), when it states when the book was
     written or published.

Rokomari wins wherever it has anything, even when it only lists a later printing. The
tempting rule -- prefer an earlier year from the flap, since a reprint date is not when
the book came out -- was tried and dropped: the flap years it promoted were mostly the
book's *subject* range ("আধুনিক ইউরোপের ইতিহাস ১৭৮৯-১৯৪৫" is not a 1945 book) rather than
a real first edition. A confirmed printing year beats a year the flap never claimed.
`is_first_edition` in the provenance file marks which Rokomari years are first editions,
and `flap_year` keeps the flap's reading for every row regardless. Books neither source
can speak to stay blank.

    python -m ingest.apply_years
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from ingest.publication_year import extract

ROOT = Path(__file__).resolve().parent.parent
LOOKUPS = ROOT / 'artifacts' / 'rokomari' / 'lookups.jsonl'
PROVENANCE = ROOT / 'artifacts' / 'publication_year_provenance.csv'
FLAP_MIN_SCORE = 6
FIRST_EDITION_HINTS = ('1st', 'first', 'প্রথম')

# Multi-volume sets are the one place the page check is not enough: "পুঁজি ১ম খণ্ড" and
# "পুঁজি ২য় খণ্ড" differ by one word out of five, so the titles score as a match while
# being different books with different years. Compare the volume words directly.
VOLUME_WORD = re.compile(r'খণ্ড|খন্ড|পর্ব|part|volume|vol\.?', re.IGNORECASE)
ORDINALS = {
    'প্রথম': 1, '১ম': 1, '1st': 1, 'দ্বিতীয়': 2, '২য়': 2, '2nd': 2,
    'তৃতীয়': 3, '৩য়': 3, '3rd': 3, 'চতুর্থ': 4, '৪র্থ': 4, '4th': 4,
    'পঞ্চম': 5, '৫ম': 5, '5th': 5, 'ষষ্ঠ': 6, '৬ষ্ঠ': 6, '6th': 6,
    'সপ্তম': 7, '৭ম': 7, '7th': 7, 'অষ্টম': 8, '৮ম': 8, '8th': 8,
    'নবম': 9, '৯ম': 9, '9th': 9, 'দশম': 10, '১০ম': 10, '10th': 10,
}


def volume_of(title: str) -> int | None:
    """Which volume of a set this title names, if it names one."""
    text = str(title)
    if not VOLUME_WORD.search(text):
        return None
    for token, number in ORDINALS.items():
        if token in text:
            return number
    digits = re.findall(r'(?<!\d)(\d{1,2})(?!\d)',
                        text.translate(str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')))
    return int(digits[0]) if digits else None


def same_volume(ours: str, theirs: str) -> bool:
    a, b = volume_of(ours), volume_of(theirs)
    return a is None or b is None or a == b


def is_first_edition(edition: str) -> bool:
    return any(h in str(edition).lower() for h in FIRST_EDITION_HINTS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT / 'books_cleaned.csv')
    args = parser.parse_args()

    books = pd.read_csv(args.csv)

    flap = [extract(d) for d in books['Description (Flap)']]
    flap_year = [y if y is not None and s >= FLAP_MIN_SCORE else None for y, s, _ in flap]

    rok: dict[int, dict] = {}
    if LOOKUPS.exists():
        with LOOKUPS.open(encoding='utf-8') as fh:
            for line in fh:
                if line.strip():
                    record = json.loads(line)
                    rok[record['row']] = record

    years, sources, editions, urls = [], [], [], []
    for i in range(len(books)):
        record = rok.get(i) or {}
        if record.get('year') and not same_volume(books['Book Name'][i], record.get('page_title', '')):
            record = {}          # right series, wrong volume -- not this book's year
        rok_year, edition = record.get('year'), record.get('edition', '')
        year, source = None, ''
        if rok_year:
            year, source = rok_year, 'rokomari'
        elif flap_year[i] is not None:
            year, source = flap_year[i], 'flap'
        years.append(year)
        sources.append(source)
        editions.append(edition)
        urls.append(record.get('url', ''))

    books = books.drop(columns=['Publication Year'], errors='ignore')
    books.insert(5, 'Publication Year', pd.array(years, dtype='Int64'))
    books.to_csv(args.csv, index=False)

    PROVENANCE.parent.mkdir(exist_ok=True)
    pd.DataFrame({
        'Book Name': books['Book Name'],
        'Author': books['Author'],
        'Publication Year': books['Publication Year'],
        'source': sources,
        'is_first_edition': [is_first_edition(e) if e else '' for e in editions],
        'rokomari_edition': editions,
        'rokomari_url': urls,
        'flap_year': pd.array(flap_year, dtype='Int64'),
        'flap_cue_score': [s for _, s, _ in flap],
        'flap_evidence': [e for _, _, e in flap],
    }).to_csv(PROVENANCE, index=False)

    filled = books['Publication Year'].notna().sum()
    counts = pd.Series(sources).value_counts()
    print(f'{args.csv.name}: {len(books)} rows, {filled} with a year ({filled / len(books):.1%})')
    print(counts[counts.index != ''].to_string())
    print(f'provenance -> {PROVENANCE.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
