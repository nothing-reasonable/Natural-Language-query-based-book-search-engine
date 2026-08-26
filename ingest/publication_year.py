"""Pull a publication year out of each book's flap copy.

The flap text is the only evidence we have, and it usually is not about publication at
all -- the years it mentions belong to the events the book covers or to the author's
life. So every year found is scored by the words sitting next to it, and only years with
a publication-shaped cue nearby are kept. Books whose copy says nothing about when they
came out are left blank; a guessed year is worse than no year for a metadata field the
index will treat as fact.

Run as a script to (re)write the Publication Year column of books_cleaned.csv:

    python -m ingest.publication_year [--min-score 6]

`artifacts/publication_year_provenance.csv` records every candidate with its score and
the surrounding text, so a different threshold can be judged without re-reading the CSV.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

BENGALI_DIGITS = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')
YEAR = re.compile(r'(?<!\d)(1[3-9]\d{2}|20[0-2]\d)(?!\d)')

# Cues that mean "the book came out then", weighted by how directly they say publication
# rather than merely sitting in a sentence about the book.
PUB_CUES: list[tuple[int, tuple[str, ...]]] = [
    (10, ('প্রথম প্রকাশ', 'প্রকাশকাল', 'প্রকাশিত হয়', 'প্রকাশ পায়', 'প্রকাশ করা হয়',
          'first published', 'was published', 'published in')),
    (7, ('প্রকাশিত', 'প্রকাশ', 'মুদ্রিত', 'সংস্করণ', 'published', 'edition')),
    (6, ('রচিত', 'রচনা কর', 'লিখিত', 'লেখা হয়', 'লিখেছিলেন', 'written in', 'wrote this')),
    (4, ('গ্রন্থটি', 'বইটি', 'বইখানি', 'গ্রন্থখানি', 'this book', 'the book')),
]
# Cues that mark the year as belonging to the subject matter or the author's life.
SUBJECT_CUES = ('যুদ্ধ', 'জন্ম', 'জন্মগ্রহণ', 'মৃত্যু', 'ইন্তেকাল', 'শহীদ', 'আন্দোলন',
                'অভ্যুত্থান', 'নির্বাচন', 'স্বাধীনতা', 'দেশভাগ', 'গণহত্যা', 'বিপ্লব',
                'সাল থেকে', 'সাল পর্যন্ত', 'born', 'died', 'war')

WINDOW = 70   # characters scanned either side of a year for cues
PENALTY = 4   # knocked off when a subject-matter cue is in range
MIN_SCORE = 6  # below this the only cue is a bare "this book", too weak to trust
MAX_YEAR = 2026


def extract(text: str, max_year: int = MAX_YEAR) -> tuple[int | None, int, str]:
    """Return (year, cue score, surrounding text) for the best-supported year in `text`."""
    normalised = str(text).translate(BENGALI_DIGITS)
    best: tuple[int | None, int, str] = (None, 0, '')
    for match in YEAR.finditer(normalised):
        year = int(match.group(1))
        if not 1400 <= year <= max_year:
            continue
        left = normalised[max(0, match.start() - WINDOW):match.start()].lower()
        right = normalised[match.end():match.end() + WINDOW].lower()
        near = f'{left} {right}'
        score = max((weight for weight, cues in PUB_CUES if any(c in near for c in cues)),
                    default=0)
        if score and any(c in near for c in SUBJECT_CUES):
            score -= PENALTY
        # Ties go to the later year: a reprint line usually follows the first-edition one.
        if score > best[1] or (score == best[1] and best[0] is not None and year > best[0]):
            best = (year, score, f'{left[-40:]}[{match.group(1)}]{right[:40]}'.strip())
    return best if best[1] > 0 else (None, 0, '')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT / 'books_cleaned.csv')
    parser.add_argument('--min-score', type=int, default=MIN_SCORE)
    args = parser.parse_args()

    books = pd.read_csv(args.csv)
    found = [extract(d) for d in books['Description (Flap)']]
    years = [y if y is not None and s >= args.min_score else None for y, s, _ in found]

    books = books.drop(columns=['Publication Year'], errors='ignore')
    books.insert(5, 'Publication Year', pd.array(years, dtype='Int64'))
    books.to_csv(args.csv, index=False)

    provenance = ROOT / 'artifacts' / 'publication_year_provenance.csv'
    provenance.parent.mkdir(exist_ok=True)
    pd.DataFrame({
        'Book Name': books['Book Name'],
        'Author': books['Author'],
        'Publication Year': books['Publication Year'],
        'candidate_year': pd.array([y for y, _, _ in found], dtype='Int64'),
        'cue_score': [s for _, s, _ in found],
        'evidence': [e for _, _, e in found],
    }).to_csv(provenance, index=False)

    filled = books['Publication Year'].notna().sum()
    print(f'{args.csv.name}: {len(books)} rows, {filled} with a year '
          f'({filled / len(books):.1%}), {len(books) - filled} blank')
    print(f'provenance -> {provenance.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
