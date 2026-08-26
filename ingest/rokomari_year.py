"""Look up each book's publication year on Rokomari.

The flap copy in books_cleaned.csv almost never says when a book came out, so the year
has to come from outside. Rokomari lists nearly every Bengali title in print and its book
pages carry an `Edition` field ("1st Published, 2020"), which is the closest thing to a
publication date this corpus can be joined against.

Discovery goes through Rokomari's own sitemaps rather than its search endpoint, which
robots.txt disallows. Sitemap slugs are romanised Bengali, so titles are transliterated
and fuzzy-matched to shortlist candidate pages; every candidate is then confirmed against
the Bengali title and author printed on the page itself before its year is accepted. A
slug that merely looks similar is not evidence -- ~15% of top slug matches are a different
book entirely, and only the page can tell them apart.

    python -m ingest.rokomari_year --limit 50      # try a slice first
    python -m ingest.rokomari_year                 # full run, resumable

Progress is appended to artifacts/rokomari/lookups.jsonl; re-running skips what is
already there, so the job can be stopped and restarted freely.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / 'artifacts' / 'rokomari'
LOOKUPS = WORK / 'lookups.jsonl'
SITEMAP_INDEX = 'https://www.rokomari.com/sitemap.xml'

UA = 'Mozilla/5.0 (compatible; book-metadata-research/1.0)'
DELAY = 1.1          # seconds between page fetches -- one connection, well under a crawl
MAX_CANDIDATES = 3   # pages tried per book before giving up
SLUG_CUTOFF = 70     # loose on purpose: the page check below is what guards precision
BENGALI_DIGITS = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')

# --------------------------------------------------------------- transliteration
CONS = {
    'ক': 'k', 'খ': 'kh', 'গ': 'g', 'ঘ': 'gh', 'ঙ': 'ng',
    'চ': 'ch', 'ছ': 'ch', 'জ': 'j', 'ঝ': 'jh', 'ঞ': 'n',
    'ট': 't', 'ঠ': 'th', 'ড': 'd', 'ঢ': 'dh', 'ণ': 'n',
    'ত': 't', 'থ': 'th', 'দ': 'd', 'ধ': 'dh', 'ন': 'n',
    'প': 'p', 'ফ': 'f', 'ব': 'b', 'ভ': 'v', 'ম': 'm',
    'য': 'j', 'র': 'r', 'ল': 'l', 'শ': 'sh', 'ষ': 'sh', 'স': 's', 'হ': 'h',
    'ড়': 'r', 'ঢ়': 'r', 'য়': 'y', 'ৎ': 't', 'ং': 'ng', 'ঃ': '', 'ঁ': '',
}
VOWELS = {'অ': 'o', 'আ': 'a', 'ই': 'i', 'ঈ': 'i', 'উ': 'u', 'ঊ': 'u', 'ঋ': 'ri',
          'এ': 'e', 'ঐ': 'oi', 'ও': 'o', 'ঔ': 'ou'}
SIGNS = {'া': 'a', 'ি': 'i', 'ী': 'i', 'ু': 'u', 'ূ': 'u', 'ৃ': 'ri',
         'ে': 'e', 'ৈ': 'oi', 'ো': 'o', 'ৌ': 'ou'}
HASANT = '্'
BENGALI_DIGITS_SET = set('০১২৩৪৫৬৭৮৯')
_SPLIT = re.compile(r'[\s/,:;।\-–—()\[\]"\'’‘]+')
_FOLD = [('kh', 'k'), ('gh', 'g'), ('ch', 'c'), ('jh', 'j'), ('th', 't'), ('dh', 'd'),
         ('ph', 'f'), ('bh', 'b'), ('sh', 's'), ('ng', 'n'), ('v', 'b'), ('z', 'j'),
         ('w', 'o'), ('y', 'i'), ('q', 'k'), ('x', 'ks')]


def _translit_word(word: str) -> str:
    out, i, n = [], 0, len(word)
    while i < n:
        ch = word[i]
        if ch in CONS:
            out.append(CONS[ch])
            nxt = word[i + 1] if i + 1 < n else ''
            if nxt == HASANT:
                i += 2
            elif nxt in SIGNS:
                out.append(SIGNS[nxt])
                i += 2
            else:
                if i + 1 < n:
                    out.append('o')      # inherent vowel, silent word-finally
                i += 1
        elif ch in VOWELS:
            out.append(VOWELS[ch])
            i += 1
        elif ch in SIGNS:
            out.append(SIGNS[ch])
            i += 1
        elif ch.isdigit() or ch in BENGALI_DIGITS_SET:
            out.append(ch.translate(BENGALI_DIGITS))   # slugs keep numerals as digits
            i += 1
        else:
            i += 1
    return ''.join(out)


def translit(text: str) -> str:
    return '-'.join(filter(None, (_translit_word(w) for w in _SPLIT.split(str(text)))))


def fold(s: str) -> str:
    """Collapse the spellings Bengali romanisation is inconsistent about."""
    s = re.sub(r'[^a-z0-9]', '', str(s).lower())
    for a, b in _FOLD:
        s = s.replace(a, b)
    s = re.sub(r'[aeiou]+', 'a', s)
    return re.sub(r'(.)\1+', r'\1', s)


def key(text: str) -> str:
    """Match key for a title: transliterate Bengali, then fold. Latin text folds directly."""
    t = translit(text)
    return fold(t if t else text)


# --------------------------------------------------------------------- catalogue
def build_catalogue(session: requests.Session) -> pd.DataFrame:
    """Every /book/<id>/<slug> URL Rokomari publishes, from its sitemaps."""
    cached = WORK / 'catalogue.parquet'
    if cached.exists():
        return pd.read_parquet(cached)

    WORK.mkdir(parents=True, exist_ok=True)
    index = session.get(SITEMAP_INDEX, timeout=60).text
    rows = []
    for loc in re.findall(r'<loc>([^<]*product_urls[^<]*\.gz)</loc>', index):
        local = WORK / loc.rsplit('/', 1)[-1]
        if not local.exists():
            local.write_bytes(session.get(loc, timeout=180).content)
        text = gzip.open(local, 'rt', encoding='utf-8').read()
        rows += [(int(i), s) for i, s in
                 re.findall(r'<loc>https://www\.rokomari\.com/book/(\d+)/([^<]*)</loc>', text)]

    cat = pd.DataFrame(rows, columns=['id', 'slug']).drop_duplicates('id')
    cat['key'] = cat['slug'].map(fold)
    cat = cat[cat['key'].str.len() >= 4].reset_index(drop=True)
    cat.to_parquet(cached)
    return cat


def shortlist(books: pd.DataFrame, cat: pd.DataFrame) -> list[list[tuple[int, str, int]]]:
    """Top slug candidates per book, best first."""
    queries = [key(t) for t in books['Book Name']]
    scores = process.cdist(queries, cat['key'].tolist(), scorer=fuzz.ratio,
                           workers=-1, dtype=np.uint8, score_cutoff=SLUG_CUTOFF)
    ids, slugs = cat['id'].values, cat['slug'].values
    out = []
    for row in scores:
        # Negate as a signed type: -row on a uint8 array wraps instead of ordering.
        ranked = np.argsort(-row.astype(np.int16))[:MAX_CANDIDATES]
        out.append([(int(ids[j]), slugs[j], int(row[j])) for j in ranked if row[j] >= SLUG_CUTOFF])
    return out


# ------------------------------------------------------------------- page access
def fetch(session: requests.Session, book_id: int, slug: str, tries: int = 3) -> str | None:
    url = f'https://www.rokomari.com/book/{book_id}/{slug}'
    for attempt in range(tries):
        try:
            r = session.get(url, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** attempt + random.random())
            continue
        if r.status_code == 200:
            return r.text
        if r.status_code in (404, 410):
            return None
        time.sleep(10 * (attempt + 1))       # rate limited or upstream trouble: back off hard
    return None


def parse(html: str) -> dict:
    soup = BeautifulSoup(html, 'lxml')
    hidden = {i.get('id'): i.get('value', '') for i in soup.select('input[type=hidden][id]')}
    spec = {}
    for tr in soup.select('tr'):
        cells = tr.find_all('td')
        if len(cells) == 2:
            spec[cells[0].get_text(strip=True)] = cells[1].get_text(' ', strip=True)

    edition = spec.get('Edition') or spec.get('সংস্করণ') or ''
    years = re.findall(r'(?<!\d)(1[3-9]\d{2}|20[0-3]\d)(?!\d)', edition.translate(BENGALI_DIGITS))
    title = hidden.get('js--product-name') or (soup.h1.get_text(' ', strip=True) if soup.h1 else '')
    return {
        'page_title': title,
        'page_author': hidden.get('js--product-author-name', ''),
        'edition': edition,
        # "3rd Published, 2019, 2022" -- the last year is the printing actually on sale.
        'year': int(years[-1]) if years else None,
        'publisher': spec.get('প্রকাশনী') or spec.get('Publisher') or '',
    }


def confirms(book: pd.Series, page: dict) -> tuple[bool, int, int]:
    """Is this page really the book in our row? Compare what the page itself prints."""
    title_sim = max(
        fuzz.ratio(re.sub(r'\s+', '', str(book['Book Name'])), re.sub(r'\s+', '', page['page_title'])),
        fuzz.ratio(key(book['Book Name']), key(page['page_title'])),
    )
    author_sim = fuzz.token_set_ratio(key(book['Author']), key(page['page_author']))
    accepted = title_sim >= 90 or (title_sim >= 80 and author_sim >= 75)
    return accepted, int(title_sim), int(author_sim)


# --------------------------------------------------------------------------- run
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=ROOT / 'books_cleaned.csv')
    parser.add_argument('--limit', type=int, default=0, help='stop after N books (0 = all)')
    parser.add_argument('--delay', type=float, default=DELAY)
    args = parser.parse_args()

    WORK.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({'User-Agent': UA, 'Accept-Language': 'bn,en;q=0.8'})

    books = pd.read_csv(args.csv)
    cat = build_catalogue(session)
    print(f'catalogue: {len(cat):,} Rokomari book pages', flush=True)

    done = set()
    if LOOKUPS.exists():
        with LOOKUPS.open(encoding='utf-8') as fh:
            done = {json.loads(line)['row'] for line in fh if line.strip()}
    print(f'already looked up: {len(done):,}', flush=True)

    candidates = shortlist(books, cat)
    todo = [i for i in range(len(books)) if i not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f'to fetch: {len(todo):,}', flush=True)

    found = 0
    with LOOKUPS.open('a', encoding='utf-8') as out:
        for n, i in enumerate(todo, 1):
            book = books.iloc[i]
            record = {'row': i, 'Book Name': book['Book Name'], 'Author': book['Author'],
                      'year': None, 'status': 'no_candidate'}
            for book_id, slug, slug_score in candidates[i]:
                html = fetch(session, book_id, slug)
                time.sleep(args.delay)
                if not html:
                    record['status'] = 'fetch_failed'
                    continue
                page = parse(html)
                accepted, title_sim, author_sim = confirms(book, page)
                if not accepted:
                    record['status'] = 'rejected'
                    continue
                record.update(page, status='matched' if page['year'] else 'matched_no_year',
                              url=f'https://www.rokomari.com/book/{book_id}/{slug}',
                              slug_score=slug_score, title_sim=title_sim, author_sim=author_sim)
                break
            found += record['year'] is not None
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            out.flush()
            if n % 50 == 0:
                print(f'{n}/{len(todo)} fetched, {found} years found', flush=True)

    print(f'done: {found} years found in {len(todo)} books', flush=True)


if __name__ == '__main__':
    main()
