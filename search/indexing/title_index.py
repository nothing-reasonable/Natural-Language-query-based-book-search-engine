"""Normalised exact-title lookup kept separate from fuzzy lexical retrieval."""

from __future__ import annotations

import re
from collections import defaultdict

from search.core import bengali
from search.core.schemas import IndexedBook

_QUOTED = re.compile(r"[\"'“‘]([^\"'”’]{2,})[\"'”’]")
_EDGE_WORDS = {
    "বই", "বইটি", "গ্রন্থ", "নামের", "নাম", "খুঁজছি", "খুঁজুন",
    "book", "title", "titled", "named", "find", "show",
}


def title_key(text: str) -> str:
    return " ".join(bengali.tokenize(text))


class ExactTitleIndex:
    def __init__(self, records: list[IndexedBook]):
        self.by_key: dict[str, list[str]] = defaultdict(list)
        for record in records:
            key = title_key(record.book.title)
            if key:
                self.by_key[key].append(record.book_id)
        for ids in self.by_key.values():
            ids.sort()

    def find(self, query: str) -> list[str]:
        keys = [title_key(query)]
        keys.extend(title_key(match) for match in _QUOTED.findall(query))
        tokens = bengali.tokenize(query)
        while tokens and tokens[0] in _EDGE_WORDS:
            tokens.pop(0)
        while tokens and tokens[-1] in _EDGE_WORDS:
            tokens.pop()
        if tokens:
            keys.append(" ".join(tokens))

        found: list[str] = []
        seen: set[str] = set()
        for key in keys:
            for book_id in self.by_key.get(key, []):
                if book_id not in seen:
                    seen.add(book_id)
                    found.append(book_id)
        return found
