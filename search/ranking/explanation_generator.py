"""Renders the Bengali "why this result" line.

Built from the `Evidence` each channel recorded, never generated free-hand, so the
explanation cannot claim a match that did not happen.

The one thing it adds to that evidence is a quotation: when a channel reports which words
matched, the sentence of the book's own flap containing one of them is shown. That is
still not generation -- the sentence is copied out of `Description (Flap)` verbatim -- and
it answers the question the reader is actually asking far better than a subject tag does.
A tag says what a model decided the book is about; the flap says what the book says.
"""

from __future__ import annotations

import re

from search.core import bengali
from search.core.schemas import Evidence, IndexedBook

# `facet` first: when the search was constrained to an author or publisher, that is the
# single most important thing to tell the user about why they are seeing this list.
CHANNEL_ORDER = ("facet", "graph", "lexical", "dense", "profile")

# Bengali sentence enders, plus the ASCII ones that show up in translated flaps.
_SENTENCE_SPLIT = re.compile(r"(?<=[।!?.])\s+")

# Long enough to be a real sentence, short enough to sit on one line of a result card.
_MIN_QUOTE, _MAX_QUOTE = 20, 220


def explain(record: IndexedBook, evidence: list[Evidence], components: dict[str, float]) -> str:
    parts: list[str] = []
    seen: set[str] = set()

    for channel in CHANNEL_ORDER:
        for item in evidence:
            if item.channel != channel or not item.detail or item.detail in seen:
                continue
            seen.add(item.detail)
            parts.append(item.detail)

    tags = record.enrichment.subjects[:3]
    if tags:
        parts.insert(0, "বইটি " + ", ".join(tags) + " বিষয়ে চিহ্নিত")

    # Ahead of the tags: the book's own words outrank anything inferred about it.
    quote = flap_quote(record, evidence)
    if quote:
        parts.insert(0, "ফ্ল্যাপে: “" + quote + "”")

    roles = record.enrichment.author_roles[:2]
    periods = record.enrichment.author_periods[:1]
    if roles:
        role_text = f"লেখক {', '.join(roles)} হিসেবে পরিচিত"
        if periods:
            role_text += f" ({periods[0]})"
        parts.append(role_text)

    if components.get("personalization", 0.0) > 0.01:
        parts.append("আপনার পঠন-অভ্যাসের সাথে সঙ্গতিপূর্ণ")

    return "; ".join(parts) + "।" if parts else "প্রশ্নের সাথে সামগ্রিক মিলের ভিত্তিতে নির্বাচিত।"


def flap_quote(record: IndexedBook, evidence: list[Evidence]) -> str:
    """The first flap sentence containing a word the query matched on, or "".

    Matching is on stems, because that is what the lexical channel matched on: the query
    word "মুক্তিযোদ্ধাদের" reaches this function as whatever `Evidence.terms` carries, and
    the flap spells it "মুক্তিযোদ্ধা". Comparing the surfaces would find nothing.

    Returns "" rather than a first-sentence fallback when no term matches. A quotation
    that has nothing to do with the query is not evidence of relevance, and putting one
    at the front of the explanation would imply it is.
    """
    flap = record.book.description.strip()
    if not flap:
        return ""

    wanted = {stem for item in evidence for term in item.terms
              for stem in bengali.analyze(term)}
    if not wanted:
        return ""

    for sentence in _SENTENCE_SPLIT.split(flap):
        sentence = sentence.strip()
        if len(sentence) < _MIN_QUOTE:
            continue
        if wanted & set(bengali.analyze(sentence)):
            return _clip(sentence, _MAX_QUOTE)
    return ""


def _clip(text: str, limit: int) -> str:
    """Cut on a word boundary and mark it, so a trimmed quote does not read as the
    author's own full stop."""
    if len(text) <= limit:
        return text
    head = text[:limit].rsplit(" ", 1)[0] or text[:limit]
    return head.rstrip(" ,;।") + "…"
