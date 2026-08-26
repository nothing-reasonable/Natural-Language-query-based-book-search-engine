"""Renders the Bengali "why this result" line.

Built from the `Evidence` each channel recorded, never generated free-hand, so the
explanation cannot claim a match that did not happen.
"""

from __future__ import annotations

from search.core.schemas import Evidence, IndexedBook

# `facet` first: when the search was constrained to an author or publisher, that is the
# single most important thing to tell the user about why they are seeing this list.
CHANNEL_ORDER = ("facet", "graph", "lexical", "dense", "profile")


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
