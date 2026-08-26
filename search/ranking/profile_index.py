"""User profile index: what someone told us they like, plus what they actually did.

Storage only -- turning these signals into a ranking adjustment happens in
`search/personalize.py`, deliberately *after* relevance has been decided.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from config import Settings, settings as default_settings

EventKind = Literal["click", "save", "rate", "query"]


class UserProfile(BaseModel):
    user_id: str

    # --- explicit preferences (set by the user) ---
    languages: list[str] = Field(default_factory=lambda: ["bn"])
    genres: list[str] = Field(default_factory=list)
    subjects: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    reading_level: str | None = None  # e.g. "সাধারণ পাঠক" | "গবেষক" | "শিশু কিশোর"

    # --- implicit behaviour ---
    clicks: dict[str, int] = Field(default_factory=dict)  # book_id -> count
    saved: list[str] = Field(default_factory=list)  # book_ids
    ratings: dict[str, float] = Field(default_factory=dict)  # book_id -> 1..5
    past_queries: list[str] = Field(default_factory=list)
    updated_at: str = ""

    def record(self, kind: EventKind, value: str, rating: float | None = None) -> None:
        if kind == "click":
            self.clicks[value] = self.clicks.get(value, 0) + 1
        elif kind == "save":
            if value not in self.saved:
                self.saved.append(value)
        elif kind == "rate" and rating is not None:
            self.ratings[value] = rating
        elif kind == "query":
            self.past_queries.append(value)
            self.past_queries = self.past_queries[-50:]
        self.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def interacted_books(self) -> Counter:
        """book_id -> strength of interest, blending the implicit signals."""
        weights: Counter = Counter()
        for book_id, count in self.clicks.items():
            weights[book_id] += min(count, 5) * 1.0
        for book_id in self.saved:
            weights[book_id] += 3.0
        for book_id, rating in self.ratings.items():
            weights[book_id] += (rating - 3.0) * 2.0  # below 3 becomes a negative signal
        return weights


class Session(BaseModel):
    """Current-session intent. Weighted above long-term taste at ranking time."""

    queries: list[str] = Field(default_factory=list)
    clicks: list[str] = Field(default_factory=list)

    def record_query(self, query: str) -> None:
        self.queries.append(query)

    def record_click(self, book_id: str) -> None:
        self.clicks.append(book_id)

    def interacted_books(self) -> Counter:
        return Counter(self.clicks)


class ProfileStore:
    """One JSON file per user. Swap this class out for a database when needed."""

    def __init__(self, settings: Settings = default_settings):
        self.directory: Path = settings.profiles_dir

    def _path(self, user_id: str) -> Path:
        return self.directory / f"{user_id}.json"

    def get(self, user_id: str) -> UserProfile:
        path = self._path(user_id)
        if path.exists():
            return UserProfile.model_validate_json(path.read_text(encoding="utf-8"))
        return UserProfile(user_id=user_id)

    def save(self, profile: UserProfile) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path(profile.user_id).write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    def record(self, user_id: str, kind: EventKind, value: str, rating: float | None = None) -> UserProfile:
        profile = self.get(user_id)
        profile.record(kind, value, rating)
        self.save(profile)
        return profile

    def list_users(self) -> list[str]:
        if not self.directory.exists():
            return []
        return sorted(p.stem for p in self.directory.glob("*.json"))
