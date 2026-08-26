"""Knowledge graph over books, authors and concepts.

This is what answers questions whose evidence is not written in any single book record,
e.g. *"বইগুলো দেখাও যেগুলোর লেখক পাকিস্তান আমলে কূটনীতিক ছিলেন"* --
"which authors were diplomats during the Pakistan era" is one hop, "what did they write"
is the next.

Nodes are `"<type>:<name>"` strings; `networkx` keeps the whole thing in memory (a few
hundred thousand edges is nothing) and it serialises to readable JSON.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import networkx as nx

from config import Settings, settings as default_settings
from search.core.schemas import GraphStep, IndexedBook

# enrichment field -> (node type, edge label)
BOOK_LINKS = {
    "subjects": ("subject", "ABOUT"),
    "topics": ("topic", "ABOUT"),
    "genres": ("genre", "HAS_GENRE"),
    "periods": ("period", "IN_PERIOD"),
    "places": ("place", "SET_IN"),
    "events": ("event", "COVERS"),
    "persons": ("person", "MENTIONS"),
}
AUTHOR_LINKS = {
    "author_roles": ("occupation", "HAS_ROLE"),
    "author_periods": ("period", "ACTIVE_IN"),
}


def node(kind: str, name: str) -> str:
    return f"{kind}:{name}"


class KnowledgeGraph:
    def __init__(self, graph: nx.DiGraph | None = None):
        self.g = graph if graph is not None else nx.DiGraph()
        self._book_total_cache: int | None = None
        # idf is a pure function of a graph that never changes after `build`, and the
        # ranking loop asks for the same handful of concepts thousands of times.
        self._idf_cache: dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, records: list[IndexedBook], settings: Settings = default_settings) -> "KnowledgeGraph":
        kg = cls()
        for record in records:
            kg._add_record(record)
        kg.save(settings.graph_path)
        return kg

    def _add_record(self, record: IndexedBook) -> None:
        self._idf_cache.clear()
        self._book_total_cache = None
        book, enrichment = record.book, record.enrichment
        book_node = node("book", book.book_id)
        self.g.add_node(book_node, kind="book", title=book.title)

        if book.author_id:
            author_node = node("author", book.author_id)
            self.g.add_node(author_node, kind="author", name=book.author)
            self.g.add_edge(author_node, book_node, label="WROTE")
            for field, (kind, label) in AUTHOR_LINKS.items():
                for value in getattr(enrichment, field, []):
                    self._link(author_node, kind, value, label)

        if book.publisher:
            self._link(book_node, "publisher", book.publisher, "PUBLISHED_BY")
        for field, (kind, label) in BOOK_LINKS.items():
            for value in getattr(enrichment, field, []):
                self._link(book_node, kind, value, label)

    def _link(self, source: str, kind: str, name: str, label: str) -> None:
        name = str(name).strip()
        if not name:
            return
        target = node(kind, name)
        self.g.add_node(target, kind=kind, name=name)
        self.g.add_edge(source, target, label=label)

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.g, edges="edges")
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, settings: Settings = default_settings) -> "KnowledgeGraph":
        data = json.loads(settings.graph_path.read_text(encoding="utf-8"))
        return cls(nx.node_link_graph(data, directed=True, edges="edges"))

    # ------------------------------------------------------------------ traversal
    def neighbours(self, source: str, kind: str) -> set[str]:
        return {t for t in self.g.successors(source) if self.g.nodes[t].get("kind") == kind}

    def sources_of(self, target_node: str, kind: str) -> set[str]:
        if target_node not in self.g:
            return set()
        return {s for s in self.g.predecessors(target_node) if self.g.nodes[s].get("kind") == kind}

    def find_authors(self, occupations: list[str] = (), periods: list[str] = (),
                     within: set[str] | None = None) -> dict[str, list[str]]:
        """Authors matching *all* given constraint groups. Returns author node -> matched labels."""
        groups: list[tuple[str, list[str]]] = []
        if occupations:
            groups.append(("occupation", list(occupations)))
        if periods:
            groups.append(("period", list(periods)))
        if not groups:
            return {a: [] for a in (within or self._all("author"))}

        matched: dict[str, list[str]] = {}
        current: set[str] | None = within
        for kind, values in groups:
            hits: dict[str, list[str]] = defaultdict(list)
            for value in values:
                for author in self.sources_of(node(kind, value), "author"):
                    hits[author].append(value)
            current = set(hits) if current is None else current & set(hits)
            for author, labels in hits.items():
                matched.setdefault(author, []).extend(labels)
        return {a: matched.get(a, []) for a in (current or set())}

    def find_books(self, subjects: list[str] = (), periods: list[str] = (),
                   places: list[str] = (), events: list[str] = (), genres: list[str] = (),
                   authors: set[str] | None = None) -> dict[str, list[str]]:
        """Books matching *all* non-empty constraint groups. Returns book node -> matched labels."""
        groups = [
            ("subject", list(subjects)), ("period", list(periods)),
            ("place", list(places)), ("event", list(events)), ("genre", list(genres)),
        ]
        groups = [(kind, values) for kind, values in groups if values]

        current: set[str] | None = None
        matched: dict[str, list[str]] = defaultdict(list)
        for kind, values in groups:
            hits: dict[str, list[str]] = defaultdict(list)
            for value in values:
                for book in self.sources_of(node(kind, value), "book"):
                    hits[book].append(value)
            current = set(hits) if current is None else current & set(hits)
            for book, labels in hits.items():
                matched[book].extend(labels)

        if authors is not None:
            from_authors = {b for a in authors for b in self.neighbours(a, "book")}
            current = from_authors if current is None else current & from_authors

        return {b: matched.get(b, []) for b in (current or set())}

    def run(self, steps: list[GraphStep]) -> dict[str, dict]:
        """Execute a decomposed multi-hop plan. Returns book_id -> evidence."""
        authors: set[str] | None = None
        author_evidence: dict[str, list[str]] = {}
        books: dict[str, list[str]] = {}

        for step in steps:
            if step.kind == "find_authors":
                found = self.find_authors(step.occupations, step.periods, within=authors)
                authors, author_evidence = set(found), found
            else:
                books = self.find_books(
                    subjects=step.subjects, periods=step.periods, places=step.places,
                    events=step.events, authors=authors,
                )

        if not books:  # plan ended on an author step -- take everything they wrote
            if authors:
                books = {b: [] for a in authors for b in self.neighbours(a, "book")}
            else:
                return {}

        out: dict[str, dict] = {}
        for book_node, labels in books.items():
            book_id = book_node.split(":", 1)[1]
            writers = self.sources_of(book_node, "author")
            author_labels = [label for a in writers for label in author_evidence.get(a, [])]
            out[book_id] = {
                "book_terms": sorted(set(labels)),
                "author_terms": sorted(set(author_labels)),
                "authors": [self.g.nodes[a].get("name", "") for a in writers],
            }
        return out

    # ------------------------------------------------------------------ specificity
    def book_count(self) -> int:
        return sum(1 for _, d in self.g.nodes(data=True) if d.get("kind") == "book")

    def idf(self, kind: str, name: str) -> float:
        """How informative is this concept? Matching "ভাষা আন্দোলন" says far more than
        matching "মুক্তিযুদ্ধ" in a catalogue that is mostly about মুক্তিযুদ্ধ.

        Memoised, and that is not a micro-optimisation. Scoring one query used to make
        ~8,800 calls here, each rescanning every book attached to the concept -- five
        million node visits, 99% of the graph channel's runtime, for perhaps thirty
        distinct answers. The cache is valid because the graph is immutable after `build`;
        `_add_record` clears it for the one caller that is not.
        """
        key = (kind, name)
        cached = self._idf_cache.get(key)
        if cached is not None:
            return cached

        target = node(kind, name)
        if target not in self.g:
            self._idf_cache[key] = 0.0
            return 0.0
        df = sum(1 for s in self.g.predecessors(target) if self.g.nodes[s].get("kind") == "book")
        total = self._book_total
        value = math.log((total + 1) / (df + 1)) if total else 0.0
        self._idf_cache[key] = value
        return value

    @property
    def _book_total(self) -> int:
        if self._book_total_cache is None:
            self._book_total_cache = self.book_count()
        return self._book_total_cache

    # ------------------------------------------------------------------ misc
    def _all(self, kind: str) -> set[str]:
        return {n for n, d in self.g.nodes(data=True) if d.get("kind") == kind}

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for _, data in self.g.nodes(data=True):
            counts[data.get("kind", "?")] += 1
        counts["edges"] = self.g.number_of_edges()
        return dict(counts)
