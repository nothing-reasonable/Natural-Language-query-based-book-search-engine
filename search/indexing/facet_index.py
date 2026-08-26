"""The metadata-filter retrieval channel.

The design document lists four parallel candidate sources: BM25, vectors, the knowledge
graph, and *metadata filters*. This is the fourth, and leaving it out has a specific,
measurable cost.

A hard constraint is not the same thing as a filter applied after the fact. When the
query is "হুমায়ূন আহমেদ এর মুক্তিযুদ্ধের বই", pruning the sixty candidates BM25 and the vector
index happened to return leaves whichever of his books were already scoring well -- two
of them, when the catalogue holds twelve. The constraint has to *drive* retrieval:
enumerate the books that satisfy it, then rank those.

Everything here is an in-memory inverted index over records already loaded, so it costs
a fraction of a second to build and nothing to store.
"""

from __future__ import annotations

from collections import defaultdict

from search.core.schemas import Filters, IndexedBook


class FacetIndex:
    def __init__(self, records: list[IndexedBook]):
        self.records = {r.book_id: r for r in records}
        self.by_author: dict[str, set[str]] = defaultdict(set)
        self.by_author_name: dict[str, set[str]] = defaultdict(set)
        self.by_publisher: dict[str, set[str]] = defaultdict(set)
        self.by_year: dict[int, set[str]] = defaultdict(set)
        self.by_language: dict[str, set[str]] = defaultdict(set)
        self.by_facet: dict[str, dict[str, set[str]]] = {
            name: defaultdict(set) for name in ("genres", "subjects", "periods", "places")
        }

        for record in records:
            book, enrichment = record.book, record.enrichment
            book_id = record.book_id
            if book.author_id:
                self.by_author[book.author_id].add(book_id)
            if book.author:
                self.by_author_name[book.author].add(book_id)
            if book.publisher:
                self.by_publisher[book.publisher].add(book_id)
            if book.publish_year:
                self.by_year[book.publish_year].add(book_id)
            if book.language:
                self.by_language[book.language].add(book_id)
            for name, table in self.by_facet.items():
                for value in getattr(enrichment, name, []):
                    table[value].add(book_id)

    # ------------------------------------------------------------------ selection
    def select(self, filters: Filters) -> set[str] | None:
        """Book ids satisfying every constraint. `None` means "no constraint given".

        Constraint groups intersect (author AND year); values inside a group union
        (either of these two publishers).
        """
        if filters is None or filters.is_empty():
            return None

        groups: list[set[str]] = []
        if filters.author_ids:
            groups.append(self._union(self.by_author, filters.author_ids))
        elif filters.authors:
            groups.append(self._union(self.by_author_name, filters.authors))
        if filters.publishers:
            groups.append(self._union(self.by_publisher, filters.publishers))
        if filters.language:
            groups.append(self._union(self.by_language, [filters.language]))
        for name, table in self.by_facet.items():
            values = getattr(filters, name, None)
            if values:
                groups.append(self._union(table, values))
        if filters.year_from is not None or filters.year_to is not None:
            low = filters.year_from if filters.year_from is not None else -10_000
            high = filters.year_to if filters.year_to is not None else 10_000
            matched: set[str] = set()
            for year, ids in self.by_year.items():
                if low <= year <= high:
                    matched |= ids
            groups.append(matched)

        if not groups:
            return None
        selected = groups[0]
        for group in groups[1:]:
            selected &= group
        return selected

    @staticmethod
    def _union(table: dict, keys) -> set[str]:
        out: set[str] = set()
        for key in keys:
            out |= table.get(key, set())
        return out

    # ------------------------------------------------------------------ ranking
    def rank(self, book_ids: set[str], concepts, limit: int) -> list[tuple[str, float]]:
        """Order a constrained set by how well it answers the *rest* of the query.

        The filter has already decided membership, so what is left to judge is the
        topical part: "his মুক্তিযুদ্ধ books" should put the মুক্তিযুদ্ধ-tagged ones first, and
        fall back on metadata completeness when nothing distinguishes them.
        """
        wanted = set()
        if concepts is not None:
            wanted = {t for t in concepts.all_terms()}

        scored = []
        for book_id in book_ids:
            record = self.records.get(book_id)
            if record is None:
                continue
            enrichment = record.enrichment
            tags = set(enrichment.subjects) | set(enrichment.genres) | \
                   set(enrichment.periods) | set(enrichment.places) | set(enrichment.topics)
            overlap = len(wanted & tags)
            # Concept agreement dominates; quality only breaks ties.
            score = overlap + 0.1 * record.book.metadata_quality
            scored.append((book_id, score))

        # Stable order for equal scores; see the note in fusion.py.
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[:limit]
