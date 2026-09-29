"""Flask reader interface and validated public search/catalogue APIs."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from config import Settings, settings as default_settings
from search.core.schemas import Filters, IndexedBook, SearchHit, SearchOptions, SearchResponse
from search.engine import SearchEngine
from search.ranking.rerank import NoOpReranker

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"
PLAN_MODES = {"never", "auto", "always"}
FILTER_KEYS = {
    "author": "authors", "authors": "authors",
    "publisher": "publishers", "publishers": "publishers",
    "subject": "subjects", "subjects": "subjects",
    "genre": "genres", "genres": "genres",
    "year_from": "year_from", "year_to": "year_to",
}


class BusyError(RuntimeError):
    pass


@dataclass
class _CacheEntry:
    created: float
    response: SearchResponse


class EngineHandle:
    """Own one engine, a bounded GPU queue and the anonymous ranked-result cache."""

    def __init__(self, settings: Settings, *, use_llm: bool):
        self.settings = settings
        self.use_llm = use_llm
        self.engine: SearchEngine | None = None
        self.error = ""
        self.stage = "ক্যাটালগ প্রস্তুত হচ্ছে"
        self.started_at = time.perf_counter()
        self.ready_in_s = 0.0
        self._search_lock = threading.Lock()
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._cache_lock = threading.Lock()
        threading.Thread(target=self._load, name="engine-load", daemon=True).start()

    def _load(self) -> None:
        try:
            self.stage = "ইনডেক্স ও মডেল লোড হচ্ছে"
            self.engine = SearchEngine.load(self.settings, use_llm=self.use_llm)
            self.stage = "প্রস্তুত"
            self.ready_in_s = round(time.perf_counter() - self.started_at, 1)
        except Exception as exc:  # noqa: BLE001
            log.exception("engine failed to load")
            self.error = str(exc)
            self.stage = "ব্যর্থ"

    @property
    def state(self) -> str:
        if self.error:
            return "failed"
        if self.engine is None:
            return "loading"
        if self.engine.retriever.vector is None or isinstance(self.engine.reranker, NoOpReranker):
            return "degraded"
        return "ready"

    def status(self, *, diagnostics: bool = False) -> dict:
        payload = {
            "state": self.state,
            "ready": self.engine is not None,
            "stage": self.stage,
            "books": len(self.engine.records) if self.engine else 0,
            "elapsed_s": round(time.perf_counter() - self.started_at, 1),
            "message": (
                "অনুসন্ধান পুরোপুরি প্রস্তুত।" if self.state == "ready"
                else "অনুসন্ধান চলছে, তবে কিছু মডেল অনুপলব্ধ।" if self.state == "degraded"
                else "ক্যাটালগ প্রস্তুত হচ্ছে।" if self.state == "loading"
                else "অনুসন্ধান চালু করা যায়নি। `python cli.py build-index` চালিয়ে ইনডেক্স পুনর্নির্মাণ করুন।"
            ),
        }
        if diagnostics:
            payload.update({
                "error": self.error,
                "ready_in_s": self.ready_in_s,
                "llm_available": bool(self.engine and self.engine.llm),
                "reranker_backend": (
                    getattr(self.engine.reranker, "name", "") if self.engine else ""
                ),
                "index_generation": self.engine.index_generation if self.engine else "",
            })
        return payload

    def search(self, query: str, *, filters: Filters, options: SearchOptions,
               anonymous: bool = True) -> tuple[SearchResponse, bool]:
        if self.engine is None:
            raise RuntimeError(self.error or "engine is still loading")
        key = self._cache_key(query, filters, options) if anonymous else ""
        if key:
            cached = self._cache_get(key)
            if cached is not None:
                return cached, True

        acquired = self._search_lock.acquire(timeout=self.settings.web_search_wait_s)
        if not acquired:
            raise BusyError("search capacity is busy")
        try:
            if key:
                cached = self._cache_get(key)
                if cached is not None:
                    return cached, True
            response = self.engine.search(
                query,
                top_k=self.settings.web_ranked_results,
                filters=filters,
                options=options.model_copy(update={
                    "shortlist_size": self.settings.web_ranked_results,
                    "trace": self.settings.web_diagnostics and bool(options.trace),
                }),
            )
            if key:
                self._cache_put(key, response)
            return response, False
        finally:
            self._search_lock.release()

    def _cache_key(self, query: str, filters: Filters, options: SearchOptions) -> str:
        generation = self.engine.index_generation if self.engine else ""
        data = {
            "query": query.strip(),
            "filters": filters.model_dump(exclude_defaults=True),
            "options": options.model_dump(exclude={"trace", "trace_rerank"}),
            "settings": {
                "channels": self.settings.enabled_channels,
                "weights": self.settings.score_weights,
                "ranked": self.settings.web_ranked_results,
                "plan_mode": self.settings.llm_query_understanding,
                "rag_fusion": self.settings.rag_fusion,
                "reranker": getattr(self.engine.reranker, "name", "") if self.engine else "",
            },
            "generation": generation,
        }
        return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _cache_get(self, key: str) -> SearchResponse | None:
        now = time.monotonic()
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if now - entry.created > self.settings.web_cache_ttl_s:
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return entry.response

    def _cache_put(self, key: str, response: SearchResponse) -> None:
        with self._cache_lock:
            self._cache[key] = _CacheEntry(time.monotonic(), response)
            self._cache.move_to_end(key)
            while len(self._cache) > self.settings.web_cache_entries:
                self._cache.popitem(last=False)


def create_app(settings: Settings = default_settings, *, use_llm: bool = True) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.json.ensure_ascii = False
    handle = EngineHandle(settings, use_llm=use_llm)
    app.extensions["booksearch"] = handle

    @app.get("/")
    @app.get("/search")
    @app.get("/books/<book_id>")
    def index(book_id: str | None = None):
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/static/<path:name>")
    def static_file(name: str):
        return send_from_directory(STATIC_DIR, name)

    @app.get("/api/status")
    def status():
        return jsonify(handle.status(diagnostics=settings.web_diagnostics))

    @app.post("/api/search")
    def search_api():
        try:
            payload = _json_object()
            query = _query(payload)
            filters = _parse_filters(payload.get("filters", {}))
            page = _positive_int(payload.get("page", 1), "page", maximum=1000)
            page_size = _positive_int(
                payload.get("page_size", settings.web_page_size), "page_size", maximum=48
            )
            options = _parse_options(payload, diagnostics=settings.web_diagnostics)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        if handle.engine is None:
            return jsonify({"error": "ইঞ্জিন এখনও প্রস্তুত নয়।", "status": handle.status()}), 503
        started = time.perf_counter()
        try:
            response, cache_hit = handle.search(query, filters=filters, options=options)
        except BusyError:
            return jsonify({
                "error": "অনুসন্ধান ব্যবস্থা ব্যস্ত। একটু পরে আবার চেষ্টা করুন।",
                "retryable": True,
                "retry_after": 2,
            }), 429, {"Retry-After": "2"}
        except Exception:  # noqa: BLE001
            log.exception("search failed")
            return jsonify({"error": "অনুসন্ধান সম্পন্ন করা যায়নি। আবার চেষ্টা করুন।"}), 500

        elapsed = round((time.perf_counter() - started) * 1000)
        return jsonify(_response_payload(
            response, page=page, page_size=page_size, total_ms=elapsed,
            cache_hit=cache_hit, diagnostics=settings.web_diagnostics,
        ))

    @app.get("/api/catalog")
    def catalog_api():
        if handle.engine is None:
            return jsonify({"error": "ক্যাটালগ এখনও প্রস্তুত নয়।"}), 503
        try:
            filters = _parse_filters({
                key: value for key, value in request.args.items()
                if key in FILTER_KEYS
            })
            page = _positive_int(request.args.get("page", 1), "page", maximum=100000)
            page_size = _positive_int(request.args.get("page_size", 12), "page_size", maximum=48)
            order = request.args.get("order", "title_asc")
            if order not in {"title_asc", "title_desc", "year_asc", "year_desc"}:
                raise ValueError("order must be title_asc, title_desc, year_asc, or year_desc")
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        records = _sort_records(_filtered_records(handle.engine, filters), order)
        start = (page - 1) * page_size
        return jsonify({
            "items": [_book_payload(record, compact=True) for record in records[start:start + page_size]],
            "total": len(records),
            "page": page,
            "page_size": page_size,
            "pages": _pages(len(records), page_size),
            "applied_filters": filters.model_dump(exclude_defaults=True),
        })

    @app.get("/api/facets")
    def facets_api():
        if handle.engine is None:
            return jsonify({"error": "ক্যাটালগ এখনও প্রস্তুত নয়।"}), 503
        return jsonify(_facet_payload(list(handle.engine.records.values())))

    @app.get("/api/books/<book_id>")
    def book_api(book_id: str):
        if handle.engine is None:
            return jsonify({"error": "ক্যাটালগ এখনও প্রস্তুত নয়।"}), 503
        record = handle.engine.records.get(book_id)
        if record is None:
            return jsonify({"error": "বইটি পাওয়া যায়নি।"}), 404
        return jsonify(_book_payload(record, compact=False))

    return app


# ---------------------------------------------------------------- request validation

def _json_object() -> dict:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object.")
    return payload


def _query(payload: dict) -> str:
    value = payload.get("query")
    if not isinstance(value, str):
        raise ValueError("query must be a string.")
    value = value.strip()
    if not value:
        raise ValueError("প্রশ্ন লিখুন।")
    if len(value) > 300:
        raise ValueError("প্রশ্নটি ৩০০ অক্ষরের মধ্যে লিখুন।")
    return value


def _parse_filters(raw) -> Filters:
    if raw is None:
        return Filters()
    if not isinstance(raw, dict):
        raise ValueError("filters must be an object.")
    unknown = set(raw) - set(FILTER_KEYS)
    if unknown:
        raise ValueError("unknown filters: " + ", ".join(sorted(unknown)))
    data: dict = {}
    for source, target in FILTER_KEYS.items():
        if source not in raw:
            continue
        value = raw[source]
        if target in {"year_from", "year_to"}:
            if value in (None, ""):
                continue
            if isinstance(value, bool):
                raise ValueError(f"{source} must be a year.")
            try:
                year = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{source} must be a year.") from exc
            if year < 1000 or year > 2200:
                raise ValueError(f"{source} is outside the supported range.")
            data[target] = year
            continue
        if isinstance(value, str):
            values = [value.strip()] if value.strip() else []
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            values = list(dict.fromkeys(item.strip() for item in value if item.strip()))
        else:
            raise ValueError(f"{source} must be a string or list of strings.")
        if len(values) > 20 or any(len(item) > 120 for item in values):
            raise ValueError(f"{source} contains too many or overly long values.")
        if values:
            data[target] = values
    if data.get("year_from") and data.get("year_to") and data["year_from"] > data["year_to"]:
        raise ValueError("year_from cannot be after year_to.")
    return Filters(**data)


def _parse_options(payload: dict, *, diagnostics: bool) -> SearchOptions:
    plan_mode = payload.get("plan_mode")
    if plan_mode is not None and (not isinstance(plan_mode, str) or plan_mode not in PLAN_MODES):
        raise ValueError("plan_mode must be never, auto, or always.")
    rerank = payload.get("rerank")
    rag_fusion = payload.get("rag_fusion")
    if rerank is not None and not isinstance(rerank, bool):
        raise ValueError("rerank must be a boolean.")
    if rag_fusion is not None and not isinstance(rag_fusion, bool):
        raise ValueError("rag_fusion must be a boolean.")
    if not diagnostics:
        plan_mode = None
        rerank = None
        rag_fusion = None
    return SearchOptions(
        plan_mode=plan_mode, rerank=rerank, rag_fusion=rag_fusion,
        trace=bool(payload.get("trace")) if diagnostics else False,
        trace_rerank=bool(payload.get("trace_rerank")) if diagnostics else False,
    )


def _positive_int(value, name: str, *, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if number < 1 or number > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}.")
    return number


# ---------------------------------------------------------------- serialisation/catalogue

def _response_payload(response: SearchResponse, *, page: int, page_size: int,
                      total_ms: int, cache_hit: bool, diagnostics: bool) -> dict:
    start = (page - 1) * page_size
    ranked_count = len(response.hits)
    payload = {
        "query": response.query,
        "hits": [
            _hit_payload(hit, rank=start + offset + 1)
            for offset, hit in enumerate(response.hits[start:start + page_size])
        ],
        "ranked_count": ranked_count,
        "count_label": "ranked_results",
        "page": page,
        "page_size": page_size,
        "pages": _pages(ranked_count, page_size),
        "applied_filters": response.applied_filters.model_dump(exclude_defaults=True),
        "rerank": {"fallback": response.rerank_fallback},
        "total_ms": total_ms,
        "cache_hit": cache_hit,
    }
    if diagnostics:
        payload["diagnostics"] = {
            "timings_ms": response.timings_ms,
            "candidate_count": len(response.candidates),
            "plan": response.plan.model_dump(),
            "query_variants": response.query_variants,
            "trace_path": response.trace_path,
            "index_generation": response.index_generation,
            "rerank_backend": response.rerank_backend,
        }
    return payload


def _hit_payload(hit: SearchHit, rank: int) -> dict:
    book, enrichment = hit.book, hit.enrichment
    return {
        "rank": rank, "book_id": book.book_id, "title": book.title,
        "author": book.author, "publisher": book.publisher,
        "publish_year": book.publish_year, "description": book.description,
        "cover_url": book.cover_url, "source_url": book.source_url,
        "match_excerpt": hit.match_excerpt, "match_source": hit.match_source,
        "inferred": {
            "subjects": enrichment.subjects[:6], "genres": enrichment.genres[:4],
            "periods": enrichment.periods[:3],
        },
        "explanation": hit.explanation,
    }


def _book_payload(record: IndexedBook, *, compact: bool) -> dict:
    book, enrichment = record.book, record.enrichment
    payload = {
        "book_id": book.book_id, "title": book.title, "author": book.author,
        "publisher": book.publisher, "publish_year": book.publish_year,
        "cover_url": book.cover_url, "source_url": book.source_url,
        "description": book.description,
        "inferred": {
            "subjects": enrichment.subjects, "genres": enrichment.genres,
            "periods": enrichment.periods, "places": enrichment.places,
        },
    }
    if not compact:
        payload.update({
            "table_of_contents": book.table_of_contents, "language": book.language,
            "isbn": book.isbn, "author_bio": book.author_bio,
        })
    return payload


def _filtered_records(engine: SearchEngine, filters: Filters) -> list[IndexedBook]:
    applied = engine._applied_filters(Filters(), filters)
    selected = engine.facets.select(applied)
    ids = selected if selected is not None else set(engine.records)
    return [engine.records[book_id] for book_id in ids if book_id in engine.records]


def _sort_records(records: list[IndexedBook], order: str) -> list[IndexedBook]:
    if order.startswith("year"):
        reverse = order.endswith("desc")
        return sorted(records, key=lambda r: (r.book.publish_year or 0, r.book.title, r.book_id),
                      reverse=reverse)
    reverse = order.endswith("desc")
    return sorted(records, key=lambda r: (r.book.title.casefold(), r.book_id), reverse=reverse)


def _facet_payload(records: list[IndexedBook]) -> dict:
    authors = Counter(r.book.author for r in records if r.book.author)
    publishers = Counter(r.book.publisher for r in records if r.book.publisher)
    subjects = Counter(value for r in records for value in r.enrichment.subjects)
    genres = Counter(value for r in records for value in r.enrichment.genres)
    years = [r.book.publish_year for r in records if r.book.publish_year]

    def values(counter: Counter) -> list[dict]:
        return [{"value": value, "count": count}
                for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]

    return {
        "authors": values(authors), "publishers": values(publishers),
        "subjects": values(subjects), "genres": values(genres),
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
    }


def _pages(total: int, page_size: int) -> int:
    return (total + page_size - 1) // page_size


def serve(settings: Settings = default_settings, *, host: str = "127.0.0.1",
          port: int = 8000, use_llm: bool = True, debug: bool = False) -> None:
    app = create_app(settings, use_llm=use_llm)
    app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)


if __name__ == "__main__":
    serve()
