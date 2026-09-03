"""Browser front end for the search engine.

    python -m booksearch serve            # then open http://127.0.0.1:8000

One Flask process holds one `SearchEngine`. Loading it costs ~30 s (indexes, embedder,
cross-encoder), so it is built on a background thread while the server is already
answering requests: the page can render, poll `/api/status`, and tell the user what the
engine is doing instead of hanging on a blank socket.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from config import Settings, settings as default_settings
from search.engine import SearchEngine
from search.ranking.profile_index import Session
from search.ranking.rerank import NoOpReranker
from search.core.schemas import SearchHit, SearchResponse

PLAN_MODES = ("never", "auto", "always")

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"


class EngineHandle:
    """Loads the engine once, off the request path, and reports progress meanwhile."""

    def __init__(self, settings: Settings, *, use_llm: bool):
        self.settings = settings
        self.use_llm = use_llm
        self.engine: SearchEngine | None = None
        self.error: str = ""
        self.stage: str = "শুরু হচ্ছে"
        self.started_at = time.perf_counter()
        self.ready_in_s: float = 0.0
        # The engine is stateful (profile store, session cache) and the reranker holds a
        # torch model; serialising searches is cheaper than making either re-entrant.
        self._search_lock = threading.Lock()
        self._noop_reranker = NoOpReranker()
        threading.Thread(target=self._load, name="engine-load", daemon=True).start()

    def _load(self) -> None:
        try:
            self.stage = "ইনডেক্স ও মডেল লোড হচ্ছে"
            engine = SearchEngine.load(self.settings, use_llm=self.use_llm)
            self.stage = "প্রস্তুত"
            self.engine = engine
            self.ready_in_s = round(time.perf_counter() - self.started_at, 1)
        except Exception as exc:  # noqa: BLE001
            log.exception("engine failed to load")
            self.error = str(exc)
            self.stage = "ব্যর্থ"

    @property
    def llm_available(self) -> bool:
        """Is there a chat model behind the engine at all?

        `--no-llm` at startup, or an LM Studio that was down when the engine loaded, both
        leave `engine.llm` as None -- and then the plan-mode control is a lie unless the
        page is told.
        """
        return self.engine is not None and self.engine.llm is not None

    def status(self) -> dict:
        return {
            "ready": self.engine is not None,
            "failed": bool(self.error),
            "stage": self.stage,
            "error": self.error,
            "elapsed_s": round(time.perf_counter() - self.started_at, 1),
            "ready_in_s": self.ready_in_s,
            "books": len(self.engine.records) if self.engine else 0,
            # --- what the controls may offer ---
            "llm_available": self.llm_available,
            "reranker_backend": self.settings.reranker_backend,
            "defaults": {
                "top_k": self.settings.final_top_k,
                "plan_mode": self.settings.llm_query_understanding,
                "rerank": self.settings.use_reranker,
            },
        }

    def search(self, query: str, *, user_id: str | None = None, session: Session | None = None,
               top_k: int | None = None, plan_mode: str | None = None,
               rerank: bool | None = None,
               trace_rerank: bool = False) -> tuple[SearchResponse, dict]:
        """Run one search under the UI's chosen options. Returns the response and what
        was *actually* applied -- the two differ when an option could not be honoured."""
        if self.engine is None:
            raise RuntimeError(self.error or "engine is still loading")
        with self._search_lock:
            with self._overrides(plan_mode, rerank) as applied:
                response = self.engine.search(query, user_id=user_id, session=session,
                                              top_k=top_k, trace_rerank=trace_rerank)
        applied["top_k"] = top_k or self.settings.final_top_k
        applied["user"] = user_id or ""
        return response, applied

    @contextmanager
    def _overrides(self, plan_mode: str | None, rerank: bool | None):
        """Apply the per-request knobs, then put the engine back as it was.

        These are the CLI's `--plan/--force-plan/--no-plan` and `--no-rerank`. Both are
        bound when the engine is built -- the mode inside `QueryUnderstanding`, the
        reranker as a chosen object -- so a per-request override has to swap and restore
        rather than pass an argument. Safe because the caller holds the search lock.
        """
        engine = self.engine
        prior_mode, prior_reranker = engine.understanding.mode, engine.reranker
        applied = {"plan_mode": prior_mode, "rerank": not isinstance(prior_reranker, NoOpReranker),
                   "notes": []}

        if plan_mode in PLAN_MODES:
            if plan_mode != "never" and not self.llm_available:
                # Honouring this would call `.structured` on a None client: the engine
                # swallows that and falls back to rules, so say so rather than let the
                # page claim the model ran.
                applied["notes"].append("LM Studio নেই — নিয়মভিত্তিক বিশ্লেষণ চলেছে।")
                plan_mode = "never"
            engine.understanding.mode = plan_mode
            applied["plan_mode"] = plan_mode

        if rerank is not None:
            if rerank and isinstance(prior_reranker, NoOpReranker):
                # Nothing to turn back on -- the engine was built without a reranker.
                applied["notes"].append("রির‍্যাঙ্কার লোড হয়নি — ফিউশন ক্রমই ব্যবহৃত হয়েছে।")
                applied["rerank"] = False
            else:
                engine.reranker = prior_reranker if rerank else self._noop_reranker
                applied["rerank"] = bool(rerank)

        try:
            yield applied
        finally:
            engine.understanding.mode, engine.reranker = prior_mode, prior_reranker


def create_app(settings: Settings = default_settings, *, use_llm: bool = True) -> Flask:
    app = Flask(__name__, static_folder=None)
    handle = EngineHandle(settings, use_llm=use_llm)
    app.extensions["booksearch"] = handle
    # Bengali labels must survive `jsonify` as characters, not \uXXXX escapes.
    app.json.ensure_ascii = False

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/static/<path:name>")
    def static_file(name: str):
        return send_from_directory(STATIC_DIR, name)

    @app.get("/api/status")
    def status():
        return jsonify(handle.status())

    @app.post("/api/search")
    def search():
        payload = request.get_json(silent=True) or {}
        query = (payload.get("query") or "").strip()
        if not query:
            return jsonify({"error": "প্রশ্ন লিখুন।"}), 400
        if handle.engine is None:
            # 503 rather than a queued wait: the page already knows how to show progress.
            return jsonify({"error": handle.error or "ইঞ্জিন এখনও প্রস্তুত নয়।",
                            "status": handle.status()}), 503

        started = time.perf_counter()
        try:
            response, applied = handle.search(
                query,
                user_id=(payload.get("user") or "").strip() or None,
                session=None,
                top_k=_top_k(payload.get("top_k")),
                plan_mode=payload.get("plan_mode"),
                rerank=_flag(payload.get("rerank")),
                # Off unless asked for: the trace carries every candidate passage in full.
                trace_rerank=bool(payload.get("trace_rerank")),
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("search failed: %s", query)
            return jsonify({"error": f"অনুসন্ধান ব্যর্থ: {exc}"}), 500

        elapsed = round((time.perf_counter() - started) * 1000)
        return jsonify(_response_payload(response, elapsed, applied))

    return app


# ------------------------------------------------------------------ request parsing

def _top_k(value) -> int | None:
    """A result count from the page, or None for the configured default."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    # Clamped, not rejected: the control only offers sane values, but the endpoint is
    # reachable directly and a huge top_k walks the whole catalogue.
    return max(1, min(number, 50)) or None


def _flag(value) -> bool | None:
    return None if value is None else bool(value)


# ------------------------------------------------------------------ serialisation

def _response_payload(response: SearchResponse, total_ms: int, applied: dict) -> dict:
    plan = response.plan
    return {
        "query": response.query,
        "total_ms": total_ms,
        "applied": applied,
        "timings_ms": response.timings_ms,
        "candidate_count": len(response.candidates),
        "plan": {
            "intent": plan.intent,
            "keywords": plan.keywords[:12],
            "expanded_terms": plan.expanded_terms[:12],
            "entities": [{"name": e.name, "kind": e.kind, "hard": e.hard} for e in plan.entities],
            "filters": plan.filters.model_dump(exclude_defaults=True),
        },
        "hits": [_hit_payload(hit, rank) for rank, hit in enumerate(response.hits, start=1)],
        # Present only when the request asked for it; `null` otherwise.
        "rerank": response.rerank.model_dump() if response.rerank else None,
    }


def _hit_payload(hit: SearchHit, rank: int) -> dict:
    book, enrichment = hit.book, hit.enrichment
    return {
        "rank": rank,
        "book_id": book.book_id,
        "title": book.title,
        "author": book.author,
        "publisher": book.publisher,
        "publish_year": book.publish_year,
        "available": book.available,
        "description": book.description,
        "summary": enrichment.summary,
        "subjects": enrichment.subjects[:6],
        "genres": enrichment.genres[:4],
        "periods": enrichment.periods[:3],
        "places": enrichment.places[:3],
        "author_roles": enrichment.author_roles[:3],
        "score": hit.score,
        "relevance": hit.relevance,
        "components": hit.components,
        "channels": hit.channels,
        "explanation": hit.explanation,
        "evidence": [{"channel": e.channel, "detail": e.detail} for e in hit.evidence],
    }


def serve(settings: Settings = default_settings, *, host: str = "127.0.0.1",
          port: int = 8000, use_llm: bool = True, debug: bool = False) -> None:
    app = create_app(settings, use_llm=use_llm)
    app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)


if __name__ == "__main__":
    serve()
