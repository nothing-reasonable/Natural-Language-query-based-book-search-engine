"""Second-stage reranking and final score assembly.

Stage 1 -- a semantic reranker re-reads the top candidates against the query. Retrieval
optimises for recall and is happy to hand up a book that merely shares a subject; this is
where that gets corrected, because the reranker sees the query and the book together
instead of comparing two independently-built vectors.

The default is a multilingual **cross-encoder** (`BAAI/bge-reranker-v2-m3`), which is the
design document's "ColBERT-style second-stage reranker" in the form that actually runs on
this hardware. The chat-model grader it replaced is still available behind
`reranker_backend="llm"`, but it earned its demotion: it returned an identical 0.5 for
every candidate -- i.e. contributed nothing to a signal weighted at 55% of the final
score -- and took 50 seconds to do it.

Stage 2 -- blend the reranker score with the other ranking signals the design calls for:
rank fusion, knowledge-graph match confidence, metadata quality, popularity, availability.

Whichever backend runs, it reads the same thing: `_passage`, which puts the book's title,
author and **flap** at the top and the model-inferred tags underneath. That ordering is
load-bearing rather than cosmetic -- see `_passage` and `_blurb` for what it fixes.
"""

from __future__ import annotations

import logging
from typing import Protocol

from pydantic import BaseModel, Field

from config import Settings, settings as default_settings
from search.llm import LMStudio
from search.core.schemas import IndexedBook
from search.retrieval.fusion import Fused

log = logging.getLogger(__name__)

RERANK_SYSTEM = """তুমি একটি বাংলা বই-অনুসন্ধান ইঞ্জিনের রির‍্যাঙ্কার।

প্রতিটি বই সম্পর্কে তুমি শুধু ক্যাটালগে যা লেখা আছে তাই দেখছ: শিরোনাম, লেখক এবং
বইয়ের ফ্ল্যাপে ছাপা বিবরণ। ঠিক এই লেখাটুকুর ভিত্তিতেই সিদ্ধান্ত নাও।

নিয়ম:
- শিরোনাম ও ফ্ল্যাপের বিবরণকেই প্রধান প্রমাণ ধরো।
- বই, লেখক বা বিষয় সম্পর্কে বাইরে থেকে জানা কোনো তথ্য ব্যবহার করবে না।
- দেখানো লেখায় প্রশ্নের উত্তরের প্রমাণ না থাকলে নম্বর কম দাও, অনুমান করে বাড়াবে না।
- ফ্ল্যাপ যদি প্রশ্নের বিষয়ের কথা স্পষ্টভাবে বলে, তবেই উঁচু নম্বর।

প্রাসঙ্গিকতা ০ থেকে ১০ স্কেলে নম্বর দাও।
১০ = হুবহু যা চাওয়া হয়েছে, ০ = সম্পূর্ণ অপ্রাসঙ্গিক।
প্রতিটি বইয়ের জন্য তার id ও score দাও। শুধু JSON ফেরত দাও।"""


class _Score(BaseModel):
    id: int = 0
    score: float = 0.0


class _Scores(BaseModel):
    scores: list[_Score] = Field(default_factory=list)


class Reranker(Protocol):
    # Which backend this is, and which weights it loaded. Reported in `RerankTrace` so a
    # silent degradation to `noop` is visible instead of being read as a real reranking.
    name: str
    model_name: str

    def score(self, query: str, records: list[IndexedBook]) -> list[float]:
        """Relevance in 0..1, aligned with `records`."""


class NoOpReranker:
    """Used when no LLM is available -- fusion order is kept."""

    name = "noop"
    model_name = ""

    def score(self, query: str, records: list[IndexedBook]) -> list[float]:
        n = len(records)
        return [1.0 - i / max(n, 1) for i in range(n)]


class LLMReranker:
    name = "llm"

    def __init__(self, llm: LMStudio, batch_size: int = 8,  # 8 books ~ 100 output tokens
                 settings: Settings = default_settings):
        self.llm = llm
        self.batch_size = batch_size
        self.settings = settings
        self.model_name = ""  # resolved server-side; asking here would cost a round trip

    def score(self, query: str, records: list[IndexedBook]) -> list[float]:
        scores = [0.5] * len(records)
        batches = [
            (start, records[start : start + self.batch_size])
            for start in range(0, len(records), self.batch_size)
        ]
        results = self.llm.map_parallel(lambda b: self._score_batch(query, b[1]), batches)
        for (start, batch), graded in zip(batches, results, strict=True):
            for offset, value in enumerate(graded):
                if offset < len(batch):
                    scores[start + offset] = value
        return scores

    def _score_batch(self, query: str, batch: list[IndexedBook]) -> list[float]:
        listing = "\n\n".join(_describe(i, r, self.settings) for i, r in enumerate(batch))
        user = f"প্রশ্ন: {query}\n\nবইয়ের তালিকা:\n{listing}"
        try:
            graded = self.llm.structured(RERANK_SYSTEM, user, _Scores, max_tokens=600)
        except Exception as exc:  # noqa: BLE001
            log.warning("rerank batch failed: %s", exc)
            return [0.5] * len(batch)
        by_id = {s.id: max(0.0, min(1.0, s.score / 10.0)) for s in graded.scores}
        return [by_id.get(i, 0.5) for i in range(len(batch))]


class CrossEncoderReranker:
    """A multilingual cross-encoder scoring (query, book) pairs jointly.

    The model is loaded once and reused. GPU is preferred but not assumed: LM Studio is
    usually holding most of this laptop's 6 GB, so a CUDA OOM falls back to CPU rather
    than taking the search down with it (~2 s for 16 candidates on CPU, which is still
    twenty-five times faster than the grader it replaces).
    """

    name = "crossencoder"

    def __init__(self, model_name: str, *, device: str = "", batch_size: int = 16,
                 max_length: int = 512, settings: Settings = default_settings):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        # Only for the flap budget in `_passage`; the model itself is configured above.
        self.settings = settings
        self.model = self._load(device)

    def _load(self, device: str):
        from sentence_transformers import CrossEncoder  # imported late: heavy

        # half precision first: this is a 568M-parameter model and LM Studio is already
        # holding most of the GPU, so fp32 simply will not fit alongside the embedder.
        # Reranking is an ordering, not an arithmetic result -- fp16 costs nothing here.
        attempts: list[tuple[str, dict]] = []
        for candidate in ([device] if device else _device_preference()):
            if candidate == "cuda":
                attempts.append(("cuda", {"torch_dtype": "float16"}))
            attempts.append((candidate, {}))

        last: Exception | None = None
        for candidate, model_kwargs in attempts:
            try:
                model = CrossEncoder(self.model_name, device=candidate,
                                     max_length=self.max_length,
                                     model_kwargs=model_kwargs or None)
                log.info("reranker %s loaded on %s%s", self.model_name, candidate,
                         " (fp16)" if model_kwargs else "")
                return model
            except Exception as exc:  # noqa: BLE001 - OOM, missing driver, absent weights
                log.warning("could not load reranker on %s%s: %s", candidate,
                            " (fp16)" if model_kwargs else "", exc)
                last = exc
                _release_cuda()
        raise RuntimeError(f"could not load reranker {self.model_name}: {last}")

    def score(self, query: str, records: list[IndexedBook]) -> list[float]:
        if not records:
            return []
        pairs = [(query, _passage(r, self.settings)) for r in records]
        try:
            raw = self.model.predict(pairs, batch_size=self.batch_size,
                                     show_progress_bar=False)
        except Exception as exc:  # noqa: BLE001 - never let reranking kill a search
            log.warning("cross-encoder scoring failed (%s); keeping fusion order", exc)
            return NoOpReranker().score(query, records)
        # bge-reranker already emits 0..1 via its sigmoid head; clamp for safety only.
        scores = [max(0.0, min(1.0, float(v))) for v in raw]
        _log_pairs(log, self.model_name, query, pairs, scores)
        return scores


def _release_cuda() -> None:
    """A failed load can leave allocations behind; the next attempt needs them back."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _device_preference() -> list[str]:
    try:
        import torch

        if torch.cuda.is_available():
            return ["cuda", "cpu"]
    except Exception:  # noqa: BLE001
        pass
    return ["cpu"]


def _log_pairs(logger: logging.Logger, model: str, query: str,
               pairs: list[tuple[str, str]], scores: list[float]) -> None:
    """Dump reranker input and output at DEBUG.

    Guarded on `isEnabledFor` because building this string is not free -- sixteen
    four-hundred-character passages per search, on a path that runs per query.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug("rerank %s | query: %s | %d pairs", model or "?", query, len(pairs))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    for rank, i in enumerate(order, start=1):
        first_line = pairs[i][1].splitlines()[0] if pairs[i][1] else ""
        logger.debug("  #%-2d in=%-2d score=%.4f  %s", rank, i + 1, scores[i], first_line)


def _blurb(record: IndexedBook) -> str:
    """The book's own words about itself, preferred over anything a model wrote.

    `Description (Flap)` is populated for every row of both source CSVs, and it is the
    only field that says what a particular book actually contains. The LLM's one-sentence
    `summary` used to win this choice, which meant that wherever enrichment had succeeded
    the reranker never saw the flap at all -- it was grading a model's paraphrase of a
    book against the query, with the catalogue text discarded.

    The summary survives only as a fallback, for the ~23% of rows whose flap was pure
    storefront advertising and was emptied by `ingest.clean.strip_boilerplate`. For those
    books it is the only description there is.
    """
    return record.book.description.strip() or record.enrichment.summary.strip()


def _passage(record: IndexedBook, settings: Settings = default_settings) -> str:
    """What the reranker reads, flap first.

    Field order is the whole point of this function, because the cross-encoder truncates
    at `reranker_max_length` and the LM Studio backend at `lmstudio_rerank_max_chars`:
    whatever sits at the bottom is what gets cut. So the catalogue's own text goes at the
    top and the inferred tags go underneath, where a tight budget costs a genre label
    rather than the paragraph that says what the book is about.

    Author is included deliberately, immediately after the title -- half the queries in
    this catalogue name a person, and a reranker that cannot see the author cannot tell
    that book apart from any other book on the same subject.
    """
    book, enrichment = record.book, record.enrichment
    parts = [f"শিরোনাম: {book.title}", f"লেখক: {book.author}"]
    if book.publish_year:
        parts.append(f"প্রকাশকাল: {book.publish_year}")

    blurb = _blurb(record)
    if blurb:
        parts.append("বিবরণ (ফ্ল্যাপ): " + blurb[: settings.rerank_flap_chars])

    # Inferred by a model, not read from the catalogue: useful for breaking ties between
    # two books whose flaps are equally close to the query, and the first thing that
    # should go when the passage has to be shortened.
    if enrichment.subjects:
        parts.append("বিষয়: " + ", ".join(enrichment.subjects[:5]))
    if enrichment.genres:
        parts.append("ধরন: " + ", ".join(enrichment.genres[:3]))
    if enrichment.author_roles:
        parts.append("লেখকের ভূমিকা: " + ", ".join(enrichment.author_roles[:3]))
    return "\n".join(parts)


def make_reranker(settings: Settings = default_settings,
                  llm: LMStudio | None = None) -> "Reranker":
    """Pick a reranker from configuration, degrading instead of failing."""
    backend = settings.reranker_backend
    if backend == "none":
        return NoOpReranker()
    if backend == "llm":
        return LLMReranker(llm, settings=settings) if llm is not None else NoOpReranker()
    try:
        return CrossEncoderReranker(
            settings.reranker_model,
            device=settings.reranker_device,
            batch_size=settings.reranker_batch_size,
            max_length=settings.reranker_max_length,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("cross-encoder unavailable (%s) -- falling back to fusion order", exc)
        return NoOpReranker()


def _describe(index: int, record: IndexedBook, settings: Settings = default_settings) -> str:
    """One book in the listwise grader's prompt. Same flap-first rule as `_passage`.

    The budget is a fraction of `_passage`'s because this backend puts `batch_size` books
    in one prompt, so the flap allowance is shared eight ways.
    """
    book, enrichment = record.book, record.enrichment
    facets = ", ".join(enrichment.subjects[:4] + enrichment.genres[:2]) or "-"
    blurb = _blurb(record)[: max(1, settings.rerank_flap_chars // 3)] or "-"
    return (
        f"id: {index}\n"
        f"শিরোনাম: {book.title}\n"
        f"লেখক: {book.author}\n"
        f"বিবরণ (ফ্ল্যাপ): {blurb}\n"
        f"বিষয়: {facets}"
    )


# --------------------------------------------------------------------------- final blend

def final_scores(fused: list[Fused], records: dict[str, IndexedBook],
                 semantic: list[float], settings: Settings = default_settings
                 ) -> list[tuple[Fused, float, dict[str, float]]]:
    """Combine every ranking signal. Returns (candidate, score, per-signal breakdown)."""
    weights = settings.score_weights
    max_graph = max((_graph_confidence(f) for f in fused), default=0.0) or 1.0

    out = []
    for candidate, semantic_score in zip(fused, semantic, strict=True):
        record = records.get(candidate.book_id)
        book = record.book if record else None
        components = {
            "semantic": semantic_score,
            "fusion": candidate.fusion_score,
            "graph": _graph_confidence(candidate) / max_graph,
            "quality": book.metadata_quality if book else 0.0,
            "popularity": book.popularity if book else 0.0,
        }
        score = sum(weights.get(name, 0.0) * value for name, value in components.items())
        if book is not None and not book.available:
            score *= 1.0 - settings.unavailable_penalty
            components["availability"] = -settings.unavailable_penalty
        out.append((candidate, score, components))

    out.sort(key=lambda item: (-item[1], item[0].book_id))
    return out


def _graph_confidence(candidate: Fused) -> float:
    """The graph channel's own specificity score: how informative its match was.

    Counting matched terms would rate "this book is tagged গল্প" -- true of a thousand
    books -- as highly as "this author was a diplomat in the Pakistan era".
    """
    return float(candidate.scores.get("graph", 0.0))
