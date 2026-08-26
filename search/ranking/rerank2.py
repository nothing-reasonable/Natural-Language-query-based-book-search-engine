"""Second-stage reranking with the local chat model, used as a cross-encoder.

`rerank.py` stays as it is; this is an alternative stage-2 backend for the machines
where the sentence-transformers cross-encoder is not an option (no GPU headroom beside
LM Studio, no model download, no torch). It reranks with the Gemma already loaded in
LM Studio -- and it does so *without* asking Gemma to be a reranker, because it is not
one out of the box.

The distinction matters, and it is the whole reason this file exists. `rerank.py`'s
`LLMReranker` shows a chat model eight books and asks it to write a JSON list of scores.
That is a generation task, and a small instruct model answers it the way it answers any
vague grading prompt: it returns the same middling number for everything. An identical
0.5 for every candidate is not a weak signal, it is *no* signal -- and it was being
weighted at 55% of the final score.

What is done here instead is the standard way a causal LM is turned into a relevance
model (monoT5 / RankLLaMA, pointwise):

    one (query, book) pair per call
        -> "Is this book relevant? Yes or No."
        -> read the *probability distribution over the first generated token*
        -> score = sigmoid((log P(Yes) - log P(No)) / T)

Nothing is asked of the model that it cannot do. It never has to invent a number, hold
sixteen books in its head at once, or emit well-formed JSON. It answers one binary
question, and the score comes from the logits behind that answer -- which are continuous,
well separated, and free. Measured here on "হুমায়ূন আহমেদের মুক্তিযুদ্ধের উপন্যাস" with
gemma-4-e4b, P(Yes) reads:

    right book (Humayun, liberation war)          1.00
    right subject, wrong author (Jahanara Imam)   0.042
    right author, wrong subject (Himu)            0.00017
    unrelated (a cookbook)                        0.00000003

i.e. it separates the two *near* misses that fusion cannot separate, which is exactly the
job of a second stage. Cost is one prompt of ~90 tokens and one decoded token per
candidate: ~0.35 s each on this laptop, ~6 s for a shortlist of 16 at `llm_workers=1`.

Two things about this particular model had to be handled, both verified against the
running server rather than assumed:

* Gemma-4 is a *thinking* model, and with reasoning on it spends the whole token budget
  in `reasoning_content` and returns `logprobs: null` -- the method silently yields
  nothing. `reasoning_effort: "none"` switches the template's thinking channel off, and
  also sharpens the answer (it stops the `<|channel>` control token from competing with
  `Yes` for first place).
* Servers that do not implement logprobs at all exist. The first call probes; if no
  distribution comes back the instance switches to asking for a 0-10 grade *per book*
  (still pointwise, still one question -- just a coarser signal), and if that fails too
  it degrades to fusion order rather than taking the search down.

Stage 2's second half -- blending the reranker score with fusion, graph, quality and
popularity -- is unchanged and imported from `rerank.final_scores`.
"""

from __future__ import annotations

import logging
import math
import re
import threading

from config import Settings, settings as default_settings
from search.llm import LLMUnavailable, LMStudio
from search.core.schemas import IndexedBook

# Re-exported so `rerank2` is a drop-in for `rerank` at the import site. `_passage` is
# what the cross-encoder reads; feeding the chat model the same text keeps the two
# backends comparable on the evaluation set.
from search.ranking.rerank import NoOpReranker, Reranker, _passage, final_scores  # noqa: F401
from search.ranking.rerank import make_reranker as _make_reranker_v1

log = logging.getLogger(__name__)

JUDGE_SYSTEM = (
    "You are a relevance judge for a Bengali book search engine. "
    "You are shown one search query and one book. "
    "Answer with exactly one word: Yes or No."
)
JUDGE_QUESTION = "এই বইটি কি প্রশ্নের উত্তর হিসেবে দেখানোর মতো প্রাসঙ্গিক? Yes বা No।"

GRADE_SYSTEM = (
    "You are a relevance judge for a Bengali book search engine. "
    "Rate how well the book answers the query on a 0-10 scale "
    "(10 = exactly what was asked for, 0 = unrelated). "
    "Reply with the number only."
)
GRADE_QUESTION = "০ থেকে ১০ স্কেলে প্রাসঙ্গিকতা কত? শুধু সংখ্যাটি লেখো।"

# First-token spellings that count as an affirmative / negative answer. Whitespace and
# case are stripped before matching; the Bengali forms show up in the top-20 often
# enough to be worth summing, given the prompt is in Bengali.
_YES = {"yes", "y", "yeah", "true", "1", "হ্যাঁ", "হ্যা", "হ্যাঁ।", "হা"}
_NO = {"no", "n", "nope", "false", "0", "না", "নো", "না।"}

_INT = re.compile(r"\d+")

# What a single judgement call came back with. `UNSUPPORTED` (the server answered but
# there was nothing to read) is a fact about the *server* and downgrades the method for
# good; `FAILED` is a fact about this one call and costs only this one candidate.
OK, UNSUPPORTED, FAILED = "ok", "unsupported", "failed"


class LMStudioReranker:
    """Pointwise (query, book) relevance from the local chat model's own logits.

    One instance per engine; it is called from `llm.map_parallel`, so the small amount of
    mutable state (which mode works, the score cache) is guarded.
    """

    def __init__(self, llm: LMStudio, settings: Settings = default_settings):
        self.llm = llm
        self.settings = settings
        # Empty means "whatever LM Studio has loaded" (see LMStudio.chat_model_param);
        # `model_label` is only for logging, and is resolved lazily so constructing a
        # reranker never depends on the server being reachable.
        self.model = settings.lmstudio_rerank_model or llm.chat_model_param
        # "logprob" -> read the answer distribution; "grade" -> ask for 0-10;
        # "off" -> the server cannot do either, keep fusion order.
        self._mode = "logprob"
        self._extras_ok = True  # server accepts reasoning_effort et al.
        self._failures = 0  # consecutive; see _judge
        self._cache: dict[tuple[str, str], float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public API
    def score(self, query: str, records: list[IndexedBook]) -> list[float]:
        if not records:
            return []
        # Fusion order, used for any candidate the model could not judge. Falling back to
        # a flat 0.5 is what made the old grader useless: it is an *assertion* of average
        # relevance, and it overrides the ordering retrieval already found.
        fallback = NoOpReranker().score(query, records)

        try:
            judged = self.llm.map_parallel(
                lambda record: self._cached(query, record), list(records)
            )
        except Exception as exc:  # noqa: BLE001 - reranking is an improvement, not a dependency
            log.warning("LM Studio reranking raised (%s) -- keeping fusion order", exc)
            return fallback

        if all(value is None for value in judged):
            log.warning("LM Studio reranker produced no usable scores -- keeping fusion order")
            return fallback
        return [fallback[i] if value is None else value for i, value in enumerate(judged)]

    def score_passages(self, query: str, passages: list[str]) -> list[float | None]:
        """Score already-rendered passages. Useful for testing without an index."""
        return self.llm.map_parallel(lambda p: self._judge(query, p), list(passages))

    # ------------------------------------------------------------------ per-candidate
    def _cached(self, query: str, record: IndexedBook) -> float | None:
        key = (query, record.book_id)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        value = self._judge(query, _truncate(_passage(record), self.settings.lmstudio_rerank_max_chars))
        with self._lock:
            if len(self._cache) > 4096:
                self._cache.clear()
            self._cache[key] = value
        return value

    def _judge(self, query: str, passage: str) -> float | None:
        """Score one candidate, downgrading the *method* only on evidence.

        Two different things can go wrong and they deserve different reactions:

        * The server answered, but there was no distribution to read (`UNSUPPORTED`).
          That is a property of the server, not of this candidate -- it will not grow
          logprobs on candidate two -- so switch the whole instance to grading rather
          than pay fifteen more wasted round trips to learn the same thing.
        * The call itself failed (`FAILED`): a timeout, a 400, a model reloading. That
          says nothing about the method. Skip this candidate and try the next one, and
          only give up on the backend after `_failure_limit` in a row.
        """
        if self._mode == "off":
            return None

        if self._mode == "logprob":
            outcome, value = self._by_logprob(query, passage)
            if outcome is OK:
                return self._succeeded(value)
            if outcome is UNSUPPORTED:
                with self._lock:
                    if self._mode == "logprob":
                        log.warning("%s returned no answer distribution -- falling back to "
                                    "0-10 grading", self.model)
                        self._mode = "grade"
            else:
                return self._failed()

        outcome, value = self._by_grade(query, passage)
        return self._succeeded(value) if outcome is OK else self._failed()

    def _succeeded(self, value: float | None) -> float | None:
        with self._lock:
            self._failures = 0
        return value

    def _failed(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.settings.lmstudio_rerank_failure_limit:
                log.warning("%d rerank calls failed in a row -- keeping fusion order for the "
                            "rest of this session", self._failures)
                self._mode = "off"
        return None

    def _by_logprob(self, query: str, passage: str) -> tuple[str, float | None]:
        """P(Yes) vs P(No) over the first answer token, as a calibrated 0..1 score."""
        choice = self._complete(
            JUDGE_SYSTEM, _user_prompt(query, passage, JUDGE_QUESTION),
            max_tokens=self.settings.lmstudio_rerank_answer_tokens,
            logprobs=True,
        )
        if choice is None:
            return FAILED, None
        positions = getattr(getattr(choice, "logprobs", None), "content", None) or []
        for position in positions:
            top = getattr(position, "top_logprobs", None) or []
            if not top:
                continue
            p_yes = p_no = 0.0
            floor = 1.0
            for item in top:
                probability = math.exp(item.logprob)
                floor = min(floor, probability)
                token = item.token.strip().lower()
                if token in _YES:
                    p_yes += probability
                elif token in _NO:
                    p_no += probability
            if p_yes <= 0.0 and p_no <= 0.0:
                continue  # the model started with something else; try the next position
            # Whichever side is absent from the top-k has at most the smallest listed
            # probability, so use that as its bound instead of log(0).
            p_yes = max(p_yes, floor * 1e-3)
            p_no = max(p_no, floor * 1e-3)
            log_odds = math.log(p_yes) - math.log(p_no)
            # Raw P(Yes) saturates: the gap between the two examples that matter is often
            # 1.00 vs 0.04, and every irrelevant book collapses to zero. Squashing the
            # log-odds with a temperature spreads the shortlist back out into a graded
            # signal, which is what the 0.55 `semantic` weight downstream expects.
            temperature = self.settings.lmstudio_rerank_temperature
            return OK, 1.0 / (1.0 + math.exp(-log_odds / temperature))
        return UNSUPPORTED, None

    def _by_grade(self, query: str, passage: str) -> tuple[str, float | None]:
        choice = self._complete(
            GRADE_SYSTEM, _user_prompt(query, passage, GRADE_QUESTION),
            max_tokens=self.settings.lmstudio_rerank_grade_tokens,
            logprobs=False,
        )
        if choice is None:
            return FAILED, None
        # A thinking model that ignored `reasoning_effort` puts its answer -- if it got
        # as far as one -- in `reasoning_content` and leaves `content` empty.
        text = (choice.message.content or "") or (
            getattr(choice.message, "reasoning_content", "") or ""
        )
        match = _INT.search(_ascii_digits(text))
        if not match:
            return FAILED, None
        return OK, max(0.0, min(1.0, int(match.group(0)) / 10.0))

    # ------------------------------------------------------------------ transport
    def _complete(self, system: str, user: str, *, max_tokens: int, logprobs: bool):
        """One chat call. Returns the choice, or None if this candidate cannot be judged.

        A dead server ends reranking for the whole search rather than for this candidate:
        `LMStudio._retry` is not used here (it cannot carry logprobs), so the circuit
        breaker is re-implemented -- without it, an LM Studio that went away mid-search
        costs one connect timeout per remaining candidate.
        """
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        if logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = self.settings.lmstudio_rerank_top_logprobs

        for extras in ([self._reasoning_extras(), {}] if self._extras_ok else [{}]):
            try:
                response = self.llm.client.chat.completions.create(
                    **kwargs, **({"extra_body": extras} if extras else {})
                )
                return response.choices[0]
            except Exception as exc:  # noqa: BLE001 - never let reranking kill a search
                if _is_connection_error(exc):
                    with self._lock:
                        self._mode = "off"
                    log.warning("LM Studio unreachable during reranking (%s)", exc)
                    return None
                if extras:
                    # Most likely the server rejected `reasoning_effort`. Try once
                    # without it, and stop sending it for the rest of the session.
                    with self._lock:
                        self._extras_ok = False
                    log.info("retrying rerank call without extra params: %s", exc)
                    continue
                log.warning("rerank call failed: %s", exc)
                return None
        return None

    def _reasoning_extras(self) -> dict:
        effort = self.settings.lmstudio_rerank_reasoning_effort
        return {"reasoning_effort": effort} if effort else {}


# --------------------------------------------------------------------------- helpers

def _user_prompt(query: str, passage: str, question: str) -> str:
    """Query first, book second, on purpose.

    The shortlist is judged one book at a time against the *same* query, so putting the
    query ahead of the book leaves every call in a search sharing a prompt prefix, and
    llama.cpp reuses the cached KV for it. Book-first would re-process the query sixteen
    times for nothing.
    """
    return f"প্রশ্ন: {query}\n\nবই:\n{passage}\n\n{question}"


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0]


def _ascii_digits(text: str) -> str:
    """Small models answering a Bengali prompt sometimes reply '৮' rather than '8'."""
    return text.translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789"))


def _is_connection_error(exc: Exception) -> bool:
    if isinstance(exc, LLMUnavailable):
        return True
    from openai import APIConnectionError, APITimeoutError

    return isinstance(exc, (APIConnectionError, APITimeoutError))


def make_reranker(settings: Settings = default_settings,
                  llm: LMStudio | None = None) -> Reranker:
    """`rerank.make_reranker` plus the `lmstudio` backend. Same contract: never raises.

    Importing this instead of `rerank.make_reranker` is the only change an application
    needs; every other backend is delegated untouched.
    """
    if settings.reranker_backend != "lmstudio":
        return _make_reranker_v1(settings, llm)
    if llm is None:
        log.warning("reranker_backend='lmstudio' but no LM Studio client -- keeping fusion order")
        return NoOpReranker()
    try:
        reranker = LMStudioReranker(llm, settings)
        log.info("LM Studio reranker using %s", reranker.model or "the loaded model")
        return reranker
    except Exception as exc:  # noqa: BLE001
        log.warning("LM Studio reranker unavailable (%s) -- keeping fusion order", exc)
        return NoOpReranker()


if __name__ == "__main__":  # a smoke test that needs the server but not the index
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    reranker = LMStudioReranker(LMStudio(default_settings))
    demo_query = "হুমায়ূন আহমেদের মুক্তিযুদ্ধের উপন্যাস"
    demo_books = {
        "right book": "শিরোনাম: জোছনা ও জননীর গল্প\nলেখক: হুমায়ূন আহমেদ\n"
                      "বিষয়: মুক্তিযুদ্ধ, উপন্যাস\nবিবরণ: ১৯৭১ সালের মুক্তিযুদ্ধ নিয়ে মহাকাব্যিক উপন্যাস।",
        "same subject, other author": "শিরোনাম: একাত্তরের দিনগুলি\nলেখক: জাহানারা ইমাম\n"
                                      "বিষয়: মুক্তিযুদ্ধ, স্মৃতিকথা",
        "same author, other subject": "শিরোনাম: হিমু\nলেখক: হুমায়ূন আহমেদ\nবিষয়: সমকালীন উপন্যাস",
        "unrelated": "শিরোনাম: রান্নার সহজ পদ্ধতি\nলেখক: সিদ্দিকা কবীর\nবিষয়: রান্না",
    }
    print(f"query: {demo_query}  (mode: {reranker._mode}, model: {reranker.model})")
    for label, values in zip(demo_books, reranker.score_passages(demo_query, list(demo_books.values()))):
        print(f"  {values if values is None else f'{values:.3f}'}  {label}")
