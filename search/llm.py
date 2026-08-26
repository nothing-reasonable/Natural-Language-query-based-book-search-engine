"""Thin client for a local LM Studio server (OpenAI-compatible /v1).

Everything the pipeline needs from a model lives here: chat, JSON-schema-constrained
chat, and embeddings. Swapping in another OpenAI-compatible backend is a URL change.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

import httpx
import numpy as np
from openai import APIConnectionError, OpenAI
from pydantic import BaseModel, ValidationError

from config import Settings, settings as default_settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_FENCE = re.compile(r"^\s*```(?:json)?|```\s*$")
_EMBED_HINTS = ("embed", "bge", "e5", "gte", "minilm", "nomic", "lfm2.5-embedding")


class LLMUnavailable(RuntimeError):
    pass


class LMStudio:
    def __init__(self, settings: Settings = default_settings):
        self.settings = settings
        self.client = OpenAI(
            base_url=settings.lmstudio_base_url,
            api_key=settings.lmstudio_api_key,
            # Generation can legitimately take minutes; failing to connect should not.
            timeout=httpx.Timeout(settings.llm_timeout_s, connect=settings.llm_connect_timeout_s),
            max_retries=0,  # we retry ourselves so we can log and back off
        )
        self._chat_model: str | None = settings.chat_model or None
        self._embed_model: str | None = settings.lmstudio_embedding_model or None
        self._offline = False  # circuit breaker: probe a dead server once, not once per query

    # ------------------------------------------------------------------ models
    def list_models(self) -> list[str]:
        try:
            models = [m.id for m in self.client.models.list().data]
        except Exception as exc:  # noqa: BLE001 - surface a friendly message
            self._offline = True
            raise LLMUnavailable(
                f"LM Studio unreachable at {self.settings.lmstudio_base_url}: {exc}"
            ) from exc
        self._offline = False
        return models

    def is_available(self) -> bool:
        try:
            self.list_models()
            return True
        except LLMUnavailable:
            return False

    def loaded_models(self) -> list[dict]:
        """What LM Studio currently has *in memory*, via its native `/api/v0/models`.

        `/v1/models` is the OpenAI-compatible listing and reports every model that has
        been downloaded, with no way to tell which one is actually resident. That
        distinction is the whole problem: naming a model that is merely available asks
        LM Studio to JIT-load it, and on a 6 GB card that means evicting or failing
        alongside whatever is already there -- the connection error looks like the server
        is down when in fact it is refusing to load a second model.

        Returns [] when the endpoint is missing (a non-LM-Studio OpenAI-compatible
        server) so callers fall back to the plain listing rather than breaking.
        """
        root = self.settings.lmstudio_base_url.rstrip("/")
        for suffix in ("/v1", "/api/v0"):
            if root.endswith(suffix):
                root = root[: -len(suffix)]
        try:
            response = httpx.get(f"{root}/api/v0/models",
                                 timeout=self.settings.llm_connect_timeout_s)
            response.raise_for_status()
            return [m for m in response.json().get("data", []) if m.get("state") == "loaded"]
        except Exception as exc:  # noqa: BLE001 - optional endpoint, never fatal
            log.debug("native model listing unavailable (%s)", exc)
            return []

    def _autoselect(self, want_embedder: bool) -> str | None:
        """Pick the loaded model of the right kind, or None to fall back.

        Prefers the native listing's own `type` ("embeddings" vs everything else) over
        guessing from the model's name, which is a heuristic that mislabels anything not
        on the hint list.
        """
        loaded = self.loaded_models()
        if not loaded:
            return None
        matching = [
            m["id"] for m in loaded
            if (m.get("type") == "embeddings") == want_embedder and m.get("id")
        ]
        return matching[0] if matching else None

    @property
    def chat_model_param(self) -> str:
        """What to put in a request's `model` field. **This is what calls should send.**

        Empty unless a model was explicitly configured, because LM Studio treats an empty
        `model` as "use whatever is loaded" and answers from the resident model. That is
        the only way to name a model that cannot go stale: any id resolved in advance --
        however carefully -- is a snapshot, and the moment someone swaps models in the UI
        it becomes a request to *load* a model that is not resident. On a card that fits
        one model at a time that fails, and LM Studio reports it as

            400 Failed to load model "..." -- llama-server exited before becoming healthy

        which reads like the server is down when it is running perfectly well.

        `chat_model` below still resolves a concrete name, but only for *reporting*
        (`doctor`) -- never for routing a call.
        """
        return self.settings.chat_model or ""

    @property
    def chat_model(self) -> str:
        """The configured chat model, or whichever one LM Studio has loaded.

        For display and for callers that genuinely need a name. Requests should use
        `chat_model_param` instead, so an empty setting stays empty all the way to the
        server rather than being resolved into an id that may not survive the round trip.

        Leaving `chat_model` empty is the recommended setup: the pipeline then follows
        whatever is loaded in the LM Studio UI instead of pinning an id that may not be
        resident.
        """
        if not self._chat_model:
            self._chat_model = self._autoselect(want_embedder=False)
            if self._chat_model:
                log.info("Using the chat model loaded in LM Studio: %s", self._chat_model)
        if not self._chat_model:
            # No native endpoint (or nothing loaded): fall back to the plain listing and
            # let LM Studio load on demand.
            models = [m for m in self.list_models() if not _looks_like_embedder(m)]
            if not models:
                raise LLMUnavailable("No chat model is loaded in LM Studio.")
            self._chat_model = models[0]
            log.info("Auto-selected chat model: %s", self._chat_model)
        return self._chat_model

    @property
    def embedding_model(self) -> str:
        if not self._embed_model:
            self._embed_model = self._autoselect(want_embedder=True)
            if self._embed_model:
                log.info("Using the embedding model loaded in LM Studio: %s", self._embed_model)
        if not self._embed_model:
            embedders = [m for m in self.list_models() if _looks_like_embedder(m)]
            if not embedders:
                raise LLMUnavailable(
                    "No embedding model is loaded in LM Studio "
                    "(load one, e.g. LFM2.5-Embedding / bge-m3 / multilingual-e5)."
                )
            self._embed_model = embedders[0]
            log.info("Auto-selected embedding model: %s", self._embed_model)
        return self._embed_model

    # ------------------------------------------------------------------ chat
    def chat(self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        def call():
            resp = self.client.chat.completions.create(
                model=self.chat_model_param,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""

        return self._retry(call)

    def structured(self, system: str, user: str, schema: type[T], *, max_tokens: int = 1500) -> T:
        """Chat constrained to a Pydantic schema.

        LM Studio honours `json_schema` response_format on most runtimes; we still parse
        defensively so a chatty model degrades the result instead of breaking the pipeline.

        Two things make this harder than it looks with the Gemma-4 QAT models this project
        runs against:

        * They are *thinking* models, and reasoning tokens are billed against `max_tokens`.
          They are not frugal: a trivial query measured here spent 530 of a 700-token
          budget thinking before writing a single character of JSON. Budget for both, or
          the answer is silently truncated to nothing and every query quietly degrades to
          the rule-based path while appearing to work.
        * The reasoning comes back out-of-band in `reasoning_content`, and the JSON
          sometimes lands there instead of in `content`. Both are searched.
        """
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": _json_schema(schema), "strict": False},
        }

        def call():
            resp = self.client.chat.completions.create(
                model=self.chat_model_param,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=max_tokens + self.settings.llm_reasoning_budget,
                response_format=response_format,
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            if _JSON_BLOCK.search(content):
                return content
            reasoning = getattr(choice.message, "reasoning_content", "") or ""
            if _JSON_BLOCK.search(reasoning):
                log.debug("recovered %s from reasoning_content", schema.__name__)
                return reasoning
            if choice.finish_reason == "length":
                log.warning(
                    "%s truncated -- the model used its whole budget (%s tokens) before "
                    "answering; raise BOOKSEARCH_LLM_REASONING_BUDGET",
                    schema.__name__, getattr(resp.usage, "completion_tokens", "?"),
                )
            return content

        return _parse_model(self._retry(call), schema)

    # ------------------------------------------------------------------ embeddings
    def embed(self, texts: Iterable[str], *, batch_size: int | None = None,
              on_batch: Callable[[int], None] | None = None) -> np.ndarray:
        """L2-normalised float32 matrix, one row per input (so cosine == dot product).

        `on_batch` is called with the number of texts finished, for progress reporting.
        """
        texts = [t if str(t).strip() else " " for t in texts]
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        batch = batch_size or self.settings.embed_batch_size
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch):
            chunk = texts[start : start + batch]
            resp = self._retry(
                lambda c=chunk: self.client.embeddings.create(model=self.embedding_model, input=c)
            )
            vectors.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
            if on_batch is not None:
                on_batch(len(chunk))
        matrix = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-9, None)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    # ------------------------------------------------------------------ helpers
    def map_parallel(self, fn, items: list, workers: int | None = None) -> list:
        """Run `fn` over `items` with a small thread pool, preserving order."""
        workers = workers or self.settings.llm_workers
        if workers <= 1:
            return [fn(i) for i in items]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, items))

    def _retry(self, call):
        if self._offline:
            raise LLMUnavailable("LM Studio was unreachable earlier in this session.")
        delay, last = 1.0, None
        for attempt in range(self.settings.llm_max_retries):
            try:
                return call()
            except (APIConnectionError, LLMUnavailable) as exc:
                # The server is not there; retrying only adds latency to every later call.
                self._offline = True
                raise LLMUnavailable(f"LM Studio unreachable: {exc}") from exc
            except Exception as exc:  # noqa: BLE001
                last = exc
                log.warning("LM Studio call failed (attempt %d): %s", attempt + 1, exc)
                # "Failed to load model X" is not a transient fault, and retrying the
                # same id just reproduces it three times: the model this client resolved
                # is no longer the one LM Studio has resident -- either it was swapped in
                # the UI mid-session, or the id was picked before the native listing was
                # reachable. Re-resolve once against what is loaded *now* and try again.
                if _is_model_load_error(exc) and self._reresolve_chat_model():
                    continue
                if attempt + 1 < self.settings.llm_max_retries:
                    time.sleep(delay)
                    delay *= 2
        raise LLMUnavailable(f"LM Studio call failed after retries: {last}") from last

    def _reresolve_chat_model(self) -> bool:
        """Re-pick the chat model after a load failure. True if it changed.

        Never overrides an explicitly configured `chat_model`: if someone pinned an id,
        a load failure is a real error about their choice, not something to silently
        route around.
        """
        if self.settings.chat_model:
            return False
        previous, self._chat_model = self._chat_model, None
        try:
            current = self.chat_model
        except LLMUnavailable:
            self._chat_model = previous
            return False
        if current == previous:
            return False
        log.warning("chat model changed under us: %s -> %s; retrying", previous, current)
        return True


# --------------------------------------------------------------------------- module helpers

def _is_model_load_error(exc: Exception) -> bool:
    """Whether LM Studio refused because it could not load the model that was named.

    Matched on the message because the server returns a plain 400 for this, with the
    same shape as any other bad request.
    """
    text = str(exc).lower()
    return "failed to load model" in text or "model_not_found" in text


def _looks_like_embedder(model_id: str) -> bool:
    lowered = model_id.lower()
    return any(hint in lowered for hint in _EMBED_HINTS)


def _json_schema(schema: type[BaseModel]) -> dict:
    """Pydantic JSON schema with $refs inlined -- some runtimes cannot follow them."""
    js = schema.model_json_schema()
    defs = js.pop("$defs", {})

    def inline(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return inline(defs.get(node["$ref"].rsplit("/", 1)[-1], {}))
            node = {k: inline(v) for k, v in node.items()}
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)
            return node
        if isinstance(node, list):
            return [inline(n) for n in node]
        return node

    return inline(js)


def _parse_model(raw: str, schema: type[T]) -> T:
    """Best-effort JSON -> pydantic. Falls back to the schema's defaults."""
    text = _FENCE.sub("", raw.strip()).strip()
    match = _JSON_BLOCK.search(text)
    for candidate in filter(None, [text, match.group(0) if match else None]):
        try:
            return schema.model_validate(json.loads(candidate))
        except (json.JSONDecodeError, ValidationError):
            continue
    log.warning("Could not parse %s from model output: %.200s", schema.__name__, raw)
    return schema()
