"""Where embeddings come from.

Two backends behind one small protocol:

  * `HuggingFaceEmbedder`  -- a sentence-transformers model running in this process.
    The default, because the models that actually understand Bengali live on the Hub.
  * `LMStudioEmbedder`     -- LM Studio's /v1/embeddings, for when it is serving a
    multilingual embedder and you would rather not load a second model into RAM.

Retrieval models are *asymmetric*: a query and a document are encoded differently. The
protocol keeps `embed_documents` and `embed_query` separate so that distinction cannot be
lost by accident -- getting it wrong silently costs a lot of recall.
"""

from __future__ import annotations

import logging
from typing import Callable, Protocol

import numpy as np

from config import Settings, settings as default_settings
from search.llm import LMStudio

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str], on_batch: Callable[[int], None] | None = None) -> np.ndarray:
        """L2-normalised matrix, one row per text."""

    def embed_query(self, text: str) -> np.ndarray:
        """L2-normalised vector for a search query."""


# --------------------------------------------------------------------------- HuggingFace

class HuggingFaceEmbedder:
    """A local sentence-transformers model.

    Retrieval models want their two sides marked differently -- Harrier prepends an
    instruction to queries, E5 prepends `query: ` / `passage: `, BGE-M3 wants nothing.
    `query_prompt` and `document_prompt` accept either a prompt *name* the model ships in
    its config, or the prefix text itself; both resolve to plain text here.
    """

    def __init__(self, model_name: str, *, device: str = "", batch_size: int = 8,
                 max_tokens: int = 512, query_prompt: str = "", document_prompt: str = ""):
        from sentence_transformers import SentenceTransformer  # imported late: heavy

        self.name = model_name
        self.batch_size = batch_size
        self.model = self._load(SentenceTransformer, model_name, device)
        if max_tokens:
            # Context windows of 32k are useless on CPU; book metadata fits in far less.
            self.model.max_seq_length = min(max_tokens, self.model.max_seq_length)
        self.query_prompt = self._resolve(query_prompt)
        self.document_prompt = self._resolve(document_prompt)

    @staticmethod
    def _load(SentenceTransformer, model_name: str, device: str):
        """Prefer the requested device, but never let a full GPU disable semantic search.

        This laptop shares 6 GB between LM Studio and everything here, so CUDA OOM is a
        normal event rather than an exceptional one. Falling back to CPU costs latency;
        losing the dense channel costs recall on every semantic query -- and does it
        silently, which is worse.
        """
        attempts = [device] if device else ["cuda", "cpu"]
        if "cpu" not in attempts:
            attempts.append("cpu")
        last: Exception | None = None
        for candidate in attempts:
            try:
                return SentenceTransformer(model_name, device=candidate or None)
            except Exception as exc:  # noqa: BLE001
                log.warning("embedder could not load on %s: %s", candidate, exc)
                last = exc
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
        raise RuntimeError(f"could not load embedding model {model_name}: {last}")

    def _resolve(self, prompt: str) -> str:
        """A configured prompt is a name if the model defines one, else literal text."""
        return (self.model.prompts or {}).get(prompt, prompt)

    @property
    def dimension(self) -> int:
        # `get_sentence_embedding_dimension` was renamed in sentence-transformers 5.x.
        get = getattr(self.model, "get_embedding_dimension", None)
        return int(get() if get else self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str], on_batch: Callable[[int], None] | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        chunks = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        out = []
        for chunk in chunks:
            out.append(self._encode(chunk, self.document_prompt))
            if on_batch is not None:
                on_batch(len(chunk))
        return np.vstack(out)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text], self.query_prompt)[0]

    def _encode(self, texts: list[str], prompt: str) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            prompt=prompt or None,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype="float32")


# --------------------------------------------------------------------------- LM Studio

class LMStudioEmbedder:
    """Adapts LM Studio's embeddings endpoint to the same protocol."""

    def __init__(self, llm: LMStudio, query_prefix: str = ""):
        self.llm = llm
        self.name = f"lmstudio:{llm.embedding_model}"
        self.query_prefix = query_prefix
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = int(self.llm.embed_one(" ").shape[0])
        return self._dimension

    def embed_documents(self, texts: list[str], on_batch: Callable[[int], None] | None = None) -> np.ndarray:
        return self.llm.embed(texts, on_batch=on_batch)

    def embed_query(self, text: str) -> np.ndarray:
        return self.llm.embed_one(self.query_prefix + text)


# --------------------------------------------------------------------------- factory

def make_embedder(settings: Settings = default_settings, llm: LMStudio | None = None) -> Embedder:
    if settings.embedding_backend == "lmstudio":
        if llm is None:
            llm = LMStudio(settings)
        return LMStudioEmbedder(llm)
    return HuggingFaceEmbedder(
        settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        max_tokens=settings.embedding_max_tokens,
        query_prompt=settings.embedding_query_prompt,
        document_prompt=settings.embedding_document_prompt,
    )
