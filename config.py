"""All tunable knobs live here. Override any of them with env vars (BOOKSEARCH_*) or a .env file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BOOKSEARCH_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---------------------------------------------------------------- LM Studio
    # LM Studio exposes an OpenAI-compatible server. Both chat and embeddings go through it.
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_api_key: str = "lm-studio"  # LM Studio ignores the value but the SDK requires one
    chat_model: str = ""  # empty => auto-pick the first loaded model
    lmstudio_embedding_model: str = ""  # only used when embedding_backend == "lmstudio"
    llm_timeout_s: float = 180.0  # generation can be slow on CPU
    llm_connect_timeout_s: float = 3.0
    llm_max_retries: int = 3
    # LM Studio serves one request at a time unless parallel serving is switched on in
    # its server settings. Firing concurrent requests at a single-slot server makes every
    # one of them wait, so 1 is the safe default; raise it once parallelism is enabled.
    llm_workers: int = 1
    # Extra allowance for models that "think" before answering. Reasoning tokens come out
    # of the same budget as the answer, so without headroom the JSON is cut off mid-object
    # and every structured call silently falls back to schema defaults.
    llm_reasoning_budget: int = 900

    # ---------------------------------------------------------------- Embeddings
    # "huggingface" runs a sentence-transformers model in-process; "lmstudio" uses
    # LM Studio's /v1/embeddings. The default is HuggingFace because that is where the
    # models that actually cover Bengali are.
    embedding_backend: Literal["huggingface", "lmstudio"] = "huggingface"

    # Harrier is multilingual (Bengali included), MIT-licensed, and comes in three sizes:
    #   microsoft/harrier-oss-v1-270m   640-dim,  MTEB v2 66.5  <- fits comfortably on CPU
    #   microsoft/harrier-oss-v1-0.6b  1024-dim,  MTEB v2 69.0  <- better, ~3x slower
    #   microsoft/harrier-oss-v1-27b   5376-dim,  MTEB v2 74.3  <- needs a serious GPU
    embedding_model: str = "microsoft/harrier-oss-v1-270m"
    embedding_device: str = ""  # "" => auto ("cuda" when available, else "cpu")
    embedding_batch_size: int = 8
    # These models advertise a 32k context. Book metadata needs a fraction of that, and
    # attention cost grows with sequence length, so cap it.
    embedding_max_tokens: int = 512
    # Retrieval models encode queries and documents differently. Each setting below is
    # either a prompt *name* from the model's config_sentence_transformers.json, or the
    # prefix text itself -- whichever the model documents. Empty means "no prefix".
    #   Harrier: query "web_search_query", document none
    #   E5:      query "query: ",          document "passage: "
    #   BGE-M3:  neither
    embedding_query_prompt: str = "web_search_query"
    embedding_document_prompt: str = ""

    # ---------------------------------------------------------------- Paths
    data_dir: Path = ROOT / "data"
    artifacts_dir: Path = ROOT / "artifacts"

    # ---------------------------------------------------------------- Query tracing
    # One plain-text file per search, recording what every pipeline stage handed to the
    # next (see search/trace.py). On by default because the failure mode this exists to
    # catch -- a stage silently doing nothing -- is invisible from the results alone.
    # Costs a file of roughly 10-40 kB per query; set BOOKSEARCH_TRACE_QUERIES=false to
    # turn it off, and delete artifacts/query_traces/ freely.
    trace_queries: bool = True
    # Characters of each reranker passage written to the trace. 0 means no limit.
    trace_passage_chars: int = 600

    # ---------------------------------------------------------------- Ingestion
    # Fuzzy author merging is deliberately conservative: Bengali names differ by a single
    # vowel sign more often than not ("সালেহ" vs "সালেহা"). Exact-key merges and the
    # curated data/author_aliases.yaml do the heavy lifting; every fuzzy merge is written
    # to artifacts/author_merges.json for review.
    author_alias_threshold: int = 97

    # ---------------------------------------------------------------- Retrieval
    channel_top_k: int = 60  # candidates pulled from each retrieval channel
    rerank_top_k: int = 16  # candidates handed to the (expensive) reranker
    final_top_k: int = 10
    rrf_k: int = 60  # reciprocal-rank-fusion damping constant
    # Expansion terms added per query keyword. The taxonomy holds up to eleven aliases
    # for a concept like মুক্তিযুদ্ধ; adding all of them rewrites a four-word question into
    # eighteen terms that mostly restate each other, and retrieval scores the restatement
    # instead of the question. A handful of specific, in-vocabulary terms per keyword is
    # what expansion is for.
    expansion_per_keyword: int = 4

    # A graph hit only counts if the concepts it matched are informative; matching a tag
    # that half the catalogue carries is not evidence of anything.
    graph_min_specificity: float = 0.7

    # Weight of each retrieval channel inside RRF.
    channel_weights: dict[str, float] = Field(
        # `facet` is weighted highest: it is the only channel whose hits are *known* to
        # satisfy what the user explicitly asked for, rather than inferred to be similar.
        default_factory=lambda: {"lexical": 1.0, "dense": 1.2, "graph": 1.0, "facet": 1.5}
    )

    # Blend of the final ranking score (see search/rerank.py). Should sum to ~1.
    score_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "semantic": 0.55,  # reranker score
            "fusion": 0.25,  # rank-fusion score
            "graph": 0.10,  # knowledge-graph match confidence
            "quality": 0.05,  # metadata completeness
            "popularity": 0.05,
        }
    )
    unavailable_penalty: float = 0.15

    # Personalisation is applied *after* relevance and is deliberately bounded so it can
    # never bury an objectively relevant book (see search/personalize.py).
    personalization_strength: float = 0.15
    session_weight: float = 0.7  # session intent outranks long-term taste

    # Which retrieval channels run at all. Ablation studies flip these off one at a
    # time; production keeps all three.
    enabled_channels: list[str] = Field(
        default_factory=lambda: ["lexical", "dense", "graph", "facet"]
    )

    # ---------------------------------------------------------------- RAG-Fusion
    # Retrieve for several reformulations of the query and fuse the rankings (rag_fusion.py).
    # Off by default: it costs one chat call to write the variants plus one full retrieval
    # per variant, which is the right trade only when recall matters more than latency.
    # CLI: --rag-fusion / --top N.
    rag_fusion: bool = False
    rag_fusion_variants: int = 4  # reformulations requested from the model (3-5 is the useful range)
    rag_fusion_top_n: int = 15  # fused results carried forward; --top overrides
    # The original phrasing is the only one the user actually chose; the variants are
    # guesses at what they meant, and a guess should not outvote the question.
    rag_fusion_original_weight: float = 1.5

    # ---------------------------------------------------------------- Reranking
    # "crossencoder" scores every (query, book) pair with a multilingual cross-encoder.
    # It is the design document's second stage, and unlike asking a chat model to grade
    # a list, it produces a real ordering: on "হুমায়ূন আহমেদের মুক্তিযুদ্ধের বই" it separates
    # the right book (0.98) from a same-subject book by someone else (0.07).
    # "llm" keeps the old chat-model grader; "none" leaves fusion order alone.
    reranker_backend: Literal["crossencoder", "lmstudio", "llm", "none"] = "crossencoder"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = ""  # "" => cuda when it fits, else cpu
    reranker_batch_size: int = 16
    reranker_max_length: int = 512

    # "lmstudio" (see rerank2.py) is the cross-encoder's stand-in when torch, a model
    # download or spare VRAM are not available: it judges one (query, book) pair per call
    # with the chat model already loaded, and takes the score from the probability of
    # "Yes" vs "No" as the first answer token rather than from anything the model writes.
    lmstudio_rerank_model: str = ""  # "" => whatever `chat_model` resolves to
    # Gemma-4 is a thinking model; with reasoning on it spends the whole budget in
    # `reasoning_content` and returns no logprobs at all. "" disables the parameter for
    # servers that reject it (the reranker then retries without it once, and remembers).
    lmstudio_rerank_reasoning_effort: str = "none"
    lmstudio_rerank_top_logprobs: int = 20
    lmstudio_rerank_answer_tokens: int = 3  # 1 is enough; a couple spare for a preamble
    lmstudio_rerank_grade_tokens: int = 8  # only the no-logprobs fallback path
    # Softmax temperature on the Yes/No log-odds. Raw P(Yes) saturates at 1.0 and 0.0;
    # ~4 keeps the shortlist spread across a usable range. Lower = sharper ordering.
    lmstudio_rerank_temperature: float = 4.0
    lmstudio_rerank_max_chars: int = 700  # book text sent per judgement
    # Consecutive failed judgement calls before the backend gives up for the session.
    # One 400 or one model reload should cost a candidate, not the whole stage.
    lmstudio_rerank_failure_limit: int = 4

    use_reranker: bool = True  # master switch for stage 2, whichever backend is chosen

    # How much the chat model is involved in reading the query:
    #
    #   "never"  -- rules only. The default, on evidence rather than taste: since the
    #               rule pass gained entity linking, taxonomy expansion and year
    #               detection, adding the local model measurably *lowered* quality
    #               (nDCG@10 0.836 -> 0.819, simple queries 0.883 -> 0.838) while adding
    #               ~25 s to any query that reached it.
    #   "auto"   -- rules first, and the model only for queries where they found nothing
    #               at all: no recognised name, no vocabulary hit, no year. That is the
    #               free-text case rules genuinely cannot serve, and the only place the
    #               model has been worth its latency.
    #   "always" -- call the model on every query, whatever the rules found. Independent
    #               of the gate; use it to study what the model contributes, or when
    #               running against a catalogue whose vocabulary is not curated yet.
    #
    # CLI: --no-plan / --plan / --force-plan.
    llm_query_understanding: Literal["never", "auto", "always"] = "never"

    # ---------------------------------------------------------------- Derived paths
    @property
    def books_path(self) -> Path:
        return self.artifacts_dir / "books.jsonl"

    @property
    def enrichment_path(self) -> Path:
        return self.artifacts_dir / "enrichment.jsonl"

    @property
    def lexical_dir(self) -> Path:
        return self.artifacts_dir / "lexical"

    @property
    def vector_dir(self) -> Path:
        return self.artifacts_dir / "vector"

    @property
    def graph_path(self) -> Path:
        return self.artifacts_dir / "graph.json"

    @property
    def trace_dir(self) -> Path:
        return self.artifacts_dir / "query_traces"

    @property
    def profiles_dir(self) -> Path:
        return self.artifacts_dir / "profiles"

    @property
    def taxonomy_path(self) -> Path:
        return Path(__file__).resolve().parent / "data" / "taxonomy.yaml"


settings = Settings()
