"""
query_understanding.py — Query Understanding Layer (Step 2.5).

Uses the local Gemma-4 12B model to classify raw user queries into types
and extract structured information (entities, filters) for routing to
the appropriate retrieval channels.

Query Types:
    - simple: direct keyword lookup
    - semantic: needs meaning-based search
    - filtered: has metadata constraints (publisher, etc.)
    - author_search: looking for an author's works
    - multi_hop: requires reasoning across entities

The module outputs a QueryIntent dataclass that downstream components
use for retrieval routing.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from openai import OpenAI


# Local Gemma-4 endpoint
GEMMA_BASE_URL = "http://localhost:1234/v1"
GEMMA_API_KEY = "not-needed"
GEMMA_MODEL = "local-model"

# Valid query types
VALID_QUERY_TYPES = {"simple", "semantic", "filtered", "author_search", "multi_hop"}

# Retrieval routing configuration: query_type → list of retrieval channels
RETRIEVAL_ROUTING = {
    "simple":        ["bm25"],
    "semantic":      ["dense", "bm25"],
    "filtered":      ["kg", "bm25"],
    "author_search": ["kg", "bm25"],
    "multi_hop":     ["kg", "dense", "bm25"],
}


@dataclass
class QueryIntent:
    """
    Structured representation of a parsed user query.

    Produced by the query understanding module and consumed by the
    retrieval router to decide which channels to query.
    """
    original_query: str
    query_type: str  # One of VALID_QUERY_TYPES
    normalized_query: str  # Cleaned/expanded version of the query
    expanded_terms: List[str] = field(default_factory=list)  # Synonym expansions
    entities: Dict[str, str] = field(default_factory=dict)  # Extracted entities (author, publisher)
    retrieval_channels: List[str] = field(default_factory=list)  # Which channels to use
    confidence: float = 0.0  # Classification confidence
    reasoning: str = ""  # Why this classification was chosen

    def __str__(self) -> str:
        lines = [
            f"  Query Type: {self.query_type} (confidence: {self.confidence:.2f})",
            f"  Normalized: {self.normalized_query}",
        ]
        if self.expanded_terms:
            lines.append(f"  Expanded Terms: {', '.join(self.expanded_terms)}")
        if self.entities:
            for k, v in self.entities.items():
                lines.append(f"  Entity [{k}]: {v}")
        lines.append(f"  Channels: {', '.join(self.retrieval_channels)}")
        if self.reasoning:
            lines.append(f"  Reasoning: {self.reasoning}")
        return "\n".join(lines)


def _create_client() -> OpenAI:
    """Create an OpenAI client for the local Gemma server."""
    return OpenAI(base_url=GEMMA_BASE_URL, api_key=GEMMA_API_KEY)


def _extract_json_from_response(text: str) -> Optional[Dict]:
    """
    Extract JSON from LLM response, handling markdown code blocks
    and other common formatting issues.
    """
    # Try to find JSON in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def understand_query(query: str, temperature: float = 0.2) -> QueryIntent:
    """
    Analyze a user query using the local Gemma-4 model.

    Classifies the query type, extracts entities, and determines
    which retrieval channels to use.

    Handles Gemma-4 QAT (thinking model) which puts reasoning into
    a separate `reasoning_content` field, sometimes leaving `content`
    empty. We check both fields for JSON output.

    Args:
        query: The raw user query string.
        temperature: LLM temperature (low for deterministic classification).

    Returns:
        QueryIntent with classification, entities, and routing info.
    """
    client = _create_client()

    system_prompt = """You are a query analysis engine for a Bangla book search system.
Analyze the user's query and output a JSON object with these fields:

{
    "query_type": "simple" | "semantic" | "filtered" | "author_search" | "multi_hop",
    "normalized_query": "cleaned/expanded version of the query",
    "expanded_terms": ["synonym1", "synonym2"],
    "entities": {"author": "...", "publisher": "..."},
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}

Query type definitions:
- "simple": Direct keyword search (e.g., "মুক্তিযুদ্ধ", "একাত্তর")
- "semantic": Needs meaning-based search, conceptual queries (e.g., "১৯৭১ সালের যুদ্ধের গল্প")
- "filtered": Has metadata constraints like publisher (e.g., "রকমারি থেকে প্রকাশিত বই")
- "author_search": Looking for a specific author's works (e.g., "হুমায়ূন আহমেদের বই")
- "multi_hop": Requires reasoning across entities (e.g., "মুক্তিযোদ্ধা লেখকদের বই")

Rules:
- If an author name is mentioned, extract it into entities.author
- If a publisher name is mentioned, extract it into entities.publisher
- expanded_terms should include Bangla synonyms/related terms
- Output ONLY the JSON object, nothing else"""

    user_prompt = f'Analyze this query: "{query}"'

    try:
        response = client.chat.completions.create(
            model=GEMMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            # Gemma-4 QAT uses reasoning tokens heavily — need enough room
            # for both reasoning (thinking) + actual content output.
            # With 512, the model used 509 for reasoning and had no room left.
            max_tokens=2048,
        )

        choice = response.choices[0]
        content = choice.message.content or ""
        # Gemma-4 QAT (thinking model) puts reasoning in a separate field.
        # Sometimes the JSON ends up there instead of in content.
        reasoning_content = getattr(choice.message, "reasoning_content", "") or ""

        # Try to extract JSON from content first, then reasoning_content
        parsed = None
        for text_source, source_name in [
            (content, "content"),
            (reasoning_content, "reasoning_content"),
        ]:
            if text_source:
                parsed = _extract_json_from_response(text_source)
                if parsed:
                    print(f"[query_understanding] Parsed JSON from {source_name}")
                    break

        if parsed:
            return _build_intent_from_json(query, parsed)
        else:
            # Log what we got for debugging
            content_preview = content[:150] if content else "(empty)"
            reasoning_preview = reasoning_content[:150] if reasoning_content else "(empty)"
            print(f"[query_understanding] Failed to parse LLM JSON.")
            print(f"  content: {content_preview}")
            print(f"  reasoning_content: {reasoning_preview}")
            return _fallback_classify(query)

    except Exception as e:
        print(f"[query_understanding] LLM error: {e}. Using fallback classifier.")
        return _fallback_classify(query)


def _build_intent_from_json(query: str, parsed: Dict) -> QueryIntent:
    """Build a QueryIntent from parsed LLM JSON output."""
    query_type = parsed.get("query_type", "simple")
    if query_type not in VALID_QUERY_TYPES:
        query_type = "simple"

    entities = parsed.get("entities", {})
    # Clean empty entity values
    entities = {k: v for k, v in entities.items() if v}

    intent = QueryIntent(
        original_query=query,
        query_type=query_type,
        normalized_query=parsed.get("normalized_query", query),
        expanded_terms=parsed.get("expanded_terms", []),
        entities=entities,
        retrieval_channels=RETRIEVAL_ROUTING.get(query_type, ["bm25"]),
        confidence=float(parsed.get("confidence", 0.8)),
        reasoning=parsed.get("reasoning", ""),
    )
    return intent


def _fallback_classify(query: str) -> QueryIntent:
    """
    Rule-based fallback classifier when LLM is unavailable.

    Uses simple heuristics to classify the query type.
    """
    query_lower = query.lower().strip()

    # Check for author search patterns
    author_patterns = ["এর বই", "র বই", "লেখক", "রচিত", "লিখিত"]
    for pattern in author_patterns:
        if pattern in query_lower:
            # Try to extract author name (text before the pattern)
            idx = query_lower.find(pattern)
            author_name = query[:idx].strip()
            if author_name:
                return QueryIntent(
                    original_query=query,
                    query_type="author_search",
                    normalized_query=query,
                    entities={"author": author_name},
                    retrieval_channels=RETRIEVAL_ROUTING["author_search"],
                    confidence=0.7,
                    reasoning=f"Fallback: detected author search pattern '{pattern}'",
                )

    # Check for publisher/filter patterns
    filter_patterns = ["প্রকাশনী", "প্রকাশিত", "প্রকাশ"]
    for pattern in filter_patterns:
        if pattern in query_lower:
            return QueryIntent(
                original_query=query,
                query_type="filtered",
                normalized_query=query,
                retrieval_channels=RETRIEVAL_ROUTING["filtered"],
                confidence=0.6,
                reasoning=f"Fallback: detected filter pattern '{pattern}'",
            )

    # Check for multi-hop indicators
    multi_hop_patterns = ["যাদের", "যারা", "যে লেখক", "যে সকল"]
    for pattern in multi_hop_patterns:
        if pattern in query_lower:
            return QueryIntent(
                original_query=query,
                query_type="multi_hop",
                normalized_query=query,
                retrieval_channels=RETRIEVAL_ROUTING["multi_hop"],
                confidence=0.5,
                reasoning=f"Fallback: detected multi-hop pattern '{pattern}'",
            )

    # Check query length for simple vs semantic
    words = query.split()
    if len(words) <= 2:
        return QueryIntent(
            original_query=query,
            query_type="simple",
            normalized_query=query,
            retrieval_channels=RETRIEVAL_ROUTING["simple"],
            confidence=0.6,
            reasoning="Fallback: short query → simple keyword search",
        )
    else:
        return QueryIntent(
            original_query=query,
            query_type="semantic",
            normalized_query=query,
            retrieval_channels=RETRIEVAL_ROUTING["semantic"],
            confidence=0.5,
            reasoning="Fallback: longer query → semantic search",
        )


if __name__ == "__main__":
    # Test with sample queries
    test_queries = [
        "মুক্তিযুদ্ধ",
        "হুমায়ূন আহমেদের মুক্তিযুদ্ধের বই",
        "১৯৭১ সালের যুদ্ধের গল্প",
        "অন্যপ্রকাশ থেকে প্রকাশিত বই",
        "মুক্তিযোদ্ধা লেখকদের বই",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"🔍 Query: {query}")
        print(f"{'='*60}")
        intent = understand_query(query)
        print(intent)
