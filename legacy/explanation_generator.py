"""
explanation_generator.py — Gemma-4 explanation generation (Step 2.11, without ranking).

Connects to the local Gemma-4 12B model at http://localhost:1234/v1
via the OpenAI-compatible API. Generates natural-language explanations
for why each search result is relevant, using a template-plus-LLM approach.

The explanations are grounded in actual retrieval signals (matched terms,
similarity scores, field matches) to stay factual and traceable.
"""

from typing import List, Dict, Tuple, Optional
from openai import OpenAI


# Local Gemma-4 12B endpoint
GEMMA_BASE_URL = "http://localhost:1234/v1"
GEMMA_API_KEY = "not-needed"  # Local server ignores this
GEMMA_MODEL = "local-model"  # Local server uses whatever model is loaded


def _create_client() -> OpenAI:
    """Create an OpenAI client pointing to the local Gemma server."""
    return OpenAI(
        base_url=GEMMA_BASE_URL,
        api_key=GEMMA_API_KEY,
    )


def generate_explanation(
    query: str,
    results: List[Tuple[Dict[str, str], float, str]],
    retrieval_method: str,
    top_n: int = 5,
    temperature: float = 0.4,
) -> str:
    """
    Generate a natural-language explanation for search results using Gemma-4.

    Uses a template-plus-LLM approach: structured prompt with matched signals,
    LLM polishes into readable text with reasoning.

    Args:
        query: The original search query.
        results: List of (book_dict, score, reasoning) tuples from a retriever.
        retrieval_method: Name of the retrieval method ("BM25" or "Dense/Semantic").
        top_n: Number of top results to explain.
        temperature: LLM temperature (lower = more deterministic).

    Returns:
        Formatted explanation string.
    """
    client = _create_client()

    # Build the structured prompt with retrieval signals
    results_section = _format_results_for_prompt(results[:top_n])

    system_prompt = (
        "You are a search result explanation engine for a Bangla book catalogue. "
        "Your task is to explain WHY each search result is relevant to the user's query. "
        "You MUST base your explanation on the retrieval signals provided (matched terms, "
        "similarity scores, field matches). Do NOT hallucinate or invent relevance signals. "
        "Write in a clear, concise style mixing Bangla and English as appropriate. "
        "For each result, provide a 1-2 sentence explanation of its relevance."
    )

    user_prompt = f"""Query: "{query}"
Retrieval Method: {retrieval_method}

Search Results with Retrieval Signals:
{results_section}

For each result above, explain WHY it was retrieved and how relevant it is to the query.
Include the reasoning behind the match based on the retrieval signals.
Format each explanation as:

**Result #N: [Book Name]**
- Relevance: [Your explanation based on the signals]
- Retrieval reasoning: [How the retrieval method found this result]
"""

    try:
        response = client.chat.completions.create(
            model=GEMMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    except Exception as e:
        return (
            f"[explanation_generator] Error calling Gemma-4: {e}\n"
            f"Falling back to template-based explanation.\n\n"
            f"{_template_explanation(query, results[:top_n], retrieval_method)}"
        )


def _format_results_for_prompt(
    results: List[Tuple[Dict[str, str], float, str]]
) -> str:
    """Format search results into a structured string for the LLM prompt."""
    sections = []
    for i, (book, score, reasoning) in enumerate(results):
        section = (
            f"--- Result #{i+1} ---\n"
            f"Book Name: {book.get('book_name', 'Unknown')}\n"
            f"Author: {book.get('author', 'Unknown')}\n"
            f"Publisher: {book.get('publisher', 'Unknown')}\n"
            f"Description (excerpt): {book.get('description', '')[:300]}\n"
            f"Retrieval Signal: {reasoning}\n"
        )
        sections.append(section)
    return "\n".join(sections)


def _template_explanation(
    query: str,
    results: List[Tuple[Dict[str, str], float, str]],
    retrieval_method: str,
) -> str:
    """
    Fallback template-based explanation when LLM is unavailable.
    Generates structured explanations directly from retrieval signals.
    """
    lines = [f"📋 Explanation for query: \"{query}\" ({retrieval_method})\n"]

    for i, (book, score, reasoning) in enumerate(results):
        lines.append(f"  Result #{i+1}: {book.get('book_name', 'Unknown')}")
        lines.append(f"  Author: {book.get('author', 'Unknown')}")
        lines.append(f"  Retrieval Signal: {reasoning}")
        lines.append("")

    return "\n".join(lines)


def generate_comparison_explanation(
    query: str,
    bm25_results: List[Tuple[Dict[str, str], float, str]],
    dense_results: List[Tuple[Dict[str, str], float, str]],
    top_n: int = 3,
    temperature: float = 0.4,
) -> str:
    """
    Generate a comparative explanation of BM25 vs Dense search results.

    Explains the differences and complementary strengths of each method.

    Args:
        query: The search query.
        bm25_results: Results from BM25 retriever.
        dense_results: Results from Dense retriever.
        top_n: Number of top results from each to compare.
        temperature: LLM temperature.

    Returns:
        Comparative explanation string.
    """
    client = _create_client()

    bm25_section = _format_results_for_prompt(bm25_results[:top_n])
    dense_section = _format_results_for_prompt(dense_results[:top_n])

    system_prompt = (
        "You are a search system analyst for a Bangla book catalogue. "
        "Compare the results from two different retrieval methods and explain "
        "the strengths and weaknesses of each. Be concise and insightful. "
        "Write in a clear style mixing Bangla and English as appropriate."
    )

    user_prompt = f"""Query: "{query}"

=== BM25 (Lexical) Results ===
{bm25_section}

=== Dense (Semantic) Results ===
{dense_section}

Compare these two sets of results:
1. Which method found more relevant results and why?
2. What are the unique contributions of each method?
3. Are there results that appear in both, and what does that tell us?
4. What type of query is this (keyword-based, semantic, or mixed)?
"""

    try:
        response = client.chat.completions.create(
            model=GEMMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"[explanation_generator] Error calling Gemma-4: {e}"


if __name__ == "__main__":
    # Quick test with mock data
    mock_results = [
        (
            {
                "book_name": "মুক্তিযুদ্ধের ইতিহাস",
                "author": "রফিকুল ইসলাম",
                "publisher": "বাংলা একাডেমি",
                "description": "বাংলাদেশের মুক্তিযুদ্ধের বিস্তারিত ইতিহাস।",
            },
            8.5,
            "[BM25] Score: 8.5000 | Matched terms: 'মুক্তিযুদ্ধ' (×2) | Matched in: Book Name, Description",
        ),
    ]

    explanation = generate_explanation(
        query="মুক্তিযুদ্ধ",
        results=mock_results,
        retrieval_method="BM25",
    )
    print(explanation)
