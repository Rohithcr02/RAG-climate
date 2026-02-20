"""Query expansion: generate alternative phrasings to improve retrieval recall."""

from typing import List
from groq import Groq

from config import LLM_MODEL

EXPANSION_PROMPT = """Generate 2 alternative phrasings of the following question.
The alternatives should use different vocabulary but ask the same thing.
Return only the 2 alternatives, one per line, no numbering, no explanation.

Question: {query}"""


def expand_query(query: str, groq_client: Groq) -> List[str]:
    """
    Return the original query plus 2 rephrased alternatives.
    Falls back to the original query alone if the LLM call fails.
    """
    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": EXPANSION_PROMPT.format(query=query)}],
            model=LLM_MODEL,
            temperature=0.5,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()
        alternatives = [line.strip() for line in raw.splitlines() if line.strip()][:2]
        return [query] + alternatives

    except Exception:
        return [query]
