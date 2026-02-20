"""Prompts, model settings, and constants."""

# LLM
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1024
LLM_TOP_P = 0.9

# Retrieval
RETRIEVAL_TOP_K = 5
RETRIEVAL_CANDIDATE_K = 20  # candidates fetched before reranking

# System prompt
SYSTEM_PROMPT = """You are a research assistant. Answer the question using ONLY information explicitly stated in the provided sources.

RULES:
1. Be concise. Aim for 100-200 words. No filler, no restating the question.
2. Every factual sentence must have an inline citation: "India joined in 1992 [1]." Use one citation per claim.
3. Do not state anything that is not directly supported by a cited source.
4. Use bullet points for lists. No bold text, no markdown headers, plain text only.
5. Be specific. Include dates, numbers, and names from the sources.
6. If the sources do not cover the question, say "The documents don't cover this."

Sources:
{context}

Question: {query}

Answer:"""

SYSTEM_MESSAGE = (
    "You are a concise research assistant. Answer using only the provided sources. "
    "Every factual claim must have an inline [N] citation. No bold text. No filler."
)

# Example queries shown in the UI
EXAMPLE_QUERIES = [
    "What is the Montreal Protocol and India's role in it?",
    "What are low-GWP refrigerant alternatives?",
    "What are passive cooling strategies for buildings?",
    "What training is required for RAC technicians?",
    "What is the India Cooling Action Plan?",
]
