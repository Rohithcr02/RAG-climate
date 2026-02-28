"""Prompts, model settings, and constants."""

# LLM
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 800
LLM_TOP_P = 0.9

# System prompt
SYSTEM_PROMPT = """You are a research assistant. Answer the question using ONLY the provided sources.

RULES:
1. Be concise. Get to the point. No filler, no restating the question. Aim for 100-200 words.
2. Cite inline. After a key claim, add the source number: "India joined in 1992 [1]." Use only ONE citation per claim.
3. Use bullet points for lists. Do NOT use bold or any special formatting.
4. Be specific. Include dates, numbers, names from the sources.
5. Only cite specific facts. Connecting sentences don't need citations.
6. If unsure, say "The documents don't cover this."
7. Do NOT use markdown headers or bold text. Write in plain text only.

Sources:
{context}

Question: {query}

Answer:"""

SYSTEM_MESSAGE = (
    "You are a concise research assistant. Answer directly using inline [N] citations. "
    "Be brief and specific. No filler. No bold text."
)

# Example queries shown in the UI
EXAMPLE_QUERIES = [
    "What is the Montreal Protocol and India's role in it?",
    "What are low-GWP refrigerant alternatives?",
    "What are passive cooling strategies for buildings?",
    "What training is required for RAC technicians?",
    "What is the India Cooling Action Plan?",
]

# Eval
import os

EVAL_LLM_MODEL = LLM_MODEL
EVAL_TEST_SET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "test_set.json")
EVAL_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "results")
