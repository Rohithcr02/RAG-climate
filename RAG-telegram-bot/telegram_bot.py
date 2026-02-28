"""Telegram bot for RAG-Climate Q&A — with rich HTML formatting."""

import html
import logging
import os
import re

import chromadb
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from llm import build_context, generate_answer, get_groq_client
from retrieve import HybridRetriever

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── One-time resource initialisation ────────────────────────────────────────
logger.info("Loading retriever…")
retriever = HybridRetriever()
groq_client = get_groq_client()
logger.info("Ready.")


# ── Formatting helpers ───────────────────────────────────────────────────────

def escape_html(text: str) -> str:
    """Escape characters that break Telegram HTML mode."""
    return html.escape(text)


def format_answer_html(answer: str, results: list) -> str:
    """
    Convert the plain-text RAG answer into a Telegram HTML message.

    Transformations applied:
    - [N] citation markers  →  <code>[N]</code>
    - Lines starting with - →  • bullet points
    - Source footer with doc name + page
    """
    # Escape the raw answer for HTML
    safe = escape_html(answer)

    # Convert citation markers [1], [2], ... → styled inline code
    safe = re.sub(r'\[(\d+)\]', r'<code>[\1]</code>', safe)

    # Convert markdown-style bullets "- text" at start of line → • text
    safe = re.sub(r'(?m)^- (.+)$', r'• \1', safe)

    # Build source footer
    source_lines = []
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        raw_name = meta["filename"].replace(".pdf", "").replace("_", " ").replace("-", " ")
        # Strip leading numeric prefixes like "0007 1 " or "0002 "
        clean_name = re.sub(r'^\d+\s+', '', raw_name).strip()
        doc_name = escape_html(clean_name)
        page = meta.get("page_number", "?")
        source_lines.append(f"  <code>[{i}]</code> {doc_name} — p.{page}")

    sources_block = "\n".join(source_lines)

    return (
        f"{safe}\n\n"
        f"<b>📚 Sources</b>\n"
        f"{sources_block}"
    )


def split_message(text: str, max_len: int = 4000) -> list[str]:
    """Split a message into chunks that fit within Telegram's 4096-char limit."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Try to split at a newline to avoid cutting mid-sentence
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


# ── Command handlers ──────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 <b>Hi! I'm a Climate Research Assistant.</b>\n\n"
        "I can answer questions about:\n"
        "• Refrigerants &amp; HVAC\n"
        "• Cooling policies &amp; action plans\n"
        "• Environmental regulations\n"
        "• Climate research documents\n\n"
        "<b>Try asking:</b>\n"
        "• What is the India Cooling Action Plan?\n"
        "• What are low-GWP refrigerant alternatives?\n"
        "• What is the Montreal Protocol?\n"
        "• What training is needed for RAC technicians?\n\n"
        "Type any question to get started! 🌍",
        parse_mode=ParseMode.HTML,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "<b>ℹ️ Commands</b>\n\n"
        "/start   – Welcome message\n"
        "/help    – This help message\n"
        "/sources – List all 29 PDF documents in the knowledge base\n\n"
        "<b>How it works</b>\n"
        "1. You type a question\n"
        "2. I search across climate &amp; HVAC documents\n"
        "3. I generate a cited answer using AI\n"
        "4. Sources are listed at the bottom\n\n"
        "<b>Tips</b>\n"
        "• Be specific for better answers\n"
        "• Ask about policies, chemicals, or procedures\n"
        "• Citations like <code>[1]</code> refer to the source list at the bottom",
        parse_mode=ParseMode.HTML,
    )


async def sources_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all unique PDF documents in the ChromaDB knowledge base."""
    try:
        chroma_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
        collection_name = os.getenv("CHROMA_COLLECTION_NAME", "saikiran_corpus")
        client = chromadb.PersistentClient(path=chroma_path)
        col = client.get_collection(collection_name)

        results = col.get(include=["metadatas"])
        filenames = sorted(set(m["filename"] for m in results["metadatas"]))
        total_chunks = col.count()

        lines = []
        for i, fname in enumerate(filenames, 1):
            # Clean up display name
            display = fname.replace(".pdf", "").replace("_", " ").replace("-", " ")
            # Strip leading numeric prefix like "0002 "
            display = re.sub(r'^\d+\s+', '', display).strip()
            lines.append(f"  {i}. {escape_html(display)}")

        doc_list = "\n".join(lines)
        reply = (
            f"<b>📚 Knowledge Base — {len(filenames)} Documents</b>\n"
            f"<i>{total_chunks} total searchable passages</i>\n\n"
            f"{doc_list}"
        )
        await update.message.reply_text(reply, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.error(f"Error in /sources: {e}", exc_info=True)
        await update.message.reply_text("⚠️ Could not retrieve source list.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Core handler: retrieve → generate → reply with formatted HTML."""
    query = update.message.text.strip()
    if not query:
        return

    logger.info(f"Query from {update.effective_user.id}: {query!r}")

    # Send a "Searching…" placeholder so user knows something is happening
    status_msg = await update.message.reply_text(
        "🔍 <i>Searching documents…</i>",
        parse_mode=ParseMode.HTML,
    )

    try:
        # 1. Retrieve
        await update.message.chat.send_action(ChatAction.TYPING)
        results = retriever.hybrid_search(query=query, top_k=5)

        if not results:
            await status_msg.edit_text(
                "❌ <b>No relevant documents found.</b>\n\n"
                "Try rephrasing, or ask about refrigerants, cooling policies, or climate regulations.",
                parse_mode=ParseMode.HTML,
            )
            return

        # Update status
        await status_msg.edit_text(
            "⚙️ <i>Generating answer…</i>",
            parse_mode=ParseMode.HTML,
        )
        await update.message.chat.send_action(ChatAction.TYPING)

        # 2. Generate
        context_str = build_context(results)
        answer = generate_answer(query, context_str, groq_client)

        # 3. Format
        reply = format_answer_html(answer, results)

        # 4. Send (delete the status message, send the real answer)
        await status_msg.delete()

        for chunk in split_message(reply):
            await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.error(f"Error handling message: {e}", exc_info=True)
        await status_msg.edit_text(
            "⚠️ <b>Something went wrong.</b> Please try again in a moment.",
            parse_mode=ParseMode.HTML,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in .env")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("sources", sources_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is running (polling)…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
