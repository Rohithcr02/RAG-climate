"""Telegram bot powered by Contextual AI agent (replaces local RAG pipeline)."""

import io
import logging
import os
import re
import html
import tempfile

import requests
from gtts import gTTS
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

# ── Indian language options for TTS ──────────────────────────────────────────
INDIAN_LANGUAGES = {
    "hi": "🇮🇳 Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "en": "🌐 English",
}

# Per-user preferences: {chat_id: {"lang": "hi", "tts": True}}
user_prefs: dict[int, dict] = {}

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CONTEXTUAL_API_KEY = os.getenv("CONTEXTUAL_API_KEY")
CONTEXTUAL_AGENT_ID = os.getenv("CONTEXTUAL_AGENT_ID", "840be60e-c004-4850-8fb9-980a691f431a")
CONTEXTUAL_BASE_URL = "https://api.contextual.ai/v1"


# ── Contextual AI helpers ────────────────────────────────────────────────────

def get_agent_datastores() -> dict:
    """
    Fetch the datastores (document collections) linked to the agent.
    Returns a list of datastore dicts.
    """
    url = f"{CONTEXTUAL_BASE_URL}/agents/{CONTEXTUAL_AGENT_ID}"
    headers = {"Authorization": f"Bearer {CONTEXTUAL_API_KEY}"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    agent_data = response.json()
    return agent_data


def query_contextual_agent(question: str) -> dict:
    """
    Send a question to the Contextual AI agent.
    Returns the full response dict with 'message' and 'retrieval_contents'.
    """
    url = f"{CONTEXTUAL_BASE_URL}/agents/{CONTEXTUAL_AGENT_ID}/query/acl"
    headers = {
        "Authorization": f"Bearer {CONTEXTUAL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [{"role": "user", "content": question}],
        "stream": False,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


# ── Formatting ────────────────────────────────────────────────────────────────

def escape_html(text: str) -> str:
    return html.escape(str(text))


def format_response(data: dict) -> str:
    """
    Format the Contextual AI response into a Telegram HTML message.
    Includes the answer and a source footer.
    """
    answer = data.get("message", {}).get("content", "No answer returned.")
    retrieval = data.get("retrieval_contents", [])

    # Escape for HTML
    safe_answer = escape_html(answer)

    # Convert [N] citation markers to monospace
    safe_answer = re.sub(r'\[(\d+)\]', r'<code>[\1]</code>', safe_answer)

    # Convert markdown bullets to •
    safe_answer = re.sub(r'(?m)^[-*] (.+)$', r'• \1', safe_answer)

    # Build source footer from retrieval_contents
    if retrieval:
        seen = {}
        for item in retrieval:
            num = item.get("number", "?")
            # Try ctxl_metadata first, then top-level fields
            meta = item.get("ctxl_metadata", {})
            doc_name = (
                meta.get("document_title")
                or item.get("doc_name")
                or "Unknown document"
            )
            page = meta.get("page") or item.get("page") or "?"
            # Clean up numeric prefixes from filenames
            doc_name = re.sub(r'^\d+[\s_]+', '', doc_name).strip()
            if num not in seen:
                seen[num] = (escape_html(doc_name), page)

        source_lines = [
            f"  <code>[{num}]</code> {name} — p.{page}"
            for num, (name, page) in sorted(seen.items())
        ]
        sources_block = "\n".join(source_lines)
        footer = f"\n\n<b>📚 Sources</b>\n{sources_block}"
    else:
        footer = ""

    return safe_answer + footer


def split_message(text: str, max_len: int = 4000) -> list:
    """Split long messages to fit Telegram's 4096-char limit."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
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
        "• What is the Montreal Protocol?\n\n"
        "Type any question to get started! 🌍",
        parse_mode=ParseMode.HTML,
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "<b>ℹ️ Commands</b>\n\n"
        "/start    – Welcome message\n"
        "/help     – This help message\n"
        "/sources  – Show connected data sources\n"
        "/language – Choose voice language for audio replies\n\n"
        "<b>How it works</b>\n"
        "1. You type a question\n"
        "2. The AI searches across climate &amp; HVAC documents\n"
        "3. You get a cited text answer\n"
        "4. A voice MP3 is sent in your chosen language\n\n"
        "Powered by <b>Contextual AI</b> + <b>gTTS</b> 🤖🔊",
        parse_mode=ParseMode.HTML,
    )


async def language_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show inline keyboard to pick TTS language."""
    chat_id = update.effective_chat.id
    current = user_prefs.get(chat_id, {}).get("lang", "en")

    # Build 2-column keyboard
    buttons = []
    items = list(INDIAN_LANGUAGES.items())
    for i in range(0, len(items), 2):
        row = []
        for code, name in items[i:i+2]:
            label = f"✅ {name}" if code == current else name
            row.append(InlineKeyboardButton(label, callback_data=f"lang:{code}"))
        buttons.append(row)

    await update.message.reply_text(
        "🔊 <b>Choose your voice language</b>\n"
        "Audio will be sent after every answer in the selected language.",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(buttons),
    )


async def language_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle language selection from inline keyboard."""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    lang_code = query.data.split(":")[1]
    lang_name = INDIAN_LANGUAGES.get(lang_code, lang_code)

    user_prefs.setdefault(chat_id, {})["lang"] = lang_code

    # Rebuild keyboard with updated checkmark
    buttons = []
    items = list(INDIAN_LANGUAGES.items())
    for i in range(0, len(items), 2):
        row = []
        for code, name in items[i:i+2]:
            label = f"✅ {name}" if code == lang_code else name
            row.append(InlineKeyboardButton(label, callback_data=f"lang:{code}"))
        buttons.append(row)

    await query.edit_message_text(
        f"🔊 <b>Choose your voice language</b>\n"
        f"Audio will be sent after every answer in the selected language.\n\n"
        f"✅ Now set to: <b>{lang_name}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(buttons),
    )


def translate_text(text: str, target_lang: str) -> str:
    """Translate text to the target language using Google Translate. Skip if English."""
    if target_lang == "en":
        return text
    try:
        translated = GoogleTranslator(source="auto", target=target_lang).translate(text)
        return translated or text
    except Exception as e:
        logger.warning(f"Translation failed ({target_lang}): {e}")
        return text  # fall back to English audio


def generate_voice(text: str, lang: str) -> io.BytesIO:
    """Translate text to target language, then convert to MP3 using gTTS."""
    # Strip HTML tags and citation markers for clean input
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"\[(\d+)\]", "", clean).strip()

    # Step 1: translate to target language
    translated = translate_text(clean, lang)

    # Step 2: speak the translated text
    tts = gTTS(text=translated, lang=lang, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    buf.name = "answer.mp3"
    return buf



async def sources_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show which datastores are connected to the Contextual AI agent."""
    status_msg = await update.message.reply_text(
        "🔍 <i>Fetching data sources…</i>", parse_mode=ParseMode.HTML
    )
    try:
        agent_data = get_agent_datastores()

        # Agent metadata
        agent_name = escape_html(agent_data.get("name", "Unknown Agent"))
        agent_id   = escape_html(agent_data.get("id", CONTEXTUAL_AGENT_ID))

        # Datastores list
        datastores = agent_data.get("datastore_ids", []) or agent_data.get("datastores", [])

        lines = [f"<b>🤖 Agent:</b> {agent_name}", f"<b>🆔 ID:</b> <code>{agent_id}</code>", ""]

        if datastores:
            lines.append(f"<b>📂 Connected Datastores ({len(datastores)}):</b>")
            for ds in datastores:
                if isinstance(ds, dict):
                    ds_id   = escape_html(str(ds.get("id", "?")))
                    ds_name = escape_html(ds.get("name", "Unnamed"))
                    lines.append(f"  • <b>{ds_name}</b> — <code>{ds_id}</code>")
                else:
                    lines.append(f"  • <code>{escape_html(str(ds))}</code>")
        else:
            lines.append("⚠️ No datastores found in agent metadata.")
            lines.append("(Documents may be embedded directly in the agent)")

        await status_msg.edit_text("\n".join(lines), parse_mode=ParseMode.HTML)

    except requests.HTTPError as e:
        logger.error(f"Error fetching agent info: {e}")
        await status_msg.edit_text(
            f"⚠️ <b>API Error:</b> {escape_html(str(e))}",
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        logger.error(f"Error in /sources: {e}", exc_info=True)
        await status_msg.edit_text(
            "⚠️ <b>Could not fetch data sources.</b> Please try again.",
            parse_mode=ParseMode.HTML,
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Core handler: query Contextual AI → format → reply → voice."""
    question = update.message.text.strip()
    if not question:
        return

    chat_id = update.effective_chat.id
    logger.info(f"Query from {update.effective_user.id}: {question!r}")

    status_msg = await update.message.reply_text(
        "🔍 <i>Searching documents…</i>",
        parse_mode=ParseMode.HTML,
    )

    try:
        await update.message.chat.send_action(ChatAction.TYPING)
        await status_msg.edit_text(
            "⚙️ <i>Generating answer…</i>",
            parse_mode=ParseMode.HTML,
        )
        data = query_contextual_agent(question)

        reply = format_response(data)
        await status_msg.delete()

        for chunk in split_message(reply):
            await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)

        # ── TTS: send voice MP3 ──────────────────────────────────────────────
        lang = user_prefs.get(chat_id, {}).get("lang", "en")
        try:
            await update.message.chat.send_action(ChatAction.UPLOAD_VOICE)
            voice_buf = generate_voice(reply, lang)
            lang_name = INDIAN_LANGUAGES.get(lang, lang)
            await update.message.reply_audio(
                audio=voice_buf,
                filename="answer.mp3",
                title=f"Answer ({lang_name})",
                performer="Climate AI",
            )
        except Exception as tts_err:
            logger.warning(f"TTS failed (non-fatal): {tts_err}")
            await update.message.reply_text(
                f"🔇 <i>Voice generation failed: {escape_html(str(tts_err))}</i>",
                parse_mode=ParseMode.HTML,
            )

    except requests.HTTPError as e:
        logger.error(f"Contextual AI API error: {e}")
        await status_msg.edit_text(
            f"⚠️ <b>API Error:</b> {escape_html(str(e))}\n\nCheck your API key and agent ID.",
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        logger.error(f"Error handling message: {e}", exc_info=True)
        await status_msg.edit_text(
            "⚠️ <b>Something went wrong.</b> Please try again.",
            parse_mode=ParseMode.HTML,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN not set in .env")
    if not CONTEXTUAL_API_KEY:
        raise ValueError("CONTEXTUAL_API_KEY not set in .env")

    logger.info(f"Using Contextual AI agent: {CONTEXTUAL_AGENT_ID}")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("sources", sources_command))
    app.add_handler(CommandHandler("language", language_command))
    app.add_handler(CallbackQueryHandler(language_callback, pattern=r"^lang:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot is running (polling)…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
