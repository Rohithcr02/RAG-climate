import streamlit as st
import streamlit.components.v1 as components
import os
import re
import html as html_lib
from dotenv import load_dotenv
from groq import Groq
from retrieve import HybridRetriever

load_dotenv()

# Page configuration — no sidebar
st.set_page_config(
    page_title="RAG for Climate Challenges",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Hide sidebar completely + clean minimal styling
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    .stApp { background-color: #ffffff; }
    .block-container {
        max-width: 720px;
        padding-top: 3rem;
        padding-bottom: 2rem;
    }
    h1 { font-weight: 500; font-size: 1.6rem; color: #111; letter-spacing: -0.02em; }
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 12px 16px;
        font-size: 15px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #999;
        box-shadow: none;
    }
    .stButton > button {
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        background: #fafafa;
        color: #333;
        font-size: 13px;
        padding: 8px 14px;
        font-weight: 400;
    }
    .stButton > button:hover {
        background: #f0f0f0;
        border-color: #ccc;
    }
    .stSpinner > div { color: #666; }
</style>
""", unsafe_allow_html=True)

# System prompt — concise, no bold
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


@st.cache_resource
def load_retriever():
    return HybridRetriever()


def build_answer_html(answer_text, results):
    """
    Build a self-contained HTML block with the answer, inline clickable citations,
    and collapsible source cards.
    """
    # Build source data
    sources = []
    for i, result in enumerate(results, 1):
        meta = result['metadata']
        display_name = meta['filename'].replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        sources.append({
            'num': i,
            'filename': meta['filename'],
            'display_name': display_name,
            'page': meta['page_number'],
            'text': html_lib.escape(result['document'])
        })

    # Escape the answer text for HTML
    safe_answer = html_lib.escape(answer_text)

    # Strip any bold markers the LLM might still produce
    safe_answer = re.sub(r'\*\*(.+?)\*\*', r'\1', safe_answer)

    # Convert bullet points
    safe_answer = re.sub(r'^- (.+)$', r'<li>\1</li>', safe_answer, flags=re.MULTILINE)
    safe_answer = re.sub(r'^(\* )(.+)$', r'<li>\2</li>', safe_answer, flags=re.MULTILINE)
    safe_answer = re.sub(r'((?:<li>.*?</li>\n?)+)', r'<ul>\1</ul>', safe_answer)

    # Convert numbered lists
    safe_answer = re.sub(r'^(\d+)\. (.+)$', r'<li>\2</li>', safe_answer, flags=re.MULTILINE)

    # Replace [N] with clickable citation pills
    def replace_citation(match):
        num = match.group(1)
        return f'<span class="cite" onclick="showSource({num})">{num}</span>'

    safe_answer = re.sub(r'\[(\d+)\]', replace_citation, safe_answer)

    # Convert newlines to paragraphs
    paragraphs = safe_answer.split('\n')
    formatted = []
    for p in paragraphs:
        p = p.strip()
        if p and not p.startswith('<ul>') and not p.startswith('<li>') and not p.startswith('</ul>'):
            formatted.append(f'<p>{p}</p>')
        elif p:
            formatted.append(p)
    safe_answer = '\n'.join(formatted)

    # Build source cards
    source_cards_html = ""
    for src in sources:
        source_cards_html += f"""
        <div class="source-card" id="source-{src['num']}">
            <div class="source-header" onclick="toggleSource({src['num']})">
                <div class="source-num">{src['num']}</div>
                <div class="source-meta">
                    <div class="source-title">{src['display_name']}</div>
                    <div class="source-page">Page {src['page']}</div>
                </div>
                <div class="source-chevron" id="chevron-{src['num']}">
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                        <path d="M4 6L8 10L12 6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
            </div>
            <div class="source-body" id="body-{src['num']}">
                <div class="source-text">{src['text']}</div>
                <div class="source-file">{src['filename']}</div>
            </div>
        </div>
        """

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            color: #222;
            line-height: 1.7;
            font-size: 14.5px;
            background: transparent;
            -webkit-font-smoothing: antialiased;
        }}

        .answer-content {{
            padding: 0 0 20px 0;
        }}
        .answer-content p {{
            margin-bottom: 8px;
        }}
        .answer-content ul {{
            margin: 6px 0 10px 18px;
            padding: 0;
        }}
        .answer-content li {{
            margin-bottom: 5px;
            line-height: 1.6;
        }}

        .cite {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            font-weight: 600;
            min-width: 15px;
            height: 15px;
            padding: 0 3px;
            border-radius: 3px;
            background: #eee;
            color: #666;
            cursor: pointer;
            vertical-align: super;
            margin: 0 1px;
            line-height: 1;
            transition: all 0.15s ease;
            position: relative;
            top: -1px;
        }}
        .cite:hover {{
            background: #ddd;
            color: #333;
        }}
        .cite.active {{
            background: #333;
            color: #fff;
        }}

        .sources-section {{
            border-top: 1px solid #eee;
            padding-top: 16px;
            margin-top: 4px;
        }}
        .sources-label {{
            font-size: 11px;
            font-weight: 500;
            color: #999;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-bottom: 10px;
        }}

        .source-card {{
            border: 1px solid #eee;
            border-radius: 8px;
            margin-bottom: 6px;
            overflow: hidden;
            transition: border-color 0.2s ease;
            background: #fff;
        }}
        .source-card:hover {{
            border-color: #ccc;
        }}
        .source-card.highlighted {{
            border-color: #333;
        }}

        .source-header {{
            display: flex;
            align-items: center;
            padding: 10px 14px;
            cursor: pointer;
            user-select: none;
            gap: 10px;
        }}
        .source-header:hover {{
            background: #fafafa;
        }}

        .source-num {{
            display: flex;
            align-items: center;
            justify-content: center;
            width: 20px;
            height: 20px;
            border-radius: 4px;
            background: #f5f5f5;
            color: #666;
            font-size: 11px;
            font-weight: 500;
            flex-shrink: 0;
        }}

        .source-meta {{
            flex: 1;
            min-width: 0;
        }}
        .source-title {{
            font-size: 13px;
            font-weight: 400;
            color: #333;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .source-page {{
            font-size: 11px;
            color: #999;
            margin-top: 1px;
        }}

        .source-chevron {{
            color: #999;
            transition: transform 0.2s ease;
            flex-shrink: 0;
        }}
        .source-chevron.open {{
            transform: rotate(180deg);
        }}

        .source-body {{
            display: none;
            padding: 0 14px 12px 44px;
        }}
        .source-body.open {{
            display: block;
        }}
        .source-text {{
            font-size: 12.5px;
            line-height: 1.55;
            color: #555;
            white-space: pre-wrap;
            max-height: 250px;
            overflow-y: auto;
            padding: 10px;
            background: #fafafa;
            border-radius: 6px;
            border: 1px solid #f0f0f0;
        }}
        .source-file {{
            font-size: 11px;
            color: #aaa;
            margin-top: 6px;
        }}
    </style>
    </head>
    <body>
        <div class="answer-content">
            {safe_answer}
        </div>

        <div class="sources-section">
            <div class="sources-label">Sources</div>
            {source_cards_html}
        </div>

        <script>
            function toggleSource(num) {{
                const body = document.getElementById('body-' + num);
                const chevron = document.getElementById('chevron-' + num);
                const isOpen = body.classList.contains('open');
                if (isOpen) {{
                    body.classList.remove('open');
                    chevron.classList.remove('open');
                }} else {{
                    body.classList.add('open');
                    chevron.classList.add('open');
                }}
            }}

            function showSource(num) {{
                const card = document.getElementById('source-' + num);
                const body = document.getElementById('body-' + num);
                const chevron = document.getElementById('chevron-' + num);
                if (!card) return;

                document.querySelectorAll('.source-card.highlighted').forEach(el => {{
                    el.classList.remove('highlighted');
                }});
                document.querySelectorAll('.cite.active').forEach(el => {{
                    el.classList.remove('active');
                }});

                event.target.classList.add('active');

                if (!body.classList.contains('open')) {{
                    body.classList.add('open');
                    chevron.classList.add('open');
                }}

                card.classList.add('highlighted');
                card.scrollIntoView({{ behavior: 'smooth', block: 'center' }});

                setTimeout(() => {{
                    card.classList.remove('highlighted');
                }}, 2500);
            }}
        </script>
    </body>
    </html>
    """
    return full_html


def generate_answer(query: str, context: str, groq_client: Groq) -> str:
    prompt = SYSTEM_PROMPT.format(context=context, query=query)

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise research assistant. Answer directly using inline [N] citations. Be brief and specific. No filler. No bold text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=800,
            top_p=0.9,
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Error generating answer: {str(e)}"


def main():
    st.title("Retrieval Augmented Generation for Climate Challenges")
    st.caption("Search across your document collection")

    # Initialize Groq client
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found. Please add it to your .env file.")
        st.stop()

    groq_client = Groq(api_key=groq_api_key)

    # Load retriever
    try:
        with st.spinner("Loading document index..."):
            retriever = load_retriever()
    except Exception as e:
        st.error(f"Error connecting to ChromaDB: {str(e)}")
        st.stop()

    # Read query from URL params (set by example buttons)
    default_query = st.query_params.get("q", "")

    # Search input
    query = st.text_input(
        "Ask a question",
        value=default_query,
        placeholder="e.g. What is India's cooling action plan?",
        label_visibility="collapsed"
    )

    if query:
        with st.spinner("Searching..."):
            try:
                results = retriever.hybrid_search(
                    query=query,
                    top_k=5,
                    brand_filter=None
                )
            except Exception as e:
                st.error(f"Search error: {str(e)}")
                st.stop()

            if not results:
                st.info("No relevant documents found. Try a different query.")
                st.stop()

            # Build context
            context_parts = []
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                context_parts.append(
                    f"[Source {i}] (Document: {metadata['filename']}, Page: {metadata['page_number']})\n"
                    f"{result['document']}\n"
                )
            context = "\n---\n".join(context_parts)

            # Generate answer
            answer = generate_answer(query, context, groq_client)

        # Render answer with citations
        answer_html = build_answer_html(answer, results)

        answer_lines = answer.count('\n') + 1
        estimated_height = 350 + (answer_lines * 22) + (len(results) * 55)
        estimated_height = min(max(estimated_height, 450), 1800)

        components.html(answer_html, height=estimated_height, scrolling=True)

    # Example queries when no question asked
    if not query:
        st.markdown("")
        st.markdown("##### Try asking")
        example_queries = [
            "What is the Montreal Protocol and India's role in it?",
            "What are low-GWP refrigerant alternatives?",
            "What are passive cooling strategies for buildings?",
            "What training is required for RAC technicians?",
            "What is the India Cooling Action Plan?",
        ]

        cols = st.columns(2)
        for idx, example in enumerate(example_queries):
            with cols[idx % 2]:
                if st.button(example, key=f"ex_{idx}", use_container_width=True):
                    st.query_params.update({"q": example})
                    st.rerun()


if __name__ == "__main__":
    main()
