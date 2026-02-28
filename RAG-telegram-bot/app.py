import streamlit as st
import streamlit.components.v1 as components

from retrieve import HybridRetriever
from llm import get_groq_client, build_context, generate_answer
from html_renderer import build_answer_html
from config import EXAMPLE_QUERIES
import voice

st.set_page_config(
    page_title="RAG for Climate Challenges",
    layout="centered",
    initial_sidebar_state="collapsed"
)

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
    .voice-status {
        font-size: 12px;
        color: #888;
        margin-top: 2px;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_retriever():
    return HybridRetriever()


@st.cache_resource
def load_whisper_model():
    return voice.load_model()


def _init_voice_state():
    defaults = {
        "voice_query": "",
        "voice_status": "",
        "show_recorder": False,
        "just_transcribed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_voice_recorder(whisper_model):
    """Render the mic button and audio recorder. Runs transcription when audio is ready."""
    col_input, col_mic = st.columns([11, 1])

    with col_input:
        url_query = st.query_params.get("q", "")
        if url_query:
            st.session_state["query_input"] = url_query
        elif st.session_state["voice_query"]:
            st.session_state["query_input"] = st.session_state["voice_query"]
            st.session_state["voice_query"] = ""

        query = st.text_input(
            "Ask a question",
            placeholder="e.g. What is India's cooling action plan?",
            label_visibility="collapsed",
            key="query_input",
        )

    with col_mic:
        mic_clicked = st.button(
            "Mic",
            key="mic_btn",
            help="Record a question. Supports English and Indian languages.",
            use_container_width=True,
            disabled=(whisper_model is None),
        )

    if mic_clicked:
        st.session_state["show_recorder"] = not st.session_state["show_recorder"]

    if st.session_state["show_recorder"]:
        audio_value = st.audio_input(
            "Speak your question",
            label_visibility="collapsed",
            key="voice_recorder",
        )
        if audio_value is not None:
            with st.spinner("Transcribing..."):
                try:
                    audio_np = voice.decode_audio(audio_value)
                    text, status = voice.transcribe(whisper_model, audio_np)
                    if text:
                        st.session_state["voice_query"] = text
                        st.session_state["voice_status"] = status
                    else:
                        st.session_state["voice_status"] = "No speech detected. Please try again."
                    st.session_state["show_recorder"] = False
                    st.session_state["just_transcribed"] = True
                except Exception as exc:
                    st.session_state["voice_status"] = f"Transcription error: {exc}"
            st.rerun()

    if st.session_state["voice_status"]:
        st.markdown(
            f'<p class="voice-status">{st.session_state["voice_status"]}</p>',
            unsafe_allow_html=True,
        )

    return query


def _render_answer(query, retriever, groq_client):
    """Run retrieval and generation, then render the answer."""
    with st.spinner("Searching..."):
        try:
            results = retriever.hybrid_search(query=query, top_k=5, brand_filter=None)
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            st.stop()

        if not results:
            st.info("No relevant documents found. Try a different query.")
            st.stop()

        context = build_context(results)
        answer = generate_answer(query, context, groq_client)

    answer_html = build_answer_html(answer, results)
    answer_lines = answer.count("\n") + 1
    estimated_height = 350 + (answer_lines * 22) + (len(results) * 55)
    estimated_height = min(max(estimated_height, 450), 1800)
    components.html(answer_html, height=estimated_height, scrolling=True)


def _render_example_queries():
    """Show example query buttons when the input is idle."""
    st.markdown("")
    st.markdown("##### Try asking")
    cols = st.columns(2)
    for idx, example in enumerate(EXAMPLE_QUERIES):
        with cols[idx % 2]:
            if st.button(example, key=f"ex_{idx}", use_container_width=True):
                st.query_params.update({"q": example})
                st.rerun()


def main():
    st.title("Retrieval Augmented Generation for Climate Challenges")
    st.caption("Search across your document collection")

    try:
        groq_client = get_groq_client()
    except ValueError as e:
        st.error(str(e))
        st.stop()

    try:
        with st.spinner("Loading document index..."):
            retriever = load_retriever()
    except Exception as e:
        st.error(f"Error connecting to ChromaDB: {str(e)}")
        st.stop()

    with st.spinner("Loading voice model..."):
        try:
            whisper_model = load_whisper_model()
        except Exception:
            whisper_model = None

    _init_voice_state()
    query = _render_voice_recorder(whisper_model)

    if st.session_state["just_transcribed"]:
        st.session_state["just_transcribed"] = False
        query = ""

    if query:
        _render_answer(query, retriever, groq_client)
    else:
        _render_example_queries()


if __name__ == "__main__":
    main()
