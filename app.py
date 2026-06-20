import hashlib
import io

import streamlit as st
from google.api_core.exceptions import ResourceExhausted

from utils.chunker import create_chunks
from utils.embeddings import get_embedding_model
from utils.gemini_helper import get_answer
from utils.pdf_reader import extract_text
from utils.quiz import generate_quiz
from utils.summary import generate_summary
from utils.vector_store import create_vector_store


st.set_page_config(
    page_title="AI Study Buddy",
    page_icon=":books:",
    layout="wide",
)

st.markdown("""
<style>
:root {
    --bg: #f3f7fb;
    --card: #ffffff;
    --text: #111827;
    --muted: #6b7280;
    --border: #dbe4ee;
    --primary: #2563eb;
    --secondary: #3b82f6;
}

/* Main app */
.stApp {
    background: linear-gradient(135deg, #eef4ff 0%, #f8fbff 50%, #edf8f3 100%);
    color: var(--text);
    font-family: "Segoe UI", sans-serif;
}

/* Global text */
html, body, p, span, div, label, li, h1, h2, h3, h4, h5 {
    color: #111827 !important;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6);
    color: white !important;
    padding: 2rem;
    border-radius: 20px;
    margin-bottom: 20px;
    box-shadow: 0 15px 35px rgba(37,99,235,0.25);
}

.hero h1, .hero p {
    color: white !important;
}

/* Cards */
.card {
    background: white;
    border: 1px solid #dbe4ee;
    border-radius: 18px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}

/* Chips */
.chip {
    display: inline-block;
    padding: 8px 14px;
    margin-right: 8px;
    border-radius: 999px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1d4ed8 !important;
    font-weight: 600;
}

/* Metrics */
.metric {
    background: white;
    border-radius: 16px;
    padding: 18px;
    text-align: center;
    border: 1px solid #dbe4ee;
    box-shadow: 0 6px 20px rgba(0,0,0,0.04);
}

.metric .label {
    color: #6b7280 !important;
}

.metric .value {
    color: #111827 !important;
    font-size: 26px;
    font-weight: bold;
}

/* Buttons */
.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 12px !important;
}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    box-shadow: 0 12px 20px rgba(37,99,235,0.25);
    transform: translateY(-2px);
}

/* Input */
.stTextInput input,
.stTextArea textarea {
    background: white !important;
    color: #111827 !important;
    border: 2px solid #cbd5e1 !important;
    border-radius: 12px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 2px dashed #3b82f6 !important;
    border-radius: 16px !important;
    padding: 20px !important;
}

[data-testid="stFileUploader"] section {
    background: #ffffff !important;
}

/* Tabs */
[data-baseweb="tab-list"] {
    gap: 12px !important;
    background: transparent !important;
    padding: 8px 0 !important;
}

[data-baseweb="tab"] {
    height: 50px !important;
    padding: 0 24px !important;
    border-radius: 14px !important;
    background: white !important;
    border: 1px solid #dbe4ee !important;
    color: #475569 !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

[data-baseweb="tab"]:hover {
    background: #eff6ff !important;
    color: #2563eb !important;
}

[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    color: white !important;
    border: none !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a !important;
}

/* Sidebar text only */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white !important;
}

/* Sidebar arrow / hamburger icon */
button[kind="header"] {
    color: white !important;
}

button[kind="header"] svg {
    fill: white !important;
    stroke: white !important;
    color: white !important;
    opacity: 10 !important;
}

/* Chunk cards */
.chunk-card {
    background: #f8fafc;
    border-left: 4px solid #2563eb;
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 12px;
}

/* Success box */
[data-baseweb="notification"] {
    background: #eff6ff !important;
    border-left: 5px solid #2563eb !important;
}

/* Spinner */
[data-testid="stSpinner"] * {
    stroke: #2563eb !important;
    fill: #2563eb !important;
    color: #2563eb !important;
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def extract_text_from_bytes(file_bytes):
    return extract_text(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def chunk_document(text, chunk_size, chunk_overlap):
    return create_chunks(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


@st.cache_resource(show_spinner=False)
def get_cached_embeddings():
    return get_embedding_model()


@st.cache_resource(show_spinner=False)
def build_vector_store(chunks_tuple, doc_key):
    embeddings = get_cached_embeddings()
    persist_path = f"chroma_db/{doc_key}"
    return create_vector_store(list(chunks_tuple), embeddings, persist_directory=persist_path)


def update_state(key, value):
    st.session_state[key] = value


st.markdown(
    """
    <div class="hero">
        <h1>AI Study Buddy</h1>
        <p>Turn heavy PDF notes into focused revision: ask questions, generate summaries, and test yourself in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <span class="chip">PDF to Notes</span>
    <span class="chip">RAG Q&A</span>
    <span class="chip">Auto Summary</span>
    <span class="chip">Quiz Generation</span>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Study Setup")
    chunk_size = st.slider("Chunk size", min_value=500, max_value=1800, value=1000, step=100)
    max_overlap = max(50, chunk_size - 100)
    chunk_overlap = st.slider("Chunk overlap", min_value=50, max_value=max_overlap, value=min(200, max_overlap), step=25)
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=6, value=3, step=1)
    preview_chars = st.slider("Preview length", min_value=600, max_value=6000, value=2400, step=200)
    show_context = st.checkbox("Show retrieved context", value=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
st.markdown("</div>", unsafe_allow_html=True)

if not uploaded_file:
    st.info("Upload a PDF to start generating summaries, quizzes, and answers.")
    st.stop()

file_bytes = uploaded_file.getvalue()
if not file_bytes:
    st.error("Uploaded file appears empty. Please upload a valid PDF.")
    st.stop()

with st.spinner("Extracting text from PDF..."):
    text = extract_text_from_bytes(file_bytes)

if not text or not text.strip():
    st.error("No readable text was extracted from this PDF. Try another file.")
    st.stop()

with st.spinner("Preparing searchable chunks..."):
    chunks = chunk_document(text, chunk_size, chunk_overlap)



if not chunks:
    st.error("No valid text chunks found in uploaded document.")
    st.stop()

doc_key = hashlib.sha1(file_bytes).hexdigest()[:12]

with st.spinner("Building vector index..."):
    db = build_vector_store(tuple(chunks), doc_key)

st.success(f"Document ready: {uploaded_file.name}")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="metric"><div class="label">Characters</div><div class="value">{len(text):,}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric"><div class="label">Chunks</div><div class="value">{len(chunks):,}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric"><div class="label">Retriever Top-K</div><div class="value">{top_k}</div></div>', unsafe_allow_html=True)

summary_tab, quiz_tab, qa_tab, preview_tab = st.tabs(["Summary", "Quiz", "Ask AI", "Preview"])

with summary_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if st.button("Generate Summary", use_container_width=True):
        with st.spinner("Generating summary..."):
            try:
                update_state(f"summary_{doc_key}", generate_summary(text))
            except ResourceExhausted as exc:
                st.warning(f"Gemini quota is temporarily exhausted. {exc}")

    saved_summary = st.session_state.get(f"summary_{doc_key}")
    if saved_summary:
        st.markdown(saved_summary)
    st.markdown("</div>", unsafe_allow_html=True)

with quiz_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Generate Quiz Button
    if st.button("Generate Quiz", use_container_width=True):
        with st.spinner("Generating quiz..."):
            try:
                quiz_data = generate_quiz(text)
                st.session_state[f"quiz_{doc_key}"] = quiz_data
            except ResourceExhausted as exc:
                st.warning(f"Gemini quota is temporarily exhausted. {exc}")
            except Exception as e:
                st.error(f"Error generating quiz: {e}")

    saved_quiz = st.session_state.get(f"quiz_{doc_key}")

    if saved_quiz:

        # ===============================
        # CASE 1: Quiz is JSON / dict
        # ===============================
        if isinstance(saved_quiz, dict):

            # MCQs
            st.subheader("📝 Multiple Choice Questions")

            for i, mcq in enumerate(saved_quiz.get("mcqs", []), 1):
                st.markdown(f"### Q{i}. {mcq['question']}")

                options = [
                    f"A. {mcq['options']['A']}",
                    f"B. {mcq['options']['B']}",
                    f"C. {mcq['options']['C']}",
                    f"D. {mcq['options']['D']}",
                ]

                st.radio(
                    "Choose one:",
                    options,
                    key=f"mcq_{i}"
                )

                if st.button(f"Show Answer {i}", key=f"ans_{i}"):
                    st.success(
                        f"Correct Answer: {mcq['correct_answer']}"
                    )

                st.divider()

            # True / False
            st.subheader("✅ True / False")

            for i, tf in enumerate(saved_quiz.get("true_false", []), 1):
                st.markdown(f"**{i}. {tf['question']}**")
                st.radio(
                    "Select",
                    ["True", "False"],
                    key=f"tf_{i}"
                )

            # Fill in the blanks
            st.subheader("✍ Fill in the Blanks")

            for i, blank in enumerate(saved_quiz.get("fill_blanks", []), 1):
                st.markdown(f"**{i}. {blank['question']}**")
                st.text_input(
                    "Your Answer",
                    key=f"blank_{i}"
                )

        # ===============================
        # CASE 2: Gemini returned STRING
        # ===============================
        else:
            st.warning("Gemini returned plain text. Formatting output...")

            formatted = saved_quiz

            # Basic beautification
            for i in range(1, 11):
                formatted = formatted.replace(
                    f"\n{i}.",
                    f"\n\n### Q{i}."
                )

            formatted = formatted.replace("* ", "\n• ")

            st.markdown(formatted)

    st.markdown("</div>", unsafe_allow_html=True)

with qa_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    with st.form("qa_form"):
        question = st.text_input(
            "Ask a question about this document",
            placeholder="What are the key takeaways from this chapter?",
        )
        submit_question = st.form_submit_button(
            "Get Answer",
            use_container_width=True
        )

    if submit_question and question.strip():


        # ===============================
        # Similarity Search
        # ===============================
        with st.spinner("Searching relevant chunks..."):
            try:
                

                results = db.similarity_search(question, k=top_k)

                st.write("Similarity search finished")
                st.write("Retrieved chunks:", len(results))

                context = "\n\n".join(
                    [doc.page_content for doc in results]
                )

                st.write("Context length:", len(context))

            except Exception as e:
                st.error(f"Similarity search failed: {e}")
                results = []
                context = ""

        # ===============================
        # Gemini Answer
        # ===============================
        if context:
            with st.spinner("Generating answer..."):
                try:
                    st.write("Calling Gemini")

                    answer = get_answer(context, question)

                    

                    if not answer:
                        st.warning("Gemini returned empty response")
                    else:
                        st.markdown(
                            f"""
                            <div style="
                                background:white;
                                padding:22px;
                                border-radius:16px;
                                border-left:5px solid #2563eb;
                                box-shadow:0 8px 25px rgba(0,0,0,0.06);
                                line-height:1.8;
                                font-size:16px;
                                color:#111827;
                            ">
                            <b>&#129302; AI Answer</b><br><br>
                            {answer}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    update_state(
                        f"last_results_{doc_key}",
                        results
                    )

                except ResourceExhausted as exc:
                    st.warning(
                        f"Gemini quota exhausted: {exc}"
                    )

                except Exception as e:
                    st.error(f"Gemini failed: {e}")

        else:
            st.warning("No context found for this question.")

    if show_context:
        last_results = st.session_state.get(
            f"last_results_{doc_key}",
            []
        )

        if last_results:
            with st.expander(
                "View Retrieved Context",
                expanded=False
            ):
                for idx, doc in enumerate(last_results, start=1):
                    st.markdown(
                        f"""
                        <div class="chunk-card">
                            <b>Chunk {idx}</b><br>
                            {doc.page_content}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    st.markdown("</div>", unsafe_allow_html=True)

with preview_tab:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.text_area(
        "Extracted text preview",
        value=text[:preview_chars],
        height=380,

        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)


















