import streamlit as st

from utils.pdf_reader import extract_text
from utils.chunker import create_chunks
from utils.embeddings import get_embedding_model
from utils.vector_store import create_vector_store
from google.api_core.exceptions import ResourceExhausted
from utils.gemini_helper import get_answer
from utils.summary import generate_summary
from utils.quiz import generate_quiz

st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="📚",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0f1117;
        font-family: 'Segoe UI', sans-serif;
        color: #f0f2f6;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #6a5cff 0%, #9b7bff 100%);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 1.75rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(106, 92, 255, 0.25);
    }

    .app-header h1 {
        margin-bottom: 0.35rem;
        font-size: 2.4rem;
        color: #ffffff;
    }

    .app-header p {
        margin: 0;
        opacity: 0.95;
        font-size: 1.05rem;
        color: #f0eaff;
    }

    /* Section cards */
    .section-card {
        background-color: #1a1d29;
        padding: 1.75rem;
        border-radius: 16px;
        box-shadow: 0 2px 14px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.5rem;
        border: 1px solid #2a2e3f;
    }

    .section-card h3, .section-card h4 {
        color: #ffffff;
        margin-top: 0;
    }

    /* General text contrast fixes */
    .stApp p, .stApp span, .stApp label, .stApp div {
        color: #f0f2f6;
    }

    .stMarkdown, .stMarkdown p, .stMarkdown li {
        color: #f0f2f6 !important;
    }

    /* Captions */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #a0a6b8 !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.65rem 1.3rem;
        border: none;
        background: linear-gradient(135deg, #6a5cff 0%, #8f7bff 100%);
        color: #ffffff;
        transition: transform 0.12s ease-in-out, box-shadow 0.12s ease-in-out;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(106, 92, 255, 0.4);
        color: #ffffff;
    }

    .stButton > button p {
        color: #ffffff !important;
        font-weight: 600;
    }

    /* Text inputs */
    .stTextInput > div > div > input {
        border-radius: 10px;
        background-color: #20242f;
        color: #f0f2f6;
        border: 1px solid #383d4f;
    }

    .stTextInput > div > div > input::placeholder {
        color: #8088a0;
    }

    /* Text area */
    .stTextArea textarea {
        background-color: #20242f;
        color: #f0f2f6;
        border: 1px solid #383d4f;
        border-radius: 10px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #1a1d29;
        border: 1px dashed #4a4f66;
        border-radius: 12px;
        padding: 1rem;
    }

    [data-testid="stFileUploader"] section {
        background-color: #1a1d29;
    }

    [data-testid="stFileUploaderDropzone"] div {
        color: #f0f2f6 !important;
    }

    /* Expander */
    .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        background-color: #20242f !important;
        color: #f0f2f6 !important;
        border-radius: 10px !important;
        font-weight: 600;
    }

    [data-testid="stExpander"] {
        background-color: #1a1d29;
        border: 1px solid #2a2e3f;
        border-radius: 10px;
    }

    /* Alerts: info / success */
    .stAlert {
        border-radius: 10px;
    }

    div[data-baseweb="notification"] p {
        color: #0f1117 !important;
        font-weight: 500;
    }

    /* Chunk cards */
    .chunk-card {
        background-color: #20243a;
        border-left: 4px solid #8f7bff;
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        margin-bottom: 0.7rem;
        color: #f0f2f6;
    }

    .chunk-card b {
        color: #b3a3ff;
    }

    /* Subheaders */
    h1, h2, h3, h4, h5 {
        color: #f5f6fa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>📚 AI Study Buddy</h1>
        <p>Upload a PDF, get instant summaries, quizzes, and answers to your questions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# File upload section
# ---------------------------------------------------------------------------
text = ""

st.markdown('<div class="section-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"],
)

if uploaded_file:
    st.success("✅ PDF Uploaded Successfully!")

    text = extract_text(uploaded_file)

    with st.expander("📖 Extracted Text Preview", expanded=False):
        st.text_area(
            "Content",
            text[:5000],
            height=300,
            label_visibility="collapsed",
        )
else:
    st.info("👆 Upload a PDF to extract text, generate summaries, quizzes, and answers.")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Action buttons: Summary & Quiz
# ---------------------------------------------------------------------------
if uploaded_file:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🚀 Quick Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📝 Generate Summary", use_container_width=True):
            with st.spinner("Generating Summary..."):
                try:
                    summary = generate_summary(text)
                except ResourceExhausted as exc:
                    st.warning(f"Gemini quota is temporarily exhausted. {exc}")
                else:
                    st.subheader("📋 Document Summary")
                    st.write(summary)

    with col2:
        if st.button("🧠 Generate Quiz", use_container_width=True):
            with st.spinner("Generating Quiz..."):
                try:
                    quiz = generate_quiz(text)
                except ResourceExhausted as exc:
                    st.warning(f"Gemini quota is temporarily exhausted. {exc}")
                else:
                    st.subheader("📝 Generated Quiz")
                    st.markdown(quiz)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Q&A section
# ---------------------------------------------------------------------------
if uploaded_file and text:
    chunks = create_chunks(text)

    embeddings = get_embedding_model()
    db = create_vector_store(
        chunks,
        embeddings,
    )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 💬 Ask Your Document")
    st.caption(f"Document indexed into {len(chunks)} searchable chunks.")

    query = st.text_input(
        "Ask a question about your PDF",
        placeholder="e.g. What are the key takeaways of chapter 2?",
    )

    if query:
        with st.spinner("Searching document..."):
            results = db.similarity_search(
                query,
                k=3,
            )
            context = "\n\n".join(
                [doc.page_content for doc in results]
            )

        with st.spinner("Generating answer..."):
            try:
                answer = get_answer(
                    context,
                    query,
                )
            except ResourceExhausted as exc:
                st.warning(f"Gemini quota is temporarily exhausted. {exc}")
            else:
                st.markdown("#### 🤖 AI Answer")
                st.success(answer)

        with st.expander("🔍 View Retrieved Context"):
            for i, doc in enumerate(results):
                st.markdown(
                    f'<div class="chunk-card"><b>Chunk {i + 1}</b><br>{doc.page_content}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)


