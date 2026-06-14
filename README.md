# 📚 AI Study Buddy

An AI-powered Study Assistant that helps students learn smarter by allowing them to upload study materials and interact with them using Retrieval-Augmented Generation (RAG).

The application can answer questions from uploaded PDFs, generate concise summaries, and create quizzes automatically using Large Language Models (LLMs).

---

## 🚀 Features

### 📄 PDF Upload & Processing

- Upload PDF study materials
- Extract text automatically
- Process multi-page documents

### 🔍 Intelligent Document Search

- Semantic search using vector embeddings
- Context-aware retrieval
- Fast document querying

### 🤖 AI Question Answering (RAG)

- Ask questions from uploaded documents
- Answers generated using retrieved context
- Reduces hallucinations through Retrieval-Augmented Generation

### 📝 Smart Summary Generator

- Generate concise study notes
- Key concepts extraction
- Revision-ready summaries

### 🧠 Quiz Generator

- Multiple Choice Questions (MCQs)
- True/False Questions
- Fill in the Blanks
- Automated answer generation

### 🔄 Multi-LLM Support (Upcoming)

- Google Gemini
- Ollama Local Models
- Llama 3.2
- Mistral
- Phi-3
- Gemma

---

## 🏗️ System Architecture

```text
User Upload PDF
        │
        ▼
Text Extraction
        │
        ▼
Text Chunking
        │
        ▼
Embedding Generation
        │
        ▼
ChromaDB Vector Store
        │
        ▼
Similarity Search
        │
        ▼
Relevant Context Retrieval
        │
        ▼
Large Language Model
        │
        ▼
Final Response
```

---

## 🔍 RAG Workflow

```text
Question
   │
   ▼
Embedding Conversion
   │
   ▼
ChromaDB Search
   │
   ▼
Top Relevant Chunks
   │
   ▼
LLM (Gemini / Ollama)
   │
   ▼
Answer Generation
```

---

## 🛠️ Tech Stack

### Frontend

- Streamlit

### Backend

- Python

### LLM

- Google Gemini API
- Ollama (Planned)

### Vector Database

- ChromaDB

### Embedding Model

- all-MiniLM-L6-v2

### Frameworks

- LangChain

### PDF Processing

- PyPDF

### Environment Management

- Python Virtual Environment
- dotenv

---

## 📂 Project Structure

```text
AIStudyBuddy/

├── app.py
├── utils/
│   ├── pdf_reader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── gemini_helper.py
│   ├── summary.py
│   └── quiz.py
├── uploads/
├── chroma_db/
├── .env
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/AI-Study-Buddy.git

cd AI-Study-Buddy
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

Application will open automatically in your browser.

---

## 📖 Usage

### Step 1

Upload a PDF document.

### Step 2

Wait for:

- Text Extraction
- Chunk Creation
- Embedding Generation
- Vector Database Creation

### Step 3

Ask questions from your document.

Examples:

```text
What is NLP?

Explain Machine Translation.

What are the course outcomes?

Summarize the syllabus.
```

### Step 4

Generate:

- Summary
- Quiz
- Revision Notes

---

## 🎯 Current Features

- PDF Upload
- Text Extraction
- Chunking
- Embeddings
- ChromaDB Integration
- Semantic Search
- RAG Question Answering
- Summary Generation
- Quiz Generation

---

## 🚧 Future Enhancements

- Multi-LLM Support
- Ollama Integration
- Local Model Selection
- Voice-to-Text
- Flashcard Generation
- Chat History
- User Authentication
- Multi-PDF Support
- PDF Highlighting
- Export Summary to PDF

---

## 🎓 Learning Outcomes

This project demonstrates:

- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Vector Databases
- Semantic Search
- Embedding Models
- Prompt Engineering
- LangChain
- Streamlit Development
- AI Application Deployment

---

## 👨‍💻 Author

Mukund Timbadiya

AI & Machine Learning Student

Passionate about:

- Artificial Intelligence
- Machine Learning
- Generative AI
- LLM Applications
- MLOps

---

## ⭐ If you found this project useful

Give it a star on GitHub and share it with others.
