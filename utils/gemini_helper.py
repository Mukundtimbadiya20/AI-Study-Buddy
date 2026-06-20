import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Normal text model
text_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# JSON model (only for quiz)
json_model = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config={
        "temperature": 0.2,
        "response_mime_type": "application/json"
    }
)


def get_answer(context, question):
    prompt = f"""
You are an expert study assistant helping engineering students.

Answer the question in a detailed but easy-to-understand way.

Rules:
1. Use ONLY the provided context.
2. Answer in 4-8 sentences if enough information exists.
3. Explain concepts like a teacher.
4. Use bullet points if helpful.
5. If the context has only partial information, mention that and answer with available details.
6. Never give one-line answers unless question is yes/no.

If information is not available, reply exactly:
Information not found in uploaded document.

Context:
{context}

Question:
{question}

Detailed Answer:
"""
    response = text_model.generate_content(prompt)
    return response.text