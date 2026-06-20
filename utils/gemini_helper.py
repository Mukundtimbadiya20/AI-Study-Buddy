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
    You are an expert study assistant.

    Answer in natural readable text.
    Do NOT return JSON.
    Do NOT use brackets or dictionaries.

    Use only the provided context.

    If information is not available,
    reply:
    Information not found in uploaded document.

    Context:
    {context}

    Question:
    {question}
    """

    response = text_model.generate_content(prompt)
    return response.text