import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

def get_answer(context, question):

    prompt = f"""
    You are an expert study assistant.

    Use only the provided context.

    If information is not available,
    reply:
    'Information not found in uploaded document.'

    Context:
    {context}

    Question:
    {question}
    """

    response = model.generate_content(prompt)

    return response.text