from utils.gemini_helper import model

def generate_quiz(text):
    prompt = f"""
Generate:

- 10 MCQs
- 5 True/False Questions
- 5 Fill in the Blanks

from the provided study material.

For MCQs provide:
Question
4 Options
Correct Answer

Document:
{text}
"""

    response = model.generate_content(prompt)

    return response.text