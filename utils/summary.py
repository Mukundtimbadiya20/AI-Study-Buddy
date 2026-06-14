from utils.gemini_helper import model

def generate_summary(text):

    prompt = f"""
    Summarize the following document.

    Provide:
    1. Key Concepts
    2. Important Topics
    3. Short Revision Notes

    Document:
    {text}
    """

    response = model.generate_content(prompt)

    return response.text