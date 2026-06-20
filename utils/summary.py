from utils.gemini_helper import text_model
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

    response = text_model.generate_content(prompt)

    return response.text
