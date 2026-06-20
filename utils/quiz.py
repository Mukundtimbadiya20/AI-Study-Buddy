from utils.gemini_helper import json_model
import json
import re


def generate_quiz(text):
    prompt = f"""
Generate study questions from the provided study material.

Return STRICT JSON ONLY.
No markdown.
No code blocks.
No extra explanation.

JSON format:
{{
  "mcqs": [
    {{
      "question": "Question text",
      "options": {{
        "A": "Option A",
        "B": "Option B",
        "C": "Option C",
        "D": "Option D"
      }},
      "correct_answer": "A"
    }}
  ],
  "true_false": [
    {{
      "question": "Statement text",
      "answer": true
    }}
  ],
  "fill_blanks": [
    {{
      "question": "_____ is example.",
      "answer": "test"
    }}
  ]
}}

Study Material:
{text}
"""

  

    response = json_model.generate_content(prompt)
    raw = response.text

    print("RAW:", raw)

    try:
        return json.loads(raw)
    except:
        cleaned = raw.replace("```json", "").replace("```", "").strip()

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass

    return raw

    print("RAW GEMINI OUTPUT:", raw_text)

    # Remove ```json ... ```
    cleaned = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        quiz_data = json.loads(cleaned)
        return quiz_data
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                return None

    return None