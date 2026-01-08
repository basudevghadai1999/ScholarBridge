import json
import re

def extract_json(text: str) -> dict:
    """
    Robustly extracts JSON from a string, handling markdown blocks and surrounding text.
    """
    # 1. Try straightforward parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Try removing markdown fences
    clean_text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        pass

    # 3. Regex search for object {...}
    # This matches the first outer opening brace and the last closing brace
    try:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except (json.JSONDecodeError, AttributeError):
        pass

    # 4. Fail
    print(f"Failed to extract JSON from: {text[:100]}...")
    return {}
