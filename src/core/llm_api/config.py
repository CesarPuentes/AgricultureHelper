import os
import google.generativeai as genai

def _configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        return True
    return False

def is_available() -> bool:
    """
    Returns True if GEMINI_API_KEY is configured in the environment.
    """
    return bool(os.getenv("GEMINI_API_KEY"))
