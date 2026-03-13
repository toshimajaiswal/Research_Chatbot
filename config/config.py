import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def _get_secret(key: str) -> str:
    """Gets secret from Streamlit Cloud secrets or local .env — whichever is available."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, "")

GROQ_API_KEY    = _get_secret("GROQ_API_KEY")
GROQ_MODEL      = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE      = 600
CHUNK_OVERLAP   = 75
TOP_K_RESULTS   = 3