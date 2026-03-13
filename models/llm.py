from langchain_groq import ChatGroq
from config.config import GROQ_API_KEY, GROQ_MODEL

def get_chatgroq_model(mode: str = "detailed") -> ChatGroq:
    """
    Returns a LangChain-compatible Groq chat model.
    mode controls temperature and token output length.
    """
    try:
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=0.3 if mode == "concise" else 0.7,
            max_tokens=200  if mode == "concise" else 1000,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load Groq model: {e}")