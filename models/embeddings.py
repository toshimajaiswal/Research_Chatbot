from sentence_transformers import SentenceTransformer
from config.config import EMBEDDING_MODEL

_model = None

def get_embedding_model() -> SentenceTransformer:
    """Singleton — loads BGE-small once and reuses it across all calls."""
    global _model
    try:
        if _model is None:
            _model = SentenceTransformer(EMBEDDING_MODEL)
        return _model
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {e}")

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    BGE models perform better with a query prefix for retrieval tasks.
    We add it only for single queries, not for document chunks.
    """
    try:
        if not texts:
            raise ValueError("embed_texts received empty list")

        model = get_embedding_model()

        # BGE-specific: prefix single queries with instruction for better retrieval
        if len(texts) == 1:
            texts = [f"Represent this sentence for searching relevant passages: {texts[0]}"]

        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Embedding model returned empty result")

        return embeddings.tolist()
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")