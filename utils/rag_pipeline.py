import io
import fitz                          # PyMuPDF for PDF text extraction
import faiss
import numpy as np
from models.embeddings import embed_texts
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS

# Global in-memory stores
_faiss_index   = None
_doc_store     = []       # [{"text": ..., "source": ...}, ...]
_indexed_files = set()    # tracks filenames already indexed to prevent duplicates


def _get_index(dim: int):
    """
    Creates FAISS index on first call using actual embedding dimension.
    IndexFlatIP = cosine similarity (works when vectors are L2-normalized).
    """
    global _faiss_index
    if _faiss_index is None:
        _faiss_index = faiss.IndexFlatIP(dim)
    return _faiss_index


def _ocr_page(page) -> str:
    """OCR fallback for scanned/image-based PDF pages."""
    try:
        import pytesseract
        from PIL import Image
        pix = page.get_pixmap(dpi=200)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


def extract_text_from_pdf(file) -> str:
    """
    Extracts text from uploaded PDF.
    Falls back to block-level extraction then OCR if no text layer found.
    """
    try:
        doc        = fitz.open(stream=file.read(), filetype="pdf")
        text_parts = []

        for page in doc:
            text = page.get_text("text").strip()
            if text:
                text_parts.append(text)
            else:
                # Try block-level before OCR
                blocks     = page.get_text("blocks")
                block_text = " ".join(b[4] for b in blocks if b[6] == 0).strip()
                if block_text:
                    text_parts.append(block_text)
                else:
                    ocr = _ocr_page(page)
                    if ocr.strip():
                        text_parts.append(ocr)

        full_text = "\n\n".join(text_parts).strip()

        if not full_text:
            raise ValueError(
                "Could not extract text. PDF appears image-based. "
                "Download from arXiv for a text-based version."
            )
        return full_text

    except ValueError:
        raise
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {e}")


def chunk_text(text: str) -> list[str]:
    """
    Splits text into overlapping chunks.
    Overlap ensures context at chunk boundaries is not lost.
    """
    chunks, start = [], 0
    while start < len(text):
        end   = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 100:    # filter chunks too short to be meaningful
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def add_document_to_db(filename: str, file) -> int:
    """
    Full RAG indexing pipeline:
    PDF → text → chunks → BGE embeddings → FAISS index.
    Skips silently if file already indexed this session.
    """
    global _doc_store, _indexed_files

    # Prevent double-indexing same file
    if filename in _indexed_files:
        return 0

    try:
        text   = extract_text_from_pdf(file)
        chunks = chunk_text(text)

        if not chunks:
            raise ValueError("No valid chunks could be extracted")

        # Embed all chunks in one batch
        embeddings = embed_texts(chunks)

        if not embeddings:
            raise ValueError("Embedding returned empty result")

        # Convert to float32 — required by FAISS
        vectors = np.array(embeddings, dtype="float32")

        # Infer dimension from actual embeddings and init FAISS index
        index = _get_index(dim=vectors.shape[1])
        index.add(vectors)

        # Store metadata aligned with FAISS vector indices
        for chunk in chunks:
            _doc_store.append({"text": chunk, "source": filename})

        _indexed_files.add(filename)
        return len(chunks)

    except Exception as e:
        raise RuntimeError(f"Indexing failed: {e}")


def retrieve_relevant_chunks(query: str) -> list[dict]:
    """
    Embeds the query and searches FAISS for top-K most similar chunks.
    Returns empty list gracefully if no docs indexed yet.
    """
    try:
        if _faiss_index is None or _faiss_index.ntotal == 0:
            return []

        # is_query=True applies BGE prefix for better retrieval accuracy
        query_vector           = np.array(embed_texts([query], is_query=True), dtype="float32")
        distances, indices     = _faiss_index.search(query_vector, TOP_K_RESULTS)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(_doc_store):   # skip empty FAISS slots
                continue
            entry = _doc_store[idx]
            results.append({
                "text":      entry["text"],
                "source":    entry["source"],
                "relevance": round(float(dist), 3)
            })
        return results

    except Exception:
        return []


def get_indexed_files() -> list[str]:
    """Returns list of filenames currently indexed — used by app.py sidebar."""
    return list(_indexed_files)